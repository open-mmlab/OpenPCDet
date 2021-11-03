from .detector3d_template import Detector3DTemplate
import os
import gc
import torch
import math
import time
import copy
import json
import numpy as np

class PointPillarImprecise(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.vfe, self.map_to_bev, self.backbone_2d, self.dense_head = self.module_list

        # imprecise computation specific
        #self._kitti = (dataset.dataset_cfg.DATASET == 'KittiDataset') # not needed
        if 'SEPARATE_MULTIHEAD' in dir(model_cfg.DENSE_HEAD):
            self._sep_mhead = model_cfg.DENSE_HEAD.SEPARATE_MULTIHEAD
        else:
            self._sep_mhead = False

        # available methods:
        self.BASELINE1 = 1
        self.BASELINE2 = 2
        self.BASELINE3 = 3
        self.IMP_NOSLICE = 4
        self.IMP_SLICE = 5
        self.SKIP_NOSLICE = 6
        #self.SKIP_SLICE = 7

        self._default_method = int(model_cfg.METHOD)
        print('Default method is:', self._default_method)
        self._enable_slicing = not (self._sep_mhead or self._default_method >= self.SKIP_NOSLICE)

        self._eval_dict['method'] = self._default_method
        self._eval_dict['rpn_stg_exec_seqs'] = []

        # Stuff that might come as a parameter inside data_dict

        # Stuff that might pass using the model conf
        #self._merge_preds=True  # Run NMS on each slice if False
        self._slice_size_perc = 27.77777777777778
        self._min_slice_overlap_perc = 2 
        self._measure_time = True
        self._cudnn_benchmarking=True
        self._cudnn_deterministic=False

        # Auxiliary variables
        self._H_dim = 3 
        self._cls_H_dim = 2
        self._W_dim = 2
        self._calibration = False
        if self._default_method < self.SKIP_NOSLICE:
            self._num_stages = len(self.backbone_2d.num_bev_features)
        else:
            self._num_stages = len(self.backbone_2d.layer_nums)

        self._score_threshold= float(model_cfg.POST_PROCESSING.SCORE_THRESH)

        # Debug/plot related
        self._pc_range = dataset.point_cloud_range

        # determinism
        torch.backends.cudnn.benchmark = self._cudnn_benchmarking
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        if self._cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.set_deterministic(True)

        self.update_time_dict( {
                'VFE': [], #'PillarFeatureNet': [],
                'MapToBEV': [], #'PillarScatter': [],
                'AnchorMask': [],
                'RPN-stage-1': [],
                'RPN-stage-2': [],
                'RPN-stage-3': [],
                'RPN-finalize': [],
                'RPN-total': [],
                'Pre-stage-1': [],
                'Post-stage-1': [],
                'PostProcess': [],})

        print('Model:')
        print(self)

    def forward(self, data_dict):
        if not self.training:
            return self.eval_forward(data_dict)
        else:
            data_dict = self.vfe(data_dict)
            data_dict = self.map_to_bev(data_dict)
            data_dict["stage0"] = data_dict["spatial_features"]
            data_dict["stages_executed"] = 0
            # Execute all stages
            losses=[]

            if  self._default_method <  self.SKIP_NOSLICE:
                for s in range(self._num_stages):
                    data_dict = self.backbone_2d(data_dict)
                    data_dict = self.dense_head(data_dict)
                    if self.dense_head.predict_boxes_when_training:
                        data_dict = self.dense_head.gen_pred_boxes(data_dict)
                    loss, tb_dict, disp_dict = self.get_training_loss()
                    losses.append(loss)
                # Tried to implement training method in ABC paper
                tperc = self.dataset.cur_epochs / self.dataset.total_epochs
                if tperc <= 0.25:
                    weights = [0.98, 0.01, 0.01]
                elif tperc <= 0.5:
                    weights = [0.1, 0.8, 0.1]
                elif tperc <= 0.75:
                    weights = [0.1, 0.2, 0.7]
                else:
                    weights = [0.05, 0.15, 0.85]
                total_loss = .0
                for i, l in enumerate(losses):
                    total_loss += l * weights[i]
            else:
                data_dict = self.backbone_2d(data_dict)
                data_dict = self.dense_head(data_dict)
                if self.dense_head.predict_boxes_when_training:
                    data_dict = self.dense_head.gen_pred_boxes(data_dict)
                loss, tb_dict, disp_dict = self.get_training_loss()
                total_loss = loss

            ret_dict = {
                'loss': total_loss,
            }

            return ret_dict, tb_dict, disp_dict

    def eval_forward(self, data_dict):
        # defaults
        if 'method' not in data_dict:
            data_dict['method'] = self._default_method

        if data_dict['method'] < self.IMP_SLICE:
            return self.noslicing_forward(data_dict)
        if data_dict['method'] == self.IMP_SLICE:
            return self.slicing_forward(data_dict)
        elif data_dict['method'] == self.SKIP_NOSLICE:
            return self.skip_noslicing_forward(data_dict)

    def pre_rpn_forward(self, data_dict):
        self.measure_time_start('VFE')
        data_dict = self.vfe(data_dict) # pillar feature net
        self.measure_time_end('VFE')
        self.measure_time_start('MapToBEV')
        data_dict = self.map_to_bev(data_dict) # pillar scatter
        self.measure_time_end('MapToBEV')
        data_dict["stage0"] = data_dict["spatial_features"]
        data_dict["stages_executed"] = 0
        return data_dict

    # Can run baselines and imprecise noslice
    def noslicing_forward(self, data_dict):
        self.measure_time_start('Pre-stage-1')
        data_dict = self.pre_rpn_forward(data_dict)

        self.measure_time_start('RPN-total')
        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-stage-1')
        stg_seq=[1]
        #data_dict['score_thresh'] = 0.1
        
        if data_dict['method'] == self.IMP_NOSLICE:
            #torch.cuda.synchronize()
            num_stgs_to_run = self.sched_stages(data_dict['abs_deadline_sec'])
        else:
            num_stgs_to_run = data_dict['method']
            
        self.measure_time_end('Pre-stage-1')
        self.measure_time_start('Post-stage-1')
        
        if num_stgs_to_run >= 2:
            self.measure_time_start('RPN-stage-2')
            data_dict = self.backbone_2d(data_dict)
            self.measure_time_end('RPN-stage-2')
            stg_seq.append(2)
            #data_dict['score_thresh'] = 0.2

        if num_stgs_to_run == 3:
            self.measure_time_start('RPN-stage-3')
            data_dict = self.backbone_2d(data_dict)
            self.measure_time_end('RPN-stage-3')
            stg_seq.append(3)
            #data_dict['score_thresh'] = 0.1

        self.measure_time_end('RPN-total')
        self.measure_time_start('RPN-finalize')
        #torch.cuda.nvtx.range_push('Head')
        data_dict = self.dense_head(data_dict)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end('RPN-finalize')

        self.measure_time_start("PostProcess")
        # Now do postprocess and finish
        #torch.cuda.nvtx.range_push('PostProcess')
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        torch.cuda.nvtx.range_pop()
        self.measure_time_end("PostProcess")
        self.measure_time_end("Post-stage-1")
        self._eval_dict['rpn_stg_exec_seqs'].append(stg_seq)
        return det_dicts, recall_dict

    def skip_noslicing_forward(self, data_dict):
        data_dict = self.pre_rpn_forward(data_dict)
        #torch.cuda.synchronize()
        #num_layers_to_run = self.sched_stages(data_dict['abs_deadline_sec'])
        #num_layers_to_run = self.backbone_2d.layer_nums.copy()
        num_layers_to_run = [3, 5, 5]
        data_dict['layer_nums'] = num_layers_to_run

        self.measure_time_start('RPN-total')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-total')
        self.measure_time_start('RPN-finalize')
        #torch.cuda.nvtx.range_push('Head')
        data_dict = self.dense_head(data_dict)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end('RPN-finalize')

        self.measure_time_start("PostProcess")
        # Now do postprocess and finish
        #torch.cuda.nvtx.range_push('PostProcess')
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end("PostProcess")
        self._eval_dict['rpn_stg_exec_seqs'].append(num_layers_to_run)
        return det_dicts, recall_dict

    def slicing_forward(self, data_dict):
        self.measure_time_start('Pre-stage-1')
        data_dict = self.pre_rpn_forward(data_dict)

        # Calculate anchor mask
        self.measure_time_start('AnchorMask')
        stg0_sum = torch.sum(data_dict['stage0'], 1, keepdim=False)
        sum_mask = torch.nn.functional.max_pool2d(stg0_sum, 15,
                stride=int(self._stg0_pred_scale_rate), padding=7).type(torch.bool)
        #example['anchors_mask'] = sum_mask.expand(
        #        self._box_k:preds_size[:-1]).contiguous()
        sum_del_mask = torch.unsqueeze(torch.logical_not(sum_mask), -1)
        sum_del_mask = sum_del_mask.expand(self._cls_preds_size).contiguous()
        self.measure_time_end('AnchorMask')

        self.measure_time_start("RPN-total")
        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        data_dict = self.dense_head.forward_cls_preds(data_dict)

        # Calculate sum of class scores within each slice
        # but only use class scores positioned close to pillar locations
        # use stg0 to create the pillar mask which will be used for slicing
        # Apply sigmoid and mask values below nms threshold
        cls_scores = torch.sigmoid(data_dict["cls_preds"])
        cls_scores_del = cls_scores <= self._score_threshold

        cls_scores_del = torch.logical_or(cls_scores_del, sum_del_mask)
        cls_scores.masked_scatter_(cls_scores_del, self._pred_zeroer)
        cls_scores = torch.sum(cls_scores, [0, 1, 3]) # reduce to H, since class is dim 2
        csa = self.slice_with_ranges(cls_scores.cpu(), self._cls_scr_ranges)
        # LOOKS LIKE IT IS SYNCHED AT THIS POINT

        #slice
        slice_data_dicts = []
        for i in range(self._num_slices):
            slice_data_dicts.append( {} )

        # Get the cls mask of each slice, also the overlapped regions explicitly
        # stg1 class scores will be enough for everything
        cls_scr_sums = torch.empty(2 * len(slice_data_dicts) - 1,
                dtype=cls_scores.dtype, device='cpu')
        for i, cs in enumerate(csa):
            cls_scr_sums[i] = torch.sum(cs)

        zerocuk_tensor = cls_scr_sums.new_zeros((1,))
        slice_data_dicts[0]['cls_scores'] = torch.cat((zerocuk_tensor, cls_scr_sums[:2]))
        slice_data_dicts[-1]['cls_scores'] = torch.cat((cls_scr_sums[-2:], zerocuk_tensor))
        for i, dt_d in zip(range(1, len(cls_scr_sums)-2 ,2), slice_data_dicts[1:-1]):
            dt_d['cls_scores'] = cls_scr_sums[i:i+3]

        # I DON'T NEED TO CALL SYNC BECAUSE IT IS ALREADY SYNCED
        # FROM WHAT I SAW BUT DO IT ANYWAY, NO BIG LOSS
        torch.cuda.synchronize()

        # Now decide the slice forwarding pattern
        # This algorithm takes 0.5 ms
        slices_to_exec = self.sched_slices_v2(slice_data_dicts, data_dict['abs_deadline_sec'])
        stg2_slices, stg3_slices = slices_to_exec
        data_dict['score_thresh'] = 0.3
        if len(stg2_slices) > 0:
            data_dict['score_thresh'] -= 0.1
        if len(stg3_slices) > 0:
            data_dict['score_thresh'] -= 0.1

        stg_seq=[1]
        self.measure_time_end('RPN-stage-1')
        self.measure_time_end('Pre-stage-1') 
        self.measure_time_start('Post-stage-1')

        data_sliced=False
        if len(stg2_slices) == self._num_slices:
            # Since we are going to execute all slices,
            # Don't do slicing and run the whole stage
            self.measure_time_start("RPN-stage-2")
            data_dict = self.backbone_2d(data_dict)
            self.measure_time_end(f"RPN-stage-2")
        elif len(stg2_slices) > 0:
            data_sliced=True
            # Slice the tensors
            sa  = self.slice_with_ranges(data_dict["stage1"], self._stg1_slc_ranges)
            ua  = self.slice_with_ranges(data_dict["up1"], self._up1_slc_ranges)
            cpa = self.slice_preds_with_ranges(data_dict["cls_preds"], self._preds_slc_ranges)

            for i in range(self._num_slices):
                slice_data_dicts[i]["stages_executed"]       = 1
                slice_data_dicts[i]["stage1"]                = sa[i]
                slice_data_dicts[i]["up1"]                   = ua[i]
                slice_data_dicts[i]["spatial_features_2d_1"] = ua[i]
                slice_data_dicts[i]["cls_preds"]             = cpa[i]

            # We have slices to exec through stage 2
            # batch the chosen slices
            batch_data_dict = { "stages_executed": 1, }
            batch_data_dict["stage1"] = torch.cat(
                    [slice_data_dicts[s]["stage1"] for s in stg2_slices])
            batch_data_dict["up1"] = torch.cat(
                    [slice_data_dicts[s]["up1"] for s in stg2_slices])

            self.measure_time_start("RPN-stage-2")
            batch_data_dict = self.backbone_2d(batch_data_dict)
            self.measure_time_end(f"RPN-stage-2")

            # Scatter the results anyway
            #if len(stg3_slices) < len(stg2_slices):
            stg2_chunks = torch.chunk(batch_data_dict["stage2"], len(stg2_slices))
            up2_chunks = torch.chunk(batch_data_dict["up2"], len(stg2_slices))
            for i, s in enumerate(stg2_slices):
                slice_data_dicts[s]["stage2"] = stg2_chunks[i]
                slice_data_dicts[s]["up2"] = up2_chunks[i]
                slice_data_dicts[s]["stages_executed"] = 2

        stg_seq.extend([2] * len(stg2_slices))

        if len(stg3_slices) == self._num_slices:
            # data_sliced will be always false
            # at this point since spatial_features2d2 slices
            # will be also equal to _num_slices
            self.measure_time_start("RPN-stage-3")
            data_dict = self.backbone_2d(data_dict)
            self.measure_time_end(f"RPN-stage-3")
        elif len(stg3_slices) > 0: # that means stg2_slices was also > 0
            data_sliced=True
            if len(stg2_slices) == self._num_slices:
                # Slice the tensors if they were not sliced during stage 2
                sa  = self.slice_with_ranges(data_dict["stage2"], self._stg2_slc_ranges)
                ua1  = self.slice_with_ranges(data_dict["up1"], self._up1_slc_ranges)
                ua2  = self.slice_with_ranges(data_dict["up2"], self._up2_slc_ranges)

                for i in range(self._num_slices):
                    slice_data_dicts[i]["stage2"] = sa[i]
                    slice_data_dicts[i]["up1"]          = ua1[i]
                    slice_data_dicts[i]["up2"]          = ua2[i]
                    slice_data_dicts[i]["stages_executed"] = 2

                batch_data_dict = { "stages_executed": 2, }

            # We have slices to exec through stage 3
            # batch chosen slices
            batch_data_dict["stage2"] = torch.cat(
                    [slice_data_dicts[s]["stage2"] for s in stg3_slices])
            #batch_data_dict["up2"] = torch.cat(
            #        [slice_data_dicts[s]["up2"] for s in stg3_slices])

            self.measure_time_start("RPN-stage-3")
            batch_data_dict = self.backbone_2d(batch_data_dict)
            self.measure_time_end(f"RPN-stage-3")

            # Scatter the results
            up3_chunks = torch.chunk(batch_data_dict["up3"], len(stg3_slices))
            for i, s in enumerate(stg3_slices):
                slice_data_dicts[s]["up3"] = up3_chunks[i]
                slice_data_dicts[s]["stages_executed"] = 3

            stg_seq.extend([3] * len(stg3_slices))

        self.measure_time_end("RPN-total")
        self.measure_time_start("RPN-finalize")
        if not data_sliced:
            # No slicing were used
            if data_dict['stages_executed'] == 1:
                data_dict = self.dense_head.forward_remaining_preds(data_dict)
            else:
                data_dict = self.dense_head(data_dict)
            #preds_dict = data_dict
        else:
            # We used slicing, now we need to merge the slices
            # After detection heads
            # This part can be batched too but it is okay to
            # stay like this
            # Another optimization could be using cuda streams
            for i, dt_d in enumerate(slice_data_dicts):
                cur_stg = dt_d["stages_executed"] 
                if dt_d["stages_executed"] == 1:
                    # stage 1 slices already has cls preds
                    dt_d[f'stage{cur_stg}'] = \
                            dt_d[f'stage{cur_stg}'].contiguous()
                    dt_d = self.dense_head.forward_remaining_preds(dt_d)
                else:
                    dt_d = self.dense_head(dt_d)
                slice_data_dicts[i] = dt_d

            # if two overlapped regions went through same number of 
            # stages, get half of the overlapped region from each
            # neighboor io dict
            # Otherwise, select the one with more stages executed
            preds_dict = {}
            for k,v in self._pred_dict_copy.items():
                preds_dict[k] = v.clone().detach()

            # every slice has a big middle range and two (or one)
            # small overlap ranges
            slc_r = self._cls_scr_ranges[0]
            for k in preds_dict.keys():
                preds_dict[k][..., :slc_r[1], :] = \
                        slice_data_dicts[0][k][..., :slc_r[1], :]

            for i in range(len(slice_data_dicts)-1):
                dt_d1, dt_d2 = slice_data_dicts[i],  slice_data_dicts[i+1]
                se1, se2 = dt_d1["stages_executed"], dt_d2["stages_executed"]
                ovl_r = self._cls_scr_ranges[i*2 + 1]
                ovl_len = ovl_r[1] - ovl_r[0]
                for k in preds_dict.keys():
                    if se1 > se2:
                        preds_dict[k][..., ovl_r[0]:ovl_r[1], :] = \
                                dt_d1[k][..., -ovl_len:, :]
                    elif se1 < se2:
                        preds_dict[k][..., ovl_r[0]:ovl_r[1], :] = \
                                dt_d2[k][..., :ovl_len, :]
                    else:
                        mid = ovl_len//2
                        preds_dict[k][..., ovl_r[0]:(ovl_r[0]+mid), :] = \
                                dt_d1[k][..., -ovl_len:(-ovl_len+mid), :]
                        preds_dict[k][..., (ovl_r[0]+mid):ovl_r[1], :] = \
                                dt_d2[k][..., mid:ovl_len, :]
                    slc_r = self._cls_scr_ranges[i*2 + 2]
                    slc_len = slc_r[1] - slc_r[0]
                    preds_dict[k][..., slc_r[0]:slc_r[1], :] = \
                            dt_d2[k][..., ovl_len:(ovl_len+slc_len) , :]

            for k, v in preds_dict.items():
                preds_dict[k] = v.contiguous()

            data_dict.update(preds_dict)
        self.measure_time_end("RPN-finalize")

        self.measure_time_start("PostProcess")
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        # Now do postprocess and finish
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        self.measure_time_end("PostProcess")
        self.measure_time_end("Post-stage-1")
        self._eval_dict['rpn_stg_exec_seqs'].append(stg_seq)
        return det_dicts, recall_dict

    def slice_with_ranges(self, tensor, ranges):
        slices = [ tensor[..., b:e] for b, e in ranges ]
        return slices

    def slice_preds_with_ranges(self, tensor, ranges):
        slices = [ tensor[..., b:e, :] for b, e in ranges ]
        return slices

    def print_dict(self, d):
        for k, v in d.items():
            print(k, ':', end=' ')
            if torch.is_tensor(v):
                print(v.size())
            elif isinstance(v, list) and torch.is_tensor(v[0]):
                for e in v:
                    print(e.size(), end=' ')
                print()
            else:
                print(v)


    def calibrate(self):
        data_dict = self.load_data_with_ds_index(0)
        print('\ndata_dict:') 
        self.print_dict(data_dict)

        # just do a regular forward first
        data_dict = self.vfe(data_dict)
        data_dict = self.map_to_bev(data_dict)
        data_dict["stage0"] = data_dict["spatial_features"]
        data_dict["stages_executed"] = 0
        if  self._default_method <  self.SKIP_NOSLICE:
            for s in range(self._num_stages):
                data_dict = self.backbone_2d(data_dict)
        else:
            data_dict = self.backbone_2d(data_dict)
        data_dict = self.dense_head(data_dict)
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        det_dicts, recall_dict = self.post_processing(data_dict)

        #Print full tensor sizes
        print('\nTensors:') 
        self.print_dict(data_dict)

        print('\nDetections:')
        for pd in det_dicts:
            self.print_dict(pd)
        
        print('\nRecall dict:')
        self.print_dict(recall_dict)

        if self._sep_mhead:
            self._box_preds_size = [bp.size() for bp in data_dict['box_preds']]
            self._cls_preds_size = [cp.size() for cp in data_dict['cls_preds']]
        else:
            self._box_preds_size = data_dict['box_preds'].size()
            self._cls_preds_size = data_dict['cls_preds'].size()

        print('\nDense head return:\nbox_preds', self._box_preds_size)
        print('cls_preds', self._cls_preds_size)


        if self._default_method >= self.SKIP_NOSLICE:
            return
        
        # needed for anchor mask
        if self._sep_mhead:
            self._pred_zeroer = [torch.zeros_like(cp) for cp in data_dict['cls_preds']]
        else:
            self._pred_zeroer = torch.zeros_like(data_dict['cls_preds'])

        if self._enable_slicing:
            # this is needed for merging slices
            self._pred_dict_copy = {"cls_preds":None, "box_preds": None,
                        "dir_cls_preds": None}
            for k in self._pred_dict_copy.keys():
                self._pred_dict_copy[k] = torch.zeros_like(data_dict[k])

            det = det_dicts[0]
            self.init_empty_det_dict(det)

            pred_sz = self._box_preds_size[self._cls_H_dim]
            stg0_sz = data_dict["stage0"].size()[self._H_dim]
            self._stg0_pred_scale_rate = stg0_sz / pred_sz

            self._preds_slc_ranges, self._preds_ovl_slc_ranges = \
                    self.get_slice_ranges_v3(self._cls_preds_size[self._cls_H_dim])

            # This part is for slicing
            # Slicing can happen after stage 1 or stage 2
            stg1_sz = data_dict["stage1"].size()[self._H_dim]
            up1_sz = data_dict["up1"].size()[self._H_dim]
            scale_rate = stg1_sz / pred_sz
            self._stg1_slc_ranges = [(int(r[0] * scale_rate), int(r[1] * scale_rate))
                    for r in self._preds_slc_ranges]
            scale_rate = up1_sz / pred_sz
            self._up1_slc_ranges = [(int(r[0] * scale_rate), int(r[1] * scale_rate))
                    for r in self._preds_slc_ranges]

            stg2_sz = data_dict["stage2"].size()[self._H_dim]
            up2_sz = data_dict["up2"].size()[self._H_dim]
            scale_rate = stg2_sz / pred_sz
            self._stg2_slc_ranges = [(int(r[0] * scale_rate), int(r[1] * scale_rate))
                    for r in self._preds_slc_ranges]
            scale_rate = up2_sz / pred_sz
            self._up2_slc_ranges = [(int(r[0] * scale_rate), int(r[1] * scale_rate))
                    for r in self._preds_slc_ranges]

            self._num_slices = len(self._stg1_slc_ranges)
            # split overlapped regions for class scores
            posr = self._preds_ovl_slc_ranges
            self._cls_scr_ranges = [(0, posr[0][0])]
            for i in range(len(posr)-1):
                ro1, ro2 = posr[i], posr[i+1]
                self._cls_scr_ranges.append(ro1)
                self._cls_scr_ranges.append((ro1[1], ro2[0]))
            self._cls_scr_ranges.append(posr[-1])
            self._cls_scr_ranges.append((posr[-1][1], pred_sz))

            print("VAL min_slice_overlap_perc", self._min_slice_overlap_perc)
            self._eval_dict["min_slice_overlap_perc"] = self._min_slice_overlap_perc
            print("VAL num_slices", self._num_slices)
            self._eval_dict["num_slices"] = self._num_slices
            print("2D_LIST _stg1_slc_ranges", self._stg1_slc_ranges)
            print("2D_LIST _stg2_slc_ranges", self._stg2_slc_ranges)
            print("2D_LIST _preds_slc_ranges", self._preds_slc_ranges)
            print('2D_LIST _cls_scr_ranges', self._cls_scr_ranges)
            print("VAL slice_size_perc", self._slice_size_perc)
            self._eval_dict["slice_size_perc"] = self._slice_size_perc

            if self._cudnn_benchmarking:
                # This will trigger all possible slice inputs
                print('Starting dry run for cudnn benchmarking')
                self._calibration=True
                self._dry_run_slices = []
                #WARNING! This part assumes there are three rpn stages
                for i in range(self._num_slices+1):
                    for j in range(i+1):
                        self._dry_run_slices.append((i,j))
                for drs in self._dry_run_slices:
                    self._cur_calib_tuple = drs
                    self.load_and_infer(0, {'method': self.IMP_SLICE})
                self.clear_stats()
                self._calibration=False
                print('Dry run finished')

        # generate 50 random indexes to use for calibration, if required
        samples = np.random.randint(0, len(self.dataset)-1, 50)

        m = np.finfo(np.single).max
        self._post_stg1_noslice_table = np.full((self._num_stages,), m)
        fnames= [f"calib_dict_noslice_{self.dataset.dataset_cfg.DATASET}.json"]
        if self._enable_slicing:
            self._post_stg1_slice_table = np.full((self._num_slices+1, self._num_slices+1), m)
            fnames.append(f"calib_dict_slice{self._slice_size_perc}_" \
                    f"{self.dataset.dataset_cfg.DATASET}.json")
        for fname in fnames:
            sliced = not bool('noslice' in fname)
            try:
                with open(fname, 'r') as handle:
                    calib_dict = json.load(handle)
            except FileNotFoundError:
                print(f'Calibration file {fname} not found, running calibration') 
                calib_dict = self.do_calibration(fname, sliced, samples)

            # Use 99 percentile Post-stage-1 times
            stat_dict = calib_dict['stats']
            for k, v in stat_dict.items():
                if sliced:
                    r, c = k.replace('(', '').replace(')', '').replace(',', '').split()
                    r, c = int(r), int(c)
                    self._post_stg1_slice_table[r,c] = v['Post-stage-1'][3]
                else:
                    r = k.replace('(', '').replace(')', '').replace(',', '').split()
                    r = int(r[0])
                    self._post_stg1_noslice_table[r-1] = v['Post-stage-1'][3]

        if self._enable_slicing:
            print('Post stage 1 slice table:')
            print(self._post_stg1_slice_table)
        print('Post stage 1 noslice table:')
        print(self._post_stg1_noslice_table)

    def do_calibration(self, fname, sliced, sample_indexes):
        self._calibration = True
        self._cur_calib_tuple = None
        self._calib_test_cases=[]
        calib_dict = {"data":{}, "stats":{}}
        if sliced:
            for i in range(self._num_slices+1):
                for j in range(i+1):
                    self._calib_test_cases.append((i,j))
        else:
            for i in range(1, self._num_stages+1):
                self._calib_test_cases.append((i,))


        gc.disable()
        for stages in self._calib_test_cases:
            print('Calibrating test case', stages)
            self._cur_calib_tuple = stages
            
            for i in sample_indexes:  #range(len(self.dataset)):
                self.load_and_infer(i, {
                        'method': (self.IMP_SLICE if sliced else self.IMP_NOSLICE) })

            calib_dict["data"][str(stages)]  = copy.deepcopy(self.get_time_dict())
            stats = self.get_time_dict_stats()
            calib_dict["stats"][str(stages)] = stats
            self.print_time_stats()
            self.clear_stats()
            gc.collect()
            torch.cuda.empty_cache()
        gc.enable()

        with open(fname, 'w') as handle:
            json.dump(calib_dict, handle, indent=4)

        self._calibration = False
        return calib_dict   

    def get_slice_ranges_v3(self, tensor_H_size):
        # maintain minimum overlap while sqeezing more if necessary
        h_sz = tensor_H_size
        slice_sz = math.ceil(h_sz * self._slice_size_perc / 100)
        slice_ovl = math.ceil(h_sz * self._min_slice_overlap_perc / 100)

        slice_ranges, bi, ei = [], 0, slice_sz
        step = slice_sz - slice_ovl
        slice_ranges.append([bi, ei])
        while ei < h_sz:
            bi += step
            ei += step
            slice_ranges.append([bi, ei])

        # Slide with small steps
        s = 1
        while slice_ranges[-1][1] > h_sz:
            for rng in slice_ranges[s:]:
                rng[0] -= 1
                rng[1] -= 1
            s += 1
            if s == len(slice_ranges):
                s = 1 # reset

        overlap_ranges = []
        for i in range(len(slice_ranges)-1):
            overlap_ranges.append((slice_ranges[i+1][0],
                slice_ranges[i][1]))

        return slice_ranges, overlap_ranges

    def sched_stages(self, dline):
        if self._calibration:
            return self._cur_calib_tuple[0]
        else:
            time_left = (dline - time.time()) * 1000 # to ms
            tbl = self._post_stg1_noslice_table
            num_stages_to_run = 1
            for tm in self._post_stg1_noslice_table[1:]:  # skip the first as its already executed
                if tm < time_left:
                    num_stages_to_run += 1

        return num_stages_to_run

    # WARNING! This function assumes there are three stages
    # prioritize stage 2 no matter what
    def sched_slices_v2(self, slice_data_dicts, dline):
        cls_scores = np.array([float(torch.sum(s["cls_scores"])) \
                for s in slice_data_dicts])
        #if self._dry_run:
        #    ret = self._dry_run_slices[self._dry_run_idx]
        #    self._dry_run_idx += 1
        #    return ret
        if self._calibration:
            stg2, stg3 = self._cur_calib_tuple
            # choose slices that has highest cls scores
            indexes = np.argsort(cls_scores)
            ret1 = [] if stg2 == 0 else indexes[-stg2:]
            ret2 = [] if stg3 == 0 else indexes[-stg3:]
            return ret1, ret2
        else:
            #Run the actual decision algorithm
            time_left = (dline - time.time()) * 1000 # to ms
            tbl = self._post_stg1_slice_table
            stg2_slices, stg3_slices = [], []
            stg2, stg3 = 0, 0

            sorted_indexes = np.argsort(cls_scores)
            si = sorted_indexes.tolist()
            while len(si) > 0 and time_left > tbl[stg2+1, stg3] * 1.03:  # add %3 pessimism
                selected_slc = si.pop(si.index(min(si)))
                stg2_slices.append(selected_slc)
                stg2 += 1

            si = sorted_indexes.tolist()
            while len(si) > 0 and time_left > tbl[stg2, stg3+1] * 1.03: # add 3% pessimism
                selected_slc = si.pop(si.index(min(si)))
                stg3_slices.append(selected_slc)
                stg3 += 1

            return [sorted(stg2_slices), sorted(stg3_slices)]

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
                }

        loss = loss_rpn
        return loss, tb_dict, disp_dict


    def clear_stats(self):
        super().clear_stats()
        self._eval_dict['rpn_stg_exec_seqs'] = []
        
