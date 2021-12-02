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

        if 'SEPARATE_MULTIHEAD' in dir(model_cfg.DENSE_HEAD):
            self._sep_mhead = model_cfg.DENSE_HEAD.SEPARATE_MULTIHEAD
        else:
            self._sep_mhead = False

        # available methods:
        self.BASELINE1 = 1
        self.BASELINE2 = 2
        self.BASELINE3 = 3
        self.IMPRECISE = 4

        self._default_method = int(model_cfg.METHOD)
        print('Default method is:', self._default_method)

        self._eval_dict['method'] = self._default_method
        self._eval_dict['rpn_stg_exec_seqs'] = []

        # Stuff that might come as a parameter inside data_dict
        self._cudnn_benchmarking=True
        self._cudnn_deterministic=False

        self._calibrating_now = False

        self._num_stages = len(self.backbone_2d.num_bev_features)
        print('num rpn stages:', self._num_stages)

        self._score_threshold= float(model_cfg.POST_PROCESSING.SCORE_THRESH)

        # Debug/plot related
        # self._pc_range = dataset.point_cloud_range

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
                'RPN-stage-1': [],
                'Forward-cls': [],
                'Sched': [],
                'RPN-stage-2': [],
                'RPN-stage-3': [],
                'RPN-finalize': [],
                'RPN-total': [],
                'Pre-stage-1': [],
                'Post-stage-1': [],
                'PostProcess': [],})

        if self._sep_mhead:
            # Times below include postprocess
            # This table is overwritten by the calibration
            self.post_sync_time_table_ms = [
            # heads    1    2    3    4    5    6
                    [ 10., 20., 30., 40., 50., 60.], # no more rpn stage
                    [ 25., 35., 45., 55., 65., 75.], # 1  more rpn stage
                    [ 40., 50., 60., 70., 80., 90.], # 2  more rpn stages
            ]

        #print('Model:')
        #print(self)

        self.scores_that_zeros_list=[]
        self.total_missed_classes=0

    def forward(self, data_dict):
        if not self.training:
            return self.eval_forward(data_dict)
        else:
            data_dict = self.pre_rpn_forward(data_dict)

            # Execute all stages
            losses=[]
            for s in range(self._num_stages):
                data_dict = self.backbone_2d(data_dict)
                data_dict = self.dense_head(data_dict)
                if self.dense_head.predict_boxes_when_training:
                    data_dict = self.dense_head.gen_pred_boxes(data_dict)
                loss, tb_dict, disp_dict = self.get_training_loss()
                losses.append(loss)
            # Tried to implement training method in ABC paper
            # THIS PART ASSUMES THERE ARE THREE RPN STAGES
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

            ret_dict = {
                'loss': total_loss,
            }

            return ret_dict, tb_dict, disp_dict

    # Can run baselines and imprecise
    def eval_forward(self, data_dict):
        if 'method' not in data_dict:
            data_dict['method'] = self._default_method

        self.measure_time_start('Pre-stage-1')
        data_dict = self.pre_rpn_forward(data_dict)
        self.measure_time_start('RPN-total')
        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-stage-1')
        stg_seq=[1]
        data_dict['score_thresh'] = self._score_threshold
        if data_dict['method'] == self.IMPRECISE:
            self.measure_time_start('Forward-cls')
            data_dict = self.dense_head.forward_cls_preds(data_dict)
            self.measure_time_end('Forward-cls')
            self.measure_time_start('Sched')
            data_dict = self.sched_stages_and_heads(data_dict)
            self.measure_time_end('Sched')
            num_stgs_to_run = data_dict['num_stgs_to_run']
        else:
            num_stgs_to_run = data_dict['method']
            self.measure_time_end('Pre-stage-1')
            self.measure_time_start('Post-stage-1')

        if num_stgs_to_run >= 2:
            self.measure_time_start('RPN-stage-2')
            data_dict = self.backbone_2d(data_dict)
            self.measure_time_end('RPN-stage-2')
            stg_seq.append(2)

        if num_stgs_to_run == 3:
            self.measure_time_start('RPN-stage-3')
            data_dict = self.backbone_2d(data_dict)
            self.measure_time_end('RPN-stage-3')
            stg_seq.append(3)

        self.measure_time_end('RPN-total')
        self.measure_time_start('RPN-finalize')
        #torch.cuda.nvtx.range_push('Head')
        if num_stgs_to_run > 1:
            data_dict = self.dense_head(data_dict)
        else:
            data_dict = self.dense_head.forward_remaining_preds(data_dict)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end('RPN-finalize')

        self.measure_time_start("PostProcess")
        # Now do postprocess and finish
        #torch.cuda.nvtx.range_push('PostProcess')
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end("PostProcess")
        self.measure_time_end("Post-stage-1")
        if data_dict['method'] == self.IMPRECISE:
            self._eval_dict['rpn_stg_exec_seqs'].append((stg_seq, data_dict['heads_to_run'].tolist()))
        else:
            self._eval_dict['rpn_stg_exec_seqs'].append(stg_seq)
        return det_dicts, recall_dict

    def pre_rpn_forward(self, data_dict):
        self.measure_time_start('VFE')
        data_dict = self.vfe(data_dict) # pillar feature net
        self.measure_time_end('VFE')
        self.measure_time_start('MapToBEV')
        data_dict = self.map_to_bev(data_dict) # pillar scatter
        self.measure_time_end('MapToBEV')
        data_dict["stage0"] = data_dict["spatial_features"]
        data_dict["stages_executed"] = 0
        data_dict['score_thresh'] = self._score_threshold

        return data_dict

    def sched_stages_and_heads(self, data_dict):

        stg0_sum = torch.sum(data_dict['spatial_features'], 1, keepdim=True)
        sum_mask = torch.nn.functional.max_pool2d(stg0_sum, 19,
                stride=4, padding=9).unsqueeze(-1)

        # each cls score: 1, 2x, 128, 128, x
        expanded_sum_masks = {}
        for i, cp in enumerate(data_dict['cls_preds']):
            k = cp.size()[-1]
            if k not in expanded_sum_masks:
                expanded_sum_masks[k] = \
                        sum_mask.expand(cp.size()).type(torch.bool)

        cls_scores = []
        cls_score_sums = []
        cutoff_threshold = 0.5
        cutoff_scores = torch.zeros((self.dense_head.num_heads,), \
                device=data_dict['cls_preds'][0].device)
        prio_scores = torch.zeros((self.dense_head.num_heads,))
        for i, cp in enumerate(data_dict['cls_preds']):
            cls_scores.append(torch.sigmoid(cp))

            # first,calculate the score which determines whether
            # this class must be included in the head
            cutoff_scores[i] = torch.sum((cls_scores[-1] > cutoff_threshold) \
                    * cls_scores[-1])

            # Second, calculate the priorities to include
            cls_scores_keep = cls_scores[-1] > self._score_threshold
            cls_scores_filtered = cls_scores[-1] * \
                    cls_scores_keep.logical_and(expanded_sum_masks[cp.size()[-1]])
            cls_scores[-1] = cls_scores_filtered
            cls_score_sums.append(torch.sum(cls_scores_filtered, \
                    [0, 1, 2, 3], keepdim=False))

        if self._calibrating_now:
            # I hope this gpu to cpu transfer is not going to affect calibration
            calib_prios = data_dict['gt_counts'].cpu()[0].argsort(descending=True)

        # Here it will sync
        cutoff_scores = cutoff_scores.cpu()
        cls_score_sums = [css.cpu() for css in cls_score_sums]

        self.measure_time_end('Pre-stage-1')
        self.measure_time_start('Post-stage-1')

        for i, s in enumerate(cls_score_sums):
            for j, c in enumerate(s):
                prio_scores[i] += c / self.dense_head.anchor_area_coeffs[i][j]

        # this operation boosts the priority of the classes that definetely
        # has some corresponding objects, we will call them king classes
        prio_scores = (cutoff_scores * 100.0) + prio_scores
        prios = prio_scores.argsort(descending=True)
        #print('prios', prios)
        #print('gtc  ', data_dict['gt_counts'].argsort(descending=True))

        rem_ms = (data_dict['abs_deadline_sec'] - time.time()) * 1000.0

        if self._calibrating_now:
            num_stages, num_heads = self._cur_calib_tuple
            prios = calib_prios
        else:
            # First, check if we can finish the line with executing no more rpn
            # If we can't, we need to decrease the head count until
            # we meet the deadline
            num_stages = 1
            num_heads = max(torch.count_nonzero(cutoff_scores), 1)
            if rem_ms < self.post_sync_time_table_ms[0][num_heads-1]:
                # Oh no, we can't meet the deadline, gotta decrase the number 
                # of heads
                while num_heads > 1 and \
                        rem_ms < self.post_sync_time_table_ms[0][num_heads-1]:
                    num_heads -= 1
            else:
                # Second, increase the number of rpn stages by one if possible
                if rem_ms > self.post_sync_time_table_ms[1][num_heads-1]:
                    num_stages += 1

                # Third,increase the number of heads if there is time remaining
                while num_heads < len(self.post_sync_time_table_ms[1]) and \
                        rem_ms > self.post_sync_time_table_ms[num_stages-1][num_heads]:
                    num_heads += 1

                # Fourth, add more rpn stages if there is time remaining
                while num_stages < len(self.post_sync_time_table_ms) and \
                        rem_ms > self.post_sync_time_table_ms[num_stages][num_heads-1]:
                    num_stages += 1

        data_dict['num_stgs_to_run'] = num_stages
        data_dict['heads_to_run'] = prios[:num_heads]

        #print('num_stages:', num_stages, 
        #        'heads_to_run:', data_dict['heads_to_run'],
        #        'gt:', data_dict['gt_counts'][0].cpu())

        # utilize precomputed class scores if we are gonna run a single rpn stage
        new_cls_preds = []
        for h in data_dict['heads_to_run']:
            if num_stages == 1:
                new_cls_preds.append(cls_scores[h])
            else:
                new_cls_preds.append(data_dict['cls_preds'][h])
        data_dict['cls_preds'] = new_cls_preds

        if num_stages == 1:
            data_dict['cls_preds_normalized'] = True

        return data_dict

    def calibrate(self):
        super().calibrate()
        
        # generate 1000 random indexes to use for calibration
        #samples = np.random.randint(0, len(self.dataset)-1, 1000)
        #samples.sort()

        fname = f"calib_dict_{self.dataset.dataset_cfg.DATASET}.json"
        try:
            with open(fname, 'r') as handle:
                calib_dict = json.load(handle)
        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration') 
            calib_dict = self.do_calibration(fname) #, samples)
        # Use 99 percentile Post-stage-1 times
        #m = np.finfo(np.single).max
        stat_dict = calib_dict['stats']
        for k, v in stat_dict.items():
            r,c = k.replace('(', '').replace(')', '').replace(',', '').split()
            r,c = int(r), int(c)
            self.post_sync_time_table_ms[r-1][c-1] = v['Post-stage-1'][3]

        print('Post stage 1 wcet table:')
        for row in self.post_sync_time_table_ms:
            print(row)

    def do_calibration(self, fname): #, sample_indexes):
        self._calibrating_now = True
        self._cur_calib_tuple = None  # (num_stages, num_heads)
        self._calib_test_cases=[]
        calib_dict = {"data":{}, "stats":{}, "eval":{}}
        for i in range(1, self._num_stages+1):
            for j in range(1, self.dense_head.num_heads+1):
                self._calib_test_cases.append((i,j))

        gc.disable()

        for cur_calib_conf in self._calib_test_cases:
            print('Calibrating test case', cur_calib_conf)
            self._cur_calib_tuple = cur_calib_conf

            det_annos = []
            #for i in sample_indexes:
            for i in range(len(self.dataset)):
                with torch.no_grad():
                    batch_dict, pred_dicts, ret_dict = self.load_and_infer(i,
                            {'method': self.IMPRECISE })
                #if i not in sample_indexes:
                #    pred_dicts = [ self.get_empty_det_dict() for p in pred_dicts ]
                annos = self.dataset.generate_prediction_dicts(
                    batch_dict, pred_dicts, self.dataset.class_names,
                    output_path='./temp_results'
                )
                det_annos += annos
                if i % (len(self.dataset)//10) == 0:
                    print(i, '/', len(self.dataset), ' done')
            calib_dict["data"][str(cur_calib_conf)]  = copy.deepcopy(self.get_time_dict())
            stats = self.get_time_dict_stats()
            calib_dict["stats"][str(cur_calib_conf)] = stats
            self.print_time_stats()
            self.clear_stats()
            gc.collect()
            torch.cuda.empty_cache()

            result_str, result_dict = self.dataset.evaluation(
                det_annos, self.dataset.class_names,
                eval_metric=self.model_cfg.POST_PROCESSING.EVAL_METRIC,
                output_path='./temp_results'
            )
            calib_dict['eval'][str(cur_calib_conf)]  = result_dict # mAP or NDS will be enough
            gc.collect()
            torch.cuda.empty_cache()
        gc.enable()

        #calib_dict['sample_indexes'] = sample_indexes.tolist()

        with open(fname, 'w') as handle:
            json.dump(calib_dict, handle, indent=4)

        self._calibrating_now = False
        return calib_dict

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
