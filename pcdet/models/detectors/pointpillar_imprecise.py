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
        self.IMPRECISE_A_P   = 4  # With aging and prediction
        self.IMPRECISE_NA_NP = 5
        self.IMPRECISE_A_NP  = 6
        self.IMPRECISE_NA_P  = 7

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
                #'Forward-cls': [],
                'Sched': [],
                'RPN-stage-2': [],
                'RPN-stage-3': [],
                'RPN-finalize': [],
                'RPN-total': [],
                #'Pre-RPN': [],
                'Post-PFE': [],
                'PostProcess': [],})

        if self._sep_mhead:
            # Times below include postprocess
            # These tables are overwritten by the calibration
            self.post_sync_time_table_ms = [
            # heads    1    2    3    4    5    6
                    [ 10., 20., 30., 40., 50., 60.], # no more rpn stage
                    [ 25., 35., 45., 55., 65., 75.], # 1  more rpn stage
                    [ 40., 50., 60., 70., 80., 90.], # 2  more rpn stages
            ]
            self.cfg_to_NDS = [
                    [.2] * 6,
                    [.3] * 6,
                    [.4] * 6,
            ]
            self.cur_scene_str = ''
            #self._eval_dict['gt_counts'] = []

        #print('Model:')
        #print(self)

        self.latest_timestamp = None
        self.last_skipped_heads= []
        self.det_hist_queue = [] # list of (timestamp, last_skipped_heads, det_dicts)
        self.max_queue_size = 5
        self.dets_to_migrate = []

        # from ego pose json file, create ts_to_egopose dictionary
        self.ts_to_egopose={}
        print('Reading timestamp to egopose dictionary')
        fname = f"ts_to_egopose.json"
        try:
            with open(fname, 'r') as handle:
                self.ts_to_egopose = json.load(handle)
        except FileNotFoundError:
            print(f'Couldnt open file {fname}')


    def forward(self, data_dict):
        if not self.training:
            # det_dicts length is equal to batch size
            det_dicts, recall_dict = self.eval_forward(data_dict)

            self.det_hist_queue.append((self.latest_timestamp, \
                    self.last_skipped_heads.copy(), det_dicts))

            if len(self.det_hist_queue) > self.max_queue_size:
                self.det_hist_queue.pop(0)

            if self.dets_to_migrate:
                for dd in det_dicts:
                    for k in dd.keys():
                        dd[k] = torch.cat([dd[k]] + [dtm[k] \
                                for dtm in self.dets_to_migrate])
                self.dets_to_migrate = []

            return det_dicts, recall_dict
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

        data_dict = self.pre_rpn_forward(data_dict)

        post_pfe_event = torch.cuda.Event()

        self.measure_time_start('Post-PFE')
        self.measure_time_start('RPN-total')
        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-stage-1')
        stg_seq=[1]
        data_dict['score_thresh'] = self._score_threshold

        if data_dict['method'] >= self.IMPRECISE_A_P:
            post_pfe_event.synchronize()
            data_dict = self.sched_stages_and_heads(data_dict)
            num_stgs_to_run = data_dict['num_stgs_to_run']
        else:
            num_stgs_to_run = data_dict['method']

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
        if data_dict['method'] >= self.IMPRECISE_A_P:
            data_dict = self.dense_head.forward_cls_preds(data_dict)
            self.measure_time_start('Sched')
            data_dict = self.prioritize_heads(data_dict)
            self.measure_time_end('Sched')
            data_dict = self.dense_head.forward_remaining_preds(data_dict)
        else:
            data_dict = self.dense_head.forward(data_dict)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end('RPN-finalize')

        self.measure_time_start("PostProcess")
        # Now do postprocess and finish
        #torch.cuda.nvtx.range_push('PostProcess')
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end("PostProcess")
        self.measure_time_end("Post-PFE")
        if data_dict['method'] >= self.IMPRECISE_A_P:
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
        rem_ms = (data_dict['abs_deadline_sec'] - time.time()) * 1000.0

        if self._calibrating_now:
            selected_cfg = self._cur_calib_tuple
            #prios = calib_prios
        else:
            # Traverse the table and find the config which will meet the
            # deadline and would give the highest NDS
            selected_cfg = (1, 1)
            best_NDS = self.cfg_to_NDS[0][0]
            for i in range(len(self.post_sync_time_table_ms)):
                for j in range(len(self.post_sync_time_table_ms[0])):
                    if self.post_sync_time_table_ms[i][j] <= rem_ms and \
                            self.cfg_to_NDS[i][j] > best_NDS:
                        best_NDS = self.cfg_to_NDS[i][j]
                        selected_cfg = (i+1,j+1)

            # this method prioritizes having more heads
#            selected_cfg = (1, 1)
#            selected = False
#            for c in range(len(self.post_sync_time_table_ms[0])-1, -1, -1):
#                for r in range(len(self.post_sync_time_table_ms)-1, -1, -1):
#                    if self.post_sync_time_table_ms[r][c] <= rem_ms:
#                        selected_cfg = (r+1,c+1)
#                        selected = True
#                        break
#                if selected:
#                    break

        data_dict['sh_config'] = selected_cfg
        data_dict['num_stgs_to_run'] = selected_cfg[0]
        return data_dict

    def prioritize_heads(self, data_dict):
        ##n015-2018-07-24-10-42-41+0800__LIDAR_TOP__1532400184198068.pcd
        frame_id = data_dict['frame_id'][0].split('_')
        self.latest_timestamp = int(frame_id[5].split('.')[0])
        scene_str = frame_id[0]
        if scene_str != self.cur_scene_str:
            self.det_hist_queue = [] # reset queue
            self.cur_scene_str = scene_str

        selected_cfg = data_dict['sh_config']

        if selected_cfg[1] == len(self.post_sync_time_table_ms[0]): # max num heads
            self.last_skipped_heads= []
            data_dict['heads_to_run'] = torch.arange(selected_cfg[1])
            return data_dict

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

        #if self._calibrating_now:
            # I hope this gpu to cpu transfer is not going to affect calibration significantly
            #calib_prios = data_dict['gt_counts'].cpu()[0].argsort(descending=True)

        cutoff_scores = cutoff_scores.cpu()
        cls_score_sums = [css.cpu() for css in cls_score_sums]

        for i, s in enumerate(cls_score_sums):
            for j, c in enumerate(s):
                prio_scores[i] += c / self.dense_head.anchor_area_coeffs[i][j]

        if data_dict['method'] == self.IMPRECISE_A_P or \
                data_dict['method'] == self.IMPRECISE_A_NP:
            # priority boost using history, most skipped will be most boosted
            prio_boost = [1.] * self.dense_head.num_heads
            head_skips = [dh[1] for dh in self.det_hist_queue]
            for h in range(self.dense_head.num_heads):
                boost_amount = 100.
                for i, hs in enumerate(head_skips):
                    if h not in hs:
                        boost_amount=1.
                        break
                prio_boost[h] = boost_amount

            for i, pb in enumerate(prio_boost):
                prio_scores[i] *= pb

        prio_scores = (cutoff_scores * 50.0) + prio_scores
        #print('prio_boost', prio_boost)
        #print('prio_scores', prio_scores)
        # this operation boosts the priority of the classes that definetely
        # has some corresponding objects
        prios = prio_scores.argsort(descending=True)
        #print('prios', prios)
        #print('gtc  ', data_dict['gt_counts'].argsort(descending=True))

        data_dict['heads_to_run'] = prios[:selected_cfg[1]]

        self.last_skipped_heads = prios[selected_cfg[1]:].tolist()
        skipped_labels=[]
        for h in self.last_skipped_heads:
            skipped_labels.extend(self.dense_head.head_to_labels[h])

        #print('Skipped labels:', skipped_labels)
        # migrate detections from previous frame if possible

        if data_dict['method'] == self.IMPRECISE_A_P or \
                data_dict['method'] == self.IMPRECISE_NA_P:
            for h in self.last_skipped_heads:
                for ts, hs, det_dicts in reversed(self.det_hist_queue):
                    if h not in hs:
                        # get the corresponding detections if they exist
                        skipped_labels = self.dense_head.head_to_labels[h]
                        for dd in det_dicts:
                            prev_det_labels = dd['pred_labels'].cpu().tolist()
                            prev_det_scores = dd['pred_scores'].cpu().tolist()
                            indexes_to_migrate = []
                            i = 0
                            for lbl, score in zip(prev_det_labels, prev_det_scores):
                                if score >= 0.3 and lbl in skipped_labels:  # migrate confident ones
                                    indexes_to_migrate.append(i)
                                i += 1

                            if indexes_to_migrate:

                                # We need the velocity of the car to calculate the global
                                # speeds of the detected objects, IF THEY ARE NOT GLOBAL
                                past_egopose = self.ts_to_egopose[str(ts)]
                                cur_egopose = self.ts_to_egopose[str(self.latest_timestamp)]
                                tdiff = (self.latest_timestamp - ts) / 1000000.0 # convert it to second
                                #egov_x_vel = (cur_egopose[0] - past_egopose[0]) / tdiff
                                #egov_y_vel = (cur_egopose[1] - past_egopose[1]) / tdiff

                                #print('tdiff:', tdiff)
                                #print('Ego vehicle velocity:', egov_x_vel, egov_y_vel)

                                pred_boxes = dd['pred_boxes'][indexes_to_migrate]
                                #print('pred_boxes size', pred_boxes.size())
                                # 0:x 1:y z dx dy dz ori 7:velx 8:vely
                                # I am assuming the network output velocity as meters/second
                                world_x = pred_boxes[:, 0] + past_egopose[0]
                                #world_x += (egov_x_vel + pred_boxes[:, 7]) * tdiff
                                world_x += pred_boxes[:, 7] * tdiff
                                pred_boxes[:, 0] = world_x - cur_egopose[0]

                                world_y = pred_boxes[:, 1] + past_egopose[1]
                                #world_y += (egov_y_vel + pred_boxes[:, 8]) * tdiff
                                world_y += pred_boxes[:, 8] * tdiff
                                pred_boxes[:, 1] = world_y - cur_egopose[1]

                                self.dets_to_migrate.append({
                                    'pred_boxes': pred_boxes,
                                    'pred_scores': dd['pred_scores'][indexes_to_migrate],
                                    'pred_labels': dd['pred_labels'][indexes_to_migrate],
                                })

                                #print('Labels:', self.dets_to_migrate[-1]['pred_labels'])
                                #print('Velocity:',  self.dets_to_migrate[-1]['pred_boxes'][:,7:])
                        break

        #print('num_stages:', num_stages, 
        #        'heads_to_run:', data_dict['heads_to_run'],
        #        'gt:', data_dict['gt_counts'][0].cpu())

        new_cls_preds = []
        for h in data_dict['heads_to_run']:
            new_cls_preds.append(cls_scores[h])

        data_dict['cls_preds'] = new_cls_preds

        data_dict['cls_preds_normalized'] = True

        return data_dict

    def calibrate(self):
        super().calibrate()
        
        fname = f"calib_dict_{self.dataset.dataset_cfg.DATASET}" \
                f"_m{self._default_method}.json"
        try:
            with open(fname, 'r') as handle:
                calib_dict = json.load(handle)
            if calib_dict['method'] != self._default_method:
                print('**********************************************************')
                print("WARNING! Calibration data is not based on the configuration\n" \
                        "of model configuration file!")
                print('**********************************************************')
        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration') 
            calib_dict = self.do_calibration(fname) #, samples)
        # Use 99 percentile Post-PFE times
        def get_rc(k):
            r,c = k.replace('(', '').replace(')', '').replace(',', '').split()
            r,c = int(r), int(c)
            return r,c

        for k, v in calib_dict['stats'].items():
            r,c = get_rc(k)
            self.post_sync_time_table_ms[r-1][c-1] = v['Post-PFE'][3]

        for k, v in calib_dict['eval'].items():
            r,c = get_rc(k)
            self.cfg_to_NDS[r-1][c-1] = round(v['NDS'], 3)

        for i in range(len(self.cfg_to_NDS)):
            for j in range(1, len(self.cfg_to_NDS[0])):
                if self.cfg_to_NDS[i][j-1] >= self.cfg_to_NDS[i][j]:
                    self.cfg_to_NDS[i][j] = self.cfg_to_NDS[i][j-1] + 0.001

        print('Post PFE wcet table:')
        for row in self.post_sync_time_table_ms:
            print(row)

        print('Stage/Head configuration to NDS table:')
        for row in self.cfg_to_NDS:
            print(row)

    def do_calibration(self, fname): #, sample_indexes):
        self._calibrating_now = True
        self._cur_calib_tuple = None  # (num_stages, num_heads)
        self._calib_test_cases=[]
        calib_dict = {"data":{}, "stats":{}, "eval":{}, "method":self._default_method}
        for i in range(1, self._num_stages+1):
            for j in range(1, self.dense_head.num_heads+1):
                self._calib_test_cases.append((i,j))


        gc.disable()

        for cur_calib_conf in self._calib_test_cases:
            print('Calibrating test case', cur_calib_conf)
            self._cur_calib_tuple = cur_calib_conf

            self.last_skipped_heads= []
            self.det_hist_queue = [] # list of (timestamp, last_skipped_heads, det_dicts)
            self.max_queue_size = 5
            self.dets_to_migrate = []

            det_annos = []
            #for i in sample_indexes:
            for i in range(len(self.dataset)):
                with torch.no_grad():
                    batch_dict, pred_dicts, ret_dict = self.load_and_infer(i,
                            {'method': self._default_method})
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
