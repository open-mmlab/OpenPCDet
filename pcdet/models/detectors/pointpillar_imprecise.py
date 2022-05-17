from .detector3d_template import Detector3DTemplate
import os
import gc
import torch
import math
import time
import copy
import json
import tqdm
import heapq
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from multiprocessing import Pool
from collections import deque

def calcCurPose(pbs_np, pose_idx_start, idx_to_pose, cur_pose_inv):
    ts_cur = cur_pose_inv['ts']
    for i in range(pbs_np.shape[0]):
        pb_np = pbs_np[i]
        # The velocities output by the network are wrt ego vehicle coordinate frame,
        # but they appear to be global velocities.
        box = Box(pb_np[:3], pb_np[3:6],
                #Quaternion([np.cos(pb_np[6]/2.), 0., 0., np.sin(pb_np[6]/2.)]),
                Quaternion(axis=[0, 0, 1], radians=pb_np[6]),
                velocity=np.append(pb_np[7:], .0))
        # Move from sensor coordinate to global
        pose = idx_to_pose[pose_idx_start]
        pose_idx_start += 1
        box.rotate(pose['csr'])
        box.translate(pose['cst'])
        box.rotate(pose['epr'])
        box.translate(pose['ept'])

        #if USE_VEL:
        elapsed_sec = (ts_cur - pose['ts']) / 1000000.
        pose_diff = box.velocity * elapsed_sec
        if not np.any(np.isnan(pose_diff)):
            box.translate(pose_diff)

        # Move from global to predicted sensor coordinate
        box.translate(cur_pose_inv['ept_neg'])
        box.rotate(cur_pose_inv['epr_inv'])
        box.translate(cur_pose_inv['cst_neg'])
        box.rotate(cur_pose_inv['csr_inv'])

        pb_np[:3] = box.center
        r, i, j, k = box.orientation.elements
        pb_np[6] = 2. * math.atan2(math.sqrt(i*i+j*j+k*k),r)
        pb_np[7:] = box.velocity[:2]

    return pbs_np

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
        self.IMPR_MultiStage = 4
        self.IMPR_StaticHeadSel = 5
        self.IMPR_RRHeadSel = 6
        self.IMPR_HistoryHeadSel = 7
        self.IMPR_PCHeadSel_Prj = 8       # Projection calibrated selection
        self.IMPR_RRHeadSel_Prj = 9
        self.IMPR_HistoryHeadSel_Prj = 10
        self.IMPR_NearoptHeadSel_Prj = 11
        self.IMPR_CSSHeadSel = 12
        self.IMPR_CSSHeadSel_Prj = 13
        self.IMPR_NearoptHeadSel = 14

        self.IMPR_Dryrun = 19
        self.IMPR_PTEST = 20              # Projection test

        self._default_method = int(model_cfg.METHOD)
        print('Default method is:', self._default_method)

        self._eval_dict['method'] = self._default_method
        self._eval_dict['rpn_stg_exec_seqs'] = []
        self._eval_dict['projections_per_task'] = []

        # Stuff that might come as a parameter inside data_dict
        self._cudnn_benchmarking=True
        self._cudnn_deterministic=False
        self._calibrating_now = False
        self._num_stages = len(self.backbone_2d.num_bev_features)
        print('num rpn stages:', self._num_stages)

        self._score_threshold= float(model_cfg.POST_PROCESSING.SCORE_THRESH)

        # determinism
        torch.backends.cudnn.benchmark = self._cudnn_benchmarking
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        if self._cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.set_deterministic(True)

        self.update_time_dict({
            'VFE': [],
            'MapToBEV': [],
            'Sync': [],
            'RPN-stage-1': [],
            'Prioritize': [],
            'SmartSel': [],
            'PrePrediction': [],
            'RPN-stage-2': [],
            'RPN-stage-3': [],
            'RPN-finalize': [],
            'PostProcess': [],
            'PostPrediction': [],
            'Post-PFE': [],})

        self.post_sync_time_table_ms = np.full(\
                (self._num_stages, self.dense_head.num_heads), 999.)
        self.cfg_to_NDS = np.zeros(\
                (self._num_stages, self.dense_head.num_heads))
        self.cur_scene_token = ''

        #print('Model:')
        #print(self)

        #self.head_proj_calib_table = np.array([
        #    # [car], [truck, construction_vehicle], [bus, trailer], 
        #    # [barrier], [motorcycle, bicycle], [pedestrian, traffic_cone]
        #    [0.980, 0.982, 0.987, 0.983, 0.970, 0.970], #150ms
        #    [0.960, 0.969, 0.983, 0.967, 0.946, 0.945], #300ms
        #    [0.936, 0.947, 0.969, 0.951, 0.915, 0.916], #450ms
        #    [0.905, 0.931, 0.957, 0.936, 0.884, 0.894], #600ms
        #    [0.871, 0.910, 0.935, 0.921, 0.846, 0.868]  #750ms
        #], dtype=np.float32)

        # These lines are interpolated from the values above with the x values 1 2 3 4 5
        #1: y = -0.0273*x + 1.0123
        #2: y = -0.0182*x + 1.0024
        #3: y = -0.0130*x + 1.0052
        #4: y = -0.0155*x + 0.9981
        #5: y = -0.0310*x + 1.0052 
        #6: y = -0.0255*x + 0.9951
        self.head_proj_calib_mult= np.array( \
            [-0.0273, -0.0182, -0.013, -0.0155, -0.031, -0.0255], dtype=np.float32)
        self.head_proj_calib_add = np.array( \
            [1.0123,   1.0024, 1.0052,  0.9981, 1.0052,  0.9951], dtype=np.float32)

        # AP scores from trainval evaluation
        #self.AP scores = [
        #        [0.438, 0.114, 0.006, 0.127, 0.028, 0.107, 0.116, 0.000, 0.216, 0.101],
        #        [0.527, 0.321, 0.029, 0.446, 0.172, 0.131, 0.151, 0.003, 0.242, 0.129],
        #        [0.593, 0.357, 0.042, 0.485, 0.267, 0.210, 0.185, 0.007, 0.262, 0.157],
        #]

        self.head_static_prios = [
                # [car], [truck, construction_vehicle], [bus, trailer], 
                # [barrier], [motorcycle, bicycle], [pedestrian, traffic_cone]
                np.flip(np.array([0.438, (0.114+0.006)/2, (0.127+0.028)/2,
                    0.107, (0.116+0.000)/2, (0.216+0.101)/2]).argsort()),
                np.flip(np.array([0.527, (0.321+0.029)/2, (0.446+0.172)/2,
                    0.131, (0.151+0.003)/2, (0.242+0.129)/2]).argsort()),
                np.flip(np.array([0.593, (0.357+0.042)/2, (0.485+0.267)/2,
                    0.210, (0.185+0.007)/2, (0.262+0.157)/2]).argsort()),
        ]

        self.css_coeffs = np.array([ \
                [0.07, 0.06, 0.07, 0.07, 0.19, 0.08], \
                [0.06, 0.11, 0.11, 0.07, 0.22, 0.07], \
                [0.05, 0.13, 0.16, 0.07, 0.21, 0.07]], dtype=np.float32)

        self.css_bef_nms_coeffs = torch.tensor([ \
                [0.02, 0.01, 0.01, 0.02, 0.11, 0.04], \
                [0.02, 0.02, 0.01, 0.02, 0.13, 0.03], \
                [0.02, 0.02, 0.01, 0.02, 0.12, 0.03]], \
                dtype=torch.float32, device='cuda')

        self.latest_token = None
        self.max_queue_size = 10 # 1-1.5 second

        # holds the time passed since a head was used for all heads
        self.head_age_arr = np.full(self.dense_head.num_heads, \
                self.max_queue_size+1, dtype=np.int8)
        self.head_scores_arr = np.zeros(self.dense_head.num_heads, \
                dtype=np.float32)
        self.cls_score_sums = np.zeros(self.dense_head.num_heads, \
                dtype=np.float32)
        self.det_dicts_queue = deque(maxlen=self.max_queue_size)
        self.pose_dict_queue = deque(maxlen=self.max_queue_size)
        self.last_skipped_heads= np.array([], dtype=np.uint8)
        self.hist_tuple = None

        #RR
        self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)

        # prediction
        self.pool_size = 6  # 6 appears to give best results on jetson-agx
        self.pred_box_pool = Pool(self.pool_size)
        self.chosen_det_dicts, self.all_indexes = [], []
        self.all_async_results = []

        self.all_heads = np.arange(self.dense_head.num_heads, dtype=np.uint8)

        self.CSS_calib=0
        self.CSS_calib_thresholds = [0.1]
        self.gt_and_css_tuples=[]

        self.history_calib=0

        self.dryrun_heads=self.all_heads
        self.config_tuples=[]

        with open('token_to_pos.json', 'r') as handle:
            self.token_to_pos = json.load(handle)

        self.use_oracle = False
        if self.use_oracle:
            with open('token_to_anns.json', 'r') as handle:
                self.token_to_anns= json.load(handle)


    def reset_queues_and_arrays(self):
        self.head_age_arr.fill(self.max_queue_size+1)
        self.head_scores_arr.fill(.0)
        self.det_dicts_queue.clear()
        self.pose_dict_queue.clear()
        self.last_skipped_heads= np.array([], dtype=np.uint8)
        self.hist_tuple = None

        #RR
        self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)
        self.last_head = self.dense_head.num_heads -1
        self.rrhq_restart = 0


    def use_projection(self, data_dict):
        m = data_dict['method'] 
        return (m == self.IMPR_CSSHeadSel_Prj or \
                m == self.IMPR_NearoptHeadSel_Prj or \
                m == self.IMPR_HistoryHeadSel_Prj or \
                m == self.IMPR_PCHeadSel_Prj or \
                m == self.IMPR_RRHeadSel_Prj or \
                m == self.IMPR_PTEST)

    def forward(self, data_dict):
        if not self.training:
            self.gt_counts = data_dict['gt_counts'][0]
            if self._calibrating_now:
                self.all_gt_counts += self.gt_counts
            if 'method' not in data_dict:
                data_dict['method'] = self._default_method
            data_dict['score_thresh'] = self._score_threshold
            self.latest_token = data_dict['metadata'][0]['token']

            # det_dicts length is equal to batch size
            det_dicts, recall_dict = self.eval_forward(data_dict)

            dd = det_dicts[0] # Assume batch size is 1
            if data_dict['method'] == self.IMPR_HistoryHeadSel or \
                    data_dict['method'] == self.IMPR_HistoryHeadSel_Prj:
                self.hist_tuple = \
                        (data_dict['num_stgs_to_run'] - 1, data_dict['heads_to_run'], 
                         dd['pred_score_sizes'])

                del dd['pred_score_sizes']

            for k,v in dd.items():
                dd[k] = v.cpu()

            if self.use_oracle:
                oracle_dd = self.token_to_anns[self.latest_token]
                oracle_dd['pred_boxes'] = \
                        torch.as_tensor(oracle_dd['pred_boxes'])
                oracle_dd['pred_scores'] = \
                        torch.as_tensor(oracle_dd['pred_scores'])
                oracle_dd['pred_labels'] = \
                        torch.as_tensor(oracle_dd['pred_labels'])
                det_dicts = [oracle_dd] * data_dict['batch_size']

            det_dicts_ret = det_dicts

            if data_dict['method'] == self.IMPR_HistoryHeadSel:
                self.det_dicts_queue.appendleft(det_dicts)
            elif self.use_projection(data_dict):
                self.measure_time_start('PostPrediction', False)
                pose = self.token_to_pos[self.latest_token]
                pose_dict = { 'ts' : int(pose['timestamp']),
                        'cst' : np.array(pose['cs_translation']),
                        'csr' : Quaternion(pose['cs_rotation']),
                        'ept' : np.array(pose['ep_translation']),
                        'epr' : Quaternion(pose['ep_rotation'])
                }
                self.pose_dict_queue.appendleft(pose_dict)
                self.det_dicts_queue.appendleft(det_dicts)
                # Now, collect the predictions calculated in the background
                if self.all_async_results:
                    self.projected_boxes = torch.from_numpy(np.concatenate( \
                            [ar.get() for ar in self.all_async_results]))
                    if self.projected_boxes.size()[0] > 0:
                        det_to_migrate = {'pred_boxes': self.projected_boxes}
                        for k in ['pred_scores', 'pred_labels']:
                            det_to_migrate[k]= torch.cat( \
                                    [dd[k][i] for dd, i in zip( \
                                    self.chosen_det_dicts, self.all_indexes)])

                        if data_dict['method'] == self.IMPR_CSSHeadSel_Prj:
                            # weed out the projections which are unnecessary
                            dd = det_dicts[0]
                            det_labels = set(dd['pred_labels'].tolist())
                            indexes = [i for i,l in enumerate(\
                                    det_to_migrate['pred_labels'].tolist()) \
                                    if l not in det_labels]
                            for k,v in det_to_migrate.items():
                                det_to_migrate[k] = v[indexes]

                        if data_dict['method'] == self.IMPR_PTEST:
                            det_dicts_ret = [det_to_migrate] * len(det_dicts)
                        else:
                            det_dicts_ret = []
                            for dd in det_dicts:
                                dd_ = {k:torch.cat([dd[k], det_to_migrate[k]]) \
                                        for k in dd.keys()}
                                det_dicts_ret.append(dd_)
                    self.all_async_results.clear()
                self.measure_time_end('PostPrediction', False)

            if data_dict['method'] >= self.IMPR_MultiStage:
                for h in data_dict['heads_to_run']:
                    self.head_age_arr[h] = 1
                for h in self.last_skipped_heads:
                    self.head_age_arr[h] += 1

                self.measure_time_end("Post-PFE", data_dict['method'] == self.IMPR_Dryrun)

            return det_dicts_ret, recall_dict
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
        data_dict = self.pre_rpn_forward(data_dict)

        #if data_dict['method'] >= self.IMPR_MultiStage:
        #    post_pfe_event = torch.cuda.Event()

        if data_dict['method'] >= self.IMPR_MultiStage:
            #post_pfe_event.synchronize()
            #self.measure_time_start('Sync', False)
            if data_dict['method'] != self.IMPR_Dryrun:
                torch.cuda.synchronize()
            #self.measure_time_end('Sync', False)
            rem_ms = (data_dict['abs_deadline_sec'] - time.time()) * 1000.0
            self.measure_time_start('Post-PFE', data_dict['method'] == self.IMPR_Dryrun)

        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-stage-1')
        stg_seq=[1]

        # try to overlap rpn stage-1 execution with prioritization
        if data_dict['method'] >= self.IMPR_MultiStage:
            self.measure_time_start('Prioritize', False)
            data_dict = self.sched_stages_and_heads(data_dict, rem_ms)
            data_dict = self.prioritize_heads(data_dict)
            self.measure_time_end('Prioritize', False)
            num_stgs_to_run = data_dict['num_stgs_to_run']
        else:
            num_stgs_to_run = data_dict['method']

        # migrate detections from previous frames if possible
        # while GPU runs
        if self.use_projection(data_dict):
            # Start projection
            self.measure_time_start('PrePrediction', False)
            self.start_projections(data_dict)
            self.measure_time_end('PrePrediction', False)

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

        self.measure_time_start('RPN-finalize')
        data_dict = self.dense_head.forward_cls_preds(data_dict)
        if self.CSS_calib > 0:
            self.measure_time_start('SmartSel')
            data_dict = self.css_calib_calc(data_dict)
            self.measure_time_end('SmartSel')
        elif (data_dict['method'] == self.IMPR_CSSHeadSel_Prj or \
                data_dict['method'] == self.IMPR_CSSHeadSel) \
                and ('must_have_heads' in data_dict):
            self.measure_time_start('SmartSel')
            data_dict = self.smart_head_selection(data_dict)
            self.measure_time_end('SmartSel')

        data_dict = self.dense_head.forward_remaining_preds(data_dict)
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        self.measure_time_end('RPN-finalize')

        # Now do postprocess and finish
        self.measure_time_start("PostProcess")
        ss = (data_dict['method'] == self.IMPR_HistoryHeadSel or \
                    data_dict['method'] == self.IMPR_HistoryHeadSel_Prj)
        det_dicts, recall_dict = self.post_processing(data_dict, False, ss)
        self.measure_time_end("PostProcess")
        if data_dict['method'] > self.IMPR_MultiStage:
            heads = data_dict['heads_to_run'].tolist()
        else:
            heads = self.all_heads.tolist()
        self._eval_dict['rpn_stg_exec_seqs'].append((stg_seq, heads))
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

        return data_dict

    def sched_stages_and_heads(self, data_dict, rem_ms):
        if self._calibrating_now:
            selected_cfg = self._cur_calib_tuple
        elif self.CSS_calib > 0:
            selected_cfg = (self.CSS_calib, 6)
        elif self.history_calib > 0:
            selected_cfg = (self.history_calib, 6)
        elif data_dict['method'] == self.IMPR_PTEST:
            selected_cfg = (3, 6)
        else:
            if data_dict['method'] == self.IMPR_MultiStage:
                selected_cfg=(1,6)
            else:
                selected_cfg = (1, 1)
            for t in self.config_tuples:
                if t[0] < rem_ms:
                    selected_cfg = (t[2],t[3])
                    break

        data_dict['num_stgs_to_run'] = selected_cfg[0]
        data_dict['num_heads_to_run'] = selected_cfg[1]
        return data_dict

    def start_projections(self, data_dict):
        do_ptest = (data_dict['method'] == self.IMPR_PTEST)
        if do_ptest:
            proj_scr_thres = 0.
        else:
            #proj_scr_thres = 0.1 * self.last_skipped_heads.shape[0]
            proj_scr_thres = 0.2

        self.chosen_det_dicts, prev_pose_dicts, self.all_indexes = [], [], []
        total_num_of_migrations = 0
        indexes_to_migrate = []
        for h in self.last_skipped_heads:
            if data_dict['method'] == self.IMPR_NearoptHeadSel_Prj and \
                    self.gt_counts[h] == 0:
                continue
            age = self.head_age_arr[h] - 1
            if do_ptest:
                age = 2
            if age < len(self.det_dicts_queue):
                skipped_labels = self.dense_head.heads_to_labels[h]
                dd = self.det_dicts_queue[age][0]
                pred_labels = dd['pred_labels'].tolist()
                pred_scores = dd['pred_scores'].tolist()
                indexes_to_migrate = []
                for i, lbl in enumerate(pred_labels):
                    # migrate confident ones
                    if lbl in skipped_labels and \
                            pred_scores[i] >= proj_scr_thres:
                        indexes_to_migrate.append(i)

                if indexes_to_migrate:
                    self.chosen_det_dicts.append(dd)
                    prev_pose_dicts.append(self.pose_dict_queue[age])
                    self.all_indexes.append(indexes_to_migrate)
                    total_num_of_migrations += len(indexes_to_migrate)

        if total_num_of_migrations == 0:
            self._eval_dict['projections_per_task'].append([0] * self.pool_size)
            return

        # This is where the overhead is, 
        # Create a 2D numpy array for all boxes to be predicted
        # 9 is the single pred box size
        all_pred_boxes = torch.empty((total_num_of_migrations, 9))

        # Generate the dicts for index to cst csr ept epr
        idx_to_pose, i = {}, 0
        for pose_dict, dd, indexes in zip(prev_pose_dicts, \
                self.chosen_det_dicts, self.all_indexes):
            all_pred_boxes[i:i+len(indexes)] = dd['pred_boxes'][indexes]
            for j in range(i, i+len(indexes)):
                idx_to_pose[j] = pose_dict
            i += len(indexes)

        pose = self.token_to_pos[self.latest_token]
        cur_pose_inv = { 'ts' : int(pose['timestamp']),
                'cst_neg' : -np.array(pose['cs_translation']),
                'csr_inv' : Quaternion(pose['cs_rotation']).inverse,
                'ept_neg' : -np.array(pose['ep_translation']),
                'epr_inv' : Quaternion(pose['ep_rotation']).inverse,
        }

        pred_boxes_chunks = np.array_split(all_pred_boxes.numpy(), \
                self.pool_size)

        pose_idx_start = 0
        sizes = []
        for i, pred_boxes_chunk in enumerate(pred_boxes_chunks):
            ar = self.pred_box_pool.apply_async(calcCurPose, \
                    (pred_boxes_chunk, \
                    pose_idx_start, idx_to_pose, cur_pose_inv))
            self.all_async_results.append(ar)
            sz = pred_boxes_chunk.shape[0]
            pose_idx_start += sz
            sizes.append(sz)

        self._eval_dict['projections_per_task'].append(sizes)
        return

    def prioritize_heads(self, data_dict):
        scene_token = self.token_to_pos[self.latest_token]['scene']
        if scene_token != self.cur_scene_token:
            self.cur_scene_token = scene_token
            self.reset_queues_and_arrays()
        num_stages_to_run = data_dict['num_stgs_to_run']
        num_heads_to_run = data_dict['num_heads_to_run']
        method = data_dict['method']

        if self.history_calib > 0:
            data_dict['heads_to_run'] = self.all_heads
            self.last_skipped_heads= np.array([], dtype=np.uint8)
        elif method == self.IMPR_Dryrun:
            data_dict['heads_to_run'] = np.array(self.dryrun_heads)
            self.last_skipped_heads = np.setdiff1d(self.all_heads, \
                   self.dryrun_heads) 
        elif method == self.IMPR_PTEST:
            data_dict['heads_to_run'] = self.all_heads
            self.last_skipped_heads = self.all_heads
        elif method == self.IMPR_NearoptHeadSel_Prj or method == self.IMPR_NearoptHeadSel:
            haa = self.head_age_arr
            age_heap = [(-haa[h], h) for h, gtc in enumerate(self.gt_counts) if gtc > 0]
            if len(age_heap) > num_heads_to_run:
                heapq.heapify(age_heap)
                age_heap = heapq.nsmallest(num_heads_to_run, age_heap)

            data_dict['heads_to_run'] = np.array([t[1] for t in age_heap])
            self.last_skipped_heads = np.setdiff1d(self.all_heads, \
                    data_dict['heads_to_run'])
        elif num_heads_to_run == self.dense_head.num_heads:
            #includes self.IMPR_MultiStage
            data_dict['heads_to_run'] = self.all_heads
            self.last_skipped_heads= np.array([], dtype=np.uint8)
        elif method == self.IMPR_RRHeadSel or method == self.IMPR_RRHeadSel_Prj:
            data_dict['heads_to_run'] = self.rr_heads_queue[:num_heads_to_run]
            self.last_skipped_heads = self.rr_heads_queue[num_heads_to_run:]
            self.rr_heads_queue = np.concatenate((self.last_skipped_heads, \
                    data_dict['heads_to_run']))
        elif method == self.IMPR_PCHeadSel_Prj:
            # Pritoritize the heads to maximize projection performance
            perfs = self.head_age_arr * self.head_proj_calib_mult + \
                    self.head_proj_calib_add
            inv_prios = perfs.argsort()
            data_dict['heads_to_run'] = inv_prios[:num_heads_to_run]
            self.last_skipped_heads = inv_prios[num_heads_to_run:]
        elif method == self.IMPR_HistoryHeadSel or method == self.IMPR_HistoryHeadSel_Prj:
            mqs = len(self.det_dicts_queue)
            if mqs > 0:
                # We have some history that we can use
                ns, htr, pred_score_sizes = self.hist_tuple
                pred_scores = self.det_dicts_queue[0][0]['pred_scores']
                si = 0

                for pss, h in zip(pred_score_sizes, htr):
                    ei = si + pss
                    self.head_scores_arr[h] = torch.sum(pred_scores[si:ei]) * \
                            self.css_coeffs[ns,h]
                    si = ei

            # Build the max heap
            haa = self.head_age_arr
            hsa = self.head_scores_arr
            scr_heap = [(-scr*haa[h], h) \
                    if haa[h] < mqs else (-1024.*haa[h], h)\
                    for h, scr in enumerate(hsa)]
            heapq.heapify(scr_heap)
            heads_to_run, i = np.empty(num_heads_to_run, dtype=np.uint8), 0
            # Pick num_heads_to_run heads having the highest scores
            while i < num_heads_to_run:
                heads_to_run[i] = heapq.heappop(scr_heap)[1]
                i +=1
            data_dict['heads_to_run'] = heads_to_run
            self.last_skipped_heads = np.array([t[1] for t in scr_heap], dtype=np.uint8)

        elif method == self.IMPR_StaticHeadSel:
            prios = self.head_static_prios[num_stages_to_run-1]
            data_dict['heads_to_run'] = prios[:num_heads_to_run].copy()
            self.last_skipped_heads = prios[num_heads_to_run:].copy()
        elif method == self.IMPR_CSSHeadSel_Prj or method == self.IMPR_CSSHeadSel:
            must_have_heads = []
            for h, age in enumerate(self.head_age_arr):
                if age > self.max_queue_size:
                    must_have_heads.append((-age, h))
            if len(must_have_heads) >= num_heads_to_run:
                # No need for smart head selection
                heapq.heapify(must_have_heads)
                mhh = heapq.nsmallest(num_heads_to_run, must_have_heads)
                data_dict['heads_to_run'] = np.array([head[1] for head in mhh])
                self.last_skipped_heads = np.setdiff1d(self.all_heads, \
                        data_dict['heads_to_run'])
            else:
                # Run the classification part of all head, afterwards do
                # prioritization and modify heads_to_run for the rest of
                # the detection heads
                data_dict['must_have_heads'] = set([head[1] for head in must_have_heads])
                data_dict['heads_to_run'] = self.all_heads
                self.last_skipped_heads = self.all_heads # for projection
        data_dict['heads_to_run'].sort()
        self.last_skipped_heads.sort()

        return data_dict

    # CSS
    def smart_head_selection(self, data_dict):
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
        must_have_heads = data_dict['must_have_heads']
        cls_score_sums = torch.full((self.dense_head.num_heads,), 1024., \
                dtype=torch.float32, device='cuda')
        for i, cp in enumerate(data_dict['cls_preds']):
            cls_scores.append(torch.sigmoid(cp))

            cls_scores_keep = cls_scores[-1] > self._score_threshold
            cls_scores_filtered = cls_scores[-1] * \
                    cls_scores_keep.logical_and(expanded_sum_masks[cp.size()[-1]])
            cls_scores[-1] = cls_scores_filtered

            if i not in must_have_heads:
                cls_score_sums[i] = torch.sum(cls_scores_filtered)
        cls_score_sums *= torch.tensor(self.head_age_arr, dtype=torch.float32,\
                device='cuda') * self.css_bef_nms_coeffs[data_dict['num_stgs_to_run']-1]

        prios = cls_score_sums.cpu().argsort(descending=True)

        num_heads_to_run = data_dict['num_heads_to_run']
        data_dict['heads_to_run'] = prios[:num_heads_to_run].numpy()
        data_dict['heads_to_run'].sort()
        self.last_skipped_heads = prios[num_heads_to_run:].numpy()
        self.last_skipped_heads.sort()

        data_dict['cls_preds'] = [cls_scores[h] for h in data_dict['heads_to_run']]
        data_dict['cls_preds_normalized'] = True

        return data_dict

    def css_calib_calc(self, data_dict):
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

        cls_score_sums = torch.empty((len(self.CSS_calib_thresholds), 10) \
                , device='cuda')  #10 is num classes
        for j, thr in enumerate(self.CSS_calib_thresholds):
            i=0
            for cp in data_dict['cls_preds']:
                cls_scores = torch.sigmoid(cp)
                cls_scores_keep = cls_scores > thr
                cls_scores_filtered = cls_scores * \
                        cls_scores_keep.logical_and(expanded_sum_masks[cp.size()[-1]])
                css = torch.sum(cls_scores_filtered, \
                        [0, 1, 2, 3], keepdim=False)
                css_size = css.size()[0]
                cls_score_sums[j, i:i+css_size] = css
                i += css_size

        cls_score_sums = cls_score_sums.cpu().tolist()
        self.gt_and_css_tuples.append((self.gt_counts.tolist(), cls_score_sums))

        data_dict['heads_to_run'] = np.array([0])
        self.last_skipped_heads = np.array([1,2,3,4,5])
        data_dict['cls_preds'] = [data_dict['cls_preds'][0]]

        return data_dict

    def calibrate(self):
        super().calibrate()
        self.clear_stats()

        if self._default_method == self.IMPR_PTEST or \
                self._default_method <= self.BASELINE3:
                    return
        method = self._default_method
        fname = f"calib_dict_{self.dataset.dataset_cfg.DATASET}" \
                f"_m{method}.json"
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
            self.cfg_to_NDS[r-1][c-1] = v['NDS']

        for i in range(len(self.post_sync_time_table_ms)):
            for j in range(len(self.post_sync_time_table_ms[0])):
                self.config_tuples.append(\
                        (self.post_sync_time_table_ms[i][j],\
                        self.cfg_to_NDS[i][j], i+1, j+1))

        self.config_tuples.sort(key=lambda x: -x[1])
        t = 0
        while t < len(self.config_tuples)-1:
            if self.config_tuples[t][0] < self.config_tuples[t+1][0]:
                self.config_tuples.pop(t+1)
            else:
                t += 1
	
        print('Post PFE wcet table:')
        for row in self.post_sync_time_table_ms:
            print(row)

        print('Stage/Head configuration to NDS table:')
        for row in self.cfg_to_NDS:
            print('\t'.join([str(round(nds,4)) for nds in row]))

        print('Config tuples:')
        for t in self.config_tuples:
            print('\t'.join([str(e) for e in t]))

    def do_calibration(self, fname): #, sample_indexes):
        self._calibrating_now = True
        self._cur_calib_tuple = None  # (num_stages, num_heads)
        self._calib_test_cases=[]
        calib_dict = {"data":{}, "stats":{}, "eval":{}, \
                "method":self._default_method}

        if self._default_method == self.IMPR_MultiStage:
            hstart = self.dense_head.num_heads
        else:
            hstart = 1

        for i in range(1, self._num_stages+1):
            for j in range(hstart, self.dense_head.num_heads+1):
                self._calib_test_cases.append((i,j))
        nusc = NuScenes(version='v1.0-mini', \
                dataroot='../data/nuscenes/v1.0-mini', verbose=True)
        gc.disable()

        if self._default_method == self.IMPR_Dryrun:
            from itertools import chain, combinations
            def powerset(iterable):
                "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
                s = list(iterable)
                return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

            all_subsets = list(powerset(self.all_heads))
            subsets_by_len = []
            for sz in range(1,self.dense_head.num_heads+1):
                group = [subset for subset in all_subsets if len(subset) == sz]
                subsets_by_len.append(group)

        for cur_calib_conf in self._calib_test_cases:
            print('Calibrating test case', cur_calib_conf)
            self._cur_calib_tuple = cur_calib_conf

            self.reset_queues_and_arrays()
            self.all_gt_counts = np.zeros(self.dense_head.num_heads, dtype=np.int32)

            if self._default_method == self.IMPR_Dryrun:
                num_heads = self._cur_calib_tuple[1]
                selected_group = subsets_by_len[num_heads-1]
                dryrun_cnt = 0

            det_annos = []
            #progress_bar = tqdm.tqdm(total=len(self.dataset), \
            #        leave=True, desc='eval', dynamic_ncols=True)
            for i in range(len(self.dataset)):
                if self._default_method == self.IMPR_Dryrun:
                    self.dryrun_heads = selected_group[dryrun_cnt]
                    dryrun_cnt += 1
                    if dryrun_cnt == len(selected_group):
                        dryrun_cnt = 0

                with torch.no_grad():
                    batch_dict, pred_dicts, ret_dict = self.load_and_infer(i,
                            {'method': self._default_method})
                annos = self.dataset.generate_prediction_dicts(
                        batch_dict, pred_dicts, self.dataset.class_names,
                        output_path='./temp_results')
                det_annos += annos
                #progress_bar.update()
            #progress_bar.close()
            calib_dict["data"][str(cur_calib_conf)]  = copy.deepcopy(self.get_time_dict())
            stats = self.get_time_dict_stats()
            calib_dict["stats"][str(cur_calib_conf)] = stats
            self.print_time_stats()

            print('All gt counts:', self.all_gt_counts)
            self.print_head_usages()

            self.clear_stats()
            gc.collect()
            torch.cuda.empty_cache()

            result_str, result_dict = self.dataset.evaluation(
                    det_annos, self.dataset.class_names,
                    eval_metric=self.model_cfg.POST_PROCESSING.EVAL_METRIC,
                    output_path='./temp_results', nusc=nusc
                    )
            calib_dict['eval'][str(cur_calib_conf)]  = result_dict # mAP or NDS will be enough
            gc.collect()
            torch.cuda.empty_cache()

        del nusc
        gc.enable()

        with open(fname, 'w') as handle:
            json.dump(calib_dict, handle, indent=4)

        self._calibrating_now = False
        return calib_dict

    def print_head_usages(self):
        head_usage = np.zeros(self.dense_head.num_heads, dtype=np.uint32)
        for es in self._eval_dict['rpn_stg_exec_seqs']:
            head_invocations = es[1]
            for hi in head_invocations:
                head_usage[hi] += 1
        print('Head usages:', head_usage)


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
        self.reset_queues_and_arrays()
        self._eval_dict['rpn_stg_exec_seqs'] = []
        self._eval_dict['projections_per_task'] = []

    def post_eval(self):
        #ppt_arrs = [np.array(ppt) for ppt in self._eval_dict['projections_per_task']]
        #print('Total projections per task:',sum(ppt_arrs))
        self.print_head_usages()
        self.pred_box_pool.close()
        self.pred_box_pool.join()

        if self.CSS_calib > 0 or self.history_calib > 0:
            calib_info = {'tuples': self.gt_and_css_tuples,}
            if self.CSS_calib > 0:
                calib_info['thresholds'] = self.CSS_calib_thresholds
                fname = f'css_calib_{self.CSS_calib}.json'
            else:
                fname = f'history_calib_{self.history_calib}.json'
            with open(fname, 'w') as handle:
                json.dump(calib_info, handle, indent=4)
