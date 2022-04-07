from .detector3d_template import Detector3DTemplate
import os
import gc
import torch
import math
import time
import copy
import json
import tqdm
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
        self.IMPR_RRHeadSel = 5           # head select round-robin
        self.IMPR_PCHeadSel = 6           # Projection calibrated selection
        self.IMPR_DynamicHeadSel = 7      # (Not so) smart history
        self.IMPR_RRHeadSel_Prj = 8       # 5 + Projection
        self.IMPR_PCHeadSel_Prj = 9       # Projection calibrated selection
        self.IMPR_DynamicHeadSel_Prj = 10 # (Not so) smart history

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

        # Debug/plot related
        # self._pc_range = dataset.point_cloud_range

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

        if self._sep_mhead:
            # Times below include postprocess
            # These tables are overwritten by the calibration
            self.post_sync_time_table_ms = [
                    # heads    1    2    3    4    5    6
                    [ 10000. ] * self.dense_head.num_heads,
                    [ 10000. ] * self.dense_head.num_heads,
                    [ 10000. ] * self.dense_head.num_heads,
            ]
            self.cfg_to_NDS = [
                    [ 0. ] * self.dense_head.num_heads,
                    [ 0. ] * self.dense_head.num_heads,
                    [ 0. ] * self.dense_head.num_heads,
            ]
            self.cur_scene_token = ''

        #print('Model:')
        #print(self)


        # AP scores from trainval evaluation
        #self.det_conf_table = torch.tensor([
        #        [0.438, 0.114, 0.006, 0.127, 0.028, 0.107, 0.116, 0.000, 0.216, 0.101],
        #        [0.527, 0.321, 0.029, 0.446, 0.172, 0.131, 0.151, 0.003, 0.242, 0.129],
        #        [0.593, 0.357, 0.042, 0.485, 0.267, 0.210, 0.185, 0.007, 0.262, 0.157],
        #], dtype=torch.float32, device='cuda')

        self.head_proj_calib_table = np.array([
            # [car], [truck, construction_vehicle], [bus, trailer], 
            # [barrier], [motorcycle, bicycle], [pedestrian, traffic_cone]
            [0.980, 0.982, 0.987, 0.983, 0.970, 0.970], #150ms
            [0.960, 0.969, 0.983, 0.967, 0.946, 0.945], #300ms
            [0.936, 0.947, 0.969, 0.951, 0.915, 0.916], #450ms
            [0.905, 0.931, 0.957, 0.936, 0.884, 0.894], #600ms
            [0.871, 0.910, 0.935, 0.921, 0.846, 0.868]  #750ms
        ], dtype=np.float32)
        self.head_proj_calib_table = np.concatenate((self.head_proj_calib_table,
            np.zeros((6,6), dtype=np.float32)))

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

        self.latest_token = None
        if self._default_method == self.IMPR_DynamicHeadSel or \
                self._default_method == self.IMPR_DynamicHeadSel_Prj:
            self.max_queue_size = self.dense_head.num_heads
        else:
            self.max_queue_size = self.dense_head.num_heads - 1
        self.hist_cnt = 1

        # holds the time passed since a head was used for all heads
        self.head_age_arr = np.full(self.dense_head.num_heads, \
                self.max_queue_size, dtype=np.uint8)
        self.head_scores_arr = np.zeros(self.dense_head.num_heads, \
                dtype=np.uint8)
        self.det_dicts_queue = deque(maxlen=self.max_queue_size)
        self.pose_dict_queue = deque(maxlen=self.max_queue_size)
        self.last_skipped_heads= np.array([], dtype=np.uint8)

        #RR
        self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)
        self.last_head = self.dense_head.num_heads -1
        self.rrhq_restart = 0

        # prediction
        self.pool_size = 6  # 6 appears to give best results on jetson-agx
        self.pred_box_pool = Pool(self.pool_size)
        self.chosen_det_dicts, self.all_indexes = [], []
        self.all_async_results = []

        with open('token_to_pos.json', 'r') as handle:
            self.token_to_pos = json.load(handle)

        self.use_oracle = False
        if self.use_oracle:
            with open('token_to_anns.json', 'r') as handle:
                self.token_to_anns= json.load(handle)


    def reset_queues_and_arrays(self):
        self.head_age_arr.fill(self.max_queue_size)
        self.head_scores_arr.fill(.0)
        self.det_dicts_queue.clear()
        self.pose_dict_queue.clear()
        self.last_skipped_heads= np.array([], dtype=np.uint8)

        #RR
        self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)
        self.last_head = self.dense_head.num_heads -1
        self.rrhq_restart = 0


    def use_projection(self, data_dict):
        m = data_dict['method'] 

        return (m == self.IMPR_DynamicHeadSel_Prj or \
                m == self.IMPR_PCHeadSel_Prj or \
                m == self.IMPR_RRHeadSel_Prj or \
                m == self.IMPR_PTEST)
#                m == self.IMPR_StaticHeadSel_Prj or \

    def forward(self, data_dict):
        if not self.training:
            if self._calibrating_now:
                self.all_gt_counts += data_dict['gt_counts'][0]
            if 'method' not in data_dict:
                data_dict['method'] = self._default_method
            data_dict['score_thresh'] = self._score_threshold
            self.latest_token = data_dict['metadata'][0]['token']

            # det_dicts length is equal to batch size
            det_dicts, recall_dict = self.eval_forward(data_dict)

            dd = det_dicts[0] # Assume batch size is 1
            for k,v in dd.items():
                dd[k] = v.cpu()

            if data_dict['method'] == self.IMPR_DynamicHeadSel or \
                    data_dict['method'] == self.IMPR_DynamicHeadSel_Prj:
                self.measure_time_start('SmartSel')
                label_counts= torch.bincount(dd['pred_labels'], \
                        minlength=11)[1:] # num labels + 1
                #label_scores = label_counts * \
                #        self.det_conf_table[data_dict['num_stgs_to_run']-1]

                for h in data_dict['heads_to_run']:
                    self.head_scores_arr[h] = 0.
                for l in range(len(label_scores)):
                    h = self.dense_head.labels_to_heads[l]
                    self.head_scores_arr[h] += label_scores[l]
                self.measure_time_end('SmartSel')

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
            if self.use_projection(data_dict):
                self.all_async_results.clear()
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
                        #print('all_pred_boxes final   size:', all_pred_boxes.size())
                        for k in ['pred_scores', 'pred_labels']:
                            det_to_migrate[k]= torch.cat( \
                                    [dd[k][i] for dd, i in zip( \
                                    self.chosen_det_dicts, self.all_indexes)])

                        if data_dict['method'] == self.IMPR_PTEST:
                            det_dicts_ret = [det_to_migrate] * len(det_dicts)
                        else:
                            det_dicts_ret = []
                            for dd in det_dicts:
                                dd_ = {k:torch.cat([dd[k], det_to_migrate[k]]) \
                                        for k in dd.keys()}
                                det_dicts_ret.append(dd_)
                self.measure_time_end('PostPrediction', False)

            for h in data_dict['heads_to_run']:
                self.head_age_arr[h] = 0
            for h in self.last_skipped_heads:
                self.head_age_arr[h] += 1

            self.measure_time_end("Post-PFE")

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

        if data_dict['method'] >= self.IMPR_MultiStage:
            post_pfe_event = torch.cuda.Event()

        self.measure_time_start('Post-PFE')
        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-stage-1')
        stg_seq=[1]

        if data_dict['method'] >= self.IMPR_MultiStage:
            post_pfe_event.synchronize()
            # The overhead of sched can be ignored
            self.measure_time_start('Prioritize', False)
            data_dict = self.sched_stages_and_heads(data_dict)
            data_dict = self.prioritize_heads(data_dict)
            self.measure_time_end('Prioritize', False)

            # migrate detections from previous frame if possible
            # while detection head is running on GPU
            if self.use_projection(data_dict):
                        # Early projection start possible as no smart select
                self.measure_time_start('PrePrediction', False)
                self.start_projections(data_dict)
                self.measure_time_end('PrePrediction', False)
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

        self.measure_time_start('RPN-finalize')
        data_dict = self.dense_head.forward_cls_preds(data_dict)
        data_dict = self.dense_head.forward_remaining_preds(data_dict)
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        self.measure_time_end('RPN-finalize')

        # Now do postprocess and finish
        self.measure_time_start("PostProcess")
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        self.measure_time_end("PostProcess")
        if data_dict['method'] > self.IMPR_MultiStage:
            heads = data_dict['heads_to_run'].tolist()
        else:
            heads = np.arange(self.dense_head.num_heads).tolist()
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

    def sched_stages_and_heads(self, data_dict):
        rem_ms = (data_dict['abs_deadline_sec'] - time.time()) * 1000.0

        if self._calibrating_now:
            selected_cfg = self._cur_calib_tuple
        elif data_dict['method'] == self.IMPR_PTEST:
            selected_cfg = (3, 6)
        elif data_dict['method'] == self.IMPR_MultiStage:
            selected_cfg = (1, 6)
            best_NDS = self.cfg_to_NDS[0][-1]
            for i in range(1, len(self.post_sync_time_table_ms)):
                if self.post_sync_time_table_ms[i][-1] <= rem_ms and \
                        self.cfg_to_NDS[i][-1] > best_NDS:
                    best_NDS = self.cfg_to_NDS[i][-1]
                    selected_cfg = (i+1, 6)
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

        data_dict['num_stgs_to_run'] = selected_cfg[0]
        data_dict['num_heads_to_run'] = selected_cfg[1]
        return data_dict

    def start_projections(self, data_dict):
        do_ptest = (self._default_method == self.IMPR_PTEST)
        if do_ptest:
            proj_scr_thres = data_dict['score_thresh']
        else:
            proj_scr_thres = 0.1 * self.last_skipped_heads.shape[0]

#        for h in self.last_skipped_heads:
#            for pose_dict, hs, det_dicts in reversed(dhq):
#                if h not in hs or self._default_method == self.IMPR_PTEST:
#                    # get the corresponding detections if they exist
#                    skipped_labels = self.dense_head.heads_to_labels[h]
#                    dd = det_dicts[0]
#                    prev_det_labels = dd['pred_labels'].tolist()
#                    prev_det_scores = dd['pred_scores'].tolist()
#                    indexes_to_migrate = []
#                    i = 0
#                    for lbl, score in zip(prev_det_labels, prev_det_scores):
#                        # migrate confident ones
#                        if score >= proj_scr_thres and lbl in skipped_labels:
#                            #if lbl in skipped_labels:
#                            indexes_to_migrate.append(i)
#                        i += 1
#
#                    if indexes_to_migrate:
#                        self.chosen_det_dicts.append(dd)
#                        prev_pose_dicts.append(pose_dict)
#                        self.all_indexes.append(indexes_to_migrate)
#                        total_num_of_migrations += len(indexes_to_migrate)
#                    break

        self.chosen_det_dicts, prev_pose_dicts, self.all_indexes = [], [], []
        total_num_of_migrations = 0
        indexes_to_migrate = []
        for h in self.last_skipped_heads:
            age = self.head_age_arr[h]
            if age < len(self.det_dicts_queue) or do_ptest:
                if do_ptest:
                    age = self.hist_cnt
                skipped_labels = self.dense_head.heads_to_labels[h]
                dd = self.det_dicts_queue[age][0]
                #if not (type(dd['pred_labels']).__module__ == np.__name__):
                if not isinstance(dd['pred_labels'], list):
                    dd['pred_labels'] = dd['pred_labels'].tolist()
                    dd['pred_scores'] = dd['pred_scores'].tolist()
                    dd['pred_boxes']  = dd['pred_boxes'].numpy()
                pred_labels, pred_scores = dd['pred_labels'], dd['pred_scores']
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
        all_pred_boxes = np.empty((total_num_of_migrations, 9))

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

        pred_boxes_chunks = np.array_split(all_pred_boxes, \
                self.pool_size)

        pose_idx_start = 0
        sizes = []
        for pred_boxes_chunk in pred_boxes_chunks:
            ar = self.pred_box_pool.apply_async(calcCurPose, \
                    (pred_boxes_chunk, \
                    pose_idx_start, idx_to_pose, cur_pose_inv))
            self.all_async_results.append(ar)
            sz = pred_boxes_chunk.shape[0]
            pose_idx_start += sz
            sizes.append(sz)

        self._eval_dict['projections_per_task'].append(sizes)
        return

#    def estimate_proj_perf(self):
#        head_skips = [dh[1] for dh in self.det_hist_queue]
#            for i, hs_arr in enumerate(reversed(head_skips)):
#                if h not in hs_arr: # if not skipped
#                    perf[h] = self.head_proj_calib_table[i][h]
#                    break
#        return perfs

#    def get_aging_prios(self):
#        prios = np.ones(self.dense_head.num_heads, dtype=np.float32)
#        head_skips = [dh[1] for dh in self.det_hist_queue]
#
#        for h in range(self.dense_head.num_heads):
#            p=1.
#            for hs_arr in reversed(head_skips):
#                if h not in hs_arr: # if not skipped
#                    break
#                p*=2
#            prios[h] = p
#        return prios

#    def get_totally_skipped_heads(self, aging_prios):
#        max_prio = pow(2, len(self.det_hist_queue))
#        heads = np.arange(self.dense_head.num_heads)
#        return heads[aging_prios==max_prio]

    def prioritize_heads(self, data_dict):
        scene_token = self.token_to_pos[self.latest_token]['scene']
        if scene_token != self.cur_scene_token:
            self.cur_scene_token = scene_token
            self.reset_queues_and_arrays()
#        num_stages_to_run = data_dict['num_stages_to_run']
        num_heads_to_run = data_dict['num_heads_to_run']
        method = data_dict['method']

        if method == self.IMPR_PTEST:
            data_dict['heads_to_run'] = np.arange(self.dense_head.num_heads, dtype=np.uint8)
            self.last_skipped_heads = np.arange(self.dense_head.num_heads, dtype=np.uint8)
        elif num_heads_to_run == self.dense_head.num_heads:
            #includes self.IMPR_MultiStage
            data_dict['heads_to_run'] = np.arange(self.dense_head.num_heads, dtype=np.uint8)
            self.last_skipped_heads= np.array([], dtype=np.uint8)
#        elif data_dict['method'] == self.IMPR_StaticHeadSel or \
#                data_dict['method'] == self.IMPR_StaticHeadSel_Prj:
#            # First, include whatever is missing in the history
#            aging_prios = self.get_aging_prios()
#            hprio_heads = self.get_totally_skipped_heads(aging_prios)
#            if hprio_heads.shape[0] >= num_heads_to_run:
#                # No need for prioritization
#                data_dict['heads_to_run'] = hprio_heads[:num_heads_to_run]
#                data_dict['heads_to_run'].sort()
#            else:
#                other_heads = self.head_static_prios[num_stgs_to_run-1]
#                other_heads = np.setdiff1d(other_heads, hprio_heads)
#                rem_heads_num = num_heads_to_run - hprio_heads.shape[0]
#                data_dict['heads_to_run'] = np.concatenate((hprio_heads,
#                    other_heads[:rem_heads_num]))
#                data_dict['heads_to_run'].sort()
#            heads = np.arange(self.dense_head.num_heads)
#            self.last_skipped_heads = np.setdiff1d(heads, data_dict['heads_to_run'])
#            self.last_skipped_heads.sort()
#
        elif method == self.IMPR_RRHeadSel or method == self.IMPR_RRHeadSel_Prj:
            data_dict['heads_to_run'] = self.rr_heads_queue[:num_heads_to_run]
            self.last_skipped_heads = self.rr_heads_queue[num_heads_to_run:]
            self.rr_heads_queue = np.concatenate((self.last_skipped_heads, \
                    data_dict['heads_to_run']))

        elif method == self.IMPR_PCHeadSel or method == self.IMPR_PCHeadSel_Prj:
            # Pritoritize the heads to maximize projection performance
            perfs = np.empty(self.dense_head.num_heads, dtype=np.float32)
            for h in range(self.dense_head.num_heads):
                age = self.head_age_arr[h]
                perfs[h] = self.head_proj_calib_table[age][h]
            inv_prios = perfs.argsort()
            data_dict['heads_to_run'] = inv_prios[:num_heads_to_run]
            self.last_skipped_heads = inv_prios[num_heads_to_run:]

        elif method == self.IMPR_DynamicHeadSel or method == self.IMPR_DynamicHeadSel_Prj:
            if self.rrhq_restart >= 1:
                #Make prioritization based on history
                self.rrhq_restart = 0

                prios = self.head_scores_arr.argsort()
                data_dict['heads_to_run'] = prios[-num_heads_to_run:]
                self.last_skipped_heads = prios[:-num_heads_to_run]
                # modify rr queue in a way that the selected heads are put to end
                # this is done to make projection more effective
                self.rr_heads_queue = \
                        np.setdiff1d(self.rr_heads_queue, data_dict['heads_to_run'])
                self.rr_heads_queue = np.concatenate((self.rr_heads_queue, \
                        data_dict['heads_to_run']))
                self.last_head=self.rr_heads_queue[-1]
            else:
                # Do round robin
                data_dict['heads_to_run'] = self.rr_heads_queue[:num_heads_to_run]
                # Check whether all heads in the rr queue are executed
                #if self.last_head in data_dict['heads_to_run']: 
                if self.last_head in data_dict['heads_to_run']: 
                    self.rrhq_restart += 1
                self.last_skipped_heads = self.rr_heads_queue[num_heads_to_run:]
                self.rr_heads_queue = np.concatenate(\
                        (self.last_skipped_heads, data_dict['heads_to_run']))

        data_dict['heads_to_run'].sort()
        self.last_skipped_heads.sort()

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
            self.post_sync_time_table_ms[r-1][c-1] *= 1.05 # add some pessimism

        for k, v in calib_dict['eval'].items():
            r,c = get_rc(k)
            self.cfg_to_NDS[r-1][c-1] = round(v['NDS'], 3)

#        for i in range(len(self.cfg_to_NDS)):
#            for j in range(1, len(self.cfg_to_NDS[0])):
#                if self.cfg_to_NDS[i][j-1] >= self.cfg_to_NDS[i][j]:
#                    self.cfg_to_NDS[i][j] = self.cfg_to_NDS[i][j-1] + 0.001

        print('Post PFE wcet table:')
        for row in self.post_sync_time_table_ms:
            print(row)

        print('Stage/Head configuration to NDS table:')
        for row in self.cfg_to_NDS:
            print(row)

        #print('Detection confidence tables (stage, num heads, num heads)')
        #for i, row in enumerate(self.det_conf_table):
        #    print(f'Stage {i}:\n', row)

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
        nusc = NuScenes(version='v1.0-mini', dataroot='../data/nuscenes/v1.0-mini', verbose=True)
        gc.disable()

        for cur_calib_conf in self._calib_test_cases:
            print('Calibrating test case', cur_calib_conf)
            self._cur_calib_tuple = cur_calib_conf


            self.reset_queues_and_arrays()
            self.all_gt_counts = np.zeros(self.dense_head.num_heads, dtype=np.uint32)

            det_annos = []
            #for i in sample_indexes:
            progress_bar = tqdm.tqdm(total=len(self.dataset), \
                    leave=True, desc='eval', dynamic_ncols=True)
            for i in range(len(self.dataset)):
                with torch.no_grad():
                    batch_dict, pred_dicts, ret_dict = self.load_and_infer(i,
                            {'method': self._default_method})
                annos = self.dataset.generate_prediction_dicts(
                        batch_dict, pred_dicts, self.dataset.class_names,
                        output_path='./temp_results')
                det_annos += annos
                progress_bar.update()
            progress_bar.close()
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
        self._eval_dict['rpn_stg_exec_seqs'] = []


    def post_eval(self):
        self.print_head_usages()
#        keys = sorted(self.prediction_timing.keys())
#        print('Prediction times:')
#        for k in keys:
#            print(k, ':', self.prediction_timing[k])
        self.pred_box_pool.close()
        self.pred_box_pool.join()
