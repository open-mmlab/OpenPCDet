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


def calcCurPose(pbs_np, pose_idx_start, idx_to_pose, cur_pose_inv):
    #USE_VEL = True
    #if USE_VEL:
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
        self.IMPR_AgingHeadSel = 6        # head select aging
        self.IMPR_SmartHeadSel = 7        # head select smart
        self.IMPR_AgingHeadSel_Prj = 8    # 6 + Projection
        self.IMPR_SmartHeadSel_Prj = 9    # 7 + Projection
        self.IMPR_RRHeadSel_Prj = 10      # 5 + Projection
        self.IMPR_SmarterHeadSel_Prj = 11 # Not smart actually
        self.IMPR_SmartRRHeadSel_Prj = 12 # Smart history
        self.IMPR_PTEST = 13              # Projection test

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
            'Prioritize': [],
            'SmartSel': [],
            'PrePrediction': [],
            'PostPrediction': [],
            'RPN-stage-2': [],
            'RPN-stage-3': [],
            'RPN-finalize': [],
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
            self.cur_scene_token = ''
            #self._eval_dict['data_dict['gt_counts']'] = []

        #print('Model:')
        #print(self)

        self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)
        self.last_head = self.dense_head.num_heads -1
        self.rrhq_restart = 0
        #self.det_conf_table= np.zeros((self._num_stages, \
        #        self.dense_head.num_heads, self.dense_head.num_heads,), dtype=np.float32)

        # AP scores from trainval evaluation
        self.det_conf_table = torch.tensor([
                # [car], [truck, construction_vehicle], [bus, trailer], 
                # [barrier], [motorcycle, bicycle], [pedestrian, traffic_cone]
                [0.438, 0.114, 0.006, 0.127, 0.028, 0.107, 0.116, 0.000, 0.216, 0.101],
                [0.527, 0.321, 0.029, 0.446, 0.172, 0.131, 0.151, 0.003, 0.242, 0.129],
                [0.593, 0.357, 0.042, 0.485, 0.267, 0.210, 0.185, 0.007, 0.262, 0.157],
        ], dtype=torch.float32, device='cuda')

        self.latest_token = None
        self.last_skipped_heads= np.array([], dtype=np.uint8)
        self.last_cls_scores= {}
        self.det_hist_queue = [] # list of (pose_dict, last_skipped_heads, det_dicts)
        self.max_queue_size = self.dense_head.num_heads
        self.hist_cnt = 1

        # prediction
        #self.migrate_scr_thres = 0.1  # .2 avoids overhead
        self.pool_size = 6  # 6 appears to give best results on jetson-agx
        self.pred_box_pool = Pool(self.pool_size)
        self.prediction_timing = {}
        self.chosen_det_dicts, self.all_indexes = [], []
        self.all_async_results = []

        with open('token_to_pos.json', 'r') as handle:
            self.token_to_pos = json.load(handle)

        self.use_oracle = False
        if self.use_oracle:
            with open('token_to_anns.json', 'r') as handle:
                self.token_to_anns= json.load(handle)

    def use_projection(self, data_dict):
        m = data_dict['method'] 

        return (m == self.IMPR_SmartHeadSel_Prj or \
                m == self.IMPR_SmarterHeadSel_Prj or \
                m == self.IMPR_SmartRRHeadSel_Prj or \
                m == self.IMPR_AgingHeadSel_Prj or \
                m == self.IMPR_RRHeadSel_Prj or \
                m == self.IMPR_PTEST)

    def forward(self, data_dict):
        if not self.training:
            if self._calibrating_now:
                self.all_gt_counts += data_dict['gt_counts'][0]
            self.latest_token = data_dict['metadata'][0]['token']
            self.all_async_results = []
            self.proj_started=False

            # det_dicts length is equal to batch size
            det_dicts, recall_dict = self.eval_forward(data_dict)

            dd = det_dicts[0] # Assume batch size is 1
            if data_dict['method'] == self.IMPR_SmartRRHeadSel_Prj:
                self.measure_time_start('SmartSel')
                label_counts= torch.bincount(dd['pred_labels'], \
                        minlength=11)[1:] # num labels + 1
                label_scores = label_counts * \
                        self.det_conf_table[data_dict['num_stgs_to_run']-1]
                label_scores = label_scores.cpu()
                self.last_cls_scores={h:.0 for h in range(self.dense_head.num_heads)}
                for l in range(len(label_scores)):
                    h = self.dense_head.labels_to_heads[l]
                    self.last_cls_scores[h] += label_scores[l]

                for h in self.last_skipped_heads:
                    del self.last_cls_scores[h]
                self.measure_time_end('SmartSel')

            for k,v in dd.items():
                dd[k] = v.cpu()

            if data_dict['method'] == self.IMPR_SmarterHeadSel_Prj:
                self.last_cls_scores={}
                heads_to_run = data_dict['heads_to_run']
                #prio_scores = torch.zeros((heads_to_run.shape[0],))

                cp_dim1_sizes = [cp.size()[1] for cp in data_dict['cls_preds']]
                css_final = []
                self.cls_score_sums = self.cls_score_sums.cpu()
                idx=0
                for i, dim1sz in enumerate(cp_dim1_sizes):
                    if dim1sz == 2:
                        css_final.append(self.cls_score_sums[idx] / \
                                self.dense_head.anchor_area_coeffs[i][0])
                        idx += 1
                    else:
                        css1 = sum(self.cls_score_sums[idx:idx+2]) / \
                                self.dense_head.anchor_area_coeffs[i][0]
                        css2 = sum(self.cls_score_sums[idx+2:idx+4]) / \
                                self.dense_head.anchor_area_coeffs[i][1]
                        css_final.append(css1+css2)
                        idx += 4

                for i, h in enumerate(heads_to_run):
                    self.last_cls_scores[h] = css_final[i]

#            elif data_dict['method'] == self.IMPR_SmartRRHeadSel_Prj and \
#                    data_dict['calc_css']:
#                for h, scr in zip(data_dict['heads_to_run'], dd['cls_score_sums']):
#                    self.last_cls_scores[h] = scr
#                    if self._calibrating_now:
#                        gtc = data_dict['gt_counts'][0][h]
#                        if gtc == 0. and scr > 0.:
#                            conf = 0.
#                        elif gtc == 0. and scr == 0.:
#                            conf = 1.
#                        elif scr <= gtc:
#                            conf = scr / gtc
#                        else:
#                            conf = (2*gtc - scr) / gtc
#                            if conf < 0.:
#                                conf = 0.
#                        ns, nh = self._cur_calib_tuple
#                        self.det_conf_table[ns-1, nh-1, h] += conf
#                del dd['cls_score_sums']
#                self.measure_time_end('SmartSel', False)

            if self.use_oracle:
                oracle_dd = self.token_to_anns[self.latest_token]
                oracle_dd['pred_boxes'] = \
                        torch.as_tensor(oracle_dd['pred_boxes'])
                oracle_dd['pred_scores'] = \
                        torch.as_tensor(oracle_dd['pred_scores'])
                oracle_dd['pred_labels'] = \
                        torch.as_tensor(oracle_dd['pred_labels'])
                det_dicts = [oracle_dd] * data_dict['batch_size']

            pose_dict = None
            det_dicts_ret = det_dicts
            if self.use_projection(data_dict):
                self.measure_time_start('PostPrediction', False)
                pose = self.token_to_pos[self.latest_token]
                pose_dict = { 'ts' : int(pose['timestamp']),
                        'cst' : np.array(pose['cs_translation']),
                        'csr' : Quaternion(pose['cs_rotation']),
                        'ept' : np.array(pose['ep_translation']),
                        'epr' : Quaternion(pose['ep_rotation'])
                        }

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

            self.det_hist_queue.append((pose_dict, \
                    self.last_skipped_heads.copy(), det_dicts, \
                    self.last_cls_scores.copy()))


            # backup1 : fixed queue size

            # backup2 : Dynamically adjust queue size
            #self.max_queue_size = self.dense_head.num_heads // \
                    #        data_dict['heads_to_run'].shape[0]

            while len(self.det_hist_queue) > self.max_queue_size:
                self.det_hist_queue.pop(0)

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
        if 'method' not in data_dict:
            data_dict['method'] = self._default_method

        data_dict = self.pre_rpn_forward(data_dict)

        if data_dict['method'] >= self.IMPR_MultiStage:
            post_pfe_event = torch.cuda.Event()

        self.measure_time_start('Post-PFE')
        self.measure_time_start('RPN-stage-1')
        data_dict = self.backbone_2d(data_dict)
        self.measure_time_end('RPN-stage-1')
        stg_seq=[1]
        data_dict['score_thresh'] = self._score_threshold

        if data_dict['method'] >= self.IMPR_MultiStage:
            post_pfe_event.synchronize()
            # The overhead of sched can be ignored
            self.measure_time_start('Prioritize', False)
            data_dict = self.sched_stages_and_heads(data_dict)
            data_dict = self.prioritize_heads(data_dict)
            self.measure_time_end('Prioritize', False)

            if self.use_projection(data_dict) and \
                    data_dict['do_smart_select'] == False:
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
        if data_dict['method'] >= self.IMPR_MultiStage and data_dict['do_smart_select']:
            self.measure_time_start('SmartSel')
            data_dict = self.smart_head_selection(data_dict)
            self.measure_time_end('SmartSel')
        data_dict = self.dense_head.forward_remaining_preds(data_dict)
        # migrate detections from previous frame if possible
        # while detection head is running on GPU
        if data_dict['method'] == self.IMPR_SmarterHeadSel_Prj:
            # could be a better place to do this
            self.measure_time_start('SmartSel')
            data_dict = self.calc_cls_score_sums(data_dict)
            self.measure_time_end('SmartSel')
        if not self.proj_started and self.use_projection(data_dict):
            self.measure_time_start('PrePrediction', False)
            self.start_projections(data_dict)
            self.measure_time_end('PrePrediction', False)
        #torch.cuda.nvtx.range_pop()
        data_dict = self.dense_head.gen_pred_boxes(data_dict)
        self.measure_time_end('RPN-finalize')

        #if self.all_async_results:
        #    self.measure_time_start('PostPrediction', False)
        #    self.projected_boxes = torch.from_numpy(np.concatenate( \
                #            [ar.get() for ar in self.all_async_results]))
        #    self.measure_time_end('PostPrediction', False)

        self.measure_time_start("PostProcess")
        # Now do postprocess and finish
        #torch.cuda.nvtx.range_push('PostProcess')

        #ss = (data_dict['method'] == self.IMPR_SmartRRHeadSel_Prj)
        #det_dicts, recall_dict = self.post_processing(data_dict, False, ss)
        det_dicts, recall_dict = self.post_processing(data_dict, False)
        #torch.cuda.nvtx.range_pop()
        self.measure_time_end("PostProcess")
        if data_dict['method'] > self.IMPR_MultiStage:
            self._eval_dict['rpn_stg_exec_seqs'].append( \
                    (stg_seq, data_dict['heads_to_run'].tolist()))
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

        data_dict['sh_config'] = selected_cfg
        data_dict['num_stgs_to_run'] = selected_cfg[0]
        return data_dict

    def start_projections(self, data_dict):
        self.proj_started=True

        self.chosen_det_dicts, prev_pose_dicts, self.all_indexes = [], [], []
        total_num_of_migrations = 0
        if self._default_method == self.IMPR_PTEST:
            dhq = self.det_hist_queue[:self.hist_cnt]
            self.hist_cnt += 1
            if self.hist_cnt == self.max_queue_size+1:
                self.hist_cnt = 1
        else:
            dhq = self.det_hist_queue

        proj_scr_thres = 0.1 * self.last_skipped_heads.shape[0]
        #proj_scr_thres = self._score_threshold

        for h in self.last_skipped_heads:
            for pose_dict, hs, det_dicts, _ in reversed(dhq):
                if h not in hs or self._default_method == self.IMPR_PTEST:
                    # get the corresponding detections if they exist
                    skipped_labels = self.dense_head.heads_to_labels[h]
                    for dd in det_dicts:
                        prev_det_labels = dd['pred_labels'].tolist()
                        prev_det_scores = dd['pred_scores'].tolist()
                        indexes_to_migrate = []
                        i = 0
                        for lbl, score in zip(prev_det_labels, prev_det_scores):
                            # migrate confident ones
                            if score >= proj_scr_thres and lbl in skipped_labels:
                                #if lbl in skipped_labels:
                                indexes_to_migrate.append(i)
                            i += 1

                        if indexes_to_migrate:
                            self.chosen_det_dicts.append(dd)
                            prev_pose_dicts.append(pose_dict)
                            self.all_indexes.append(indexes_to_migrate)
                            total_num_of_migrations += len(indexes_to_migrate)
                    break

        if total_num_of_migrations == 0:
            return

        # This is where the overhead is, 
        # Create a 2D numpy array for all boxes to be predicted
        # 9 is a single pred box size
        all_pred_boxes = torch.zeros((total_num_of_migrations, 9))

        # Generate the dicts for index to cst csr ept epr
        idx_to_pose = {}
        i = 0
        for pose_dict, dd, indexes_to_migrate in zip(prev_pose_dicts, \
                self.chosen_det_dicts, self.all_indexes):
            all_pred_boxes[i:i+len(indexes_to_migrate)] = \
                    dd['pred_boxes'][indexes_to_migrate]
            for j in range(i, i+len(indexes_to_migrate)):
                idx_to_pose[j] = pose_dict
            i += len(indexes_to_migrate)

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
        for pred_boxes_chunk in pred_boxes_chunks:
            ar = self.pred_box_pool.apply_async(calcCurPose, \
                    (pred_boxes_chunk, \
                    pose_idx_start, idx_to_pose, cur_pose_inv))
            self.all_async_results.append(ar)
            pose_idx_start += pred_boxes_chunk.shape[0]

    def get_aging_prios(self):
        prios = np.ones(self.dense_head.num_heads, dtype=np.float32)
        head_skips = [dh[1] for dh in self.det_hist_queue]

        for h in range(self.dense_head.num_heads):
            p=1.
            for hs_arr in reversed(head_skips):
                if h not in hs_arr: # if not skipped
                    break
                p*=2
            prios[h] = p
        return prios

    def get_totally_skipped_heads(self, aging_prios):
        max_prio = pow(2, len(self.det_hist_queue))
        heads = np.arange(self.dense_head.num_heads)
        return heads[aging_prios==max_prio]

    def prioritize_heads(self, data_dict):
        #scene_token = self.nusc.get('sample', self.latest_token)['scene_token']
        scene_token = self.token_to_pos[self.latest_token]['scene']
        if scene_token != self.cur_scene_token:
            self.det_hist_queue = [] # reset queue
            self.last_cls_scores=  np.array([], dtype=np.float32)
            self.last_skipped_heads= np.array([], dtype=np.uint8)
            self.cur_scene_token = scene_token

            self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)
            self.rrhq_restart = 0
            self.last_head = self.dense_head.num_heads -1

        selected_cfg = data_dict['sh_config']

        if data_dict['method'] == self.IMPR_PTEST:
            self.last_skipped_heads = np.arange(self.dense_head.num_heads, dtype=np.uint8)
            data_dict['heads_to_run'] = np.arange(self.dense_head.num_heads, dtype=np.uint8)
            data_dict['do_smart_select'] = False
        elif selected_cfg[1] == self.dense_head.num_heads:
            #includes self.IMPR_MultiStage
            data_dict['heads_to_run'] = np.arange(self.dense_head.num_heads, dtype=np.uint8)
            self.last_skipped_heads= np.array([], dtype=np.uint8)
            data_dict['do_smart_select'] = False
            #data_dict['calc_css'] = False
        elif data_dict['method'] == self.IMPR_RRHeadSel or \
                data_dict['method'] == self.IMPR_RRHeadSel_Prj:
            data_dict['heads_to_run'] = self.rr_heads_queue[:selected_cfg[1]]
            self.last_skipped_heads = self.rr_heads_queue[selected_cfg[1]:]
            self.rr_heads_queue = np.concatenate((self.last_skipped_heads, data_dict['heads_to_run']))
            data_dict['heads_to_run'].sort()
            self.last_skipped_heads.sort()
            data_dict['do_smart_select'] = False

        elif data_dict['method'] == self.IMPR_AgingHeadSel or \
                data_dict['method'] == self.IMPR_AgingHeadSel_Prj:
            prios = self.get_aging_prios()
            prios = prios.argsort()
            data_dict['heads_to_run'] = prios[-selected_cfg[1]:]
            data_dict['heads_to_run'].sort()
            self.last_skipped_heads = prios[:-selected_cfg[1]]
            self.last_skipped_heads.sort()
            data_dict['do_smart_select'] = False

        elif data_dict['method'] == self.IMPR_SmartHeadSel or \
                data_dict['method'] == self.IMPR_SmartHeadSel_Prj:
            aging_prios = self.get_aging_prios()
            hprio_heads = self.get_totally_skipped_heads(aging_prios)
            if hprio_heads.shape[0] >= selected_cfg[1]:
                # No need for smart head selection
                data_dict['heads_to_run'] = hprio_heads[:selected_cfg[1]]
                data_dict['heads_to_run'].sort()
                heads = np.arange(self.dense_head.num_heads)
                self.last_skipped_heads = np.setdiff1d(heads, data_dict['heads_to_run'])
                self.last_skipped_heads.sort()
                data_dict['do_smart_select'] = False
            else:
                data_dict['must_have_heads'] = hprio_heads.tolist()
                data_dict['aging_prios'] = torch.tensor(aging_prios, dtype=torch.float32)
                data_dict['do_smart_select'] = True
        elif data_dict['method'] == self.IMPR_SmarterHeadSel_Prj:
            aging_prios = self.get_aging_prios() # good
            hprio_heads = self.get_totally_skipped_heads(aging_prios) # good
            #print('Stats:')
            #print(aging_prios)
            #print(hprio_heads)
            heads = np.arange(self.dense_head.num_heads)
            if hprio_heads.shape[0] >= selected_cfg[1]:
                # No need for smarter head selection
                data_dict['heads_to_run'] = hprio_heads[:selected_cfg[1]]
                self.last_skipped_heads = np.setdiff1d(heads, data_dict['heads_to_run'])
            else:
                # We have a class score history for the heads_to_compare
                # since they were not skipped totally, use that to
                # make priorization
                prio_scores = np.full((self.dense_head.num_heads,),
                        np.finfo(np.float32).max, dtype=np.float32)
                for dh in self.det_hist_queue:
                    cls_scores_dict = dh[-1]
                    for h, score in cls_scores_dict.items():
                        # The recent one will override the older one
                        prio_scores[h] = score
                prios = prio_scores.argsort()
                selected_cfg = data_dict['sh_config']
                data_dict['heads_to_run'] = prios[-selected_cfg[1]:]
                self.last_skipped_heads = prios[:-selected_cfg[1]]
                #print(prio_scores) 
            data_dict['heads_to_run'].sort()
            self.last_skipped_heads.sort()
            #print('heads_to_run', data_dict['heads_to_run']) 
            data_dict['do_smart_select'] = False
        elif data_dict['method'] == self.IMPR_SmartRRHeadSel_Prj:
            if (self.rrhq_restart == 1 and selected_cfg[1] <= 3) or \
                    (self.rrhq_restart >= 2 and selected_cfg[1] > 3):
                #Make prioritization based on history
                self.rrhq_restart = 0

                prio_scores = np.zeros((self.dense_head.num_heads,),
                        dtype=np.float32)
                for dh in self.det_hist_queue:
                    cls_scores_dict = dh[-1]
                    for h, score in cls_scores_dict.items():
                        # The recent one will override the older one
                        prio_scores[h] = score
                
                prios = prio_scores.argsort()
                selected_cfg = data_dict['sh_config']
                data_dict['heads_to_run'] = prios[-selected_cfg[1]:]
                self.last_skipped_heads = prios[:-selected_cfg[1]]
                # modify rr queue in a way that the selected heads are put to end
                # this is done to make projection more effective
                self.rr_heads_queue = \
                        np.setdiff1d(self.rr_heads_queue, data_dict['heads_to_run'])
                self.rr_heads_queue = np.concatenate((self.rr_heads_queue, \
                        data_dict['heads_to_run']))
                self.last_head=self.rr_heads_queue[-1]
            else:
                # Do round robin
                data_dict['heads_to_run'] = self.rr_heads_queue[:selected_cfg[1]]
                # Check whether all heads in the rr queue are executed
                #if self.last_head in data_dict['heads_to_run']: 
                if self.last_head in data_dict['heads_to_run']: 
                    self.rrhq_restart += 1
                self.last_skipped_heads = self.rr_heads_queue[selected_cfg[1]:]
                self.rr_heads_queue = np.concatenate(\
                        (self.last_skipped_heads, data_dict['heads_to_run']))

            #data_dict['calc_css'] = True
            data_dict['heads_to_run'].sort()
            #print('heads_to_run', data_dict['heads_to_run'], end="\n\n")
            self.last_skipped_heads.sort()
            data_dict['do_smart_select'] = False
        else:
            data_dict['do_smart_select'] = False

        return data_dict

    def calc_cls_score_sums(self, data_dict):
        stg0_sum = torch.sum(data_dict['spatial_features'], 1, keepdim=True)
        sum_mask = torch.nn.functional.max_pool2d(stg0_sum, 19,
                stride=4, padding=9).unsqueeze(-1)

        cls_scores = []
        self.cls_score_sums = []
        self.cutoff_scores = []
        cutoff_threshold = 0.5

        cp_dim1_sizes = [cp.size()[1] for cp in data_dict['cls_preds']]
        cls_preds_chunks = []
        for cp, dim1sz in zip(data_dict['cls_preds'], cp_dim1_sizes):
            if dim1sz == 2:
                cls_preds_chunks.append(cp)
            else:
                chunks = torch.chunk(cp, 2, dim=1)
                for c in chunks:
                    chunks = torch.chunk(c, 2, dim=4)
                    cls_preds_chunks.extend(chunks)

        cls_preds_cat = torch.cat(cls_preds_chunks, dim=0)
        cls_scores = torch.sigmoid(cls_preds_cat)

        cls_scores_keep = cls_scores > self._score_threshold
        expanded_sum_mask = sum_mask.expand(cls_scores.size()).type(torch.bool)
        cls_scores_filtered = cls_scores * \
                cls_scores_keep.logical_and(expanded_sum_mask)

        self.cls_score_sums = torch.sum(cls_scores_filtered, \
                [1, 2, 3, 4], keepdim=False)

        cutoff_scores = torch.sum((cls_scores_filtered > cutoff_threshold) \
                * cls_scores_filtered, [1,2,3,4], keepdim=False) * 100.0

        self.cls_score_sums += cutoff_scores

        return data_dict

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
        cls_score_sums = []
        cutoff_threshold = 0.5
        cutoff_scores = torch.zeros((self.dense_head.num_heads,), \
                device=data_dict['cls_preds'][0].device)
        must_have_heads = data_dict['must_have_heads']
        for i, cp in enumerate(data_dict['cls_preds']):
            cls_scores.append(torch.sigmoid(cp))

            # first,calculate the score which determines whether
            # this class has high importance
            if i not in must_have_heads:
                cutoff_scores[i] = torch.sum((cls_scores[-1] > cutoff_threshold) \
                        * cls_scores[-1])

                # Second, calculate the priorities to include
            cls_scores_keep = cls_scores[-1] > self._score_threshold
            cls_scores_filtered = cls_scores[-1] * \
                    cls_scores_keep.logical_and(expanded_sum_masks[cp.size()[-1]])
            cls_scores[-1] = cls_scores_filtered

            if i in must_have_heads:
                cls_score_sums.append(torch.finfo(torch.float32).max)
            else:
                cls_score_sums.append(torch.sum(cls_scores_filtered, \
                        [0, 1, 2, 3], keepdim=False))

                prio_scores = torch.zeros((self.dense_head.num_heads,))
        for i, s in enumerate(cls_score_sums):
            if i in must_have_heads:
                prio_scores[i] = s
            else:
                for j, c in enumerate(s.cpu()):
                    prio_scores[i] += c / self.dense_head.anchor_area_coeffs[i][j]

        prio_scores = (cutoff_scores.cpu() * 100) + prio_scores
        prios = prio_scores.argsort(descending=True)

        selected_cfg = data_dict['sh_config']
        data_dict['heads_to_run'] = prios[:selected_cfg[1]]
        self.last_skipped_heads = prios[selected_cfg[1]:]

#        new_cls_preds = []
#        for h in data_dict['heads_to_run']:
#            new_cls_preds.append(cls_scores[h])
#        data_dict['cls_preds'] = new_cls_preds
        data_dict['cls_preds'] = cls_scores
        data_dict['cls_preds_normalized'] = True

        return data_dict

    def calibrate(self):
        super().calibrate()

        if self._default_method == self.IMPR_PTEST or \
                self._default_method <= self.BASELINE3:
                    return

        method  = self.IMPR_AgingHeadSel if self._default_method == \
                self.IMPR_MultiStage else self._default_method
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
            self.post_sync_time_table_ms[r-1][c-1] *= 1.03 # add some pessimism

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

        print('Detection confidence tables (stage, num heads, num heads)')
        for i, row in enumerate(self.det_conf_table):
            print(f'Stage {i}:\n', row)

    def do_calibration(self, fname): #, sample_indexes):
        self._calibrating_now = True
        self._cur_calib_tuple = None  # (num_stages, num_heads)
        self._calib_test_cases=[]
        calib_dict = {"data":{}, "stats":{}, "eval":{}, \
                "det_conf":[], "method":self._default_method}
        for i in range(1, self._num_stages+1):
            for j in range(1, self.dense_head.num_heads+1):
                self._calib_test_cases.append((i,j))
        nusc = NuScenes(version='v1.0-mini', dataroot='../data/nuscenes/v1.0-mini', verbose=True)
        gc.disable()

        for cur_calib_conf in self._calib_test_cases:
            print('Calibrating test case', cur_calib_conf)
            self._cur_calib_tuple = cur_calib_conf

            self.det_hist_queue = []
            self.last_skipped_heads= np.array([], dtype=np.uint8)
            self.last_cls_scores= {}

            if self._default_method == self.IMPR_SmartRRHeadSel_Prj:
                self.rr_heads_queue = np.arange(self.dense_head.num_heads, dtype=np.uint8)
                self.last_head = self.dense_head.num_heads -1
                self.rrhq_restart = 0
            self.all_gt_counts = np.zeros(self.dense_head.num_heads, dtype=np.uint32)

            det_annos = []
            #for i in sample_indexes:
            progress_bar = tqdm.tqdm(total=len(self.dataset), \
                    leave=True, desc='eval', dynamic_ncols=True)
            for i in range(len(self.dataset)):
                with torch.no_grad():
                    batch_dict, pred_dicts, ret_dict = self.load_and_infer(i,
                            {'method': self._default_method})
                    #    pred_dicts = [ self.get_empty_det_dict() for p in pred_dicts ]
                annos = self.dataset.generate_prediction_dicts(
                        batch_dict, pred_dicts, self.dataset.class_names,
                        output_path='./temp_results'
                        )
                det_annos += annos
                progress_bar.update()
                #if i % (len(self.dataset)//4) == 0:
                #    print(i, '/', len(self.dataset), ' done')
            progress_bar.close()
            calib_dict["data"][str(cur_calib_conf)]  = copy.deepcopy(self.get_time_dict())
            stats = self.get_time_dict_stats()
            calib_dict["stats"][str(cur_calib_conf)] = stats
            self.print_time_stats()

            print('All gt counts:', self.all_gt_counts)
            head_usage = np.zeros(self.dense_head.num_heads, dtype=np.uint32)
            for es in self._eval_dict['rpn_stg_exec_seqs']:
                head_invocations = es[1]
                for hi in head_invocations:
                    head_usage[hi] += 1
            print('Head usages:', head_usage)
#            if self._default_method == self.IMPR_SmartRRHeadSel_Prj:
#                s, h = cur_calib_conf
#                for i, hu in enumerate(head_usage):
#                    self.det_conf_table[s-1, h-1, i] /= hu
#                print('Detection confidence tables (stage, num heads, num heads)')
#                for i, table in enumerate(self.det_conf_table):
#                    print(f'Stage {i+1}:\n', table)

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

        #calib_dict['sample_indexes'] = sample_indexes.tolist()
        #calib_dict["det_conf"] = self.det_conf_table.tolist()
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


    def post_eval(self):
        keys = sorted(self.prediction_timing.keys())
        print('Prediction times:')
        for k in keys:
            print(k, ':', self.prediction_timing[k])
        self.pred_box_pool.close()
        self.pred_box_pool.join()
