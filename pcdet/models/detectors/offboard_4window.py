import torch
from .detector3d_template import Detector3DTemplate
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import os
import numpy as np
import time
from ...utils import common_utils
from pcdet.datasets.augmentor import augmentor_utils, database_sampler

class Offboard_4window(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):

        start_time1 = time.time()
        batch_dict['proposals_list'] = batch_dict['roi_boxes']

        if self.model_cfg.ROI_HEAD.get('USE_BEV_FEAT',None):
            with torch.no_grad():
                for cur_module in self.module_list[:-1]:
                    batch_dict = cur_module(batch_dict)
            batch_dict =  self.module_list[-1](batch_dict)
            
        else:
            for cur_module in self.module_list[:]:
                batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:
            """
            if self.model_cfg.POST_PROCESSING.get('SAVE_BBOX',False):
                pred_dicts, recall_dicts = {},{}
                for bs_idx in range(batch_dict['batch_size']):
                    
                    cur_boxes = batch_dict['trajectory_rois'][bs_idx,0,:,:7]
                    motion_mask = batch_dict['motion_mask'].long()
                    motion_mask = motion_mask[bs_idx].bool()

                    cur_boxes = cur_boxes[motion_mask]
                    if cur_boxes.shape[0] > 0 and batch_dict['gt_norm_boxes'][bs_idx,:,:7].shape[0] > 0:
                        trajectory_rois = batch_dict['trajectory_rois'][bs_idx,:,motion_mask,:7]
                        iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_boxes, batch_dict['gt_norm_boxes'][bs_idx,:,:7])  # (M, N)
            
                        max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
                        fg_inds = ((max_overlaps >= 0.55)).nonzero().view(-1)
                        cur_gtboxes = batch_dict['gt_norm_boxes'][bs_idx][gt_assignment[fg_inds]]
                        # cur_boxes = cur_boxes[fg_inds]
                        # cur_gtpreboxes = batch_dict['gt_pre_boxes'][bs_idx,:,gt_assignment[fg_inds]]

                        trajectory_rois = trajectory_rois[:,fg_inds]
                        roi_points = batch_dict['roi_traj_points'][bs_idx][fg_inds]

                        traj_scores = batch_dict['trajectory_roi_scores'][bs_idx,:,motion_mask][:,fg_inds,None]
                        # from tools.visual_utils import ss_visual_utils as V
                        # V.draw_scenes(roi_points.view(-1,6)[:,:3], cur_gtpreboxes.view(-1,7))
                        #cur_labels = pred_dicts[bs_idx]['pred_labels'][motion_mask]
                        cur_boxes = torch.cat([trajectory_rois,traj_scores],-1)

                        # cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
                        # for i in range(trajectory_rois.shape[0]):
                        #     cur_radiis = torch.sqrt((trajectory_rois[i,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.2
                        #     dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
                        #     point_mask = (dis <= cur_radiis.unsqueeze(-1))


                        # cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
                        # cur_radiis = torch.sqrt((cur_boxes[:,3]/2) ** 2 + (cur_boxes[:,4]/2) ** 2) * 1.2
                        # dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
                        # point_mask = (dis <= cur_radiis.unsqueeze(-1))
                        # non_empty_idx = []



                        #trajectory_roi_scores[fg_inds] = roi_scores_list[idx][i][gt_assignment[fg_inds]]

                        for k, roi_box_idx in enumerate(range(roi_points.shape[0])):
                            keypoints = roi_points[k]
                            if keypoints.shape[0] >0:


                                # non_empty_idx.append(k)
                                # import pdb; pdb.set_trace()
                                # sampled_points = cur_points[point_mask[roi_box_idx]].unsqueeze(dim=0)  # (1, N, 3)

                                # cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3].contiguous(), 512).long()

                                # if sampled_points.shape[1] < 512:
                                #     times = int(512 / sampled_points.shape[1]) + 1
                                #     non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                                #     cur_pt_idxs[0] = non_empty.repeat(times)[:512]

                                # keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

                                path = '/result/4frame_1point_normsup_car_motion_point_6/%s/' % (batch_dict['frame_id'][bs_idx][:-4])
                                if not os.path.exists(path):
                                    try:
                                        os.makedirs(path)
                                    except:
                                        pass
                                point_path = path + ' %s_%d.npy' % (batch_dict['frame_id'][bs_idx][-3:],k)
                                np.save(point_path, keypoints.cpu().numpy())
                                path = '/result/4frame_1point_normsup_car_motion_bbox_6/%s/' % (batch_dict['frame_id'][bs_idx][:-4])
                                if not os.path.exists(path):
                                    try:
                                        os.makedirs(path)
                                    except:
                                        pass
                                bbox_path = path + '%s_%d.npy' % (batch_dict['frame_id'][bs_idx][-3:],k)
                        
                                #cur_gtboxes = torch.cat([cur_gtboxes[k:k+1],torch.ones([1,1]).cuda()],-1)
                                pre_gt_boxes = torch.cat([cur_boxes[:,k],cur_gtboxes[k:k+1,:8]],dim=0)
                                np.save(bbox_path, pre_gt_boxes.cpu().numpy())
                                # db_info = {'path': '%s/%s_%d.npy' % ((batch_dict['frame_id'][bs_idx][:-4]),batch_dict['frame_id'][bs_idx][-3:],k), 'sequence_name': batch_dict['frame_id'][bs_idx][:-4],
                                #     'pred_box3d':cur_boxes[k:k+1].cpu().numpy(),'gt_box3d': cur_gtboxes[k:k+1].cpu().numpy(), 'num_points_in_gt': keypoints.shape[0]}
                                # self.all_db_info['Vehicle'].append(db_info)
            """
            # import pdb;pdb.set_trace()
            if self.model_cfg.POST_PROCESSING.get('SAVE_BBOX',False):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
               
                for bs_idx in range(batch_dict['batch_size']):
                    # cur_boxes = pred_dicts[bs_idx]['pred_boxes']
                    # cur_scores = pred_dicts[bs_idx]['pred_scores']
                    # cur_labels = pred_dicts[bs_idx]['pred_labels']
                    # cur_superboxes = pred_dicts[bs_idx]['pred_superboxes']

                    cur_boxes = batch_dict['batch_box_preds'][bs_idx]
                    cur_scores = batch_dict['batch_cls_preds'][bs_idx]
                    cur_labels = batch_dict['roi_labels'][bs_idx]
                    cur_superboxes = batch_dict['pred_superboxes'][bs_idx]
                    


                    path = '/home/xschen/OpenPCDet_xuesong/iter_6933/%s/' % (batch_dict['frame_id'][bs_idx][:-4])

                    if not os.path.exists(path):
                        try:
                            os.makedirs(path)
                        except:
                            pass
                    bbox_path = path + '%s.npy' % (batch_dict['frame_id'][bs_idx][-3:])
            
                    #cur_gtboxes = torch.cat([cur_gtboxes[k:k+1],torch.ones([1,1]).cuda()],-1)
                    pred_boxes = torch.cat([cur_boxes,cur_scores,cur_labels[:,None],cur_superboxes],dim=-1)
                    np.save(bbox_path, pred_boxes.cpu().numpy())

            else:
                start_time = time.time()
                pred_dicts, recall_dicts = self.post_processing(batch_dict,nms=self.model_cfg.POST_PROCESSING.get('USE_NMS',True))
                # from tools.visual_utils import ss_visual_utils as V
                # import pdb
                # pdb.set_trace()

                if self.model_cfg.POST_PROCESSING.get('TTA', False):
                    points = batch_dict['points'][:1,1:].cpu().numpy()
                    pred_bbox = pred_dicts[0]['pred_boxes'].cpu().numpy()
                    # V.draw_scenes(batch_dict['points'][:,1:], pred_bbox, batch_dict['gt_boxes'][0,:])
                    # if isinstance(pred_bbox,torch.Tensor):
                    #     pred_bbox = pred_bbox.cpu().numpy()
                    if   'flip_x_enabled' in batch_dict.keys() and not 'flip_y_enabled' in batch_dict.keys(): 
                        cur_axis = 'x'
                        pred_bbox, points,enabled = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(pred_bbox, points,1)
                        # pred_dicts[0]['pred_boxes'] = gt_boxes
                    elif 'flip_y_enabled' in batch_dict.keys() and not 'flip_x_enabled' in batch_dict.keys():
                        cur_axis = 'y'
                        pred_bbox, points,enabled = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(pred_bbox, points,1)
                        # pred_dicts[0]['pred_boxes'] = gt_boxes

                    elif 'flip_x_enabled' in batch_dict.keys() and 'flip_y_enabled' in batch_dict.keys():
                        cur_axis = 'x'
                        pred_bbox, points,enabled = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(pred_bbox, points,1)
                        cur_axis = 'y'
                        pred_bbox, points,enabled = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(pred_bbox, points,1)
                        # pred_dicts[0]['pred_boxes'] = gt_boxes

                    elif 'yaw' in batch_dict.keys(): 
                        yaw = batch_dict['yaw'][0].cpu().numpy()
                        pred_bbox, points, _ = augmentor_utils.global_rotation(pred_bbox, points, rot_range=[-yaw,-yaw])
                        # pred_dicts[0]['pred_boxes'] = gt_boxes

                    elif 'z_shift' in batch_dict.keys():
                        pred_bbox, points, z_shift = augmentor_utils.translation_along_z(pred_bbox,points, z_shift=-batch_dict['z_shift'][0].cpu().numpy(), prob=None)
                        # pred_dicts[0]['pred_boxes'] = gt_boxes

                    elif 'scale' in batch_dict.keys():
                        scale = 1/batch_dict['scale'].cpu().numpy()
                        pred_bbox, points, _ = augmentor_utils.global_scaling(pred_bbox, points, [scale,scale])


                    pred_dicts[0]['pred_boxes'] = torch.from_numpy(pred_bbox).cuda()
                    
                
                # print(batch_dict['frame_id'],batch_dict['pred_boxes'].shape)
                # from tools.visual_utils import ss_visual_utils as V
                # import pdb
                # pdb.set_trace()
                # V.draw_scenes(batch_dict['points'][:,1:], pred_dicts[0]['pred_boxes']
            torch.cuda.empty_cache()
            batch_dict['post_time'] = time.time() - start_time
            batch_dict['4window_time'] = time.time() - start_time
            return pred_dicts, recall_dicts, batch_dict

    def get_training_loss(self):
        disp_dict = {}  
        if self.model_cfg.ONLY_TRAIN_RCNN:
            tb_dict ={}
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rcnn
        elif self.model_cfg.ONLY_TRAIN_RPN:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss = loss_rpn
        else:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

