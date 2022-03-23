import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import torch.nn.functional as F


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # metric = {
    #     'gt_num': 0,
    # }
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] = 0
    #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    # class_names = dataset.class_names
    # det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    rot_loss = 0.0
    shift_loss = 0.0
    num_count = 0
    disp_dict = {}
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        num_count += 1
        with torch.no_grad():
            # pred_dicts, ret_dict = model(batch_dict)
            return_dict = model(batch_dict)
            rot_preds = return_dict['batch_rot_preds']
            shift_preds = return_dict['batch_shift_preds']
            dir_preds = return_dict['batch_dir_preds']
            
            rot_labels = batch_dict['rot_labels']
            shift_labels = batch_dict['shift_labels']
            dir_labels = batch_dict['dir_labels']
            
            if rot_preds is not None:
                if dir_preds is not None:
                    negative_index = dir_preds < 0.5
                    rot_preds[negative_index] = rot_preds[negative_index] * -1
                    rot_loss_iter = F.l1_loss(rot_preds, rot_labels).data.cpu().numpy()
                    rot_loss += rot_loss_iter
                else:
                    rot_loss_iter = F.l1_loss(rot_preds, rot_labels).data.cpu().numpy()
                    rot_loss += rot_loss_iter
                disp_dict['rot_loss'] = rot_loss_iter
            if shift_preds is not None:
                shift_loss_iter = F.l1_loss(shift_preds, shift_labels).data.cpu().numpy()
                shift_loss += shift_loss_iter
                disp_dict['shift_loss'] = shift_loss_iter
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    
    shift_avg_loss = shift_loss / num_count
    rot_avg_loss = rot_loss / num_count
    logger.info('finished evaluation')
    logger.info(shift_avg_loss)
    logger.info(rot_avg_loss)
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    return 




if __name__ == '__main__':
    pass
