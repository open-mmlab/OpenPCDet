import pickle
import time

import numpy as np
import torch
import tqdm
import os

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def inference_one_epoch(inference_data_path, inference_results, cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, inference=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s inference *****************' % epoch_id)
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
    print("*" * 60)
    inference_data_ls = sorted(os.listdir(inference_data_path))
    for i in range(len(inference_data_ls)):
        inference_pc = inference_data_path + inference_data_ls[i]
        batch_dict = dataset.inference_getitem(inference_pc)
        print("batch_dict", batch_dict)
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None, inference=inference, inference_results=inference_results
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    # for i, batch_dict in enumerate(dataloader):
    #     load_data_to_gpu(batch_dict)
    #     with torch.no_grad():
    #         pred_dicts, ret_dict = model(batch_dict)
    #     disp_dict = {}
    #
    #     statistics_info(cfg, ret_dict, metric, disp_dict)
    #     annos = dataset.generate_prediction_dicts(
    #         batch_dict, pred_dicts, class_names,
    #         output_path=final_output_dir if save_to_file else None, inference=inference
    #     )
    #     det_annos += annos
    #     if cfg.LOCAL_RANK == 0:
    #         progress_bar.set_postfix(disp_dict)
    #         progress_bar.update()


    return ret_dict


if __name__ == '__main__':
    pass
