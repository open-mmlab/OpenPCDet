import pickle
import time

import numpy as np
import torch
import tqdm

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


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
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
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_one_epoch_memorybank(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
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
    memory_dict = {}
    infer_time = []
    feature_bank = []
    io_time = []
    aug_time = []
    traj_time  = []
    crop_point = []
    feature_time = []
    voxelize_time = []
    transformer_time = []
    post_time = []
    static_time= []
    pvrcnn_time = []
    point_mask = []
    rois_mask = []
    time_mask = []
    window_time = []
    grid_time = []
    forward_time = []
    match_time = []
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):

        time1 = time.time()
        # io_time.append(batch_dict['io_time'].item())
        # aug_time.append(batch_dict['loader_time'].item())
        load_data_to_gpu(batch_dict)
        # import pdb;pdb.set_trace()
        # batch_dict['sample_idx'][0] = i

        if batch_dict['sample_idx'][0] >=1:
            #if i >=4:
            batch_dict['grid_feature_memory'] = memory_dict['grid_feature_memory']
            batch_dict['feature_bank'] = feature_bank

            # batch_dict['traj_memory'] = memory_dict['traj_memory']
            # batch_dict['pos_fea_memory'] = memory_dict['pos_fea_memory']
            # batch_dict['pos_fea_ori_memory'] = memory_dict['pos_fea_ori_memory']
            # batch_dict['voxel_point_memory'] = memory_dict['voxel_point_memory']
            # batch_dict['src_ori_memory'] = memory_dict['src_ori_memory']
        eval_time= time.time()
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)

        infer_time.append(time.time() - eval_time)
        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)
        static_time.append(time.time() - time1)
        # pvrcnn_time.append(batch_dict['pvrcnn_time'])
        try:
            feature_time.append(batch_dict['feature_time'])
            traj_time.append(batch_dict['traj_time'])
            crop_point.append(batch_dict['crop_time'])
            voxelize_time.append(batch_dict['voxelize_time1'])
            transformer_time.append(batch_dict['transformer_time'])
            point_mask.append(batch_dict['point_mask_time'])
            rois_mask.append(batch_dict['rois_crop_time'])
            time_mask.append(batch_dict['time_mask_time'])
            time_mask.append(batch_dict['time_mask_time'])
            window_time.append(batch_dict['4window_time'])
            forward_time.append(batch_dict['forward_time'])
            grid_time.append(batch_dict['voxelize_time2'])
            post_time.append(batch_dict['post_time'])
            match_time.append(batch_dict['match_time'])
        except:
            pass

        if batch_dict['sample_idx'][0] >=0:
            #if i >=3:
            # memory_dict['traj_memory'] = batch_dict['trajectory_rois']
            if len(feature_bank) <=(cfg.MODEL.ROI_HEAD.Transformer.num_frames-1) :
                feature_bank.insert(0,batch_dict['grid_feature_memory'][:,:64])
            else:
                feature_bank.pop()
                feature_bank.insert(0,batch_dict['grid_feature_memory'][:,:64])
            memory_dict['grid_feature_memory'] = batch_dict['grid_feature_memory']
            # memory_dict['pos_fea_memory'] = batch_dict['pos_fea_memory']
            # memory_dict['pos_fea_ori_memory'] = batch_dict['pos_fea_ori_memory']
            # memory_dict['src_ori_memory'] = batch_dict['src_ori_memory']
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    logger.info('Infer_Time %.4f ' % (np.array(infer_time).mean()))
    logger.info('IO_Time %.4f loader_time %.4f traj_time %.4f crop_point %.4f point_mask %.4f time_mask %.4f rois_mask %.4f feature_time %.4f  voxelize1 %.4f  match %.4f voxelize2 %.4f transformer_time %.4f post_time %.4f window %.4f forword %.4f static_time %.4f).' % \
                 (np.array(io_time).mean(),np.array(aug_time).mean(),np.array(traj_time).mean(),np.array(crop_point).mean(), np.array(point_mask).mean(),
                  np.array(time_mask).mean(),np.array(rois_mask).mean(), np.array(feature_time).mean(),np.array(voxelize_time).mean(),np.array(match_time).mean(),np.array(grid_time).mean(),
                  np.array(transformer_time).mean(), np.array(post_time).mean(),np.array(window_time).mean(),np.array(forward_time).mean(),np.array(static_time).mean()))


    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )


    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict



if __name__ == '__main__':
    pass
