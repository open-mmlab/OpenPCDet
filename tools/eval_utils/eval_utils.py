import pickle
import time

import numpy as np
import torch
import tqdm
import gc

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

#from torch.profiler import profile, record_function, ProfilerActivity

speed_test=False
visualize=False

if visualize:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def dataset_eval_from_file(saved_dets, dataset, cfg, logger, result_dir):
    with open(saved_dets, 'rb') as f:
        det_annos = pickle.load(f)

    result_str, result_dict = dataset.evaluation(
        det_annos, dataset.class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=result_dir
    )

    logger.info(result_str)
    print(result_str)

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, saved_dets=None):
    global visualize
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    if saved_dets is not None:
        dataset_eval_from_file(saved_dets, dataloader.dataset, cfg,
                logger, final_output_dir)
        return

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

    # Forward once for initialization and calibration
    if 'calibrate' in dir(model):
        with torch.no_grad():
            torch.cuda.cudart().cudaProfilerStop()
            model.calibrate()
            torch.cuda.cudart().cudaProfilerStart()
            print("Calibration complete.")


    start_time = time.time()
    gc.disable()
    # Currently, batch size of 1 is supported only
    #if cfg.MODEL.get('DEADLINE_SEC', None) is not None:
    #    dl = float(cfg.MODEL.DEADLINE_SEC)
    global speed_test
    num_samples = 100 if speed_test and len(dataset) >= 100 else len(dataset)
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=num_samples, leave=True, desc='eval', dynamic_ncols=True)

    ###################################
    # Code for drawing the input region
#    vs_ = 0.2
#    gs_ = 102.4 / vs_
#    grid_size = gs_ * gs_
#    prev_z_vals = torch.zeros(int(grid_size), device='cuda')
#    prev_mask = torch.full((int(grid_size),), True, device='cuda')
#
#    grid2d = open3d.geometry.LineSet()
#    grid2d.points = open3d.utility.Vector3dVector( \
#            np.array(((-51.2, -51.2, -3), (51.2, -51.2, -3), \
#            (-51.2, 51.2, -3), (51.2, 51.2, -3)), dtype=np.double))
#    grid2d.lines = open3d.utility.Vector2iVector( \
#            np.array(((0,1),(1,3),(3,2),(2,0)), dtype=np.int))
#    grid2d.colors =  open3d.utility.Vector3dVector(np.ones((4, 3)))
    ###################################
    #with profile(record_shapes=True) as prof:
    for i in range(num_samples):
        with torch.no_grad():
            batch_dict, pred_dicts, ret_dict = model.load_and_infer(i)
        #            {'deadline_sec':dl})
        if visualize:
            V.draw_scenes(
                points=batch_dict['points'][:, 1:], ref_boxes=batch_dict['gt_boxes'][0],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
            )
            #point_colors=point_colors, grid2d=grid2d,

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

    if 'post_eval' in dir(model):
        model.post_eval()

    gc.collect()
    gc.enable()

    #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

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

    model.print_time_stats()

    if speed_test:
        model.dump_eval_dict(ret_dict)
        model.clear_stats()
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
    ret_dict['result_str'] = result_str

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    model.dump_eval_dict(ret_dict)
    model.clear_stats()

    return ret_dict


if __name__ == '__main__':
    pass
