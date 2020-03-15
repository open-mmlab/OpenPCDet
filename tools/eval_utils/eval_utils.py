import tqdm
import time
import pickle
from pcdet.config import cfg
from pcdet.models import example_convert_to_torch


def statistics_info(ret_dict, metric, disp_dict):
    if cfg.MODEL.RCNN.ENABLED:
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] += ret_dict['roi_%s' % str(cur_thresh)]
            metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict['rcnn_%s' % str(cur_thresh)]
        metric['gt_num'] += ret_dict['gt']
        min_thresh = cfg.MODEL.TEST.RECALL_THRESH_LIST[0]
        disp_dict['recall_%s' % str(min_thresh)] = \
            '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(model, dataloader, epoch_id, logger, save_to_file=False, result_dir=None, test_mode=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    if save_to_file:
        final_output_dir = result_dir / 'final_result' / 'data'
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        final_output_dir = None

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, data in enumerate(dataloader):
        input_dict = example_convert_to_torch(data)
        pred_dicts, ret_dict = model(input_dict)
        disp_dict = {}

        statistics_info(ret_dict, metric, disp_dict)
        annos = dataset.generate_annotations(input_dict, pred_dicts, class_names,
                                             save_to_file=save_to_file, output_dir=final_output_dir)
        det_annos += annos
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

    progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    ret_dict = {}
    if cfg.MODEL.RCNN.ENABLED:
        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall_roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall_rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['num_example']
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(det_annos, class_names, eval_metric=cfg.MODEL.TEST.EVAL_METRIC)

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
