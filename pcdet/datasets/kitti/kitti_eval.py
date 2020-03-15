import copy
from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
import argparse
import pickle


def evaluation(det_annos, gt_infos, class_names, **kwargs):
    if 'annos' not in gt_infos[0]:
        return 'None', {}

    eval_det_annos = copy.deepcopy(det_annos)
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_infos]
    ap_result_str, ap_dict = get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

    return ap_result_str, ap_dict


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=list, default=['Car', 'Pedestrian', 'Cyclist'], help='')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the kitti format results...')
    ap_result_str, ap_dict = evaluation(
        pred_infos, gt_infos,
        class_names=['Car'] # , 'Pedestrian', 'Cyclist']
    )
    print(ap_result_str)


if __name__ == '__main__':
    main()
