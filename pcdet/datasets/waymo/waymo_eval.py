# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import numpy as np
import pickle
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2
import argparse


tf.get_logger().setLevel('INFO')


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class OpenPCDetWaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

    def generate_waymo_type_results(self, infos, class_names, is_gt=False, fake_gt_infos=True):
        def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
            """
            Args:
                boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

            Returns:
                boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            """
            w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
            boxes3d_lidar[:, 2] += h[:, 0] / 2
            return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []
        for frame_index, info in enumerate(infos):
            if is_gt:
                box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
                if 'num_points_in_gt' in info:
                    zero_difficulty_mask = info['difficulty'] == 0
                    info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
                    info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
                    nonzero_mask = info['num_points_in_gt'] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    print('Please provide the num_points_in_gt for evaluating on Waymo Dataset '
                          '(If you create Waymo Infos before 20201126, please re-create the validation infos '
                          'with version 1.2 Waymo dataset to get this attribute). SSS of OpenPCDet')
                    raise NotImplementedError

                num_boxes = box_mask.sum()
                box_name = info['name'][box_mask]

                difficulty.append(info['difficulty'][box_mask])
                score.append(np.ones(num_boxes))
                if fake_gt_infos:
                    info['gt_boxes_lidar'] = boxes3d_kitti_fakelidar_to_lidar(info['gt_boxes_lidar'])

                if info['gt_boxes_lidar'].shape[-1] == 9:
                    boxes3d.append(info['gt_boxes_lidar'][box_mask][:, 0:7])
                else:
                    boxes3d.append(info['gt_boxes_lidar'][box_mask])
            else:
                num_boxes = len(info['boxes_lidar'])
                difficulty.append([0] * num_boxes)
                score.append(info['score'])
                boxes3d.append(np.array(info['boxes_lidar'][:, :7]))
                box_name = info['name']
                if boxes3d[-1].shape[-1] == 9:
                    boxes3d[-1] = boxes3d[-1][:, 0:7]

            obj_type += [self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)]
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

    def build_config(self):
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """

        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config

    def build_graph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)

            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self.build_config(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=self._pd_overlap_nlz,
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
            )
            return metrics

    def run_eval_ops(
        self,
        sess,
        graph,
        metrics,
        prediction_frame_id,
        prediction_bbox,
        prediction_type,
        prediction_score,
        prediction_overlap_nlz,
        ground_truth_frame_id,
        ground_truth_bbox,
        ground_truth_type,
        ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._pd_overlap_nlz: prediction_overlap_nlz,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            },
        )

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def waymo_evaluation(self, prediction_infos, gt_infos, class_name, distance_thresh=100, fake_gt_infos=True):
        print('Start the waymo evaluation...')
        assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

        tf.compat.v1.disable_eager_execution()
        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ = self.generate_waymo_type_results(
            prediction_infos, class_name, is_gt=False
        )
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty = self.generate_waymo_type_results(
            gt_infos, class_name, is_gt=True, fake_gt_infos=fake_gt_infos
        )

        pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = self.mask_by_distance(
            distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz
        )
        gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty = self.mask_by_distance(
            distance_thresh, gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty
        )

        print('Number: (pd, %d) VS. (gt, %d)' % (len(pd_boxes3d), len(gt_boxes3d)))
        print('Level 1: %d, Level2: %d)' % ((gt_difficulty == 1).sum(), (gt_difficulty == 2).sum()))

        if pd_score.max() > 1:
            # assert pd_score.max() <= 1.0, 'Waymo evaluation only supports normalized scores'
            pd_score = 1 / (1 + np.exp(-pd_score))
            print('Warning: Waymo evaluation only supports normalized scores')

        graph = tf.Graph()
        metrics = self.build_graph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self.run_eval_ops(
                sess, graph, metrics, pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz,
                gt_frameid, gt_boxes3d, gt_type, gt_difficulty,
            )
            with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
                aps = self.eval_value_ops(sess, graph, metrics)
        return aps


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=5, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the waymo format results...')
    eval = OpenPCDetWaymoDetectionMetricsEstimator()

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        cur_info['frame_id'] = gt_infos[idx]['frame_id']
        gt_infos_dst.append(cur_info)

    waymo_AP = eval.waymo_evaluation(
        pred_infos, gt_infos_dst, class_name=args.class_names, distance_thresh=1000, fake_gt_infos=False
    )

    print(waymo_AP)


if __name__ == '__main__':
    main()
