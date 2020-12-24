import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import onnx
from onnx import helper

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, object3d_neolix


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def export_onnx(model):
    print('-------------- network readable visiual --------------')
    # print(model.module_list[0].pfn_layers[0])
    vfe_input = torch.ones([12000, 32, 10], dtype=torch.float32, device='cuda')
    torch.onnx.export(model.module_list[0].pfn_layers[0], vfe_input, "vfe.onnx", verbose=False, input_names=['features'],
                      output_names=['pillar_features'])
    print('vfe.onnx transfer success ...')

    spatial_features = torch.ones([1, 64, 288, 288], dtype=torch.float32, device='cuda')
    torch.onnx.export(model.module_list[2], spatial_features, "backbone.onnx", verbose=False,input_names=['spatial_features'],
                      output_names=['spatial_features_2d'])
    print('backbone.onnx transfer success ...')

    spatial_features_2d = torch.ones([1, 384, 144, 144], dtype=torch.float32, device='cuda')
    torch.onnx.export(model.module_list[3], spatial_features_2d, "head.onnx", verbose=False,
                      input_names=["spatial_features_2d"], output_names=['cls', 'bbox', 'dir'])
    print('head.onnx transfer success ...')

    return 0

def createGraphMemberMap(graph_member_list):
    member_map=dict()
    for n in graph_member_list:
        member_map[n.name]=n
    return member_map

def merge_onnx():
    backbone_path = 'backbone.onnx'
    backbone_model = onnx.load(backbone_path)
    backbone_graph = backbone_model.graph
    backbone_node = backbone_model.graph.node
    backbone_initializer = backbone_model.graph.initializer
    backbone_input = backbone_model.graph.input

    backbone_output_map = createGraphMemberMap(backbone_graph.output)
    # print("before merge, the output of backbone: ")
    # print(backbone_graph.output)

    backbone_graph.output.remove(backbone_output_map["spatial_features_2d"])

    head_path = 'head.onnx'
    head_model = onnx.load(head_path)
    head_graph = head_model.graph
    head_node = head_model.graph.node
    head_input = head_model.graph.input

    # node_map = createGraphMemberMap(head_graph.node)
    head_input_map = createGraphMemberMap(head_graph.input)
    head_output_map = createGraphMemberMap(head_graph.output)
    head_initializer_map = createGraphMemberMap(head_graph.initializer)

    # merge node
    for i in range(len(head_node)):
        backbone_node.append(head_node[i])

    # merge initializer
    for i in range(len(head_graph.initializer)):
        backbone_initializer.append(head_graph.initializer[i])

    for i in range(1, len(head_input)):
        # print(head_input[i].name)
        # if head_input[i].name is not "spatial_features_2d":
        backbone_input.append(head_input[i])

    backbone_graph.output.extend(head_graph.output)

    # for i in range(len(backbone_node)):
    #     print(i)
    #     print(backbone_node[i].op_type)
    #     print(backbone_node[i].input)
    #     print(backbone_node[i].output)

    onnx.save(backbone_model, 'rpn.onnx')


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Export ONNX Model-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    dt_annos = []
    dt_annos = export_onnx(model)

    logger.info('Export onnx done.')

    logger.info('-----------------Merge ONNX Model-------------------------')
    merge_onnx()
    logger.info('Merge onnx done.')

if __name__ == '__main__':
    main()
