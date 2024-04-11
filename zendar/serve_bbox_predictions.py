"""
Listen for data packets sent over a TCP port,
interpret them as a numpy point cloud,
and return with bounding boxes
"""
import argparse
import logging
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils.common_utils import create_logger

from zendar.sockets import Streamer

LOG = create_logger(log_level=logging.DEBUG)


def predict_one_batch(model, batch, mock_dataset):
    """
    Args:
        model: an instantiated PyTorch model
        batch: a list of numpy arrays
        mock_dataset: a DatasetTemplate object, just used to package
            methods and parameters for the model
    """
    LOG.debug(f"Running predictions on {len(batch)} samples")
    batched_data_dicts = [
        mock_dataset.prepare_data({'points': batch[idx], 'frame_id': idx})
        for idx in range(len(batch))
    ]
    batch_for_network = DatasetTemplate.collate_batch(batched_data_dicts)
    load_data_to_gpu(batch_for_network)
    pred_dicts, _ = model.forward(batch_for_network)
    predictions = [
        (
            pred['pred_boxes'].cpu().numpy(),
            pred['pred_labels'].cpu().numpy(),
        ) for pred in pred_dicts
    ]
    return predictions


def main(cfg, ckpt):
    LOG.info(f'Running {cfg.MODEL.NAME}:{ckpt} in OpenPCDet')

    # set up a TCP streamer, acting as a predictions server
    streamer = Streamer(log=LOG, mode='server', port=6060)

    # set up a Dataset object, just to carry parameters around:
    dataset = DatasetTemplate(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path='streamed://',
        logger=LOG,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(
        filename=ckpt,
        logger=LOG,
        to_cpu=True,
    )
    model.cuda()
    model.eval()

    LOG.info("Ready to run")
    streamer.handshake()
    with torch.inference_mode():
        while True:
            # enter a forever loop, creating a new connection for
            # each data packet and waiting to receive data
            with streamer.get_socket() as sock:
                with streamer.get_connection(sock) as connection:
                    batch = streamer.receive_object(connection)
                    preds = predict_one_batch(model, batch, dataset)
                    streamer.send_object(preds, connection)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        type=str,
        default='/OpenPCDet/tools/cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml',
        help='specify the network config file to use',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='/OpenPCDet/weights/VoxelNeXT_Argo2.pth',
        help='specify the pretrained model weights to use',
    )
    args = parser.parse_args()

    config = cfg_from_yaml_file(args.config_file, cfg)
    main(config, args.ckpt)
