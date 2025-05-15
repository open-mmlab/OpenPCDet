from dataclasses import dataclass

import numpy as np
import yaml
from easydict import EasyDict as edict

from pcdet.models import build_network


@dataclass
class EncoderData:
    num_point_features: int


@dataclass
class DatasetData:
    point_feature_encoder: EncoderData
    grid_size: np.ndarray
    point_cloud_range: list
    voxel_size: list
    depth_downsample_factor: None
    class_names: list


def load_config(path):
    with open(path, "r") as f:
        return edict(yaml.safe_load(f))


def main():
    model_cfg_path = "pillarnet.yaml"
    cfg = load_config(model_cfg_path)

    dataset = DatasetData(
        point_feature_encoder=EncoderData(num_point_features=3),
        grid_size=np.array([2000, 800, 30]),
        point_cloud_range=[-100.0, -40.0, -2.0, 100.0, 40.0, 2.5],
        voxel_size=[0.1, 0.1, 0.15],
        depth_downsample_factor=None,
        class_names=cfg.CLASS_NAMES,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
    print("success")
