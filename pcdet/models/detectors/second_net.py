import spconv
from .detector3d import Detector3D
from ...config import cfg


class SECONDNet(Detector3D):
    def __init__(self, num_class, dataset):
        super().__init__(num_class, dataset)

        self.sparse_shape = dataset.voxel_generator.grid_size[::-1] + [1, 0, 0]
        self.build_networks(cfg.MODEL)

    def forward_rpn(self, voxels, num_points, coordinates, batch_size, voxel_centers, **kwargs):
        voxel_features = self.vfe(
            features=voxels,
            num_voxels=num_points,
            coords=coordinates
        )

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=coordinates,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        backbone_ret_dict = self.rpn_net(
            input_sp_tensor,
            **{'voxel_centers': voxel_centers}
        )

        rpn_preds_dict = self.rpn_head(
            backbone_ret_dict['spatial_features'],
            **{'gt_boxes': kwargs.get('gt_boxes', None)}
        )
        rpn_preds_dict.update(backbone_ret_dict)

        rpn_ret_dict = {
            'rpn_cls_preds': rpn_preds_dict['cls_preds'],
            'rpn_box_preds': rpn_preds_dict['box_preds'],
            'rpn_dir_cls_preds': rpn_preds_dict.get('dir_cls_preds', None),
            'anchors': rpn_preds_dict['anchors']
        }
        return rpn_ret_dict

    def forward(self, input_dict):
        rpn_ret_dict = self.forward_rpn(**input_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.predict_boxes(rpn_ret_dict, rcnn_ret_dict=None, input_dict=input_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_anchor_box, tb_dict = self.rpn_head.get_loss()
        loss_rpn = loss_anchor_box
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
