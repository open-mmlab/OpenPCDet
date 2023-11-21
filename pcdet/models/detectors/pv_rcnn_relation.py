from .detector3d_template import Detector3DTemplate

from pcdet.models.object_relation.gnn import GNN
from ..object_relation import build_object_relation_module


class PVRCNNRelation(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.object_relation = build_object_relation_module(model_cfg.OBJECT_RELATION)
        self.frozen = model_cfg.FROZEN if "FROZEN" in model_cfg.keys() else False

    def forward(self, batch_dict):
        # MeanVFE: Voxelisation
        batch_dict = self.vfe(batch_dict)
        # VoxelBackBone8x: 3D Backbone
        batch_dict = self.backbone_3d(batch_dict)
        # HeightCompression(): 3D to BEV
        batch_dict = self.map_to_bev_module(batch_dict)
        # VoxelSetAbstraction: Aggregation of raw points with 3D features and BEV features
        batch_dict = self.pfe(batch_dict)
        # BaseBEVBackbone: 2D Backbone
        batch_dict = self.backbone_2d(batch_dict)
        # AnchorHeadSingle: Proposal generation for each voxel 
        batch_dict = self.dense_head(batch_dict)
        # PointHeadSimple: prediction of a 
        batch_dict = self.point_head(batch_dict)
        # PVRCNNHead: Proposal refinement
        batch_dict = self.roi_head(batch_dict)
        # GNN: Object relation
        batch_dict = self.object_relation(batch_dict)

        batch_dict = self.roi_head.final_predictions(batch_dict)


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        if self.frozen:
            loss = loss_rcnn
        else:
            loss = loss_rpn + loss_point + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
        
        return loss, tb_dict, disp_dict
