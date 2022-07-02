from collections import OrderedDict
from pathlib import Path
from torch import hub

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SegTemplate(nn.Module):
    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer

        return_layers = {_layer:_layer for _layer in feat_extract_layer}
        self.model.backbone.return_layers.update(return_layers)


    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        # Get model
        model = constructor(pretrained=False,
                            pretrained_backbone=False,
                            num_classes=self.num_classes,
                            aux_loss=self.aux_loss)
        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()

            # Download pretrained model if not available yet
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            #pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict, pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        return model.cuda()

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        model_num_classes = model_dict["classifier.4.weight"].shape[0]
        pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop("classifier.4.weight")
            pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """

        # Preprocess images
        if self.pretrained:
            images = (images - self.norm_mean[None, :, None, None].type_as(images)) / self.norm_std[None, :, None, None].type_as(images)
        x = images.cuda()

        # Extract features
        result = OrderedDict()
        features = self.model.backbone(x)
        for _layer in self.feat_extract_layer:
            result[_layer] = features[_layer]
        return result

        if 'features' in features.keys():
            feat_shape = features['features'].shape[-2:]
        else:
            feat_shape = features['layer1'].shape[-2:]

        # Prediction classification logits
        x = features["out"] # comment the classifier to reduce memory
        # x = self.model.classifier(x)
        # x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result["logits"] = x

        # Prediction auxillary classification logits
        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class SemDeepLabV3(SegTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes SemDeepLabV3 model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50/ResNet101]
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == "ResNet101":
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
