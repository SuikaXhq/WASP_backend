# import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models import resnet
from models.layers.grid_attention_layer import GridAttentionBlock2D

from typing import Any, Callable, List, Optional, Tuple, Union

# import torch.nn.functional as F
from torch import nn
# from torchvision.ops import MultiScaleRoIAlign

# from torchvision.ops import misc as misc_nn_ops
# from torchvision.transforms._presets import ObjectDetection
from torchvision.models._api import register_model, Weights, WeightsEnum
# from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
# from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.resnet import resnet50, ResNet50_Weights
# from torchvision.models.detection._utils import overwrite_eps
# from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _validate_trainable_layers
# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
# from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN, _default_anchorgen, FastRCNNConvFCHead

# import warnings
from typing import Callable, Dict, List, Optional, Union

from torch import nn, Tensor
# from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.models import mobilenet, resnet
# from torchvision.models._api import _get_enum_from_fn, WeightsEnum
from torchvision.models._utils import handle_legacy_interface, IntermediateLayerGetter

class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        att_layers: Optional[List[int]] = None,
        global_layer: Optional[str] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.att_layers = att_layers
        if self.att_layers:
            assert global_layer
            self.global_layer = return_layers[global_layer]
            att_block_dict = {}
            for idx in att_layers:
                att_block_dict[list(return_layers.values())[idx]] = GridAttentionBlock2D(in_channels=in_channels_list[idx], gating_channels=in_channels_list[-1])
            self.att_blocks = nn.ModuleDict(att_block_dict)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        if self.att_layers:
            g = x[self.global_layer]
            for idx in self.att_layers:
                layer_name = list(x.keys())[idx]
                x[layer_name], att_x = self.att_blocks[layer_name](x[layer_name], g)
        x = self.fpn(x)
        return x
    
    
# class ResNetAttentionGate(resnet.ResNet):
#     def __init__(self, num_classes, att_layers=['layer3', 'layer2']):
#         super(ResNetAttentionGate, self).__init__()
#         self.base_model = create_model(num_classes)
#         channels = [64, 256, 512, 1024, 2048]
#         self.att_in_layers = {layer: None for layer in att_layers}
#         for name, module in self.base_model.named_children():
#             if name in att_layers:
#                 module.register_forward_hook(self.__provide_hook(name))
#                 print(f'RCNN_Attention: Registered hook in {name}.')
#         self.grid_att_blocks = {layer: None for layer in att_layers}

#     def __provide_hook(self, layer):
#         def hook(module, input, output):
#             self.att_in_layers[layer] = output
#         return hook

#     def forward(self, images, targets=None):
#         base_out = self.base_model(images, targets)



def create_model(num_classes, att_layers=None):
    
    # load Faster RCNN pre-trained model
    model = fasterrcnn_resnet50_fpn_v2_1(pretrained=False, att_layers=att_layers)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

@register_model()
@handle_legacy_interface(
    weights=("pretrained", FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fasterrcnn_resnet50_fpn_v2_1(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    att_layers: Optional[List[int]] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """
    Constructs an improved Faster R-CNN model with a ResNet-50-FPN backbone from `Benchmarking Detection
    Transfer Learning with Vision Transformers <https://arxiv.org/abs/2111.11429>`__ paper.

    .. betastatus:: detection module

    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights
        :members:
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d, att_layers=att_layers)
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def _resnet_fpn_extractor(
    backbone: resnet.ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    att_layers: Optional[List[int]] = None,
) -> BackboneWithFPN:

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer, att_layers=att_layers, global_layer=list(return_layers.keys())[-1]
    )
    
    
