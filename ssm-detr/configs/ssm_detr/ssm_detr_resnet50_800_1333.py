from torch import nn
from torchvision.ops import FrozenBatchNorm2d

from models.backbones.resnet import ResNetBackbone
from models.bricks.ssm_transformer import (
    SsmTransformer,
    SsmTransformerDecoder,
    SsmTransformerDecoderLayer,
    SsmTransformerEncoder,
    SsmTransformerEncoderLayer,
)
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.set_criterion import SetCriterion
from models.detectors.ssm_detr import SsmDETR, SalienceCriterion
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper

# mostly changed parameters
embed_dim = 256
num_classes = 91
num_queries = 300 # use 300/900 for default detr methods
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# instantiate model components
position_embedding = PositionEmbeddingSine(
    embed_dim // 2, temperature=10000, normalize=True, offset=-0.5
)

backbone = ResNetBackbone(
    "resnet50", norm_layer=FrozenBatchNorm2d, return_indices=(1, 2, 3), freeze_indices=(0,)
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = SsmTransformer(
    encoder=SsmTransformerEncoder(
        encoder_layer=SsmTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    decoder=SsmTransformerDecoder(
        decoder_layer=SsmTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=0.1,
            activation="relu",
            dim_feedforward=dim_feedforward,
            num_proposal=num_queries
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    level_filter_ratio=(0.05, 0.2, 0.8, 1.0),
    layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
)

matcher = HungarianMatcher(
    cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25, focal_gamma=2.0
)

# construct weight_dic
weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
aux_weight_dict = {}
for i in range(transformer.decoder.num_layers - 1):
    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
weight_dict.update({"loss_salience": 2})


criterion = SetCriterion(
    num_classes=num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    alpha=0.25,
    gamma=2.0,
    two_stage_binary_cls=True,
)
foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

# combine above components to instantiate the model
model = SsmDETR(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    focus_criterion=foreground_criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    min_size=800,
    max_size=1333,
)
