import copy
import math

import torch
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.position_encoding import get_sine_pos_embed
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
# from models.bricks.relation_transformer import (
#     PositionRelationEmbedding,
# )
from util.misc import inverse_sigmoid

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict

# 假设我们已经有了这些辅助模块
from models.bricks.basic import MLP
from models.bricks.position_encoding import get_sine_pos_embed


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class SsmTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 300,
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)

        # initialize pos_trans
        nn.init.xavier_uniform_(self.pos_trans.weight)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    # [(B,256(C),(100)H1,(140)W1),(B,256(C),(50)H2,(70)W2),(B,256(C),(25)H3,(35)W3),(B,256(C),(13)H4,(18)W4)] 4 levels
    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
    ):
        # get input for encoder
        # [(2)B, (18609)L, (256)C]: L is the total number of pixels across all feature levels (L = sum(Hi*Wi) for all levels)
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        # [(2)B, (18609)L]
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        # [(2)B, (18609)L, (256)C]: C is the embedding dimension of the feature map, the same as the input dimension of the encoder
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        # spatial_shapes: [(4)num_levels, (2)2]: Each row contains [Hi, Wi] for a specific feature level
        # level_start_index: [(4)num_levels]: The starting index of each feature level in the flattened feature map
        # valid_ratios: [(2)B, (4)num_levels, (2)2]: The ratio of the feature map size to the original image size for each feature level
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        # reference_points: [(2)B, (18609)L, (4)num_levels, (2)2]: normalized (x, y) coordinates for each spatial position adjusted by valid ratios
        # proposals: [(2)B, (18609)L, 4]
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # encoder
        # memory: [(2)B, (18609)L, (256)C]
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            query_key_padding_mask=mask_flatten,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # calculate filtered tokens numbers for each feature map
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1) # valid_token_nums: [2,4]
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int() # focus_token_nums: [2,4]: select focus tokens for each level
        level_token_nums = focus_token_nums.max(0)[0] # level_token_nums: [4] -> [5600, 2800,  875,  234]
        focus_token_nums = focus_token_nums.sum(-1) # focus_token_nums: [2] -> [9509, 9509]

        # from high level to low level
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = memory[:, start_index:end_index, :] # level_memory: [2,234,256]
            mask = mask_flatten[:, start_index:end_index]
            # update the memory using the higher-level score_prediction,特征金字塔网络(FPN)中的特征增强操作,将高层特征的语义信息传递到低层特征
            if level_idx != spatial_shapes.shape[0] - 1:
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                upsample_score = upsample_score.transpose(1, 2)
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory) # score: [2,234,1]
            valid_score = score.squeeze(-1).masked_fill(mask, score.min()) # valid_score: [2,234]
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx]) # score: [2,1,13,18]

            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1) # topk
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)

        selected_score = torch.cat(selected_score[::-1], 1) # [2,9509], from low level to high level
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index) # [2,9509]

        salience_score = salience_score[::-1] # [[2,1,100,140],...[2,1,13,18]]
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1) # [2,18609]
        foreground_score = foreground_score.masked_fill(mask_flatten, foreground_score.min()) # [2,18609]

        # get encoder output, classes and coordinates
        # output_memory: [(2)B, (18609)L, (256)C]
        # output_proposals: [(2)B, (18609)L, 4], 经过逆sigmoid变换，无效位置设为无穷大
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        # enc_outputs_class: [(2)B, (18609)L, (91)num_classes]
        enc_outputs_class = self.encoder_class_head(output_memory)
        # enc_outputs_coord: [(2)B, (18609)L, 4]
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        # enc_outputs_coord: [(2)B, (18609)L, 4]
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # select topk
        topk = self.two_stage_num_proposals
        # 索引0很可能是一个特殊的"对象性"或"前景概率"通道，而不是特定类别的概率。这个通道被用来评估一个位置包含任何对象的可能性
        # topk_index: [2,300,1]
        topk_index = torch.topk(enc_outputs_class[:, :, 0], topk, dim=1)[1].unsqueeze(-1)
        # topk_enc_outputs_coord: [(2)B, (300)topk, 4]
        topk_enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get query(target) and reference points
        # NOTE: original implementation calculates query and query_pos together.
        # To keep the interface the same with Dab, DN and DINO, we split the
        # calculation of query_pos into the DeformableDecoder
        # reference_points: [(2)B, (300)topk, 4]
        reference_points = topk_enc_outputs_coord.detach()
        # nn.Linear can not perceive the arrangement order of elements
        # so exchange_xy=True/False does not matter results
        # query_sine_embed: [(2)B, (300)topk, (512)self.embed_dim*2]
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        # target:[B, topk, self.embed_dim]: initial query using pos embed [2,900,256]
        target = self.pos_trans_norm(self.pos_trans(query_sine_embed))

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            foreground_score=foreground_score, # [2,18609]
            focus_token_nums=focus_token_nums, # [2] -> [9509, 9509]
            foreground_inds=selected_inds, # [2,9509]
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score # salience_score: [[2,1,100,140],...[2,1,13,18]]


class SsmTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
    ):
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )

        return query


class SsmTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


# class SsmTransformerDecoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, num_classes):
#         super().__init__()
#         # parameters
#         self.embed_dim = decoder_layer.embed_dim
#         self.num_heads = decoder_layer.num_heads
#         self.num_layers = num_layers
#         self.num_classes = num_classes

#         # decoder layers and embedding
#         self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
#         # NOTE: the ref_point_head of Deformable is split from pos_trans and pos_norm,
#         # which is different from DINO
#         self.ref_point_head = nn.Sequential(
#             nn.Linear(2 * self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
#         )

#         # iterative bounding box refinement
#         class_head = nn.Linear(self.embed_dim, num_classes)
#         bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
#         self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
#         self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])

#         self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)

#         self.init_weights()

#     def init_weights(self):
#         # initialize decoder layers
#         for layer in self.layers:
#             if hasattr(layer, "init_weights"):
#                 layer.init_weights()
#         # initialize decoder classification layers
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         for class_head in self.class_head:
#             nn.init.constant_(class_head.bias, bias_value)
#         # initiailize decoder regression layers
#         for bbox_head in self.bbox_head:
#             nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
#             nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

#         # initialize ref_point_head
#         nn.init.xavier_uniform_(self.ref_point_head[0].weight)

#     def forward(
#         self,
#         query,
#         reference_points,
#         value,
#         spatial_shapes,
#         level_start_index,
#         valid_ratios,
#         key_padding_mask=None,
#         attn_mask=None,
#     ):
#         # NOTE: the difference between DeformableDecoder and DabDecoder is that
#         # Deformable does not introduce reference refinement for query pos
#         query_sine_embed = get_sine_pos_embed(
#             reference_points, self.embed_dim // 2, exchange_xy=False
#         )
#         query_pos = self.ref_point_head(query_sine_embed)

#         outputs_classes, outputs_coords = [], []
#         valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

#         for layer_idx, layer in enumerate(self.layers):
#             reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale

#             query = layer(
#                 query=query,
#                 query_pos=query_pos,
#                 reference_points=reference_points_input,
#                 value=value,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 key_padding_mask=key_padding_mask,
#                 self_attn_mask=attn_mask,
#             )

#             # get output
#             output_class = self.class_head[layer_idx](query)
#             output_coord = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points)
#             output_coord = output_coord.sigmoid()
#             outputs_classes.append(output_class)
#             outputs_coords.append(output_coord)

#             if layer_idx == self.num_layers - 1:
#                 break

#             # NOTE: Here we integrate position_relation_embedding into DN-Deformable-DETR
#             src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
#             tgt_boxes = output_coord
#             pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
#             if attn_mask is not None:
#                 pos_relation.masked_fill_(attn_mask, float("-inf"))

#             # iterative bounding box refinement
#             reference_points = output_coord.detach()

#         outputs_classes = torch.stack(outputs_classes)
#         outputs_coords = torch.stack(outputs_coords)
#         return outputs_classes, outputs_coords






class Box2DDistFun(nn.Module):
    """2D版本的边界框距离函数"""
    def __init__(self, out_dim=16):
        super().__init__()
        self.out_dim = out_dim
        # 使用MLP将相对位置映射到高维空间
        self.mlp = MLP(4, out_dim, out_dim, 2)  # 输入是4维：x,y相对位置和宽高比例

    def forward(self, key_pos, query_center, query_size, query_labels=None):
        """
        计算关键点相对于查询框的空间关系编码
        Args:
            key_pos: [B, L, 2] - 关键点位置 (x,y)
            query_center: [B, Q, 2] - 查询框中心点 (x,y)
            query_size: [B, Q, 2] - 查询框尺寸 (w,h)
            query_labels: [B, Q] - 查询框标签 (可选)
        Returns:
            dist_encoding: [B, L, Q, out_dim] - 距离编码
        """
        B, L, _ = key_pos.shape
        _, Q, _ = query_center.shape

        # 计算关键点到查询框中心的相对位置
        # [B, L, 1, 2] - [B, 1, Q, 2] = [B, L, Q, 2]
        rel_pos = key_pos.unsqueeze(2) - query_center.unsqueeze(1)

        # 归一化相对位置（除以查询框尺寸）
        # 避免除零
        eps = 1e-6
        query_size = query_size.clamp(min=eps)
        # [B, L, Q, 2] / [B, 1, Q, 2] = [B, L, Q, 2]
        rel_pos_norm = rel_pos / (query_size.unsqueeze(1) + eps)

        # 计算关键点是否在框内的比例值 (0-1之间的值)
        # 计算关键点到框边界的距离
        dist_to_border = torch.abs(rel_pos_norm)
        in_box = (dist_to_border <= 0.5).all(dim=-1, keepdim=True).float()

        # 组合特征: [相对位置(归一化), 在框内的指示]
        # [B, L, Q, 4]
        combined_features = torch.cat([rel_pos_norm, in_box, 1.0-in_box], dim=-1)

        # 使用MLP映射到高维空间
        # [B, L, Q, out_dim]
        dist_encoding = self.mlp(combined_features)

        return dist_encoding


def _get_activation_fn(activation):
    """获取激活函数"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "silu":
        return F.silu
    else:
        raise RuntimeError(f"activation should be relu/gelu/silu, not {activation}")


class SsmTransformerDecoder(nn.Module):
    """SSM Transformer解码器"""
    def __init__(
        self,
        decoder_layer,
        num_layers,
        num_classes,
        serialization_strategies=None  # 新增：序列化策略列表
    ):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.d_model
        self.num_heads = decoder_layer.nhead
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.layers[-1].last_layer = True # 最后一个解码器层
        self.chunk_size = decoder_layer.chunk_size

        # 设置序列化策略
        if serialization_strategies is None:
            # 默认策略：所有层使用相同的序列化方式
            self.serialization_strategies = ['default'] * num_layers
        else:
            assert len(serialization_strategies) == num_layers, "序列化策略数量必须与层数相同"
            self.serialization_strategies = serialization_strategies
        # 为每一层设置序列化策略
        for i, layer in enumerate(self.layers):
            layer.serialization_strategy = self.serialization_strategies[i]

        # NOTE: the ref_point_head of Deformable is split from pos_trans and pos_norm,
        # which is different from DINO
        self.ref_point_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
        )

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

        # initialize ref_point_head
        nn.init.xavier_uniform_(self.ref_point_head[0].weight)

    def forward(
        self,
        query,              # [B, Q, D] - 查询
        reference_points,   # [B, Q, 4] - 参考点
        value,              # [B, L, D] - 值
        spatial_shapes,     # [num_levels, 2] - 空间形状
        level_start_index,  # [num_levels] - 层级起始索引
        valid_ratios,       # [B, num_levels, 2] - 有效比例
        key_padding_mask=None,  # [B, L] - 键填充掩码
        # salience input
        foreground_score=None, # [2,18609]
        focus_token_nums=None, # [2] -> [9509, 9509]
        foreground_inds=None, # [2,9509]
    ):
        # 确保特征点数量是查询数量的整数倍
        num_queries = query.shape[1]  # 通常是300
        # query vs chunk_size
        # 计算需要保留的点数（L向下取整为num_queries的整数倍）
        # nchunks = math.ceil(seqlen / chunk_size)
        chunk_size = self.chunk_size
        keep_points = (foreground_inds.shape[1] // chunk_size) * chunk_size # 9500
        foreground_inds = foreground_inds[:, :keep_points] # [2,9500]
            
        # 使用gather一次性选择所有batch的tokens
        # 扩展indices维度以匹配value的维度
        # foreground_inds: [B, L'] -> [B, L', 1]
        indices = foreground_inds.unsqueeze(-1)
        # 选择value tokens
        # value: [B, L, D] -> [B, L', D]
        value = torch.gather(value, 1, indices.expand(-1, -1, value.size(-1))) # [2,9500,256]
        # 选择memory_pos tokens
        # memory_pos: [B, L, 2] -> [B, L', 2]
        memory_pos = self.get_memory_pos(spatial_shapes, level_start_index, value.device)
        if memory_pos.shape[0] == 1 and value.shape[0] > 1:
            memory_pos = memory_pos.expand(value.shape[0], -1, -1)
        memory_pos = torch.gather(memory_pos, 1, indices.expand(-1, -1, memory_pos.size(-1))) # [2,9500,2]
        # 选择mask tokens
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, L']
            key_padding_mask = torch.gather(key_padding_mask, 1, foreground_inds) # [2,9500]


        # 获取memory_pos
        # if value.shape[1] % num_queries != 0: # value: [2,18609,256]
        #     # 计算需要保留的点数（向下取整为num_queries的整数倍）
        #     keep_points = (value.shape[1] // num_queries) * num_queries # 18600
            
        #     # 均匀采样（保持分布同时尽量保留顺序关系）
        #     indices = torch.linspace(0, value.shape[1]-1, keep_points, dtype=torch.long, device=value.device)
            
        #     # 调整所有相关张量
        #     value = value[:, indices] # [2,18600,256]
        #     if key_padding_mask is not None:
        #         key_padding_mask = key_padding_mask[:, indices] # [2,18600]

        #     # 获取memory_pos
        #     memory_pos = self.get_memory_pos(spatial_shapes, level_start_index, value.device)
        #     # 确保memory_pos的batch维度正确
        #     if memory_pos.shape[0] == 1 and value.shape[0] > 1:
        #         memory_pos = memory_pos.expand(value.shape[0], -1, -1)
        #     # 对memory_pos也应用相同的采样
        #     memory_pos = memory_pos[:, indices] # [2,18600,2]
            
        # else:
        #     # 如果不需要调整，正常获取memory_pos
        #     memory_pos = self.get_memory_pos(spatial_shapes, level_start_index, value.device)
        #     if memory_pos.shape[0] == 1 and value.shape[0] > 1:
        #         memory_pos = memory_pos.expand(value.shape[0], -1, -1)
        
        # query: 原有的处理逻辑 [2,300,4]
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False)
        query_pos_embed = self.ref_point_head(query_sine_embed) # [2,300,256]

        outputs_classes, outputs_coords = [], []
        # valid_ratio_scale: [(2)B, 1, (4)num_levels, 4]
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            # reference_points_input: [(2)B, (300)Q, (4)num_levels, 4]
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query, value = layer(
                query=query,
                memory=value,
                query_pos_embed=query_pos_embed,
                memory_pos=memory_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_key_padding_mask=key_padding_mask,
                layer_idx=layer_idx,
            )

            # get output
            output_class = self.class_head[layer_idx](query) # [(2)B, (300)Q, (91)C]
            output_coord = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points) # [(2)B, (300)Q, 4]
            output_coord = output_coord.sigmoid() # [(2)B, (300)Q, 4]
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = output_coord.detach() #[2,300,4]

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords

    def get_memory_pos(self, spatial_shapes, level_start_index, device):
        """
        为每个特征点生成归一化坐标

        Args:
            spatial_shapes: [num_levels, 2] - 每个特征层的空间形状 (H, W)
            level_start_index: [num_levels] - 每个特征层在展平特征中的起始索引
            device: 设备

        Returns:
            memory_pos: [B, L, 2] - 每个特征点的归一化坐标 (x, y)
        """
        num_levels = spatial_shapes.shape[0]

        # 为每个特征层生成网格坐标
        level_pos_list = []
        for level in range(num_levels):
            H, W = spatial_shapes[level]

            # 生成网格坐标
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )

            # 归一化坐标到 [0, 1]
            grid_x = (grid_x + 0.5) / W
            grid_y = (grid_y + 0.5) / H

            # 展平并堆叠
            pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            level_pos_list.append(pos)

        # 连接所有特征层的位置
        memory_pos = torch.cat(level_pos_list, dim=0)

        # 扩展批次维度 (假设批次大小为1，实际使用时会被广播)
        memory_pos = memory_pos.unsqueeze(0)

        return memory_pos


class SsmTransformerDecoderLayer(nn.Module):
    """SSM Transformer解码器层"""
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_proposal=300, # 300 two_stage_num_proposals
        ssm_expand=2,
        ssm_use_biscan=True,
        last_layer=False,
        chunk_size=128,
    ):
        super().__init__()
        self.d_model = d_model # embed_dim
        self.nhead = nhead
        self.last_layer = last_layer
        self.weight_dist = -0.1  # 距离权重衰减系数
        self.serialization_strategy = 'default'  # 默认序列化策略
        self.chunk_size = chunk_size

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ssm = MultiHead2DISSM(
            d_model=d_model,
            d_state=num_proposal, # state dimension is # of queries
            d_dist=16,  # 距离编码维度
            chunk_size=chunk_size, # power of 2; set 128 now
            nheads=nhead,
            expand=ssm_expand,
            use_biscan=ssm_use_biscan
        )
        self.spatial_dist = Box2DDistFun(out_dim=16)

        # memory: 残差连接和FFN处理
        self.dropout2_memory = nn.Dropout(dropout)
        self.norm2_memory = nn.LayerNorm(d_model)
        # 前馈网络
        self.linear1_memory = nn.Linear(d_model, dim_feedforward)
        self.dropout_memory = nn.Dropout(dropout)
        self.linear2_memory = nn.Linear(dim_feedforward, d_model)
        self.dropout3_memory = nn.Dropout(dropout)
        self.norm3_memory = nn.LayerNorm(d_model)

        # query: 残差连接和FFN处理
        self.dropout2_query = nn.Dropout(dropout)
        self.norm2_query = nn.LayerNorm(d_model)
        # 前馈网络
        self.linear1_query = nn.Linear(d_model, dim_feedforward)
        self.dropout_query = nn.Dropout(dropout)
        self.linear2_query = nn.Linear(dim_feedforward, d_model)
        self.dropout3_query = nn.Dropout(dropout)
        self.norm3_query = nn.LayerNorm(d_model)

        # 激活函数
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1_memory.weight)
        nn.init.xavier_uniform_(self.linear2_memory.weight)
        nn.init.xavier_uniform_(self.linear1_query.weight)
        nn.init.xavier_uniform_(self.linear2_query.weight)

    def with_pos_embed(self, tensor, pos):
        """添加位置编码"""
        return tensor if pos is None else tensor + pos

    def forward_ffn_memory(self, x):
        """前馈网络前向传播"""
        x2 = self.linear2_memory(self.dropout_memory(self.activation(self.linear1_memory(x))))
        x = x + self.dropout3_memory(x2) # 
        x = self.norm3_memory(x)
        return x

    def forward_ffn_query(self, x):
        """前馈网络前向传播"""
        x2 = self.linear2_query(self.dropout_query(self.activation(self.linear1_query(x))))
        x = x + self.dropout3_query(x2)
        x = self.norm3_query(x)
        return x

    def local_weight(self, memory_pos, query_center, query_size, strategy='default'):
        """
        计算2D版本的局部权重，支持不同的序列化策略

        Args:
            memory_pos: [B, L, 2] - 关键点位置
            query_center: [B, Q, 2] - 查询框中心
            query_size: [B, Q, 2] - 查询框尺寸
            strategy: 序列化策略

        Returns:
            weights: [B, L, Q] - 权重矩阵
        """
        B, L, _ = memory_pos.shape
        _, Q, _ = query_center.shape

        # 计算查询框的半径（对角线长度的一半）
        query_radius = torch.sqrt(torch.sum(query_size**2, dim=-1) / 4).clamp_min(16.0)  # 最小半径16像素

        # 计算关键点到查询中心的距离
        dist = torch.cdist(memory_pos, query_center, p=2)  # [B, L, Q]

        # 根据不同策略计算权重
        if strategy == 'default' or strategy == 'distance':
            # 基于距离的衰减权重
            weights = torch.exp(self.weight_dist * ((dist - query_radius.unsqueeze(1)).clamp_min(0.0)))

        elif strategy == 'attention':
            # 基于注意力的权重（使用softmax归一化）
            # 距离越小，注意力权重越大
            attention_scores = -dist / query_radius.unsqueeze(1)
            weights = F.softmax(attention_scores, dim=1)

        elif strategy == 'topk':
            # 只保留每个查询的Top-K个最近点
            k = min(L, 64)  # 可以根据需要调整K值
            # 对每个查询找到最近的K个点
            _, indices = torch.topk(-dist, k=k, dim=1)  # [B, k, Q]
            weights = torch.zeros_like(dist)

            # 为每个批次和查询设置权重
            for b in range(B):
                for q in range(Q):
                    weights[b, indices[b, :, q], q] = 1.0

        elif strategy == 'hybrid':
            # 混合策略：结合距离衰减和Top-K
            k = min(L, 128)  # 可以根据需要调整K值
            _, indices = torch.topk(-dist, k=k, dim=1)  # [B, k, Q]

            # 先用距离衰减计算权重
            weights = torch.exp(self.weight_dist * ((dist - query_radius.unsqueeze(1)).clamp_min(0.0)))

            # 然后只保留Top-K个点的权重
            mask = torch.zeros_like(weights)
            for b in range(B):
                for q in range(Q):
                    mask[b, indices[b, :, q], q] = 1.0

            weights = weights * mask

        else:
            # 默认回退到距离衰减
            weights = torch.exp(self.weight_dist * ((dist - query_radius.unsqueeze(1)).clamp_min(0.0)))

        return weights

    def get_serialization_order(self, memory_pos, query_center, strategy='default'):
        """
        根据不同策略获取序列化顺序

        Args:
            memory_pos: [B, L, 2] - 记忆位置
            query_center: [B, Q, 2] - 查询中心
            strategy: 序列化策略

        Returns:
            indices: [B, L] - 序列化顺序索引
        """
        B, L, _ = memory_pos.shape
        _, Q, _ = query_center.shape

        if strategy == 'default' or strategy == 'raster':
            # 默认光栅顺序（保持原始顺序）
            return torch.arange(L, device=memory_pos.device).unsqueeze(0).expand(B, -1)

        elif strategy == 'spiral':
            # 螺旋序列化：从图像中心向外螺旋
            # 计算每个点到图像中心的距离
            image_center = torch.tensor([0.5, 0.5], device=memory_pos.device).view(1, 1, 2)
            dist_to_center = torch.norm(memory_pos - image_center, dim=2)  # [B, L]

            # 按距离排序
            _, indices = torch.sort(dist_to_center, dim=1)
            return indices

        elif strategy == 'query_centered':
            # 以查询为中心的序列化
            # 对于每个批次，计算所有记忆点到所有查询点的平均距离
            dist = torch.cdist(memory_pos, query_center, p=2)  # [B, L, Q]
            avg_dist = dist.mean(dim=2)  # [B, L]

            # 按平均距离排序
            _, indices = torch.sort(avg_dist, dim=1)
            return indices

        elif strategy == 'zigzag':
            # Z字形扫描
            # 假设记忆点是按照光栅顺序排列的
            # 我们可以根据原始索引重新排列
            indices = torch.arange(L, device=memory_pos.device).unsqueeze(0).expand(B, -1)

            # 获取原始图像的高度和宽度（假设是正方形）
            side_len = int(math.sqrt(L))

            # 创建Z字形扫描索引
            zigzag_indices = torch.zeros(L, device=memory_pos.device, dtype=torch.long)
            idx = 0
            for i in range(side_len):
                if i % 2 == 0:  # 从左到右
                    for j in range(side_len):
                        if idx < L:
                            zigzag_indices[idx] = i * side_len + j
                            idx += 1
                else:  # 从右到左
                    for j in range(side_len-1, -1, -1):
                        if idx < L:
                            zigzag_indices[idx] = i * side_len + j
                            idx += 1

            # 应用Z字形扫描索引
            return zigzag_indices.unsqueeze(0).expand(B, -1)

        else:
            # 默认光栅顺序
            return torch.arange(L, device=memory_pos.device).unsqueeze(0).expand(B, -1)

    def forward(
        self,
        query,              # [B, Q, D] - 目标查询
        memory,             # [B, L, D] - 编码器记忆
        query_pos_embed,     # [B, Q, D] - 查询位置编码
        memory_pos,         # [B, L, 2] - 记忆位置编码
        reference_points,   # [B, Q, num_levels, 2] - 参考点 (x, y)
        spatial_shapes,     # [num_levels, 2] - 空间形状
        level_start_index,  # [num_levels] - 层级起始索引
        memory_key_padding_mask=None,  # [B, L] - 记忆键填充掩码
        layer_idx=None,     # 层索引，用于选择不同的序列化策略
    ):
        # 自注意力 [2,300,256]
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            need_weights=False,
        )[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query) # [(2)B, (300)Q, (256)D]

        # 提取参考点的中心和尺寸
        # 对于多尺度特征，取第一个级别的参考点
       # If reference_points has shape [B, Q, 4] with format (x,y,w,h)
        center_points = reference_points[..., 0, :2]  # Extract (x,y) -> [(2)B, (300)Q, 2]
        box_sizes = reference_points[..., 0, 2:]      # Extract (w,h) -> [(2)B, (300)Q, 2]

        # 确保memory_pos的batch维度正确
        if memory_pos.shape[0] == 1 and memory.shape[0] > 1:
            memory_pos = memory_pos.expand(memory.shape[0], -1, -1)
    
        # 选择序列化策略
        strategy = self.serialization_strategy
        # 获取序列化顺序
        # serialization_indices: [(2)B, (9500)L] - 序列化顺序索引
        # TODO: optimize
        serialization_indices = self.get_serialization_order(memory_pos, center_points, strategy)

        # 根据序列化顺序重排记忆和位置
        B = memory.shape[0] # 2
        L = memory.shape[1] # 9500
        # 使用向量化操作重排记忆和位置
        batch_indices = torch.arange(B, device=memory.device).view(-1, 1)
        reordered_memory = memory[batch_indices, serialization_indices] # [(2)B, (9500)L, (256)D]
        reordered_memory_pos = memory_pos[batch_indices, serialization_indices] # [(2)B, (9500)L, 2]
        
        # 处理填充掩码 - 重排掩码以匹配重排后的记忆
        if memory_key_padding_mask is not None:
            # 将掩码转换为浮点数，其中1表示有效位置，0表示填充位置
            # 注意：原始掩码可能是布尔型，其中True表示填充位置
            if memory_key_padding_mask.dtype == torch.bool:
                valid_mask = (~memory_key_padding_mask).float() # [(2)B, (9500)L]
            else:
                valid_mask = 1.0 - memory_key_padding_mask
                
            # 重排掩码
            reordered_valid_mask = valid_mask[batch_indices, serialization_indices] # [(2)B, (9500)L]
        else:
            # 如果没有提供掩码，则假设所有位置都有效
            reordered_valid_mask = torch.ones(B, L, device=memory.device)

        # 计算局部权重 (使用重排后的位置)
        # 将有效掩码纳入权重计算
        weights = self.local_weight(reordered_memory_pos, center_points, box_sizes, strategy) # [(2)B, (9500)L, (300)Q]
        weights = weights * reordered_valid_mask.unsqueeze(-1)  # 应用有效掩码 [(2)B, (9500)L, (300)Q]

        # 计算空间距离编码 (使用重排后的位置)
        dist = self.spatial_dist(
            key_pos=reordered_memory_pos,
            query_center=center_points,
            query_size=box_sizes,
        ) # [(2)B, (9500)L, (300)Q, (16)D]; dist embedding?

        # 应用SSM
        memory2, query2 = self.ssm(
            in_key=reordered_memory,
            in_query=query,
            dist=dist,
            key_pos=reordered_memory_pos,
            mask=weights,
        )

        # 将处理后的记忆恢复原始顺序
        restored_memory2 = torch.zeros_like(memory2) # [(2)B, (18600)L, (256)D]
        for b in range(B):
            # 创建反向索引映射
            reverse_indices = torch.zeros_like(serialization_indices[b])
            reverse_indices[serialization_indices[b]] = torch.arange(memory.shape[1], device=memory.device)
            restored_memory2[b] = memory2[b, reverse_indices]

        memory2 = restored_memory2 # [(2)B, (18600)L, (256)D]

        # update memory 残差连接和FFN处理
        if not self.last_layer:
            # 场景点特征的残差连接和FFN
            memory = memory + self.dropout2_memory(memory2) # [(2)B, (18600)L, (256)D]
            memory = self.norm2_memory(memory) # [(2)B, (18600)L, (256)D]
            memory = self.forward_ffn_memory(memory) # [(2)B, (18600)L, (256)D]

        # update query 查询特征的残差连接和FFN
        query = query + self.dropout2_query(query2) # [(2)B, (300)Q, (256)D]
        query = self.norm2_query(query) # [(2)B, (300)Q, (256)D]
        query = self.forward_ffn_query(query) # [(2)B, (300)Q, (256)D]

        return query, memory


from issm_triton.issm_combined import ISSM_chunk_scan_combined
from issm_triton.layernorm_gated import RMSNorm as RMSNormGated

class MultiHead2DISSM(nn.Module):
    """2D版本的多头ISSM扫描模块，使用ISSM_chunk_scan_combined"""
    def __init__(
        self,
        d_model: int = 256,        # 输入维度
        d_state: int = 64,         # 状态维度 same as num_proposal
        d_dist: int = 16,          # 距离编码维度
        chunk_size: int = 32,      # 使用较小的chunk_size
        nheads: int = 8,           # 注意力头数
        ngroups: int = 1,          # 组数
        expand: int = 2,           # 扩展因子
        use_biscan: bool = True,   # 是否使用双向扫描
        A_init_range=(1, 16),      # A matrix initialization range
        dt_min: float = 0.0001,    # Minimum time step
        dt_max: float = 0.1,       # Maximum time step
        dt_init_floor: float = 1e-4,# Time step initialization lower bound
        dt_limit: Tuple[float, float] = (0.0, float("inf")),  # dt限制范围
        layer_idx=None,
    ):
        super().__init__()

        # 基本参数
        self.d_model = d_model
        self.d_state = d_state
        self.d_dist = d_dist
        self.chunk_size = chunk_size
        self.nheads = nheads
        self.ngroups = ngroups
        self.expand = expand
        self.use_biscan = use_biscan
        self.d_inner = self.expand * self.d_model
        self.headdim = self.d_inner // self.nheads
        self.dt_limit = dt_limit
        self.layer_idx = layer_idx

        # 投影层
        # 输入投影: 特征 -> [z, x, b/c偏置, dt偏置]
        d_in_key_proj = 2 * self.d_inner + 2 * self.ngroups + self.nheads
        self.key_proj = nn.Linear(self.d_model, d_in_key_proj, bias=False)
        # 查询投影: 特征 -> 初始状态
        self.query_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        # 距离编码投影: 距离 -> B/C基础值
        self.bc_proj = nn.Linear(self.d_dist, 2 * self.ngroups, bias=False)
        # 距离编码投影: 距离 -> dt基础值
        self.dt_proj = nn.Linear(self.d_dist, self.nheads, bias=False)

        # 使用原始DEST的dt初始化方法
        # 初始化dt偏置 (时间步长偏置)
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Initialize state transition parameters
        # 状态空间参数
        # 初始化A矩阵 (状态转移矩阵)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty((self.nheads), dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # D "skip" parameter  初始化D矩阵 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_key_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.out_query_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        # 归一化层
        self.key_norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.query_norm = nn.LayerNorm(self.d_inner)


    def forward(self, in_key, in_query, dist, key_pos=None, mask=None):
        """
        前向传播函数
        Args:
            in_key: [B, L, D] - 输入序列特征 input
            in_query: [B, Q, D] - 查询序列特征 state
            dist: [B, L, Q, M] - 距离编码矩阵 M: 距离编码维度
            key_pos: [B, L, 2] - 关键点位置 (可选)
            mask: [B, L, Q] - 掩码矩阵 (可选)
        Returns:
            out_key: [B, L, D] - 处理后的关键点特征
            out_query: [B, Q, D] - 处理后的查询点特征
        """
        batch, seq_len, _ = in_key.shape # 9500
        _, num_queries, _ = in_query.shape 

        # 添加断言检查
        # assert seq_len % num_queries == 0, f"序列长度({seq_len})必须能被查询数量({num_queries})整除"
        if seq_len > 6000:
            print(f"seq_len: {seq_len}, num_queries: {num_queries}, chunk_size: {self.chunk_size}")

        # 计算每个查询对应的序列长度
        # chunk_size = seq_len // num_queries

        # 1. 投影变换
        # [batch_size, seq_len, 2*self.d_inner + 2*self.ngroups + self.nheads] 1034=2*(2*256)+2*1+8
        zxbcdt = self.key_proj(in_key) # [(2)B, (9500)L, (1034)D]
        # z: [(2)batch_size, (9500)seq_len, (512)self.d_inner]
        # xbc: [(2)batch_size, (9500)seq_len, (self.d_inner + 2 * self.ngroups)((512+2*1))]
        # dt_bias: [(2)batch_size, (9500)seq_len, (8)self.nheads]
        z, xbc, dt_bias = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups, self.nheads],
            dim=-1
        )

        # 分离状态和偏置
        # x: [(2)batch_size, (9500)seq_len, (512)self.d_inner]
        # b_bias: [(2)batch_size, (9500)seq_len, (1)self.ngroups]
        # c_bias: [(2)batch_size, (9500)seq_len, (1)self.ngroups]
        x, b_bias, c_bias = torch.split(
            xbc,
            [self.d_inner, self.ngroups, self.ngroups],
            dim=-1
        )

        # 如果使用双向扫描，准备反向数据
        if self.use_biscan:
            # TODO: might need to add extra conv layer to improve performance
            x_back = x.clone() # [(2)batch_size, (9500)seq_len, (512)self.d_inner]
            b_bias_back = b_bias.clone() # [(2)batch_size, (9500)seq_len, (1)self.ngroups]
            c_bias_back = c_bias.clone() # [(2)batch_size, (9500)seq_len, (1)self.ngroups]

        # 处理查询特征 - 这是关键区别
        # 将查询特征投影为初始状态
        # initial_states: [(2)batch_size, (300)num_queries, (512)self.d_inner]
        initial_states = self.query_proj(in_query) # [(2)B, (300)Q, (512)D]

        # 重排维度为 [B, H, D, Q] - 注意这里与原始DEST保持一致
        # 每个查询点都有自己的初始状态，作为一个整体传入扫描函数
        # initial_states: [(2)batch_size, (8)nheads, (64)headdim, (300)num_queries]
        initial_states = rearrange(initial_states, "b q (h d) -> b h d q", h=self.nheads)

        # 2. 生成状态空间模型参数
        # 状态转移矩阵A
        A = -torch.exp(self.A_log)  # [(8)head]
        A = repeat(A, "h -> h d", d=self.d_state)  # [(8)head, (300)num_queries]

        # 从距离编码生成B和C矩阵
        bc = self.bc_proj(dist)  # [(2)B, (9500)L, (300)Q, (2)2*ngroups]
        b_base, c_base = torch.split(bc, [self.ngroups, self.ngroups], dim=-1)

        # 组合基础值和偏置
        # 注意这里的维度变换，使B和C的形状为 [batch, seqlen, nheads/ngroups, dstate]
        # it is equivalent to 3D DEST version
        B = b_base + b_bias.unsqueeze(2)  # [(2)B, (9500)L, (300)Q, (1)ngroups]
        C = c_base + c_bias.unsqueeze(2)  # [(2)B, (9500)L, (300)Q, (1)ngroups]
        B = B.permute(0, 1, 3, 2)  # [(2)B, (9500)L, (1)ngroups, (300)Q]
        C = C.permute(0, 1, 3, 2)  # [(2)B, (9500)L, (1)ngroups, (300)Q]

        # 如果使用双向扫描，也生成反向参数
        if self.use_biscan:
            B_back = b_base + b_bias_back.unsqueeze(2) # [(2)B, (9500)L, (300)Q, (1)ngroups]
            C_back = c_base + c_bias_back.unsqueeze(2) # [(2)B, (9500)L, (300)Q, (1)ngroups]
            B_back = B_back.permute(0, 1, 3, 2) # [(2)B, (9500)L, (1)ngroups, (300)Q]
            C_back = C_back.permute(0, 1, 3, 2) # [(2)B, (9500)L, (1)ngroups, (300)Q]

        # 生成时间步长
        dt_base = self.dt_proj(dist)  # [(2)B, (9500)L, (300)Q, (8)nheads]
        dt_base = dt_base.permute(0, 1, 3, 2)  # [(2)B, (9500)L, (8)nheads, (300)Q]
        # 结合两种偏置计算最终的时间步长 dt [(2)B, (9500)L, (8)nheads, (300)Q]
        dt = F.softplus(dt_base + dt_bias.unsqueeze(-1) + self.dt_bias.reshape(1, 1, -1, 1))

        # 8. 应用mask（如果提供）
        if mask != None:
            if mask.dtype == torch.float32:
                dt = dt * mask.unsqueeze(2) # [(2)B, (9500)L, (8)nheads, (300)Q]
            else:
                dt[mask.unsqueeze(2).repeat(1, 1, self.nheads, 1)] = 0.0

        # 3. 执行扫描 - 一次性处理所有查询
        # 注意这里与原始DEST保持一致，不需要循环处理每个查询
        module_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        module_kwargs["return_final_states"] = True # {'return_final_states': True}
        # 执行扫描
        key, last_states = self.scan(x, initial_states, dt, A, B, C, module_kwargs)

        # 如果使用双向扫描
        if self.use_biscan:
            # 反转序列
            x_back = torch.flip(x_back, dims=[1])
            dt_back = torch.flip(dt, dims=[1])
            B_back = torch.flip(B_back, dims=[1])
            C_back = torch.flip(C_back, dims=[1])
            # 执行反向扫描
            key_back, last_states_back = self.scan(x_back, initial_states, dt_back, A, B_back, C_back, module_kwargs)
            # 反转回来并平均
            key_back = torch.flip(key_back, dims=[1])
            key = (key + key_back) / 2
            last_states = (last_states + last_states_back) / 2

        # 4. 输出处理
        # 重排key的维度并应用归一化
        key = rearrange(key, "b l h d -> b l (h d)") # [(2)B, (9500)L, (512)D]
        key = self.key_norm(key, z) # [(2)B, (9500)L, (512)D]
        out_key = self.out_key_proj(key) # [(2)B, (9500)L, (256)D]

        # 处理最终状态 - 注意这里的维度变换
        last_states = rearrange(last_states, "b h d q -> b q (h d)") # [(2)B, (300)Q, (512)D]
        last_states = self.query_norm(last_states) # [(2)B, (300)Q, (512)D]
        out_query = self.out_query_proj(last_states) # [(2)B, (300)Q, (256)D]

        return out_key, out_query

    def scan(self, x, initial_states, dt, A, B, C, module_kwargs):
        """
        Perform unidirectional or bidirectional scan
        """
        # 检查GPU内存使用情况
        def print_gpu_memory():
            if torch.cuda.is_available():
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                print(f"GPU Memory max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        # print("Before processing:")
        # print_gpu_memory()

        # 确保所有张量都是连续的
        x = x.contiguous()
        dt = dt.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        initial_states = initial_states.contiguous()
        
        batch_size = x.shape[0]
        
        # 如果batch_size > 3，我们需要分别处理每个batch
        # if batch_size > 3:
        #     outputs = []
        #     final_states = []
            
        #     for b in range(batch_size):
        #         print(f"\nProcessing batch {b}:")
        #         print_gpu_memory()
                
        #         # 处理单个batch
        #         x_b = x[b:b+1]  # 保持batch维度
        #         dt_b = dt[b:b+1]
        #         B_b = B[b:b+1]
        #         C_b = C[b:b+1]
        #         initial_states_b = initial_states[b:b+1]
                
        #         try:
        #             # 清理GPU缓存
        #             if torch.cuda.is_available():
        #                 torch.cuda.empty_cache()
                    
        #             y_b, last_states_b = ISSM_chunk_scan_combined(
        #                 rearrange(x_b, "b l (h p) -> b l h p", p=self.headdim),
        #                 dt_b,
        #                 A,
        #                 B_b,
        #                 C_b,
        #                 chunk_size=self.chunk_size,
        #                 D=self.D,
        #                 z=None,
        #                 initial_states=initial_states_b,
        #                 **module_kwargs,
        #             )
        #             outputs.append(y_b)
        #             final_states.append(last_states_b)
        #         except Exception as e:
        #             print(f"Error in batch {b}: {e}")
        #             print_gpu_memory()
        #             raise
            
        #     # 合并所有batch的结果
        #     y = torch.cat(outputs, dim=0)
        #     last_states = torch.cat(final_states, dim=0)
            
        #     print("\nAfter processing all batches:")
        #     print_gpu_memory()
        #     return y, last_states
        
        # 如果batch_size <= 3，直接处理
        try:
            y, last_states = ISSM_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                B,
                C,
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                initial_states=initial_states,
                **module_kwargs,
            )
            
            # print("\nAfter processing single batch:")
            # print_gpu_memory()
            return y, last_states
        except Exception as e:
            print(f"Error in ISSM_chunk_scan_combined: {e}")
            print_gpu_memory()
            raise

    # def scan(self, x, initial_states, dt, A, B, C, module_kwargs):
    #     """
    #     Perform unidirectional or bidirectional scan
    #     Args:
    #         x: (B, L, D) - Input sequence
    #         initial_states: (B, K, D) - Initial states
    #         dt: (B, L, nheads) - Time steps
    #         A, B, C: Parameters for the scan
    #         module_kwargs: Additional parameters
    #     Returns:
    #         y: (B, K, D) - Output sequence
    #         last_states: (B, K, D) - Final states
    #     """
    #     # 对于每个时间步 t：
    #     # h[t] = h[t-1] + (Ah[t-1] + Bx[t])dt  # 状态更新
    #     # y[t] = Ch[t]                         # 输出计算
    #     y, last_states = ISSM_chunk_scan_combined(
    #         rearrange(x, "b l (h p) -> b l h p", p=self.headdim), # [(2)B, (14365)L, (8)nheads, (64)headdim]
    #         dt, # [(2)B, (14365)L, (8)nheads, (300)num_queries]
    #         A, # [(8)nheads, (300)num_queries]
    #         B, # [(2)B, (14365)L, (1)ngroups, (300)num_queries]
    #         C, # [(2)B, (14365)L, (1)ngroups, (300)num_queries]
    #         chunk_size=self.chunk_size, # 300
    #         D=self.D, # [(8)nheads]
    #         z=None,
    #         initial_states=initial_states, # [(2)B, (8)nheads, (64)headdim, (300)num_queries]
    #         **module_kwargs,
    #     )
    #     return y, last_states

    # """
    # Argument:
    #     x: (batch, seqlen, nheads, headdim)
    #     dt: (batch, seqlen, nheads, dstate)
    #     A: (nheads, dstate)
    #     B: (batch, seqlen, ngroups, dstate)
    #     C: (batch, seqlen, ngroups, dstate)
    #     chunk_size: int
    #     D: (nheads, headdim) or (nheads,)
    #     z: (batch, seqlen, nheads, headdim)
    #     dt_bias: (nheads,)
    #     initial_states: (batch, nheads, headdim, dstate)
    #     seq_idx: (batch, seqlen)
    #     dt_softplus: Whether to apply softplus to dt
    # Return:
    #     out: (batch, seqlen, nheads, headdim)
    # """