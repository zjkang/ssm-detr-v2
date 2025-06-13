import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bricks.position_encoding import get_sine_pos_embed
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.basic import MLP
from util.misc import inverse_sigmoid

class SsmTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        dec_n_points=4,
        use_dab=True,
        use_pe=True,  # 是否使用位置编码
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.use_pe = use_pe
        
        # 位置编码相关
        if self.use_pe:
            self.pos_embed = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
        
        # 参考点编码
        self.ref_point_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 解码器层
        decoder_layer = SsmTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, dec_n_points, use_dab
        )
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        
        # 输出头
        self.class_embed = nn.Linear(d_model, 1)  # 二分类
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        
        # 初始化
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, query, reference_points, memory, memory_spatial_shapes, memory_level_start_index, memory_padding_mask=None):
        """
        Args:
            query: [B, num_queries, C]
            reference_points: [B, num_queries, 2]
            memory: [B, L, C]
            memory_spatial_shapes: [num_levels, 2]
            memory_level_start_index: [num_levels]
            memory_padding_mask: [B, L]
        """
        # 1. 位置编码
        if self.use_pe:
            # 生成位置编码
            pos_embed = get_sine_pos_embed(reference_points, num_pos_feats=self.d_model//2)
            pos_embed = self.pos_embed(pos_embed)
            query = query + pos_embed
            
        # 2. 参考点编码
        ref_point_embed = self.ref_point_embed(reference_points)
        query = query + ref_point_embed
        
        # 3. 解码器层
        for layer in self.decoder:
            query = layer(
                query, reference_points, memory,
                memory_spatial_shapes, memory_level_start_index,
                memory_padding_mask
            )
            
        # 4. 输出预测
        outputs_class = self.class_embed(query)
        outputs_coords = self.bbox_embed(query)
        
        return outputs_class, outputs_coords

class SsmTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        dec_n_points=4,
        use_dab=True
    ):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 交叉注意力
        self.cross_attn = MultiScaleDeformableAttention(
            d_model, nhead, num_feature_levels, dec_n_points
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 参考点更新
        self.use_dab = use_dab
        if self.use_dab:
            self.ref_point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.ref_point_norm = nn.LayerNorm(d_model)
            self.ref_point_ffn = MLP(d_model, d_model, 2, 2)
            
    def forward(self, query, reference_points, memory, memory_spatial_shapes, memory_level_start_index, memory_padding_mask=None):
        # 1. 自注意力
        q = k = v = query
        query2 = self.self_attn(q, k, v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        
        # 2. 交叉注意力
        query2 = self.cross_attn(
            query, reference_points, memory,
            memory_spatial_shapes, memory_level_start_index,
            memory_padding_mask
        )
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        
        # 3. FFN
        query2 = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(query2)
        query = self.norm3(query)
        
        # 4. 参考点更新 (如果使用DAB)
        if self.use_dab:
            # 使用自注意力机制更新参考点
            ref_point_feat = self.ref_point_attn(query, query, query)[0]
            ref_point_feat = self.ref_point_norm(ref_point_feat)
            delta_ref_points = self.ref_point_ffn(ref_point_feat)
            reference_points = reference_points + delta_ref_points
            
        return query, reference_points

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class SamplingOffsetPredictor(nn.Module):
    """预测采样点偏移量"""
    def __init__(self, d_model, n_heads, n_levels, n_points):
        super().__init__()
        self.predictor = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.predictor.weight)
        nn.init.constant_(self.predictor.bias, 0)
        
    def forward(self, query):
        """
        Args:
            query: [batch_size, num_queries, d_model]
            
        Returns:
            offsets: [batch_size, num_queries, n_heads, n_levels, n_points, 2]
        """
        batch_size = query.shape[0]
        offsets = self.predictor(query)  # [batch_size, num_queries, n_heads * n_levels * n_points * 2]
        # 重塑为 [batch_size, num_queries, n_heads, n_levels, n_points, 2]
        return offsets.view(batch_size, -1, self.n_heads, self.n_levels, self.n_points, 2)

class AttentionWeightPredictor(nn.Module):
    """改进的注意力权重预测器"""
    def __init__(self, d_model, n_heads, n_levels, n_points, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.temperature = temperature
        
        # 权重预测
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_heads * n_levels * n_points)
        )
        
        # 前景分数缩放因子
        self.fg_scale = nn.Parameter(torch.ones(1))
        
        # 层级权重
        self.level_weights = nn.Parameter(torch.ones(n_levels))
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, query, foreground_score=None):
        """
        Args:
            query: [batch_size, num_queries, d_model]
            foreground_score: [batch_size, num_keys] 可选的前景分数
            
        Returns:
            weights: [batch_size, num_queries, n_heads, n_levels, n_points]
        """
        batch_size = query.shape[0]
        
        # 1. 预测基础权重
        weights = self.weight_predictor(query)  # [batch_size, num_queries, n_heads * n_levels * n_points]
        weights = weights.view(batch_size, -1, self.n_heads, self.n_levels, self.n_points)
        
        # 2. 应用温度参数
        weights = weights / self.temperature
        
        # 3. 对采样点进行归一化
        weights = F.softmax(weights, dim=-1)  # 对n_points维度归一化
        
        # 4. 应用层级权重
        level_weights = F.softmax(self.level_weights, dim=0)  # 对n_levels维度归一化
        weights = weights * level_weights.view(1, 1, 1, -1, 1)
        
        # 5. 结合前景分数（如果提供）
        if foreground_score is not None:
            # 使用可学习的缩放因子
            fg_score = foreground_score.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            fg_score = torch.sigmoid(self.fg_scale * fg_score)  # 将前景分数映射到[0,1]
            weights = weights * fg_score
            
        return weights 