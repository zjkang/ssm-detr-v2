import torch
import unittest
from models.bricks.salience_transformer import (
    SalienceTransformer,
    SalienceTransformerEncoder,
    SalienceTransformerEncoderLayer,
    SalienceTransformerDecoder,
    SalienceTransformerDecoderLayer,
    MaskPredictor
)
from models.necks.repnet import RepVGGPluXNetwork
from torch import nn

class TestSalienceTransformer(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.batch_size = 2
        self.embed_dim = 256
        self.num_classes = 91
        self.num_feature_levels = 4
        self.num_heads = 8
        self.dim_feedforward = 1024
        self.transformer_enc_layers = 6
        self.transformer_dec_layers = 6
        self.num_queries = 900

        self.model = SalienceTransformer(
            encoder=SalienceTransformerEncoder(
                encoder_layer=SalienceTransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    n_heads=self.num_heads,
                    dropout=0.0,
                    activation=nn.ReLU(inplace=True),
                    n_levels=self.num_feature_levels,
                    n_points=4,
                    d_ffn=self.dim_feedforward,
                ),
                num_layers=self.transformer_enc_layers,
            ),
            neck=RepVGGPluXNetwork(
                in_channels_list=[256,256,256,256],
                out_channels_list=[256,256,256,256],
                norm_layer=nn.BatchNorm2d,
                activation=nn.SiLU,
                groups=4,
            ),
            decoder=SalienceTransformerDecoder(
                decoder_layer=SalienceTransformerDecoderLayer(
                    embed_dim=self.embed_dim,
                    n_heads=self.num_heads,
                    dropout=0.0,
                    activation=nn.ReLU(inplace=True),
                    n_levels=self.num_feature_levels,
                    n_points=4,
                    d_ffn=self.dim_feedforward,
                ),
                num_layers=self.transformer_dec_layers,
                num_classes=self.num_classes,
            ),
            num_classes=self.num_classes,
            num_feature_levels=self.num_feature_levels,
            two_stage_num_proposals=self.num_queries,
            level_filter_ratio=(0.4, 0.8, 1.0, 1.0),
            layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        )

    def test_forward_workflow(self):
        # Create mock inputs
        multi_level_feats = [
            torch.randn(self.batch_size, self.embed_dim, 100, 140),  # Level 0
            torch.randn(self.batch_size, self.embed_dim, 50, 70),    # Level 1
            torch.randn(self.batch_size, self.embed_dim, 25, 35),    # Level 2
            torch.randn(self.batch_size, self.embed_dim, 13, 18)     # Level 3
        ]
        
        multi_level_masks = [
            torch.zeros(self.batch_size, 100, 140, dtype=torch.bool),  # Level 0
            torch.zeros(self.batch_size, 50, 70, dtype=torch.bool),    # Level 1
            torch.zeros(self.batch_size, 25, 35, dtype=torch.bool),    # Level 2
            torch.zeros(self.batch_size, 13, 18, dtype=torch.bool)     # Level 3
        ]
        
        multi_level_pos_embeds = [
            torch.randn(self.batch_size, self.embed_dim, 100, 140),  # Level 0
            torch.randn(self.batch_size, self.embed_dim, 50, 70),    # Level 1
            torch.randn(self.batch_size, self.embed_dim, 25, 35),    # Level 2
            torch.randn(self.batch_size, self.embed_dim, 13, 18)     # Level 3
        ]
        
        # Create mock queries for denoising training
        noised_label_query = torch.randn(self.batch_size, 196, self.embed_dim)
        noised_box_query = torch.randn(self.batch_size, 196, 4)
        
        # Create mock attention mask
        attn_mask = torch.zeros(self.batch_size, 1096, 1096, dtype=torch.bool)
        
        # Forward pass
        outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score = self.model(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            noised_label_query,
            noised_box_query,
            attn_mask
        )
        
        # Check output shapes
        self.assertEqual(outputs_classes.shape, (6, self.batch_size, 1000, self.num_classes))
        self.assertEqual(outputs_coords.shape, (6, self.batch_size, 1000, 4))
        self.assertEqual(enc_outputs_class.shape, (self.batch_size, self.two_stage_num_proposals, self.num_classes))
        self.assertEqual(enc_outputs_coord.shape, (self.batch_size, self.two_stage_num_proposals, 4))
        self.assertEqual(len(salience_score), self.num_feature_levels)
        
        # Check output types
        self.assertTrue(isinstance(outputs_classes, torch.Tensor))
        self.assertTrue(isinstance(outputs_coords, torch.Tensor))
        self.assertTrue(isinstance(enc_outputs_class, torch.Tensor))
        self.assertTrue(isinstance(enc_outputs_coord, torch.Tensor))
        self.assertTrue(all(isinstance(score, torch.Tensor) for score in salience_score))

if __name__ == '__main__':
    unittest.main() 