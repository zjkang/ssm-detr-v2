import torch
import unittest
from models.bricks.ssm_transformer import (
    SsmTransformer,
    SsmTransformerDecoder,
    SsmTransformerDecoderLayer,
    SsmTransformerEncoder,
    SsmTransformerEncoderLayer,
)
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
        self.num_queries = 300 # original 300
        
        # Force CUDA usage
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
        self.device = torch.device('cuda')
        print(f"Using device: {self.device}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        self.model = SsmTransformer(
            encoder=SsmTransformerEncoder(
                encoder_layer=SsmTransformerEncoderLayer(
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
            decoder=SsmTransformerDecoder(
                decoder_layer=SsmTransformerDecoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    dropout=0.1,
                    activation="relu",
                    dim_feedforward=self.dim_feedforward,
                    num_proposal=self.num_queries,
                    ssm_use_biscan=True, # TODO: test purposes,
                    chunk_size=32,
                ),
                num_layers=self.transformer_dec_layers,
                num_classes=self.num_classes,
            ),
            num_classes=self.num_classes,
            num_feature_levels=self.num_feature_levels,
            two_stage_num_proposals=self.num_queries,
            level_filter_ratio=(0.3, 0.4, 0.8, 0.8),
            layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        ).to(self.device)

    def test_forward_workflow(self):
        # Create mock inputs and move them to GPU
        multi_level_feats = [
            torch.randn(self.batch_size, self.embed_dim, 100, 140, device=self.device),  # Level 0
            torch.randn(self.batch_size, self.embed_dim, 50, 70, device=self.device),    # Level 1
            torch.randn(self.batch_size, self.embed_dim, 25, 35, device=self.device),    # Level 2
            torch.randn(self.batch_size, self.embed_dim, 13, 18, device=self.device)     # Level 3
        ]
        
        multi_level_masks = [
            torch.zeros(self.batch_size, 100, 140, dtype=torch.bool, device=self.device),  # Level 0
            torch.zeros(self.batch_size, 50, 70, dtype=torch.bool, device=self.device),    # Level 1
            torch.zeros(self.batch_size, 25, 35, dtype=torch.bool, device=self.device),    # Level 2
            torch.zeros(self.batch_size, 13, 18, dtype=torch.bool, device=self.device)     # Level 3
        ]
        
        multi_level_pos_embeds = [
            torch.randn(self.batch_size, self.embed_dim, 100, 140, device=self.device),  # Level 0
            torch.randn(self.batch_size, self.embed_dim, 50, 70, device=self.device),    # Level 1
            torch.randn(self.batch_size, self.embed_dim, 25, 35, device=self.device),    # Level 2
            torch.randn(self.batch_size, self.embed_dim, 13, 18, device=self.device)     # Level 3
        ]
        
        # Forward pass
        outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score = self.model(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
        )
        
        # Check output shapes
        self.assertEqual(outputs_classes.shape, (6, self.batch_size, self.num_queries, self.num_classes))
        self.assertEqual(outputs_coords.shape, (6, self.batch_size, self.num_queries, 4))
        self.assertEqual(enc_outputs_class.shape, (self.batch_size, 18609, self.num_classes))
        self.assertEqual(enc_outputs_coord.shape, (self.batch_size, 18609, 4))
        self.assertEqual(len(salience_score), 4)
        

if __name__ == '__main__':
    unittest.main() 