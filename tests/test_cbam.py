"""
Unit tests for Custom CNN with CBAM (Channel & Spatial Attention) Backbone.
"""

import unittest
import torch
from core.cbam_cnn import ChannelAttention, SpatialAttention, CBAMBlock, CBAMResidualBlock, PCBCBAMNet


class TestCBAMCNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 64
        self.height, self.width = 56, 56
        self.dummy_input = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)

    def test_channel_attention(self):
        ca = ChannelAttention(in_planes=self.channels, ratio=16)
        out = ca(self.dummy_input)
        self.assertEqual(out.shape, self.dummy_input.shape)
        self.assertTrue(torch.is_tensor(out))

    def test_spatial_attention(self):
        sa = SpatialAttention(kernel_size=7)
        out = sa(self.dummy_input)
        self.assertEqual(out.shape, self.dummy_input.shape)
        self.assertTrue(torch.is_tensor(out))

    def test_cbam_block(self):
        cbam = CBAMBlock(in_planes=self.channels, ratio=16, kernel_size=7)
        out = cbam(self.dummy_input)
        self.assertEqual(out.shape, self.dummy_input.shape)

    def test_cbam_residual_block(self):
        res_block = CBAMResidualBlock(in_planes=64, out_planes=128, stride=2)
        out = res_block(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 128, 28, 28))

    def test_pcb_cbam_net_forward(self):
        model = PCBCBAMNet(num_classes=7, embedding_dim=512)
        logits, embeddings = model(self.dummy_image)
        self.assertEqual(logits.shape, (self.batch_size, 7))
        self.assertEqual(embeddings.shape, (self.batch_size, 512))

    def test_pcb_cbam_net_features(self):
        model = PCBCBAMNet(num_classes=7, embedding_dim=512)
        features = model.extract_features(self.dummy_image)
        self.assertEqual(features.shape, (self.batch_size, 512))
        # Check L2 normalization: norm should be approximately 1.0
        norms = torch.norm(features, p=2, dim=1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
