"""
Unit tests for Grad-CAM Saliency Localization and Bounding Box Extraction.
"""

import unittest
import torch
import numpy as np
import cv2
from core.cbam_cnn import PCBCBAMNet
from core.gradcam import GradCAM


class TestGradCAM(unittest.TestCase):
    def setUp(self):
        self.model = PCBCBAMNet(num_classes=7, embedding_dim=512)
        self.gradcam = GradCAM(self.model)
        self.dummy_tensor = torch.randn(1, 3, 224, 224)
        self.dummy_rgb = np.full((224, 224, 3), 128, dtype=np.uint8)

    def tearDown(self):
        self.gradcam.remove_hooks()

    def test_cam_generation(self):
        cam_map, target_class, score = self.gradcam.generate_cam(self.dummy_tensor)
        self.assertIsInstance(cam_map, np.ndarray)
        self.assertEqual(cam_map.shape, (224, 224))
        self.assertTrue(0.0 <= np.min(cam_map) <= np.max(cam_map) <= 1.0)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertIsInstance(target_class, int)

    def test_overlay_heatmap(self):
        cam_map = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)
        overlay, heatmap = GradCAM.overlay_heatmap(self.dummy_rgb, cam_map, alpha=0.5)
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(heatmap.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_bounding_box_extraction(self):
        # Create synthetic cam with a distinct hot spot
        cam_map = np.zeros((224, 224), dtype=np.float32)
        cam_map[50:100, 60:120] = 0.95
        boxes = GradCAM.extract_defect_bounding_boxes(cam_map, threshold_ratio=0.5, min_area=50)
        self.assertGreater(len(boxes), 0)
        first_box = boxes[0]
        self.assertIn("bbox", first_box)
        self.assertIn("peak_activation", first_box)
        self.assertGreaterEqual(first_box["peak_activation"], 0.9)


if __name__ == "__main__":
    unittest.main()
