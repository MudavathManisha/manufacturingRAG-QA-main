"""
Unit tests for ProtoNet Few-Shot Metric Learning Engine.
"""

import unittest
import torch
import numpy as np
from core.cbam_cnn import PCBCBAMNet
from core.protonet import ProtoNetEngine


class TestProtoNet(unittest.TestCase):
    def setUp(self):
        self.model = PCBCBAMNet(num_classes=7, embedding_dim=512)
        self.engine = ProtoNetEngine(self.model, device="cpu")
        self.dummy_support = torch.randn(3, 3, 224, 224)
        self.dummy_query = torch.randn(1, 3, 224, 224)

    def test_prototype_computation(self):
        centroid = self.engine.compute_prototype(self.dummy_support)
        self.assertIsInstance(centroid, np.ndarray)
        self.assertEqual(centroid.shape, (512,))
        # Check L2 unit length
        norm = np.linalg.norm(centroid)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_novel_defect_registration(self):
        res = self.engine.register_novel_defect(
            defect_name="Novel_Flux_Residue",
            support_tensors=self.dummy_support,
            description="Sticky flux residue across SMD pins",
            ipc_standard_ref="IPC-A-610 Clause 10.2"
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["defect_name"], "Novel_Flux_Residue")
        self.assertEqual(res["shots_registered"], 3)
        self.assertIn("Novel_Flux_Residue", self.engine.prototype_bank)

    def test_few_shot_prediction(self):
        # Register two classes
        self.engine.register_novel_defect("Class_A", torch.randn(2, 3, 224, 224))
        self.engine.register_novel_defect("Class_B", torch.randn(2, 3, 224, 224))

        pred = self.engine.predict_few_shot(self.dummy_query, top_k=2)
        self.assertIn("predicted_class", pred)
        self.assertIn(pred["predicted_class"], ["Class_A", "Class_B"])
        self.assertTrue(0.0 <= pred["confidence"] <= 1.0)
        self.assertEqual(len(pred["top_predictions"]), 2)

    def test_prototype_deletion(self):
        self.engine.register_novel_defect("To_Delete", self.dummy_support)
        self.assertIn("To_Delete", self.engine.prototype_bank)
        deleted = self.engine.delete_prototype("To_Delete")
        self.assertTrue(deleted)
        self.assertNotIn("To_Delete", self.engine.prototype_bank)


if __name__ == "__main__":
    unittest.main()
