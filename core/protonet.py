"""
Prototypical Networks (ProtoNet) Metric Learning Engine.
Enables instant registration and classification of novel PCB assembly defects
from 1-5 example shots without model retraining or weight updates.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class ProtoNetEngine:
    """
    Prototypical Network Manager:
    Maintains prototype vectors in 512-dim metric space and computes
    Euclidean distance-based probability distributions for zero-retraining few-shot inference.
    """
    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        temperature: float = 1.0,
        bank_path: Optional[str] = None
    ):
        self.backbone = backbone
        self.device = device
        self.backbone.to(self.device)
        self.backbone.eval()
        self.temperature = temperature
        self.bank_path = bank_path

        # Prototype Bank: dict of class_name -> { "centroid": np.ndarray, "shots": int, "meta": dict }
        self.prototype_bank: Dict[str, Dict[str, Any]] = {}

        if self.bank_path and os.path.exists(self.bank_path):
            self.load_prototype_bank(self.bank_path)

    def extract_embedding(self, tensor_input: torch.Tensor) -> torch.Tensor:
        """Extracts normalized embedding from (B, C, H, W) tensor."""
        if tensor_input.dim() == 3:
            tensor_input = tensor_input.unsqueeze(0)
        tensor_input = tensor_input.to(self.device)

        with torch.no_grad():
            if hasattr(self.backbone, "extract_features"):
                emb = self.backbone.extract_features(tensor_input)
            else:
                _, emb = self.backbone(tensor_input)
                emb = F.normalize(emb, p=2, dim=1)
        return emb

    def compute_prototype(self, support_tensors: torch.Tensor) -> np.ndarray:
        """
        Computes the prototype centroid vector c_k from support set tensors.
        support_tensors shape: (K, 3, H, W)
        """
        embeddings = self.extract_embedding(support_tensors)
        # Average embeddings across support shots
        centroid = torch.mean(embeddings, dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=1)
        return centroid.squeeze(0).cpu().numpy()

    def register_novel_defect(
        self,
        defect_name: str,
        support_tensors: torch.Tensor,
        description: str = "",
        ipc_standard_ref: str = "IPC-A-610 Custom",
        severity_baseline: str = "Major"
    ) -> Dict[str, Any]:
        """
        Registers a brand new defect class in the prototype memory bank.
        """
        centroid = self.compute_prototype(support_tensors)
        num_shots = support_tensors.shape[0] if support_tensors.dim() == 4 else 1

        self.prototype_bank[defect_name] = {
            "centroid": centroid.tolist(),
            "shots": int(num_shots),
            "registered_at": datetime.now().isoformat(),
            "meta": {
                "description": description,
                "ipc_standard_ref": ipc_standard_ref,
                "severity_baseline": severity_baseline
            }
        }

        if self.bank_path:
            self.save_prototype_bank(self.bank_path)

        return {
            "status": "success",
            "defect_name": defect_name,
            "shots_registered": int(num_shots),
            "embedding_dim": len(centroid),
            "timestamp": self.prototype_bank[defect_name]["registered_at"]
        }

    def predict_few_shot(
        self,
        query_tensor: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Computes metric distance to all prototypes in memory bank and
        returns softmax probabilities, ranked predictions, and nearest prototype.
        """
        if not self.prototype_bank:
            return {
                "error": "No prototypes registered in ProtoNet memory bank.",
                "top_predictions": []
            }

        query_emb = self.extract_embedding(query_tensor).cpu().numpy().squeeze(0)  # (512,)

        class_names = list(self.prototype_bank.keys())
        centroids = np.array([self.prototype_bank[c]["centroid"] for c in class_names])  # (N, 512)

        # Compute squared Euclidean distances: ||q - c_k||^2
        diff = centroids - query_emb  # (N, 512)
        sq_distances = np.sum(diff ** 2, axis=1)  # (N,)

        # Negative distance with temperature for softmax: exp(-d / tau)
        logits = -sq_distances / max(self.temperature, 1e-4)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Cosine similarity for complementary insight: dot product of unit vectors
        cosine_sims = np.dot(centroids, query_emb)

        # Rank predictions
        sorted_indices = np.argsort(-probabilities)

        results = []
        for idx in sorted_indices[:top_k]:
            c_name = class_names[idx]
            results.append({
                "class_name": c_name,
                "probability": float(probabilities[idx]),
                "euclidean_distance": float(sq_distances[idx]),
                "cosine_similarity": float(cosine_sims[idx]),
                "shots": self.prototype_bank[c_name]["shots"],
                "meta": self.prototype_bank[c_name].get("meta", {})
            })

        best_match = results[0]
        # Margin of confidence over second best
        confidence_margin = float(results[0]["probability"] - results[1]["probability"]) if len(results) > 1 else 1.0

        return {
            "predicted_class": best_match["class_name"],
            "confidence": best_match["probability"],
            "euclidean_distance": best_match["euclidean_distance"],
            "cosine_similarity": best_match["cosine_similarity"],
            "confidence_margin": confidence_margin,
            "top_predictions": results,
            "total_registered_classes": len(class_names)
        }

    def list_active_prototypes(self) -> List[Dict[str, Any]]:
        """Returns list of registered prototype classes and their metadata."""
        items = []
        for name, data in self.prototype_bank.items():
            items.append({
                "name": name,
                "shots": data["shots"],
                "registered_at": data.get("registered_at", "N/A"),
                "description": data.get("meta", {}).get("description", ""),
                "ipc_standard_ref": data.get("meta", {}).get("ipc_standard_ref", "")
            })
        return items

    def delete_prototype(self, defect_name: str) -> bool:
        """Removes a defect prototype from memory bank."""
        if defect_name in self.prototype_bank:
            del self.prototype_bank[defect_name]
            if self.bank_path:
                self.save_prototype_bank(self.bank_path)
            return True
        return False

    def save_prototype_bank(self, filepath: str):
        """Persists the prototype bank to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.prototype_bank, f, indent=2)

    def load_prototype_bank(self, filepath: str):
        """Loads prototype bank from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.prototype_bank = json.load(f)
