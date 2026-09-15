"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Visual Explainer.
Generates saliency heatmaps, alpha-blended overlays, and automated bounding box
proposals around localized PCB assembly defect regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Any


class GradCAM:
    """
    Grad-CAM implementation for PCBCBAMNet and PyTorch vision backbones.
    Hooks into intermediate/final convolutional attention layers to calculate
    activation maps weighted by class prediction gradients.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or (
            model.get_target_cam_layer() if hasattr(model, "get_target_cam_layer") else model.layer4[-1]
        )
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes the raw 2D Grad-CAM heatmap for a given input tensor and class index.
        Returns:
            cam_map: (H, W) float32 array normalized to [0, 1]
            target_class_idx: predicted or specified class index
            score: softmax confidence for target class
        """
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Requires grad for backward pass
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits, _ = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = float(probs[0, target_class_idx].item())

        # Zero gradients & backpropagate on target class score
        self.model.zero_grad()
        loss = logits[0, target_class_idx]
        loss.backward(retain_graph=True)

        # Compute gradient weights: alpha_k = global average pooling over gradients
        # gradients shape: (1, C, H, W)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Linear combination of feature maps weighted by alpha
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')

        # Apply ReLU to retain only positive influences on class score
        cam = F.relu(cam)

        # Interpolate to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class_idx, score

    @staticmethod
    def overlay_heatmap(
        original_rgb: np.ndarray,
        cam_map: np.ndarray,
        alpha: float = 0.55,
        colormap_type: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays the Grad-CAM heatmap on the original RGB image.
        Returns:
            overlay_rgb: (H, W, 3) blended RGB image
            heatmap_rgb: (H, W, 3) colored heatmap standalone
        """
        h, w, _ = original_rgb.shape
        if cam_map.shape != (h, w):
            cam_resized = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_resized = cam_map

        # Scale to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * cam_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap_type)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (alpha * heatmap_rgb + (1 - alpha) * original_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return overlay, heatmap_rgb

    @staticmethod
    def extract_defect_bounding_boxes(
        cam_map: np.ndarray,
        threshold_ratio: float = 0.55,
        min_area: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Extracts localized defect bounding boxes from high-activation Grad-CAM clusters.
        """
        binary = (cam_map >= (threshold_ratio * np.max(cam_map))).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                roi_cam = cam_map[y:y+h, x:x+w]
                peak_intensity = float(np.max(roi_cam))
                mean_intensity = float(np.mean(roi_cam))
                
                boxes.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": float(area),
                    "peak_activation": peak_intensity,
                    "mean_activation": mean_intensity,
                    "center": [int(x + w / 2), int(y + h / 2)]
                })

        # Sort by peak activation descending
        boxes.sort(key=lambda b: b["peak_activation"], reverse=True)
        return boxes
