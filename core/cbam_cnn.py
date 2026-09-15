"""
Custom Convolutional Neural Network with CBAM (Convolutional Block Attention Module).
Integrates Channel Attention and Spatial Attention mechanisms for fine-grained
PCB assembly defect localization and metric embedding feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module:
    Explores inter-channel feature relationships using joint AvgPool and MaxPool
    passed through a shared multi-layer perceptron.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention_map = self.sigmoid(out)
        return x * attention_map


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module:
    Explores spatial defect locality (solder bridge locations, missing pads)
    using 7x7 convolution over pooled channel slices.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv1(combined))
        return x * attention_map


class CBAMBlock(nn.Module):
    """
    Sequential CBAM Block combining Channel Attention followed by Spatial Attention.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CBAMResidualBlock(nn.Module):
    """
    Residual Block with integrated CBAM attention and optional downsampling.
    """
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, ratio: int = 16):
        super(CBAMResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.cbam = CBAMBlock(out_planes, ratio=ratio)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out


class PCBCBAMNet(nn.Module):
    """
    Custom Deep Attention Network for PCB Inspection:
    Extracts multi-scale representations refined by CBAM attention at each stage,
    projecting into a 512-dim metric space for ProtoNet and classification.
    """
    def __init__(
        self,
        num_classes: int = 7,
        embedding_dim: int = 512,
        base_classes: Optional[list] = None
    ):
        super(PCBCBAMNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.base_classes = base_classes or [
            "Good_Assembly",
            "Missing_Component",
            "Component_Misaligned",
            "Solder_Bridge",
            "Tombstoning",
            "Solder_Ball",
            "Insufficient_Solder"
        ]

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Residual Stages with CBAM
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global pooling & metric embedding projector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Classification Head (Base Classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(embedding_dim, self.num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_planes: int, out_planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [CBAMResidualBlock(in_planes, out_planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(CBAMResidualBlock(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dim L2-normalized embedding representation for metric learning.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.embedding_head(x)
        norm_feat = F.normalize(feat, p=2, dim=1)
        return norm_feat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (class_logits, feature_embeddings).
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Target layer for Grad-CAM
        
        pooled = self.avgpool(x)
        flattened = torch.flatten(pooled, 1)
        embeddings = self.embedding_head(flattened)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_target_cam_layer(self) -> nn.Module:
        """Returns the final convolutional block of layer4 for Grad-CAM hooks."""
        return self.layer4[-1]
