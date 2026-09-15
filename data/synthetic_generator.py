"""
Procedural PCB Defect Image Synthesizer.
Generates photorealistic PCB assemblies with high-fidelity realistic defects
(Missing components, Misaligned chips, Solder bridges, Tombstoning, Solder balls,
and Insufficient solder fillets) for training and evaluation.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, List, Any, Optional


class PCBSyntheticGenerator:
    def __init__(self, width: int = 224, height: int = 224):
        self.width = width
        self.height = height

    def _create_pcb_substrate(self, base_color: Tuple[int, int, int] = (24, 76, 36)) -> np.ndarray:
        """Generates realistic FR4 substrate with copper traces, vias, and silkscreen."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # FR4 base green with subtle texture
        img[:] = base_color
        noise = np.random.randint(-10, 10, (self.height, self.width, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw copper traces (darker gold/green)
        trace_color = (35, 105, 50)
        for _ in range(5):
            x1 = random.randint(10, self.width - 10)
            y1 = random.randint(10, self.height - 10)
            x2 = random.randint(10, self.width - 10)
            y2 = y1 if random.random() > 0.5 else random.randint(10, self.height - 10)
            cv2.line(img, (x1, y1), (x2, y2), trace_color, thickness=random.choice([2, 3, 4]))

        # Draw small vias (circular gold rings)
        via_gold = (70, 180, 210)  # BGR
        for _ in range(random.randint(4, 8)):
            vx = random.randint(15, self.width - 15)
            vy = random.randint(15, self.height - 15)
            cv2.circle(img, (vx, vy), 4, via_gold, -1)
            cv2.circle(img, (vx, vy), 2, (15, 15, 15), -1)

        # Draw silkscreen white markings
        cv2.putText(img, "R12", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "C18", (self.width - 45, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "U1", (20, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)

        return img

    def _draw_solder_pad(self, img: np.ndarray, x: int, y: int, w: int, h: int, tinned: bool = True):
        """Draws a solder land pad (ENIG gold or HASL tin)."""
        pad_color = (200, 210, 220) if tinned else (60, 180, 220)  # Silver/Gold
        cv2.rectangle(img, (x, y), (x + w, y + h), pad_color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (140, 150, 160), 1)

    def _draw_solder_fillet(self, img: np.ndarray, x: int, y: int, w: int, h: int, quality: str = "good"):
        """Draws a solder joint fillet with specular highlights."""
        if quality == "good":
            # Smooth silver shiny fillet with gradient
            solder_base = (195, 205, 215)
            cv2.rectangle(img, (x, y), (x + w, y + h), solder_base, -1)
            # Specular shiny streak
            cv2.line(img, (x + 2, y + h // 2), (x + w - 2, y + h // 2), (245, 250, 255), 1)
        elif quality == "insufficient":
            # Starved, dark gray un-wetted joint with gaps
            solder_base = (100, 105, 110)
            cv2.rectangle(img, (x, y), (x + w, y + h), solder_base, -1)
            cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + h - 2), (40, 45, 50), -1)
        elif quality == "cold":
            # Rough, grainy, non-reflective dull solder
            solder_base = (130, 135, 140)
            cv2.rectangle(img, (x, y), (x + w, y + h), solder_base, -1)
            # Add grain noise
            sub = img[y:y+h, x:x+w]
            noise = np.random.randint(-20, 20, sub.shape, dtype=np.int16)
            img[y:y+h, x:x+w] = np.clip(sub.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_sample(self, defect_type: str = "Good_Assembly") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a synthetic PCB image adhering to the requested defect type.
        Returns:
            image_rgb: (224, 224, 3) uint8 numpy array
            metadata: dict containing defect labels, ground truth bbox, and severity
        """
        img = self._create_pcb_substrate()
        cx, cy = self.width // 2, self.height // 2
        pad_w, pad_h = 24, 40
        comp_w, comp_h = 60, 32

        # Standard 2-pad SMT component setup (0805 / 1206 chip)
        left_pad_x = cx - 44
        right_pad_x = cx + 20
        pad_y = cy - pad_h // 2

        self._draw_solder_pad(img, left_pad_x, pad_y, pad_w, pad_h)
        self._draw_solder_pad(img, right_pad_x, pad_y, pad_w, pad_h)

        meta = {
            "defect_type": defect_type,
            "target_center": [cx, cy],
            "bounding_box": [cx - comp_w // 2, cy - comp_h // 2, comp_w, comp_h],
            "severity_expected": "None"
        }

        if defect_type == "Good_Assembly":
            # Perfectly centered SMD resistor body (ceramic black/dark body + silver endcaps)
            comp_x = cx - comp_w // 2
            comp_y = cy - comp_h // 2
            # Resistor black body
            cv2.rectangle(img, (comp_x + 10, comp_y), (comp_x + comp_w - 10, comp_y + comp_h), (35, 35, 38), -1)
            # Silkscreen text on resistor (e.g. "103" = 10k)
            cv2.putText(img, "103", (comp_x + 16, comp_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
            # Metal endcaps & full solder fillets
            self._draw_solder_fillet(img, comp_x, comp_y, 10, comp_h, quality="good")
            self._draw_solder_fillet(img, comp_x + comp_w - 10, comp_y, 10, comp_h, quality="good")
            meta["severity_expected"] = "None"

        elif defect_type == "Missing_Component":
            # Pads have bare solder paste / gold, but NO component body
            # Add some printed solder paste flux marks
            cv2.rectangle(img, (left_pad_x + 4, pad_y + 6), (left_pad_x + pad_w - 4, pad_y + pad_h - 6), (110, 115, 120), -1)
            cv2.rectangle(img, (right_pad_x + 4, pad_y + 6), (right_pad_x + pad_w - 4, pad_y + pad_h - 6), (110, 115, 120), -1)
            meta["bounding_box"] = [left_pad_x, pad_y, right_pad_x + pad_w - left_pad_x, pad_h]
            meta["severity_expected"] = "Critical"

        elif defect_type == "Component_Misaligned":
            # Component placed with severe angular rotation (45 deg) and side overhang
            comp_x = cx - comp_w // 2 + 12
            comp_y = cy - comp_h // 2 - 18
            # Draw rotated component
            rect = ((cx + 6, cy - 8), (comp_w, comp_h), 38.0)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(img, [box], 0, (35, 35, 38), -1)
            # Disturbed solder
            self._draw_solder_fillet(img, left_pad_x, pad_y + 10, pad_w, pad_h - 10, quality="cold")
            self._draw_solder_fillet(img, right_pad_x, pad_y - 8, pad_w, pad_h - 12, quality="insufficient")
            meta["bounding_box"] = [comp_x - 5, comp_y - 5, comp_w + 10, comp_h + 30]
            meta["severity_expected"] = "Major"

        elif defect_type == "Solder_Bridge":
            # Perfectly placed component, but a heavy conductive solder bridge links left & right pads or IC pins
            comp_x = cx - comp_w // 2
            comp_y = cy - comp_h // 2
            cv2.rectangle(img, (comp_x + 10, comp_y), (comp_x + comp_w - 10, comp_y + comp_h), (35, 35, 38), -1)
            self._draw_solder_fillet(img, comp_x, comp_y, 10, comp_h, quality="good")
            self._draw_solder_fillet(img, comp_x + comp_w - 10, comp_y, 10, comp_h, quality="good")
            
            # Massive solder bridge extending across the bottom trace/pads
            bridge_pts = np.array([
                [left_pad_x + pad_w, cy + 8],
                [right_pad_x, cy + 6],
                [right_pad_x + 4, cy + 18],
                [left_pad_x + pad_w - 4, cy + 20]
            ], np.int32)
            cv2.fillPoly(img, [bridge_pts], (210, 220, 230))
            cv2.polylines(img, [bridge_pts], True, (245, 250, 255), 1)
            meta["bounding_box"] = [left_pad_x + 10, cy + 4, right_pad_x - left_pad_x + 10, 20]
            meta["severity_expected"] = "Critical"

        elif defect_type == "Tombstoning":
            # Component lifted vertically on left pad; right pad completely empty
            comp_x = left_pad_x - 4
            comp_y = cy - comp_h
            # Draw vertical standing chip body
            cv2.rectangle(img, (comp_x, comp_y), (comp_x + pad_w + 8, comp_y + comp_h), (35, 35, 38), -1)
            # High solder meniscus holding bottom end
            cv2.rectangle(img, (comp_x, cy - 8), (comp_x + pad_w + 8, cy + 12), (200, 210, 220), -1)
            # Lifted top endcap
            cv2.rectangle(img, (comp_x, comp_y), (comp_x + pad_w + 8, comp_y + 8), (170, 180, 190), -1)
            # Right pad exposed with cold un-wetted paste
            cv2.rectangle(img, (right_pad_x + 2, pad_y + 2), (right_pad_x + pad_w - 2, pad_y + pad_h - 2), (70, 75, 80), -1)
            meta["bounding_box"] = [comp_x - 4, comp_y - 4, pad_w + 16, comp_h + 24]
            meta["severity_expected"] = "Critical"

        elif defect_type == "Solder_Ball":
            # Normal component, with 3-5 loose shiny solder spheres scattered on solder mask
            comp_x = cx - comp_w // 2
            comp_y = cy - comp_h // 2
            cv2.rectangle(img, (comp_x + 10, comp_y), (comp_x + comp_w - 10, comp_y + comp_h), (35, 35, 38), -1)
            self._draw_solder_fillet(img, comp_x, comp_y, 10, comp_h, quality="good")
            self._draw_solder_fillet(img, comp_x + comp_w - 10, comp_y, 10, comp_h, quality="good")

            # Draw stray solder balls near leads
            solder_ball_color = (220, 230, 240)
            sb_coords = [(cx - 28, cy + 24), (cx + 26, cy - 24), (cx + 32, cy + 22)]
            for bx, by in sb_coords:
                cv2.circle(img, (bx, by), 5, solder_ball_color, -1)
                cv2.circle(img, (bx - 1, by - 1), 2, (255, 255, 255), -1)
                cv2.circle(img, (bx, by), 5, (100, 110, 120), 1)

            meta["bounding_box"] = [cx - 35, cy - 30, 70, 60]
            meta["severity_expected"] = "Moderate"

        elif defect_type == "Insufficient_Solder":
            # Component present, but one or both fillets starved/dry
            comp_x = cx - comp_w // 2
            comp_y = cy - comp_h // 2
            cv2.rectangle(img, (comp_x + 10, comp_y), (comp_x + comp_w - 10, comp_y + comp_h), (35, 35, 38), -1)
            self._draw_solder_fillet(img, comp_x, comp_y, 10, comp_h, quality="insufficient")
            self._draw_solder_fillet(img, comp_x + comp_w - 10, comp_y, 10, comp_h, quality="insufficient")
            meta["bounding_box"] = [left_pad_x, pad_y, right_pad_x + pad_w - left_pad_x, pad_h]
            meta["severity_expected"] = "Major"

        elif defect_type == "Laser_Burn_Defect":
            # Novel defect: localized laser mark burning / carbonization on FR4 laminate
            comp_x = cx - comp_w // 2
            comp_y = cy - comp_h // 2
            cv2.rectangle(img, (comp_x + 10, comp_y), (comp_x + comp_w - 10, comp_y + comp_h), (35, 35, 38), -1)
            self._draw_solder_fillet(img, comp_x, comp_y, 10, comp_h, quality="good")
            self._draw_solder_fillet(img, comp_x + comp_w - 10, comp_y, 10, comp_h, quality="good")

            # Carbon burn circle
            cv2.circle(img, (cx, cy + 28), 12, (20, 20, 25), -1)
            cv2.circle(img, (cx, cy + 28), 18, (30, 60, 40), 2)
            meta["bounding_box"] = [cx - 20, cy + 10, 40, 40]
            meta["severity_expected"] = "Major"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, meta

    def generate_dataset_batch(
        self,
        output_dir: str,
        samples_per_class: int = 10,
        classes: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Generates and saves a full synthetic PCB inspection dataset on disk.
        """
        target_classes = classes or [
            "Good_Assembly",
            "Missing_Component",
            "Component_Misaligned",
            "Solder_Bridge",
            "Tombstoning",
            "Solder_Ball",
            "Insufficient_Solder"
        ]

        os.makedirs(output_dir, exist_ok=True)
        manifest = {}

        for c in target_classes:
            class_dir = os.path.join(output_dir, c)
            os.makedirs(class_dir, exist_ok=True)
            manifest[c] = []

            for i in range(samples_per_class):
                img_rgb, _ = self.generate_sample(defect_type=c)
                filename = f"{c}_{i:03d}.png"
                file_path = os.path.join(class_dir, filename)
                Image.fromarray(img_rgb).save(file_path)
                manifest[c].append(file_path)

        return manifest
