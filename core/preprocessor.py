"""
PCB Image Preprocessing and Enhancement Pipeline.
Includes CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space,
patch normalization, ROI extraction, and dataset augmentation transforms.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Union, Optional


class PCBPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def apply_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE on the L-channel of LAB color space to enhance solder joint
        and trace edge contrast without altering color hue/saturation.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def preprocess_image(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        apply_enhancement: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and prepares an image for model inference.
        Returns:
            processed_tensor_np: (3, H, W) normalized numpy array
            original_display_rgb: (H, W, 3) uint8 RGB image for visualization
        """
        # Convert input to BGR numpy array
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Could not decode image bytes")
        elif isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input.convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Resize for standard model input
        resized_bgr = cv2.resize(image_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        original_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        if apply_enhancement:
            enhanced_bgr = self.apply_clahe(resized_bgr)
            working_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        else:
            working_rgb = original_rgb

        # Normalize to float32 [0, 1] then subtract mean / std
        norm_img = working_rgb.astype(np.float32) / 255.0
        norm_img = (norm_img - self.mean) / self.std

        # Shape (H, W, C) -> (C, H, W)
        tensor_np = np.transpose(norm_img, (2, 0, 1))

        return tensor_np, original_rgb

    def extract_component_rois(
        self,
        image_bgr: np.ndarray,
        min_area: int = 400,
        max_area: int = 50000
    ) -> list:
        """
        Extracts candidate component ROIs (SMD pads, ICs, resistors) using
        adaptive thresholding and contour detection.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(h, 1)
                if 0.2 <= aspect_ratio <= 5.0:
                    roi_crop = image_bgr[y:y+h, x:x+w]
                    rois.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float(area),
                        "crop": roi_crop
                    })

        return rois

    @staticmethod
    def encode_image_base64(image_rgb: np.ndarray, format: str = "JPEG") -> str:
        """Converts an RGB numpy array to base64 data URI string."""
        pil_img = Image.fromarray(image_rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format=format, quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    @staticmethod
    def decode_base64_to_image(b64_string: str) -> np.ndarray:
        """Decodes base64 string or data URI into RGB numpy array."""
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        decoded = base64.b64decode(b64_string)
        pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")
        return np.array(pil_img)
