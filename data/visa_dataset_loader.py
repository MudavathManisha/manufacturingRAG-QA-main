"""
VisA PCB Dataset Loader + Episodic Few-Shot Sampler

Supports the official VisA split CSV files:

    1cls.csv
    2cls_highshot.csv
    2cls_fewshot.csv

For the main CBAM CNN experiment:
    2cls_highshot.csv

Labels:
    normal  -> 0 (Good_Assembly)
    anomaly -> 1 (Anomaly)
"""

import os
import csv
import random

from typing import Tuple, List, Dict, Optional, Any

import torch
from torch.utils.data import Dataset

from core.preprocessor import PCBPreprocessor


class VisAPCBDataset(Dataset):
    """
    Dataset loader specifically for VisA PCB1-PCB4.

    Uses the official split CSV located at:

        VisA_20220922/split_csv/

    Example:

        dataset = VisAPCBDataset(
            visa_root=visa_dir,
            categories=["pcb1", "pcb2", "pcb3", "pcb4"],
            split="train",
            split_csv="2cls_highshot.csv"
        )
    """

    def __init__(
        self,
        visa_root: str,
        categories: Optional[List[str]] = None,
        split: str = "all",
        split_csv: str = "1cls.csv",
        transform: bool = True
    ):

        self.visa_root = visa_root

        self.categories = categories or [
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4"
        ]

        self.split = split.lower()
        self.split_csv = split_csv

        self.preprocessor = PCBPreprocessor()
        self.transform = transform

        # Binary classification
        self.classes = [
            "Good_Assembly",
            "Anomaly"
        ]

        self.class_to_idx = {
            "Good_Assembly": 0,
            "Anomaly": 1
        }

        if self.split not in [
            "train",
            "test",
            "all"
        ]:
            raise ValueError(
                "split must be 'train', 'test', or 'all'"
            )

        self.samples = []

        self._load_annotations()

        print(
            f"[VisA Loader] Loaded "
            f"{len(self.samples)} images "
            f"from {self.categories} "
            f"(split={self.split}, "
            f"csv={self.split_csv})"
        )

    def _load_annotations(self):
        """
        Read the official VisA split CSV.
        """

        split_file = os.path.join(
            self.visa_root,
            "split_csv",
            self.split_csv
        )

        if not os.path.exists(split_file):

            raise FileNotFoundError(
                "VisA split CSV not found:\n"
                f"{split_file}"
            )

        print(
            f"[VisA Loader] Reading official split:\n"
            f"{split_file}"
        )

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:

                category = row.get(
                    "object",
                    ""
                ).strip().lower()

                row_split = row.get(
                    "split",
                    ""
                ).strip().lower()

                label = row.get(
                    "label",
                    ""
                ).strip().lower()

                image_path = row.get(
                    "image",
                    ""
                ).strip()

                mask_path = row.get(
                    "mask",
                    ""
                ).strip()

                # -------------------------------------------------
                # Category filtering
                # -------------------------------------------------

                if category not in self.categories:
                    continue

                # -------------------------------------------------
                # Split filtering
                # -------------------------------------------------

                if (
                    self.split != "all"
                    and row_split != self.split
                ):
                    continue

                if not image_path:
                    continue

                # -------------------------------------------------
                # Image path
                # -------------------------------------------------

                full_image_path = os.path.join(
                    self.visa_root,
                    image_path.replace(
                        "/",
                        os.sep
                    )
                )

                # -------------------------------------------------
                # Binary label
                # -------------------------------------------------

                if label == "normal":

                    class_label = 0
                    label_name = "Good_Assembly"

                else:

                    class_label = 1
                    label_name = "Anomaly"

                # -------------------------------------------------
                # Mask path
                # -------------------------------------------------

                full_mask_path = None

                if mask_path:

                    full_mask_path = os.path.join(
                        self.visa_root,
                        mask_path.replace(
                            "/",
                            os.sep
                        )
                    )

                # -------------------------------------------------
                # Verify image exists
                # -------------------------------------------------

                if not os.path.exists(
                    full_image_path
                ):

                    print(
                        f"[Warning] Image not found: "
                        f"{full_image_path}"
                    )

                    continue

                # -------------------------------------------------
                # Store sample
                # -------------------------------------------------

                self.samples.append(
                    {
                        "image_path":
                            full_image_path,

                        "label":
                            class_label,

                        "label_name":
                            label_name,

                        "mask_path":
                            full_mask_path,

                        "category":
                            category,

                        "original_label":
                            label,

                        "split":
                            row_split
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[
        torch.Tensor,
        int,
        str,
        Dict[str, Any]
    ]:

        sample = self.samples[idx]

        image_path = sample["image_path"]

        tensor_np, _ = (
            self.preprocessor.preprocess_image(
                image_path,
                apply_enhancement=self.transform
            )
        )

        tensor = torch.from_numpy(
            tensor_np
        ).float()

        metadata = {
            "image_path":
                sample["image_path"],

            "mask_path":
                sample["mask_path"],

            "category":
                sample["category"],

            "original_label":
                sample["original_label"],

            "split":
                sample["split"]
        }

        return (
            tensor,
            sample["label"],
            sample["label_name"],
            metadata
        )


class EpisodicFewShotSampler:
    """
    Generates N-way K-shot episodes for ProtoNet.
    """

    def __init__(
        self,
        dataset: VisAPCBDataset,
        n_way: int = 2,
        k_shot: int = 5,
        q_query: int = 15
    ):

        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.class_indices: Dict[
            int,
            List[int]
        ] = {}

        for idx, sample in enumerate(
            dataset.samples
        ):

            label = sample["label"]

            if label not in self.class_indices:
                self.class_indices[label] = []

            self.class_indices[label].append(idx)

    def sample_episode(
        self
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:

        required_samples = (
            self.k_shot +
            self.q_query
        )

        available_classes = [
            cls
            for cls, indices
            in self.class_indices.items()
            if len(indices) >= required_samples
        ]

        if len(available_classes) < self.n_way:

            available_classes = list(
                self.class_indices.keys()
            )

        if len(available_classes) == 0:

            raise RuntimeError(
                "No classes available "
                "for episodic sampling."
            )

        sampled_classes = random.sample(
            available_classes,
            min(
                self.n_way,
                len(available_classes)
            )
        )

        support_list = []
        support_labels = []

        query_list = []
        query_labels = []

        for episode_label, class_id in enumerate(
            sampled_classes
        ):

            indices = self.class_indices[
                class_id
            ]

            if len(indices) >= required_samples:

                chosen = random.sample(
                    indices,
                    required_samples
                )

            else:

                chosen = random.choices(
                    indices,
                    k=required_samples
                )

            support_indices = chosen[
                :self.k_shot
            ]

            query_indices = chosen[
                self.k_shot:
            ]

            for idx in support_indices:

                image, _, _, _ = self.dataset[idx]

                support_list.append(image)
                support_labels.append(
                    episode_label
                )

            for idx in query_indices:

                image, _, _, _ = self.dataset[idx]

                query_list.append(image)
                query_labels.append(
                    episode_label
                )

        support_tensors = torch.stack(
            support_list
        )

        support_labels = torch.tensor(
            support_labels,
            dtype=torch.long
        )

        query_tensors = torch.stack(
            query_list
        )

        query_labels = torch.tensor(
            query_labels,
            dtype=torch.long
        )

        return (
            support_tensors,
            support_labels,
            query_tensors,
            query_labels
        )


# Backward compatibility
PCBDataset = VisAPCBDataset