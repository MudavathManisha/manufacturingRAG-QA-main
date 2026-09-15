"""
Few-Shot Evaluation Script for ProtoNet on VisA PCB1-PCB4.

Evaluation:
    Official VisA 2cls_fewshot.csv
    Train split -> Support Set
    Test split  -> Query Set

Pipeline:
    PCB Image
        ↓
    CBAM CNN
        ↓
    512-D Embedding
        ↓
    ProtoNet
        ↓
    Few-Shot Prediction
"""

import os
import sys
import random

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import torch
import torch.nn.functional as F
import numpy as np

from core.cbam_cnn import PCBCBAMNet
from data.visa_dataset_loader import VisAPCBDataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VISA_ROOT = r"D:\ManufacturingRAG-QA\Visa\VisA_20220922"

CATEGORIES = [
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4"
]

SPLIT_CSV = "2cls_fewshot.csv"

MODEL_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "models",
    "cbam_cnn_visa_best.pth"
)


# ---------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------

def sample_episode(
    support_dataset,
    query_dataset,
    n_way=2,
    k_shot=5,
    q_query=10
):
    """
    Creates one few-shot episode.

    Support examples come from the training split.
    Query examples come from the test split.

    Returns:
        support_tensors
        support_labels
        query_tensors
        query_labels
    """

    # Build class index dictionaries

    support_indices = {}
    query_indices = {}

    for idx, sample in enumerate(
        support_dataset.samples
    ):

        label = sample["label"]

        if label not in support_indices:
            support_indices[label] = []

        support_indices[label].append(idx)

    for idx, sample in enumerate(
        query_dataset.samples
    ):

        label = sample["label"]

        if label not in query_indices:
            query_indices[label] = []

        query_indices[label].append(idx)

    # Classes available in BOTH splits

    available_classes = [
        c
        for c in support_indices
        if c in query_indices
        and len(support_indices[c]) >= k_shot
        and len(query_indices[c]) >= q_query
    ]

    if len(available_classes) < n_way:
        raise RuntimeError(
            f"Not enough classes for "
            f"{n_way}-way evaluation. "
            f"Available classes: {available_classes}"
        )

    # Randomly choose N classes

    selected_classes = random.sample(
        available_classes,
        n_way
    )

    support_images = []
    support_labels = []

    query_images = []
    query_labels = []

    # Create episode

    for episode_label, class_id in enumerate(
        selected_classes
    ):

        selected_support = random.sample(
            support_indices[class_id],
            k_shot
        )

        selected_query = random.sample(
            query_indices[class_id],
            q_query
        )

        # Support

        for idx in selected_support:

            image, _, _, _ = support_dataset[idx]

            support_images.append(image)
            support_labels.append(
                episode_label
            )

        # Query

        for idx in selected_query:

            image, _, _, _ = query_dataset[idx]

            query_images.append(image)
            query_labels.append(
                episode_label
            )

    support_tensors = torch.stack(
        support_images
    )

    support_labels = torch.tensor(
        support_labels,
        dtype=torch.long
    )

    query_tensors = torch.stack(
        query_images
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


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def evaluate_few_shot_accuracy(
    visa_root=VISA_ROOT,
    model_weights_path=MODEL_WEIGHTS,
    n_way=2,
    k_shot=5,
    q_query=10,
    episodes=20
):

    print("\n==============================================")
    print("      ProtoNet Few-Shot Evaluation")
    print("==============================================")

    print(f"Dataset     : VisA PCB1-PCB4")
    print(f"Split       : {SPLIT_CSV}")
    print(f"Setting     : {n_way}-Way {k_shot}-Shot")
    print(f"Query       : {q_query} per class")
    print(f"Episodes    : {episodes}")

    # -----------------------------------------------------
    # Load support dataset
    # -----------------------------------------------------

    support_dataset = VisAPCBDataset(
        visa_root=visa_root,
        categories=CATEGORIES,
        split="train",
        split_csv=SPLIT_CSV
    )

    # -----------------------------------------------------
    # Load query dataset
    # -----------------------------------------------------

    query_dataset = VisAPCBDataset(
        visa_root=visa_root,
        categories=CATEGORIES,
        split="test",
        split_csv=SPLIT_CSV
    )

    print(
        f"\nSupport images: "
        f"{len(support_dataset)}"
    )

    print(
        f"Query images  : "
        f"{len(query_dataset)}"
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device        : {device}")

    # -----------------------------------------------------
    # Load CBAM backbone
    # -----------------------------------------------------

    model = PCBCBAMNet(
        num_classes=2,
        embedding_dim=512,
        base_classes=[
            "Good_Assembly",
            "Anomaly"
        ]
    )

    # -----------------------------------------------------
    # Load trained weights
    # -----------------------------------------------------

    if not os.path.exists(
        model_weights_path
    ):

        raise FileNotFoundError(
            "\nCBAM checkpoint not found:\n"
            f"{model_weights_path}\n\n"
            "Wait for CBAM training to finish "
            "before running ProtoNet evaluation."
        )

    checkpoint = torch.load(
        model_weights_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"\n[ProtoNet] Loaded CBAM checkpoint:"
        f"\n{model_weights_path}"
    )

    model.to(device)
    model.eval()

    # -----------------------------------------------------
    # Episode evaluation
    # -----------------------------------------------------

    accuracies = []

    print(
        "\n----------------------------------------------"
    )

    for episode in range(
        1,
        episodes + 1
    ):

        (
            support_tensors,
            support_labels,
            query_tensors,
            query_labels
        ) = sample_episode(
            support_dataset,
            query_dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query
        )

        support_tensors = (
            support_tensors.to(device)
        )

        query_tensors = (
            query_tensors.to(device)
        )

        # -------------------------------------------------
        # Extract embeddings
        # -------------------------------------------------

        with torch.no_grad():

            support_embeddings = (
                model.extract_features(
                    support_tensors
                )
            )

            query_embeddings = (
                model.extract_features(
                    query_tensors
                )
            )

        # -------------------------------------------------
        # Compute class prototypes
        # -------------------------------------------------

        prototypes = []

        for class_id in range(n_way):

            class_embeddings = (
                support_embeddings[
                    support_labels == class_id
                ]
            )

            prototype = (
                class_embeddings.mean(
                    dim=0,
                    keepdim=True
                )
            )

            prototype = F.normalize(
                prototype,
                p=2,
                dim=1
            )

            prototypes.append(
                prototype
            )

        prototypes = torch.cat(
            prototypes,
            dim=0
        )

        # -------------------------------------------------
        # Compute distances
        # -------------------------------------------------

        distances = torch.cdist(
            query_embeddings,
            prototypes,
            p=2
        ) ** 2

        predictions = torch.argmin(
            distances,
            dim=1
        ).cpu()

        # -------------------------------------------------
        # Accuracy
        # -------------------------------------------------

        correct = (
            predictions == query_labels
        ).sum().item()

        total = query_labels.size(0)

        accuracy = (
            correct / total
        ) * 100.0

        accuracies.append(
            accuracy
        )

        running_mean = np.mean(
            accuracies
        )

        if (
            episode % 5 == 0
            or episode == 1
            or episode == episodes
        ):

            print(
                f"Episode [{episode:02d}/{episodes:02d}] "
                f"Accuracy: {accuracy:.2f}% "
                f"| Mean: {running_mean:.2f}%"
            )

    # -----------------------------------------------------
    # Final statistics
    # -----------------------------------------------------

    mean_accuracy = np.mean(
        accuracies
    )

    std = np.std(
        accuracies,
        ddof=1
    )

    std_error = (
        std / np.sqrt(episodes)
    )

    confidence_interval = (
        1.96 * std_error
    )

    print(
        "\n=============================================="
    )

    print(
        "           ProtoNet Evaluation"
    )

    print(
        "=============================================="
    )

    print(
        f"Dataset          : VisA PCB1-PCB4"
    )

    print(
        f"Few-Shot Setting : "
        f"{n_way}-Way {k_shot}-Shot"
    )

    print(
        f"Episodes         : {episodes}"
    )

    print(
        f"Mean Accuracy    : "
        f"{mean_accuracy:.2f}%"
    )

    print(
        f"95% CI           : "
        f"+/- {confidence_interval:.2f}%"
    )

    print(
        "==============================================\n"
    )

    return (
        mean_accuracy,
        confidence_interval
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_few_shot_accuracy(
        n_way=2,
        k_shot=5,
        q_query=10,
        episodes=100
    )