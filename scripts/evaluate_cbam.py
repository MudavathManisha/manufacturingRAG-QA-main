
"""
Evaluate trained CBAM CNN V2 on the official VisA PCB1-PCB4 test split.

IMPORTANT:
    - Uses the official 2cls_highshot.csv TEST split.
    - Does NOT modify the test data.
    - Evaluates the V2 best checkpoint selected using validation
      balanced accuracy.
"""

import os
import sys

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# IMPORTS
# =========================================================

import torch
from torch.utils.data import DataLoader

from core.cbam_cnn import PCBCBAMNet
from data.visa_dataset_loader import VisAPCBDataset


# =========================================================
# CONFIGURATION
# =========================================================

VISA_DIR = r"D:\ManufacturingRAG-QA\Visa\VisA_20220922"

# IMPORTANT:
# Evaluate the V2 model trained by train_cbam_cnn_v2.py
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cbam_cnn_visa_v2_best.pth"
)

BATCH_SIZE = 8


# =========================================================
# COLLATE FUNCTION
# =========================================================

def visa_collate_fn(batch):

    images = torch.stack([
        item[0]
        for item in batch
    ])

    labels = torch.tensor(
        [
            item[1]
            for item in batch
        ],
        dtype=torch.long
    )

    label_names = [
        item[2]
        for item in batch
    ]

    metadata = [
        item[3]
        for item in batch
    ]

    return (
        images,
        labels,
        label_names,
        metadata
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model():

    print("=" * 70)
    print("VisA PCB CBAM CNN V2 Evaluation")
    print("=" * 70)

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[Evaluation] Device: {device}"
    )

    # -----------------------------------------------------
    # Load official TEST dataset
    # -----------------------------------------------------

    test_dataset = VisAPCBDataset(
        visa_root=VISA_DIR,
        categories=[
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4"
        ],
        split="test",
        split_csv="2cls_highshot.csv"
    )

    print(
        f"[Data] Test samples: "
        f"{len(test_dataset)}"
    )

    # -----------------------------------------------------
    # Test DataLoader
    # -----------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=visa_collate_fn
    )

    print(
        f"[Data] Test batches: "
        f"{len(test_loader)}"
    )

    # =====================================================
    # LOAD V2 BEST CHECKPOINT
    # =====================================================

    # The checkpoint was created by our own training script.
    # weights_only=False is required because the checkpoint
    # contains training metadata / NumPy RNG state in addition
    # to the model state_dict.

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    print(
        f"[Model] Checkpoint: "
        f"{os.path.basename(MODEL_PATH)}"
    )

    print(
        f"[Model] Trained epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    stored_accuracy = checkpoint.get(
        "accuracy",
        None
    )

    if stored_accuracy is not None:

        print(
            f"[Model] Stored validation accuracy: "
            f"{stored_accuracy:.2f}%"
        )

    stored_balanced_accuracy = checkpoint.get(
        "validation_balanced_accuracy",
        None
    )

    if stored_balanced_accuracy is not None:

        print(
            f"[Model] Stored validation balanced accuracy: "
            f"{stored_balanced_accuracy:.2f}%"
        )

    print(
        f"[Model] Embedding dimension: "
        f"{checkpoint.get('embedding_dim', 'N/A')}"
    )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    class_names = checkpoint.get(
        "classes",
        [
            "Good_Assembly",
            "Anomaly"
        ]
    )

    model = PCBCBAMNet(
        num_classes=2,
        base_classes=class_names
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    # =====================================================
    # EVALUATION
    # =====================================================

    total = 0

    correct = 0

    # Confusion matrix:
    #
    #                  Predicted
    #                  Normal  Anomaly
    #
    # Actual Normal
    # Actual Anomaly

    confusion_matrix = torch.zeros(
        2,
        2,
        dtype=torch.long
    )

    print(
        "\n[Evaluation] Running inference..."
    )

    with torch.no_grad():

        for batch_idx, (
            images,
            labels,
            _,
            _
        ) in enumerate(
            test_loader,
            start=1
        ):

            images = images.to(device)
            labels = labels.to(device)

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            logits, _ = model(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            # -------------------------------------------------
            # Accuracy
            # -------------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # -------------------------------------------------
            # Confusion matrix
            # -------------------------------------------------

            for actual, predicted in zip(
                labels.cpu(),
                predictions.cpu()
            ):

                confusion_matrix[
                    actual,
                    predicted
                ] += 1

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                batch_idx % 10 == 0
                or
                batch_idx == len(test_loader)
            ):

                print(
                    f"Batch "
                    f"[{batch_idx:03d}/"
                    f"{len(test_loader):03d}]",
                    flush=True
                )

    # =====================================================
    # METRICS
    # =====================================================

    accuracy = (
        correct /
        max(total, 1)
    ) * 100.0

    tn = confusion_matrix[0, 0].item()
    fp = confusion_matrix[0, 1].item()
    fn = confusion_matrix[1, 0].item()
    tp = confusion_matrix[1, 1].item()

    precision = (
        tp /
        max(tp + fp, 1)
    )

    recall = (
        tp /
        max(tp + fn, 1)
    )

    f1 = (
        2 *
        precision *
        recall /
        max(precision + recall, 1e-12)
    )

    # Balanced accuracy

    normal_recall = (
        tn /
        max(tn + fp, 1)
    )

    anomaly_recall = (
        tp /
        max(tp + fn, 1)
    )

    balanced_accuracy = (
        (normal_recall + anomaly_recall)
        / 2.0
    )

    # =====================================================
    # RESULTS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OFFICIAL VisA TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Total test images       : {total}"
    )

    print(
        f"Correct predictions     : {correct}"
    )

    print(
        f"Accuracy                : {accuracy:.2f}%"
    )

    print(
        f"Balanced Accuracy       : "
        f"{balanced_accuracy * 100.0:.2f}%"
    )

    print(
        f"Precision               : {precision:.4f}"
    )

    print(
        f"Recall                  : {recall:.4f}"
    )

    print(
        f"F1 Score                : {f1:.4f}"
    )

    print(
        f"Normal Recall           : "
        f"{normal_recall * 100.0:.2f}%"
    )

    print(
        f"Anomaly Recall          : "
        f"{anomaly_recall * 100.0:.2f}%"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        "----------------"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 Normal  Anomaly"
    )

    print(
        f"Actual Normal    "
        f"{tn:6d}  {fp:7d}"
    )

    print(
        f"Actual Anomaly   "
        f"{fn:6d}  {tp:7d}"
    )

    print(
        "=" * 70
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_model()

