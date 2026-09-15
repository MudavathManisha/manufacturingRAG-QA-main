"""
Train CBAM CNN on VisA PCB1-PCB4 dataset.
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
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from core.cbam_cnn import PCBCBAMNet
from data.visa_dataset_loader import VisAPCBDataset


# =========================================================
# CUSTOM COLLATE FUNCTION
# =========================================================

def visa_collate_fn(batch):
    """
    Each dataset item contains:

        image
        label
        label_name
        metadata
    """

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
# TRAINING FUNCTION
# =========================================================

def train_model(
    visa_dir=r"D:\ManufacturingRAG-QA\Visa\VisA_20220922",
    model_save_path=None,
    epochs=10,
    batch_size=8,
    lr=1e-3
):

    # -----------------------------------------------------
    # Model save path
    # -----------------------------------------------------

    if model_save_path is None:

        model_save_path = os.path.join(
            PROJECT_ROOT,
            "models",
            "cbam_cnn_visa_best.pth"
        )

    os.makedirs(
        os.path.dirname(model_save_path),
        exist_ok=True
    )


    # -----------------------------------------------------
    # Check dataset
    # -----------------------------------------------------

    if not os.path.exists(visa_dir):

        raise FileNotFoundError(
            "VisA dataset not found:\n"
            f"{visa_dir}"
        )


    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 70)
    print("VisA PCB CBAM CNN Training")
    print("=" * 70)

    print(
        f"[Training] Device: {device}"
    )

    print(
        f"[Data] VisA directory:\n"
        f"{visa_dir}"
    )


    # =====================================================
    # DATASET
    # =====================================================

    dataset = VisAPCBDataset(
        visa_root=visa_dir,
        categories=[
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4"
        ]
    )


    if len(dataset) == 0:

        raise RuntimeError(
            "No VisA images were loaded.\n"
            "Check the dataset path and "
            "image_anno.csv files."
        )


    print(
        f"[Data] Total samples: "
        f"{len(dataset)}"
    )


    # -----------------------------------------------------
    # Classes
    # -----------------------------------------------------

    class_names = [
        "Good_Assembly",
        "Anomaly"
    ]

    print(
        f"[Data] Classes: "
        f"{class_names}"
    )


    # =====================================================
    # DATALOADER
    # =====================================================

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=visa_collate_fn
    )


    print(
        f"[Data] Number of batches: "
        f"{len(train_loader)}"
    )


    # =====================================================
    # MODEL
    # =====================================================

    model = PCBCBAMNet(
        num_classes=2,
        base_classes=class_names
    )

    model.to(device)


    print(
        f"[Model] Embedding dimension: "
        f"{model.embedding_dim}"
    )


    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss()


    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )


    # =====================================================
    # LEARNING RATE SCHEDULER
    # =====================================================

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )


    # =====================================================
    # TRAINING
    # =====================================================

    best_acc = 0.0


    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0


        print(
            f"\n--- Epoch "
            f"{epoch}/{epochs} ---"
        )


        for batch_idx, (
            images,
            labels,
            _,
            _
        ) in enumerate(
            train_loader,
            start=1
        ):

            # ---------------------------------------------
            # Move data to device
            # ---------------------------------------------

            images = images.to(device)
            labels = labels.to(device)


            # ---------------------------------------------
            # Clear gradients
            # ---------------------------------------------

            optimizer.zero_grad()


            # ---------------------------------------------
            # Forward pass
            # ---------------------------------------------

            logits, embeddings = model(
                images
            )


            # ---------------------------------------------
            # Classification loss
            # ---------------------------------------------

            loss = criterion(
                logits,
                labels
            )


            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            loss.backward()


            # ---------------------------------------------
            # Update model
            # ---------------------------------------------

            optimizer.step()


            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (
                batch_idx % 10 == 0
                or
                batch_idx == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"[{batch_idx:03d}/"
                    f"{len(train_loader):03d}] "
                    f"Loss: "
                    f"{loss.item():.4f}",
                    flush=True
                )


        # =================================================
        # EPOCH STATISTICS
        # =================================================

        scheduler.step()


        epoch_loss = (
            running_loss
            /
            max(total, 1)
        )


        epoch_acc = (
            correct
            /
            max(total, 1)
        ) * 100.0


        print(
            f"\nEpoch "
            f"[{epoch:02d}/{epochs:02d}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_acc:.2f}%"
        )


        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if epoch_acc > best_acc:

            best_acc = epoch_acc


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "classes":
                        class_names,

                    "embedding_dim":
                        model.embedding_dim,

                    "epoch":
                        epoch,

                    "accuracy":
                        epoch_acc,

                    "dataset":
                        "VisA PCB1-PCB4"
                },
                model_save_path
            )


            print(
                f"[Model] Saved best model:\n"
                f"{model_save_path}"
            )


    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Training Complete"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Accuracy: "
        f"{best_acc:.2f}%"
    )

    print(
        f"Model saved at:\n"
        f"{model_save_path}"
    )


    return model_save_path


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train_model()