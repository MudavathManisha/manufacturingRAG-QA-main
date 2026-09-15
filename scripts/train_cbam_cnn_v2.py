
"""
CBAM CNN V2 - Proper VisA Training Pipeline

Purpose:
    Train the existing PCBCBAMNet architecture correctly on VisA PCB1-PCB4.

Features:
    - Uses official 2cls_highshot TRAIN split only
    - Creates validation split from TRAIN data
    - Keeps official TEST data completely untouched
    - Training-only augmentation
    - WeightedRandomSampler for class imbalance
    - Label smoothing
    - AdamW optimizer
    - Cosine learning-rate schedule
    - Best checkpoint selected using validation balanced accuracy
    - Early stopping
    - Automatic epoch checkpointing
    - Automatic resume after system shutdown/interruption
    - Atomic checkpoint writing to reduce corruption risk

IMPORTANT:
    This script does NOT modify the existing PCBCBAMNet architecture.
"""


import os
import sys
import random
import copy


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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from torchvision import transforms

from core.cbam_cnn import PCBCBAMNet
from data.visa_dataset_loader import VisAPCBDataset


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# CONFIGURATION
# =========================================================

VISA_DIR = r"D:\ManufacturingRAG-QA\Visa\VisA_20220922"


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cbam_cnn_visa_v2_best.pth"
)


CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "cbam_cnn_visa_v2_checkpoint.pth"
)


EPOCHS = 30

BATCH_SIZE = 16

LEARNING_RATE = 3e-4

VALIDATION_RATIO = 0.10

PATIENCE = 7

NUM_WORKERS = 0


# =========================================================
# CHECKPOINT CONFIGURATION
# =========================================================

CHECKPOINT_VERSION = 2

# True:
#     If checkpoint exists, automatically resume training.
#
# False:
#     Always start a completely new training run.
AUTO_RESUME = True


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
# TRAINING AUGMENTATION
# =========================================================

class TrainAugmentation:

    def __init__(self):

        self.transform = transforms.Compose([

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

        ])

    def __call__(self, image):

        return self.transform(image)


# =========================================================
# AUGMENTED SUBSET
# =========================================================

class AugmentedSubset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        indices,
        augmentation=None
    ):

        self.dataset = dataset
        self.indices = indices
        self.augmentation = augmentation

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        original_index = self.indices[index]

        image, label, label_name, metadata = \
            self.dataset[original_index]

        if self.augmentation is not None:

            image = self.augmentation(image)

        return (
            image,
            label,
            label_name,
            metadata
        )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    predictions,
    labels
):

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    normal_total = normal_mask.sum()
    anomaly_total = anomaly_mask.sum()

    normal_correct = (
        (predictions[normal_mask] == 0).sum()
    )

    anomaly_correct = (
        (predictions[anomaly_mask] == 1).sum()
    )

    total_correct = (
        (predictions == labels).sum()
    )

    total = len(labels)

    accuracy = (
        total_correct /
        max(total, 1)
    ) * 100.0

    normal_recall = (
        normal_correct /
        max(normal_total, 1)
    )

    anomaly_recall = (
        anomaly_correct /
        max(anomaly_total, 1)
    )

    balanced_accuracy = (
        (normal_recall + anomaly_recall)
        / 2.0
    ) * 100.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "normal_recall": normal_recall * 100.0,
        "anomaly_recall": anomaly_recall * 100.0
    }


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    predictions = []
    labels_all = []

    running_loss = 0.0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for (
        images,
        labels,
        _,
        _
    ) in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)

        loss = criterion(
            logits,
            labels
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        total += labels.size(0)

    metrics = calculate_metrics(
        predictions,
        labels_all
    )

    metrics["loss"] = (
        running_loss /
        max(total, 1)
    )

    return metrics


# =========================================================
# ATOMIC CHECKPOINT SAVE
# =========================================================

def save_checkpoint_atomic(
    path,
    checkpoint
):
    """
    Save checkpoint safely.

    The checkpoint is first written to a temporary file.
    Only after successful writing is it renamed to the
    actual checkpoint path.

    This reduces the chance of losing the checkpoint if
    the system shuts down during torch.save().
    """

    temp_path = path + ".tmp"

    torch.save(
        checkpoint,
        temp_path
    )

    os.replace(
        temp_path,
        path
    )


# =========================================================
# RNG STATE
# =========================================================

def get_rng_state():

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state()
    }

    if torch.cuda.is_available():

        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state
):

    if state is None:
        return

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "numpy" in state:
        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        torch.cuda.is_available()
        and
        "cuda" in state
    ):

        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device
):

    print()
    print("=" * 75)
    print("LOADING EXISTING TRAINING CHECKPOINT")
    print("=" * 75)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    version = checkpoint.get(
        "checkpoint_version",
        None
    )

    if version != CHECKPOINT_VERSION:

        print(
            "[Resume] Existing checkpoint is from "
            "an incompatible version."
        )

        print(
            "[Resume] Starting a fresh training run."
        )

        return {
            "resume": False
        }

    # -----------------------------------------------------
    # Restore model
    # -----------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------------------------------
    # Restore optimizer
    # -----------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # -----------------------------------------------------
    # Restore scheduler
    # -----------------------------------------------------

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # -----------------------------------------------------
    # Restore RNG
    # -----------------------------------------------------

    restore_rng_state(
        checkpoint.get(
            "rng_state"
        )
    )

    # -----------------------------------------------------
    # Resume information
    # -----------------------------------------------------

    last_epoch = checkpoint.get(
        "epoch",
        0
    )

    start_epoch = last_epoch + 1

    best_val_accuracy = checkpoint.get(
        "best_val_accuracy",
        0.0
    )

    best_val_balanced_accuracy = checkpoint.get(
        "best_val_balanced_accuracy",
        0.0
    )

    best_epoch = checkpoint.get(
        "best_epoch",
        0
    )

    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0
    )

    # -----------------------------------------------------
    # Restore split
    # -----------------------------------------------------

    train_indices = checkpoint.get(
        "train_indices"
    )

    val_indices = checkpoint.get(
        "val_indices"
    )

    print(
        f"[Resume] Last completed epoch : {last_epoch}"
    )

    print(
        f"[Resume] Next epoch           : {start_epoch}"
    )

    print(
        f"[Resume] Best epoch            : {best_epoch}"
    )

    print(
        f"[Resume] Best Val Balanced Acc : "
        f"{best_val_balanced_accuracy:.2f}%"
    )

    print(
        f"[Resume] No improvement        : "
        f"{epochs_without_improvement}/{PATIENCE}"
    )

    return {
        "resume": True,
        "start_epoch": start_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_balanced_accuracy":
            best_val_balanced_accuracy,
        "best_epoch": best_epoch,
        "epochs_without_improvement":
            epochs_without_improvement,
        "train_indices": train_indices,
        "val_indices": val_indices
    }


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def train_model():

    print("=" * 75)
    print("CBAM CNN V2 - PROPER VisA TRAINING")
    print("=" * 75)

    print(
        "\n[IMPORTANT] Existing model files will NOT "
        "be blindly overwritten."
    )

    # =====================================================
    # CHECK DATASET
    # =====================================================

    if not os.path.exists(VISA_DIR):

        raise FileNotFoundError(
            f"VisA dataset not found:\n{VISA_DIR}"
        )

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\n[Device] {device}"
    )

    # =====================================================
    # BASE DATASET
    # =====================================================

    base_dataset = VisAPCBDataset(
        visa_root=VISA_DIR,
        categories=[
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4"
        ],
        split="train",
        split_csv="2cls_highshot.csv"
    )

    if len(base_dataset) == 0:

        raise RuntimeError(
            "No training images were loaded."
        )

    print(
        f"[Dataset] Official TRAIN samples: "
        f"{len(base_dataset)}"
    )

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    class_counts = [0, 0]

    for sample in base_dataset.samples:

        label = sample["label"]

        class_counts[label] += 1

    print(
        f"[Dataset] Normal  : {class_counts[0]}"
    )

    print(
        f"[Dataset] Anomaly : {class_counts[1]}"
    )

    # =====================================================
    # MODEL
    # =====================================================

    class_names = [
        "Good_Assembly",
        "Anomaly"
    ]

    model = PCBCBAMNet(
        num_classes=2,
        base_classes=class_names
    )

    model.to(device)

    print(
        f"\n[Model] PCBCBAMNet"
    )

    print(
        f"[Model] Embedding: "
        f"{model.embedding_dim}"
    )

    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # =====================================================
    # DEFAULT TRAINING VARIABLES
    # =====================================================

    start_epoch = 1

    best_val_balanced_accuracy = 0.0

    best_val_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    train_indices = None

    val_indices = None

    # =====================================================
    # TRY TO RESUME
    # =====================================================

    if (
        AUTO_RESUME
        and
        os.path.exists(CHECKPOINT_PATH)
    ):

        resume_info = load_training_checkpoint(
            CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            device
        )

        if resume_info.get("resume", False):

            start_epoch = resume_info[
                "start_epoch"
            ]

            best_val_accuracy = resume_info[
                "best_val_accuracy"
            ]

            best_val_balanced_accuracy = (
                resume_info[
                    "best_val_balanced_accuracy"
                ]
            )

            best_epoch = resume_info[
                "best_epoch"
            ]

            epochs_without_improvement = (
                resume_info[
                    "epochs_without_improvement"
                ]
            )

            train_indices = resume_info[
                "train_indices"
            ]

            val_indices = resume_info[
                "val_indices"
            ]

            print(
                "\n[Resume] Training state restored successfully."
            )

    # =====================================================
    # TRAIN / VALIDATION SPLIT
    # =====================================================

    if (
        train_indices is None
        or
        val_indices is None
    ):

        indices = list(
            range(len(base_dataset))
        )

        rng = random.Random(SEED)

        rng.shuffle(indices)

        val_size = int(
            len(indices)
            * VALIDATION_RATIO
        )

        val_indices = indices[:val_size]

        train_indices = indices[val_size:]

    print(
        f"\n[Split] Training   : "
        f"{len(train_indices)}"
    )

    print(
        f"[Split] Validation : "
        f"{len(val_indices)}"
    )

    print(
        "[Split] Official TEST remains completely untouched."
    )

    # =====================================================
    # DATASETS
    # =====================================================

    train_augmentation = TrainAugmentation()

    train_dataset = AugmentedSubset(
        base_dataset,
        train_indices,
        augmentation=train_augmentation
    )

    validation_dataset = AugmentedSubset(
        base_dataset,
        val_indices,
        augmentation=None
    )

    # =====================================================
    # CLASS WEIGHTS FOR SAMPLER
    # =====================================================

    train_class_counts = [0, 0]

    for index in train_indices:

        label = base_dataset.samples[index]["label"]

        train_class_counts[label] += 1

    print(
        "\n[Train Distribution]"
    )

    print(
        f"Normal  : {train_class_counts[0]}"
    )

    print(
        f"Anomaly : {train_class_counts[1]}"
    )

    sample_weights = []

    for index in train_indices:

        label = base_dataset.samples[index]["label"]

        weight = (
            1.0 /
            max(train_class_counts[label], 1)
        )

        sample_weights.append(
            weight
        )

    # =====================================================
    # WEIGHTED RANDOM SAMPLER
    # =====================================================

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(train_indices),
        replacement=True
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,

        # IMPORTANT:
        # Prevent final batch of size 1.
        #
        # 2385 samples / 16 =
        # 149 full batches + 1 sample.
        #
        # BatchNorm1d cannot train on that
        # single sample.
        drop_last=True,

        num_workers=NUM_WORKERS,
        collate_fn=visa_collate_fn
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,

        # Keep ALL validation samples.
        drop_last=False,

        num_workers=NUM_WORKERS,
        collate_fn=visa_collate_fn
    )

    print(
        f"\n[Loader] Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"[Loader] Validation batches: "
        f"{len(validation_loader)}"
    )

    # =====================================================
    # CHECK IF TRAINING IS ALREADY COMPLETE
    # =====================================================

    if start_epoch > EPOCHS:

        print()
        print("=" * 75)
        print("TRAINING WAS ALREADY COMPLETED")
        print("=" * 75)

        print(
            f"Last checkpoint epoch: "
            f"{start_epoch - 1}"
        )

        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best Validation Balanced Accuracy: "
            f"{best_val_balanced_accuracy:.2f}%"
        )

        print(
            f"\nBest model:"
        )

        print(
            BEST_MODEL_PATH
        )

        return BEST_MODEL_PATH

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        train_correct = 0

        train_total = 0

        print(
            f"\n{'=' * 75}"
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"{'=' * 75}"
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        for batch_idx, (
            images,
            labels,
            _,
            _
        ) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(images)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

            if (
                batch_idx % 20 == 0
                or
                batch_idx == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_idx:03d}/"
                    f"{len(train_loader):03d} | "
                    f"Loss "
                    f"{loss.item():.4f}",
                    flush=True
                )

        # -------------------------------------------------
        # TRAIN METRICS
        # -------------------------------------------------

        train_loss = (
            running_loss /
            max(train_total, 1)
        )

        train_accuracy = (
            train_correct /
            max(train_total, 1)
        ) * 100.0

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val_metrics = evaluate(
            model,
            validation_loader,
            device
        )

        # -------------------------------------------------
        # SCHEDULER
        # -------------------------------------------------

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # -------------------------------------------------
        # PRINT METRICS
        # -------------------------------------------------

        print(
            "\n"
            f"Train Loss       : {train_loss:.4f}\n"
            f"Train Accuracy   : {train_accuracy:.2f}%\n"
            f"Val Loss         : {val_metrics['loss']:.4f}\n"
            f"Val Accuracy     : {val_metrics['accuracy']:.2f}%\n"
            f"Val Balanced Acc : {val_metrics['balanced_accuracy']:.2f}%\n"
            f"Val Normal Rec.  : {val_metrics['normal_recall']:.2f}%\n"
            f"Val Anomaly Rec. : {val_metrics['anomaly_recall']:.2f}%\n"
            f"Learning Rate    : {current_lr:.8f}"
        )

        # =================================================
        # BEST MODEL
        # =================================================

        improved = (
            val_metrics["balanced_accuracy"]
            >
            best_val_balanced_accuracy
        )

        if improved:

            best_val_balanced_accuracy = (
                val_metrics["balanced_accuracy"]
            )

            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            # -------------------------------------------------
            # Save BEST model
            # -------------------------------------------------

            best_model_data = {

                "model_state_dict":
                    model.state_dict(),

                "classes":
                    class_names,

                "embedding_dim":
                    model.embedding_dim,

                "epoch":
                    epoch,

                "accuracy":
                    val_metrics["accuracy"],

                "validation_accuracy":
                    val_metrics["accuracy"],

                "validation_balanced_accuracy":
                    val_metrics["balanced_accuracy"],

                "validation_normal_recall":
                    val_metrics["normal_recall"],

                "validation_anomaly_recall":
                    val_metrics["anomaly_recall"],

                "dataset":
                    "VisA PCB1-PCB4 2cls_highshot",

                "training_method":
                    "train-validation split",

                "seed":
                    SEED,

                "checkpoint_version":
                    CHECKPOINT_VERSION
            }

            # Atomic best-model save
            temp_best = BEST_MODEL_PATH + ".tmp"

            torch.save(
                best_model_data,
                temp_best
            )

            os.replace(
                temp_best,
                BEST_MODEL_PATH
            )

            print(
                "\n[MODEL] ★ NEW BEST MODEL SAVED"
            )

            print(
                f"[MODEL] Validation Balanced Accuracy: "
                f"{best_val_balanced_accuracy:.2f}%"
            )

            print(
                f"[MODEL] {BEST_MODEL_PATH}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\n[Model] No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # =================================================
        # SAVE FULL TRAINING CHECKPOINT
        # =================================================

        checkpoint = {

            # Version
            "checkpoint_version":
                CHECKPOINT_VERSION,

            # Current epoch
            "epoch":
                epoch,

            # Model
            "model_state_dict":
                model.state_dict(),

            # Optimizer
            "optimizer_state_dict":
                optimizer.state_dict(),

            # Scheduler
            "scheduler_state_dict":
                scheduler.state_dict(),

            # Best performance
            "best_val_accuracy":
                best_val_accuracy,

            "best_val_balanced_accuracy":
                best_val_balanced_accuracy,

            "best_epoch":
                best_epoch,

            # Early stopping
            "epochs_without_improvement":
                epochs_without_improvement,

            # Current epoch metrics
            "current_train_loss":
                train_loss,

            "current_train_accuracy":
                train_accuracy,

            "current_val_loss":
                val_metrics["loss"],

            "current_val_accuracy":
                val_metrics["accuracy"],

            "current_val_balanced_accuracy":
                val_metrics["balanced_accuracy"],

            "current_val_normal_recall":
                val_metrics["normal_recall"],

            "current_val_anomaly_recall":
                val_metrics["anomaly_recall"],

            # Model information
            "classes":
                class_names,

            "embedding_dim":
                model.embedding_dim,

            # Dataset information
            "dataset":
                "VisA PCB1-PCB4 2cls_highshot",

            "training_method":
                "train-validation split",

            # Configuration
            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "total_epochs":
                EPOCHS,

            "validation_ratio":
                VALIDATION_RATIO,

            "patience":
                PATIENCE,

            "seed":
                SEED,

            # VERY IMPORTANT:
            # Save exact train/validation split.
            "train_indices":
                train_indices,

            "val_indices":
                val_indices,

            # RNG states
            "rng_state":
                get_rng_state()
        }

        save_checkpoint_atomic(
            CHECKPOINT_PATH,
            checkpoint
        )

        print(
            f"\n[Checkpoint] Epoch {epoch} saved."
        )

        print(
            f"[Checkpoint] {CHECKPOINT_PATH}"
        )

        # =================================================
        # EARLY STOPPING
        # =================================================

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\n[Early Stopping] "
                "Validation performance stopped improving."
            )

            print(
                f"[Early Stopping] "
                f"Best epoch: {best_epoch}"
            )

            print(
                f"[Early Stopping] "
                f"Best validation balanced accuracy: "
                f"{best_val_balanced_accuracy:.2f}%"
            )

            break

    # =====================================================
    # RESTORE BEST MODEL FROM DISK
    # =====================================================

    if os.path.exists(
        BEST_MODEL_PATH
    ):

        print(
            "\n[Model] Restoring best model "
            "from disk..."
        )

        best_checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ]
        )

    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "CBAM CNN V2 TRAINING COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Best Epoch                 : "
        f"{best_epoch}"
    )

    print(
        f"Best Validation Accuracy   : "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Best Validation Balanced   : "
        f"{best_val_balanced_accuracy:.2f}%"
    )

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nLatest checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Official VisA TEST was NOT used during training."
    )

    print(
        "Run the official evaluation script separately."
    )

    return BEST_MODEL_PATH


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_model()

