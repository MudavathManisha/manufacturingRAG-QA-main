"""
Rigorous Continual Few-Shot Adaptation Experiment
--------------------------------------------------

Dataset : VisA PCB1-PCB4
Tasks   : PCB1 -> PCB2 -> PCB3 -> PCB4

Learning strategy:
    - Frozen CBAM CNN
    - 512-D feature embeddings
    - 5-shot prototype adaptation
    - Category-specific prototype memory
    - No CNN retraining

Evaluation:
    - ALL test samples
    - 20 independent trials
    - Different 5-shot support samples per trial
    - Reports mean +/- std
    - Reports 95% confidence interval
    - Measures old-task retention

Important:
This is task/category-incremental few-shot adaptation.
The PCB category is known at inference time.
It is NOT claimed as class-incremental learning with unknown task identity.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
# import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from core.cbam_cnn import PCBCBAMNet
from data.visa_dataset_loader import VisAPCBDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"D:\ManufacturingRAG-QA"

VISA_ROOT = os.path.join(
    PROJECT_ROOT,
    "Visa",
    "VisA_20220922"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cbam_cnn_visa_best.pth"
)

CSV_NAME = "2cls_fewshot.csv"

TASKS = [
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4"
]

# Few-shot setting
K_SHOT = 5

# Number of independent support-set trials
TRIALS = 20

# Reproducibility
BASE_SEED = 42

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("\n[1] Loading frozen CBAM CNN...")

    model = PCBCBAMNet(
        num_classes=2,
        embedding_dim=512,
        base_classes=[
            "Good_Assembly",
            "Anomaly"
        ]
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # FREEZE ENTIRE CNN
    # --------------------------------------------------------

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("[CBAM] Loaded checkpoint:")
    print(MODEL_PATH)

    print("[CBAM] Parameters frozen.")
    print("[CBAM] Embedding dimension: 512")

    return model


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings(model, dataset):
    """
    Extract embeddings once for the complete dataset.

    Returns:
        embeddings : Tensor [N, 512]
        labels     : Tensor [N]
    """

    embeddings = []
    labels = []

    with torch.no_grad():

        for idx in range(len(dataset)):

            image, label, label_name, metadata = dataset[idx]

            image = image.unsqueeze(0).to(DEVICE)

            feature = model.extract_features(image)

            feature = F.normalize(
                feature,
                p=2,
                dim=1
            )

            embeddings.append(
                feature.squeeze(0).cpu()
            )

            labels.append(label)

    embeddings = torch.stack(embeddings)

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return embeddings, labels


# ============================================================
# LOAD TASK DATA
# ============================================================

def prepare_task_embeddings(model, category):

    print("\n" + "-" * 60)
    print(f"PREPARING {category.upper()}")
    print("-" * 60)

    # ========================================================
    # TRAIN DATA
    # ========================================================

    train_dataset = VisAPCBDataset(
        visa_root=VISA_ROOT,
        categories=[category],
        split="train",
        split_csv="2cls_fewshot.csv",
        transform=True
    )

    print(
        f"[{category}] Training images: "
        f"{len(train_dataset)}"
    )

    train_embeddings, train_labels = extract_embeddings(
        model,
        train_dataset
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    test_dataset = VisAPCBDataset(
        visa_root=VISA_ROOT,
        categories=[category],
        split="test",
        split_csv="2cls_fewshot.csv",
        transform=True
    )

    print(
        f"[{category}] Test images: "
        f"{len(test_dataset)}"
    )

    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset
    )

    # ========================================================
    # CLASS COUNTS
    # ========================================================

    train_normal = int(
        (train_labels == 0).sum().item()
    )

    train_anomaly = int(
        (train_labels == 1).sum().item()
    )

    test_normal = int(
        (test_labels == 0).sum().item()
    )

    test_anomaly = int(
        (test_labels == 1).sum().item()
    )

    print(
        f"[{category}] Train Normal  : "
        f"{train_normal}"
    )

    print(
        f"[{category}] Train Anomaly : "
        f"{train_anomaly}"
    )

    print(
        f"[{category}] Test Normal   : "
        f"{test_normal}"
    )

    print(
        f"[{category}] Test Anomaly  : "
        f"{test_anomaly}"
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if train_normal < K_SHOT:
        raise ValueError(
            f"{category}: only {train_normal} normal "
            f"training samples, but K_SHOT={K_SHOT}"
        )

    if train_anomaly < K_SHOT:
        raise ValueError(
            f"{category}: only {train_anomaly} anomaly "
            f"training samples, but K_SHOT={K_SHOT}"
        )

    if test_normal == 0 or test_anomaly == 0:
        raise ValueError(
            f"{category}: test set does not contain "
            f"both normal and anomaly samples."
        )

    return {
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "test_embeddings": test_embeddings,
        "test_labels": test_labels
    }

# ============================================================
# CREATE 5-SHOT PROTOTYPES
# ============================================================

def create_prototypes(
    train_embeddings,
    train_labels,
    k_shot,
    rng
):
    """
    Randomly select K examples from each class
    and calculate class prototypes.
    """

    prototypes = []

    for class_id in [0, 1]:

        indices = torch.where(
            train_labels == class_id
        )[0].tolist()

        if len(indices) < k_shot:

            raise ValueError(
                f"Not enough samples for class {class_id}"
            )

        selected = rng.sample(
            indices,
            k_shot
        )

        support = train_embeddings[
            selected
        ]

        prototype = support.mean(
            dim=0,
            keepdim=True
        )

        prototype = F.normalize(
            prototype,
            p=2,
            dim=1
        )

        prototypes.append(
            prototype.squeeze(0)
        )

    return torch.stack(prototypes)


# ============================================================
# PROTOTYPE CLASSIFICATION
# ============================================================

def classify_with_prototypes(
    test_embeddings,
    test_labels,
    prototypes
):

    # --------------------------------------------------------
    # Calculate squared Euclidean distances
    # --------------------------------------------------------

    distances = torch.cdist(
        test_embeddings,
        prototypes,
        p=2
    ) ** 2

    predictions = torch.argmin(
        distances,
        dim=1
    )

    accuracy = (
        predictions == test_labels
    ).float().mean().item()

    return accuracy


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    print("\n")
    print("=" * 70)
    print("       RIGOROUS CONTINUAL FEW-SHOT ADAPTATION")
    print("=" * 70)

    print("\nSequential tasks:")
    print("    PCB1 -> PCB2 -> PCB3 -> PCB4")

    print("\nLearning:")
    print(f"    {K_SHOT}-shot prototype adaptation")

    print("\nBackbone:")
    print("    Frozen CBAM CNN")

    print("\nEmbedding:")
    print("    512-D")

    print("\nRetraining:")
    print("    NONE")

    print("\nEvaluation:")
    print("    ALL test samples")

    print("\nTrials:")
    print(f"    {TRIALS}")

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PRECOMPUTE EMBEDDINGS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("       PRECOMPUTING EMBEDDINGS")
    print("=" * 70)

    task_data = {}

    for category in TASKS:

        task_data[category] = prepare_task_embeddings(
            model,
            category
        )

    print("\nAll embeddings extracted successfully.")

    # ========================================================
    # STORAGE
    # ========================================================

    # Results:
    #
    # new_task_results[trial][stage]
    #
    # old_task_results[trial][stage][category]

    new_task_results = []

    old_task_results = []

    # ========================================================
    # TRIAL LOOP
    # ========================================================

    for trial in range(TRIALS):

        print("\n")
        print("=" * 70)
        print(
            f"                     TRIAL {trial + 1}/{TRIALS}"
        )
        print("=" * 70)

        rng = random.Random(
            BASE_SEED + trial
        )

        # Prototype memory for this trial
        prototype_memory = {}

        trial_new_results = []

        trial_old_results = []

        # ----------------------------------------------------
        # SEQUENTIAL TASK ARRIVAL
        # ----------------------------------------------------

        for stage, category in enumerate(TASKS, start=1):

            print("\n" + "-" * 60)
            print(
                f"STAGE {stage}: "
                f"{category.upper()} ARRIVES"
            )
            print("-" * 60)

            data = task_data[category]

            # ------------------------------------------------
            # CREATE NEW 5-SHOT PROTOTYPES
            # ------------------------------------------------

            prototypes = create_prototypes(
                data["train_embeddings"],
                data["train_labels"],
                K_SHOT,
                rng
            )

            # Add to prototype memory
            prototype_memory[category] = prototypes

            print(
                f"[{category}] Added "
                f"{K_SHOT}-shot prototypes."
            )

            print(
                f"Prototype memory: "
                f"{len(prototype_memory)} PCB categories"
            )

            # ------------------------------------------------
            # NEW TASK PERFORMANCE
            # ------------------------------------------------

            new_accuracy = classify_with_prototypes(
                data["test_embeddings"],
                data["test_labels"],
                prototype_memory[category]
            )

            trial_new_results.append(
                new_accuracy
            )

            print(
                f"NEW TASK PERFORMANCE"
            )

            print(
                f"  {category.upper()} accuracy: "
                f"{new_accuracy * 100:.2f}%"
            )

            # ------------------------------------------------
            # EVALUATE ALL PREVIOUS TASKS
            # ------------------------------------------------

            stage_old_results = {}

            print(
                "\nCURRENT KNOWLEDGE:"
            )

            for old_category in TASKS[:stage]:

                old_data = task_data[old_category]

                old_accuracy = classify_with_prototypes(
                    old_data["test_embeddings"],
                    old_data["test_labels"],
                    prototype_memory[old_category]
                )

                stage_old_results[
                    old_category
                ] = old_accuracy

                print(
                    f"  {old_category.upper()}: "
                    f"{old_accuracy * 100:.2f}%"
                )

            trial_old_results.append(
                stage_old_results
            )

        new_task_results.append(
            trial_new_results
        )

        old_task_results.append(
            trial_old_results
        )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    new_task_results = np.array(
        new_task_results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("              FINAL CONTINUAL LEARNING RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # NEW TASK PERFORMANCE
    # --------------------------------------------------------

    print("\nNEW TASK ACCURACY")
    print("-" * 70)

    for stage, category in enumerate(TASKS):

        values = (
            new_task_results[:, stage] * 100
        )

        mean = np.mean(values)
        std = np.std(
            values,
            ddof=1
        )

        ci = (
            1.96 * std /
            np.sqrt(TRIALS)
        )

        print(
            f"{category.upper():5s} : "
            f"{mean:.2f}% ± {std:.2f}% "
            f"(95% CI ± {ci:.2f}%)"
        )

    # --------------------------------------------------------
    # OLD TASK RETENTION
    # --------------------------------------------------------

    print("\n")
    print("OLD TASK RETENTION")
    print("-" * 70)

    for stage in range(len(TASKS)):

        print(
            f"\nAfter Stage {stage + 1} "
            f"({TASKS[stage].upper()} arrival):"
        )

        for category in TASKS[:stage + 1]:

            values = np.array([
                old_task_results[
                    trial
                ][stage][category] * 100

                for trial in range(TRIALS)
            ])

            mean = np.mean(values)

            std = np.std(
                values,
                ddof=1
            )

            ci = (
                1.96 * std /
                np.sqrt(TRIALS)
            )

            print(
                f"  {category.upper():5s}: "
                f"{mean:.2f}% ± {std:.2f}% "
                f"(95% CI ± {ci:.2f}%)"
            )

    # ========================================================
    # RETENTION CHANGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                 RETENTION ANALYSIS")
    print("=" * 70)

    for task_index, category in enumerate(TASKS):

        initial_values = np.array([
            old_task_results[
                trial
            ][task_index][category]

            for trial in range(TRIALS)
        ])

        final_values = np.array([
            old_task_results[
                trial
            ][len(TASKS) - 1][category]

            for trial in range(TRIALS)
        ])

        initial_mean = (
            np.mean(initial_values) * 100
        )

        final_mean = (
            np.mean(final_values) * 100
        )

        change = (
            final_mean -
            initial_mean
        )

        print(
            f"\n{category.upper()}:"
        )

        print(
            f"  Initial accuracy : "
            f"{initial_mean:.2f}%"
        )

        print(
            f"  Final accuracy   : "
            f"{final_mean:.2f}%"
        )

        print(
            f"  Retention change : "
            f"{change:+.2f} percentage points"
        )

    # ========================================================
    # OVERALL NEW TASK PERFORMANCE
    # ========================================================

    all_new_values = (
        new_task_results.flatten() * 100
    )

    overall_mean = np.mean(
        all_new_values
    )

    overall_std = np.std(
        all_new_values,
        ddof=1
    )

    print("\n")
    print("=" * 70)
    print("             OVERALL ADAPTATION PERFORMANCE")
    print("=" * 70)

    print(
        f"\nMean accuracy across all "
        f"task arrivals/trials:"
    )

    print(
        f"  {overall_mean:.2f}% ± "
        f"{overall_std:.2f}%"
    )

    # ========================================================
    # FINAL MEMORY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                 FINAL PROTOTYPE MEMORY")
    print("=" * 70)

    for category in TASKS:

        print(
            f"{category.upper()} "
            f"-> Normal + Anomaly prototypes"
        )

    # ========================================================
    # FINAL STATEMENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("          CONTINUAL LEARNING EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nCBAM CNN remained frozen throughout.")

    print(
        "No CNN retraining was performed."
    )

    print(
        f"Each arriving PCB category used "
        f"{K_SHOT} labelled examples per class."
    )

    print(
        f"Evaluation used ALL available test images "
        f"over {TRIALS} independent trials."
    )

    print(
        "\nInterpretation:"
    )

    print(
        "This experiment demonstrates sequential "
        "task/category-incremental few-shot adaptation "
        "using prototype memory."
    )

    print(
        "Because previously learned prototypes remain "
        "frozen and independent, retention should remain "
        "stable rather than exhibiting catastrophic forgetting."
    )

    print("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    set_seed(BASE_SEED)

    run_experiment()