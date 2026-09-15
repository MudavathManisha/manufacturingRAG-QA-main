"""
Main Execution and Launch Script for ManufacturingRAG-QA.
Initializes data samples, trains/loads CBAM weights, seeds ProtoNet memory bank,
and launches the interactive web application.
"""

import os
import sys
import subprocess

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.synthetic_generator import PCBSyntheticGenerator
from server.app import app


def main():
    print("\n" + "=" * 65)
    print("      ManufacturingRAG-QA: Explainable PCB Quality System      ")
    print("=" * 65)

    # 1. Initialize dataset directories
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "dataset")
    samples_dir = os.path.join(PROJECT_ROOT, "data", "samples")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    bank_dir = os.path.join(PROJECT_ROOT, "data", "prototype_bank")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(bank_dir, exist_ok=True)

    # 2. Check and generate sample data
    gen = PCBSyntheticGenerator(224, 224)
    if len(os.listdir(dataset_dir)) == 0:
        print("[Setup] Generating synthetic PCB training dataset...")
        gen.generate_dataset_batch(dataset_dir, samples_per_class=20)
        print("[Setup] Dataset generated successfully.")

    # 3. Start web server
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[Ready] Launching Dashboard on http://localhost:{port}")
    print("[Ready] Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
