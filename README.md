# ManufacturingRAG-QA: Explainable PCB Quality Auditing & Standards-Grounded System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Standards](https://img.shields.io/badge/IPC--A--610G-Class%201%2F2%2F3-emerald.svg)](http://www.ipc.org/)

**ManufacturingRAG-QA** is an explainable quality auditing system designed for surface-mount technology (SMT) and PCB assembly lines. It detects assembly defects such as missing components, misalignment, solder bridges, tombstoning, solder balls, and insufficient fillets using optical inspection data.

The system combines:
1. **Custom CNN with CBAM (Convolutional Block Attention Module)**: Learns both *what* features and *where* subtle defect regions occur in solder fillets and FR4 substrates.
2. **ProtoNet (Prototypical Networks)**: Enables quality engineers to register and classify emerging novel defect types with 1–5 image examples without full model retraining.
3. **Grad-CAM Saliency Engine**: Computes visual heatmaps and automated bounding boxes around defect regions with an interactive alpha blend slider.
4. **IPC / ISO Standards RAG Pipeline**: Grounded in **IPC-A-610G** (Classes 1, 2, and 3), **J-STD-001H**, and **IPC-7711/7721**, computing an **Audit Severity Index (ASI)** and producing exportable formal audit reports.

---

## System Architecture

```
                                  [ PCB Optical Inspection Image ]
                                                │
                                    (CLAHE LAB Preprocessing)
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                        [ Custom CNN + CBAM ]       [ ProtoNet Few-Shot ]
                        (Channel & Spatial Attn)    (Prototype Memory Bank)
                                  │                           │
                                  ├─────────────┬─────────────┘
                                  ▼             ▼
                           [ Grad-CAM Map ] [ Defect Class ]
                                  │             │
                                  └──────┬──────┘
                                         ▼
                            [ Standards RAG Engine ]
                         (IPC-A-610G / J-STD-001 / ISO)
                                         │
                                         ▼
                        [ Audit Severity Index (ASI) ]
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
        [ Interactive Web Studio ]              [ PDF / JSON Audit Certificate ]
```

---

## Directory Structure

```
ManufacturingRAG-QA/
│
├── core/
│   ├── __init__.py
│   ├── preprocessor.py        # CLAHE, LAB enhancement, ROI normalization
│   ├── cbam_cnn.py            # Custom CNN with CBAM attention backbone
│   ├── protonet.py            # ProtoNet metric learning & prototype bank
│   ├── gradcam.py             # Grad-CAM heatmap & bounding box proposals
│   ├── standards_kb.py        # IPC-A-610G, J-STD-001, IPC-7721 knowledge base
│   ├── rag_engine.py          # Hybrid BM25 retriever & grounded QA engine
│   └── audit_reporter.py      # Severity scoring (ASI) & certificate export
│
├── data/
│   ├── synthetic_generator.py # Procedural PCB defect board synthesizer
│   ├── dataset_loader.py      # PyTorch Dataset & episodic few-shot sampler
│   └── prototype_bank/        # Persistent prototype memory bank
│
├── server/
│   └── app.py                 # REST backend & API endpoints
│
├── web/
│   ├── index.html             # Quality Engineering Dashboard UI
│   ├── css/styles.css         # Modern dark theme styles
│   └── js/app.js              # Viewport canvas controller & RAG assistant
│
├── scripts/
│   ├── train_cbam_cnn.py      # Backbone training script
│   ├── evaluate_protonet.py   # Few-shot N-way K-shot evaluation benchmark
│   └── run_system.py          # Unified system launcher
│
├── tests/                     # Comprehensive unit test suite
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest discover tests/
```

### 3. Launch System
```bash
python scripts/run_system.py
```
Open your browser at `http://localhost:5000` to access the interactive Quality Engineering Dashboard.

---

## Key Features

- **Real-Time Visual Saliency**: Interactive Grad-CAM heatmap opacity slider, colormap blending, and automatic defect bounding boxes.
- **Zero-Retraining Few-Shot Learning**: Drag and drop 1-5 sample photos of an emerging defect type to instantly register it into the active prototype bank.
- **Standards Grounding & RAG Assistant**: Ask questions against IPC-A-610G, J-STD-001, and IPC-7711/7721 with live clause citations.
- **Formal Audit Certificates**: One-click printable PDF quality certificates and structured JSON audit logs with IPC disposition sign-offs.
