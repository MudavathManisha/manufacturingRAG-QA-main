"""
ManufacturingRAG-QA Backend Server
-----------------------------------

Complete backend for:

    PCB image inspection
        ↓
    CBAM binary classification
        ↓
    ProtoNet defect classification
        ↓
    Grad-CAM visual localization
        ↓
    RAG / IPC-A-610 evidence
        ↓
    Persistent Quality Audit Report
        ↓
    Audit certificate
        ↓
    Analytics
        ↓
    Inspection Report Upload
        ↓
    Report-aware Standards Assistant
"""

import os
import sys
import json
import re
import traceback
from io import BytesIO

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# HUGGING FACE CACHE CONFIGURATION
# ============================================================

HF_CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "hf_cache"
)

os.makedirs(
    HF_CACHE_DIR,
    exist_ok=True
)

os.environ.setdefault(
    "HF_HOME",
    HF_CACHE_DIR
)

os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    os.path.join(
        HF_CACHE_DIR,
        "hub"
    )
)

os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    os.path.join(
        HF_CACHE_DIR,
        "transformers"
    )
)

os.environ.setdefault(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

os.environ.setdefault(
    "HF_HUB_OFFLINE",
    "1"
)

os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    "1"
)

print(
    "[Server] Hugging Face cache:"
)

print(
    f"         {os.environ['HF_HOME']}"
)

print(
    "[Server] RAG embedding model:"
)

print(
    f"         {os.environ['RAG_EMBEDDING_MODEL']}"
)


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    send_file
)

from flask_cors import CORS

from PIL import Image

import numpy as np
import torch
import cv2


# ============================================================
# OPTIONAL PDF READER
#
# Supports either:
#     pypdf
# or:
#     PyPDF2
# ============================================================

PdfReader = None

try:

    from pypdf import PdfReader

    print(
        "[Server] PDF reader: pypdf"
    )

except Exception:

    try:

        from PyPDF2 import PdfReader

        print(
            "[Server] PDF reader: PyPDF2"
        )

    except Exception:

        PdfReader = None

        print(
            "[Server] WARNING: No PDF reader installed."
        )

        print(
            "[Server] Install with: pip install pypdf"
        )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from core.preprocessor import PCBPreprocessor
from core.cbam_cnn import PCBCBAMNet
from core.protonet import ProtoNetEngine
from core.gradcam import GradCAM

from core.standards_kb import (
    get_all_standards,
    get_standard_by_defect_type
)

from core.rag_engine import StandardsRAGEngine
from core.audit_reporter import AuditReporter

from data.synthetic_generator import PCBSyntheticGenerator


# ============================================================
# FLASK APPLICATION
# ============================================================

ACTIVE_REPORT_STATE = {"report": None}

app = Flask(
    __name__,
    static_folder="../web",
    static_url_path=""
)

CORS(app)

# 15 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "samples"
)

BANK_PATH = os.path.join(
    BASE_DIR,
    "data",
    "prototype_bank",
    "prototypes.json"
)

AUDIT_LOG_DIR = os.path.join(
    BASE_DIR,
    "data",
    "audit_logs"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    os.path.dirname(BANK_PATH),
    exist_ok=True
)

os.makedirs(
    AUDIT_LOG_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"[Server] Using device: {device}"
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CBAM_CLASSES = [
    "normal",
    "anomaly"
]


PROTO_CLASSES = [
    "Good_Assembly",
    "Missing_Component",
    "Component_Misaligned",
    "Solder_Bridge",
    "Tombstoning",
    "Solder_Ball",
    "Insufficient_Solder"
]

BASE_CLASSES = PROTO_CLASSES


# ============================================================
# PREPROCESSOR
# ============================================================

print(
    "[Server] Initializing preprocessing..."
)

preprocessor = PCBPreprocessor(
    target_size=(224, 224)
)


# ============================================================
# RAG
# ============================================================

print(
    "[Server] RAG configured for lazy initialization."
)

print(
    "[Server] RAG cache directory:"
)

print(
    f"         {os.environ.get('HF_HOME')}"
)

print(
    "[Server] RAG embedding model:"
)

print(
    f"         {os.environ.get('RAG_EMBEDDING_MODEL')}"
)

rag_engine = None

def get_or_create_rag_engine():
    """Create and cache the Standards RAG engine."""

    global rag_engine

    if rag_engine is not None:
        return rag_engine

    print("[RAG] Initializing Standards RAG engine...")

    try:
        # ---------------------------------------------------------
        # Project paths
        # ---------------------------------------------------------

        _base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        _standards_dir = os.path.join(
            _base_dir,
            "data",
            "standards"
        )

        _rag_cache_dir = os.path.join(
            _base_dir,
            "hf_cache"
        )

        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------

        _embedding_model = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("[RAG] Standards directory:")
        print("      ", _standards_dir)

        print("[RAG] RAG cache directory:")
        print("      ", _rag_cache_dir)

        print("[RAG] Embedding model:")
        print("      ", _embedding_model)

        # ---------------------------------------------------------
        # Detect actual StandardsRAGEngine constructor
        # ---------------------------------------------------------

        import inspect

        _params = inspect.signature(
            StandardsRAGEngine.__init__
        ).parameters

        _kwargs = {}

        # Standards/PDF directory
        if "standards_dir" in _params:
            _kwargs["standards_dir"] = _standards_dir

        elif "pdf_dir" in _params:
            _kwargs["pdf_dir"] = _standards_dir

        elif "pdf_directory" in _params:
            _kwargs["pdf_directory"] = _standards_dir

        elif "data_dir" in _params:
            _kwargs["data_dir"] = _standards_dir

        # Cache directory
        if "cache_dir" in _params:
            _kwargs["cache_dir"] = _rag_cache_dir

        elif "index_cache_dir" in _params:
            _kwargs["index_cache_dir"] = _rag_cache_dir

        # Embedding model
        if "embedding_model_name" in _params:
            _kwargs["embedding_model_name"] = _embedding_model

        elif "model_name" in _params:
            _kwargs["model_name"] = _embedding_model

        elif "embedding_model" in _params:
            _kwargs["embedding_model"] = _embedding_model

        print("[RAG] Constructor arguments:")
        print("      ", _kwargs)

        # ---------------------------------------------------------
        # Initialize REAL Standards RAG
        # ---------------------------------------------------------

        rag_engine = StandardsRAGEngine(**_kwargs)

        print("[RAG] Standards RAG engine initialized successfully.")

        # ---------------------------------------------------------
        # Report index information
        # ---------------------------------------------------------

        if hasattr(rag_engine, "chunks"):
            print(
                "[RAG] Indexed chunks:",
                len(rag_engine.chunks)
            )

        return rag_engine

    except Exception as exc:
        print("[RAG] Initialization failed:", exc)
        traceback.print_exc()
        raise
def initialize_rag():

    global rag_engine
    global rag_error

    if rag_engine is not None:
        return rag_engine

    print(
        "[RAG] Lazy initialization requested."
    )

    print(
        "[RAG] Loading embedding model and building "
        "standards knowledge base..."
    )

    try:

        rag_engine = StandardsRAGEngine()

        rag_error = None

        print(
            "[RAG] Lazy initialization successful."
        )

        return rag_engine

    except Exception as e:

        rag_error = str(e)

        rag_engine = None

        print(
            "[RAG] WARNING: Lazy initialization failed."
        )

        print(
            f"[RAG] Error: {e}"
        )

        print(
            "[RAG] CBAM + ProtoNet will continue without RAG."
        )

        return None


# ============================================================
# AUDIT REPORTER
# ============================================================

print(
    "[Server] Initializing audit reporter..."
)

audit_reporter = AuditReporter(
    log_dir=AUDIT_LOG_DIR
)


# ============================================================
# SYNTHETIC DEMO GENERATOR
# ============================================================

print(
    "[Server] Initializing synthetic demo generator..."
)

synthetic_gen = PCBSyntheticGenerator(
    224,
    224
)


# ============================================================
# CBAM MODEL
# ============================================================

print(
    "[Server] Initializing PCBCBAMNet..."
)

model = PCBCBAMNet(
    num_classes=len(CBAM_CLASSES),
    base_classes=CBAM_CLASSES
)


# ============================================================
# CHECKPOINT SELECTION
# ============================================================

visa_weights_path = os.path.join(
    MODELS_DIR,
    "cbam_cnn_visa_best.pth"
)

fallback_weights_path = os.path.join(
    MODELS_DIR,
    "cbam_cnn_best.pth"
)

if os.path.exists(visa_weights_path):

    weights_path = visa_weights_path

elif os.path.exists(fallback_weights_path):

    weights_path = fallback_weights_path

else:

    weights_path = None


# ============================================================
# LOAD CHECKPOINT
# ============================================================

model_loaded = False

if weights_path is not None:

    try:

        print(
            "[Server] Loading checkpoint:"
        )

        print(
            f"         {weights_path}"
        )

        checkpoint = torch.load(
            weights_path,
            map_location=device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict,
            strict=True
        )

        model_loaded = True

        print(
            "[Server] Model weights loaded successfully."
        )

        print(
            f"[Server] CBAM classes: {CBAM_CLASSES}"
        )

    except Exception as e:

        print(
            "[Server] ERROR loading CBAM checkpoint:"
        )

        print(
            f"         {e}"
        )

        print(
            "[Server] WARNING: Using initialized model weights."
        )

else:

    print(
        "[Server] WARNING: No trained CBAM checkpoint found."
    )

    print(
        f"[Server] Expected: {visa_weights_path}"
    )

    print(
        f"[Server] Or: {fallback_weights_path}"
    )


# ============================================================
# MOVE MODEL
# ============================================================

model.to(device)
model.eval()


# ============================================================
# GRAD-CAM
# ============================================================

print(
    "[Server] Initializing Grad-CAM..."
)

gradcam = GradCAM(
    model
)


# ============================================================
# PROTONET
# ============================================================

print(
    "[Server] Initializing ProtoNet..."
)

protonet = ProtoNetEngine(
    model,
    device=str(device),
    bank_path=BANK_PATH
)


# ============================================================
# DEMONSTRATION DATA
# ============================================================

def ensure_sample_data():

    manifest_path = os.path.join(
        DATA_DIR,
        "samples_manifest.json"
    )

    if not os.path.exists(manifest_path):

        print(
            "[Server] Generating demonstration PCB samples..."
        )

        samples = {}

        for defect_type in PROTO_CLASSES:

            img_rgb, meta = (
                synthetic_gen.generate_sample(
                    defect_type=defect_type
                )
            )

            filename = (
                f"sample_{defect_type}.png"
            )

            filepath = os.path.join(
                DATA_DIR,
                filename
            )

            Image.fromarray(
                img_rgb
            ).save(filepath)

            samples[defect_type] = {

                "file":
                    filename,

                "label":
                    defect_type.replace(
                        "_",
                        " "
                    ),

                "meta":
                    meta,

                "data_uri":
                    PCBPreprocessor.encode_image_base64(
                        img_rgb
                    )
            }

        laser_rgb, laser_meta = (
            synthetic_gen.generate_sample(
                defect_type="Laser_Burn_Defect"
            )
        )

        laser_file = (
            "sample_Laser_Burn_Defect.png"
        )

        Image.fromarray(
            laser_rgb
        ).save(
            os.path.join(
                DATA_DIR,
                laser_file
            )
        )

        samples["Laser_Burn_Defect"] = {

            "file":
                laser_file,

            "label":
                "Laser Burn Defect (Novel)",

            "meta":
                laser_meta,

            "data_uri":
                PCBPreprocessor.encode_image_base64(
                    laser_rgb
                )
        }

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                samples,
                f,
                indent=2
            )

    if not protonet.prototype_bank:

        print(
            "[Server] Seeding ProtoNet prototypes..."
        )

        for defect_type in PROTO_CLASSES:

            tensors = []

            for _ in range(3):

                img_rgb, _ = (
                    synthetic_gen.generate_sample(
                        defect_type=defect_type
                    )
                )

                t_np, _ = (
                    preprocessor.preprocess_image(
                        img_rgb
                    )
                )

                tensors.append(
                    torch.from_numpy(
                        t_np
                    ).float()
                )

            support_batch = torch.stack(
                tensors
            )

            protonet.register_novel_defect(

                defect_name=
                    defect_type,

                support_tensors=
                    support_batch,

                description=(
                    "Base standard prototype for "
                    f"{defect_type.replace('_', ' ')}"
                ),

                ipc_standard_ref=
                    "IPC-A-610G",

                severity_baseline=
                    "Major"
            )


ensure_sample_data()


# ============================================================
# RAG STATUS
# ============================================================

def get_rag_status():
    """Return safe status information for the Standards RAG engine."""

    global rag_engine

    status = {
        "initialized": rag_engine is not None,
        "available": False,
        "chunks": 0,
        "pdf_count": 0,
        "error": None,
    }

    if rag_engine is None:
        return status

    try:
        status["available"] = True

        if hasattr(rag_engine, "chunks"):
            status["chunks"] = len(rag_engine.chunks)

        if hasattr(rag_engine, "pdf_files"):
            status["pdf_count"] = len(rag_engine.pdf_files)

        elif hasattr(rag_engine, "pdf_paths"):
            status["pdf_count"] = len(rag_engine.pdf_paths)

        return status

    except Exception as exc:
        status["available"] = False
        status["error"] = str(exc)
        return status
def normalize_prediction_class(class_name):

    if class_name is None:
        return "Unknown"

    raw = str(class_name).strip()

    if not raw:
        return "Unknown"

    key = raw.lower()
    key = key.replace("-", "_")
    key = key.replace(" ", "_")

    canonical_aliases = {

        "good_assembly":
            "Good_Assembly",

        "normal":
            "Good_Assembly",

        "missing_component":
            "Missing_Component",

        "component_missing":
            "Missing_Component",

        "component_misaligned":
            "Component_Misaligned",

        "misaligned_component":
            "Component_Misaligned",

        "solder_bridge":
            "Solder_Bridge",

        "solder_bridges":
            "Solder_Bridge",

        "soldier_bridge":
            "Solder_Bridge",

        "soldier_bridges":
            "Solder_Bridge",

        "tombstoning":
            "Tombstoning",

        "solder_ball":
            "Solder_Ball",

        "solder_balls":
            "Solder_Ball",

        "insufficient_solder":
            "Insufficient_Solder",

        "insufficient_soldering":
            "Insufficient_Solder"
    }

    if key in canonical_aliases:
        return canonical_aliases[key]

    if "solder" in key and "bridge" in key:
        return "Solder_Bridge"

    if "soldier" in key and "bridge" in key:
        return "Solder_Bridge"

    if "missing" in key and "component" in key:
        return "Missing_Component"

    if "misalign" in key and "component" in key:
        return "Component_Misaligned"

    if "tombston" in key:
        return "Tombstoning"

    if "solder" in key and "ball" in key:
        return "Solder_Ball"

    if "insufficient" in key and "solder" in key:
        return "Insufficient_Solder"

    return raw


# ============================================================
# INSPECTION REPORT PARSING
# ============================================================

REPORT_DEFECT_ALIASES = {

    "solder_bridge":
        "Solder_Bridge",

    "solder_bridging":
        "Solder_Bridge",

    "solder_bridge_or_solder_short":
        "Solder_Bridge",

    "solder_ball":
        "Solder_Ball",

    "tombstoning":
        "Tombstoning",

    "tombstone":
        "Tombstoning",

    "missing_component":
        "Missing_Component",

    "component_missing":
        "Missing_Component",

    "component_misaligned":
        "Component_Misaligned",

    "misaligned_component":
        "Component_Misaligned",

    "insufficient_solder":
        "Insufficient_Solder"
}


def _clean_report_value(value):

    if value is None:
        return ""

    value = str(value)

    value = value.replace(
        "\x00",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def _first_report_match(
    text,
    patterns,
    default=""
):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            try:

                value = match.group(1)

            except Exception:

                value = match.group(0)

            value = _clean_report_value(
                value
            )

            if value:
                return value

    return default


def _extract_report_section(
    text,
    heading,
    next_headings=None
):

    next_headings = (
        next_headings
        or []
    )

    escaped_heading = re.escape(
        heading
    )

    if next_headings:

        next_pattern = "|".join(
            re.escape(item)
            for item in next_headings
        )

        pattern = (
            rf"{escaped_heading}"
            rf"\s*:?\s*"
            rf"(.*?)"
            rf"(?=\n\s*(?:{next_pattern})\s*:?\s*$)"
        )

    else:

        pattern = (
            rf"{escaped_heading}"
            rf"\s*:?\s*"
            rf"(.*)$"
        )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return ""

    return _clean_report_value(
        match.group(1)
    )


def _normalize_report_defect(value):

    if not value:
        return ""

    normalized = (
        str(value)
        .strip()
        .lower()
    )

    normalized = normalized.replace(
        "-",
        "_"
    )

    normalized = re.sub(
        r"\s+",
        "_",
        normalized
    )

    if normalized in REPORT_DEFECT_ALIASES:

        return REPORT_DEFECT_ALIASES[
            normalized
        ]

    return normalize_prediction_class(
        value
    )


def extract_pdf_text(file_bytes):

    if PdfReader is None:

        raise RuntimeError(
            "PDF support is unavailable. "
            "Install pypdf with: pip install pypdf"
        )

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    pages = []

    for index, page in enumerate(
        reader.pages
    ):

        try:

            page_text = (
                page.extract_text()
                or ""
            )

        except Exception as e:

            print(
                f"[Report] Failed to extract page "
                f"{index + 1}: {e}"
            )

            page_text = ""

        pages.append(
            page_text
        )

    return "\n\n".join(
        pages
    )



def parse_inspection_report_text(text):
    """
    Parse a ManufacturingRAG-QA inspection certificate into
    structured context for the Standards Assistant.
    """

    text = text or ""

    # ---------------------------------------------------------
    # Normalize text
    # ---------------------------------------------------------
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    def first_match(pattern, flags=re.IGNORECASE | re.MULTILINE):
        match = re.search(pattern, text, flags)
        return match.group(1).strip() if match else ""

    def clean(value):
        if not value:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    # ---------------------------------------------------------
    # REPORT ID
    # ---------------------------------------------------------
    report_id = first_match(
        r"\bID\s*:\s*(AUD-[A-Z0-9-]+)"
    )

    if not report_id:
        report_id = first_match(
            r"\bReport\s*(?:ID|Identifier)\s*:\s*([A-Z0-9_-]+)"
        )

    # ---------------------------------------------------------
    # SERIAL / LOT / LINE
    # ---------------------------------------------------------
    serial = first_match(
        r"\bSerial\s*:\s*([^\n]+)"
    )

    lot = first_match(
        r"\bLot\s*:\s*([^\n|]+)"
    )

    line_name = first_match(
        r"\bLine\s*:\s*([^\n|]+)"
    )

    # ---------------------------------------------------------
    # OPERATING CLASS
    # ---------------------------------------------------------
    operating_class = first_match(
        r"\bOperating\s*(?:Level|Class)\s*:\s*(Class\s*[123])"
    )

    # ---------------------------------------------------------
    # STANDARD
    # ---------------------------------------------------------
    standard = first_match(
        r"\b(IPC-A-610[A-Z]?)\b"
    )

    # ---------------------------------------------------------
    # DEFECT
    # ---------------------------------------------------------

    defect = ""

    # 1. Reported Finding
    match = re.search(
        r"\bReported\s+Finding\s*:\s*"
        r"([A-Za-z0-9_\-/ ]+?)(?=\s*(?:\n|Severity\s*:|$))",
        text,
        re.IGNORECASE
    )

    if match:
        defect = match.group(1).strip()

    # 2. ProtoNet Category
    if not defect:
        match = re.search(
            r"\bProtoNet\s+Category\s*[:\-]?\s*"
            r"([A-Za-z0-9_\-/ ]+?)"
            r"\s*(?:\(\s*CONFIRMED\s*\)|\n|$)",
            text,
            re.IGNORECASE
        )

        if match:
            defect = match.group(1).strip()

    # 3. Inspection Finding
    if not defect:
        match = re.search(
            r"\bInspection\s+Finding\s*:?\s*\n?\s*"
            r"([A-Za-z0-9_\-/ ]+)",
            text,
            re.IGNORECASE
        )

        if match:
            defect = match.group(1).strip()

    defect = re.sub(
        r"\s*\(?CONFIRMED\)?",
        "",
        defect,
        flags=re.IGNORECASE
    )

    defect = re.sub(
        r"^(?:Confirmed|True|False)\s*:?\s*",
        "",
        defect,
        flags=re.IGNORECASE
    )

    defect = clean(defect)

    # ---------------------------------------------------------
    # CBAM
    # ---------------------------------------------------------
    cbam_decision = first_match(
        r"\bCBAM\s+Binary\s+Decision\s+([A-Za-z]+)"
    )

    cbam_confidence_raw = first_match(
        r"\bCBAM\s+Confidence\s*:?\s*([0-9.]+)\s*%"
    )

    cbam_confidence = None

    if cbam_confidence_raw:
        try:
            cbam_confidence = float(cbam_confidence_raw)
        except Exception:
            cbam_confidence = None

    # ---------------------------------------------------------
    # PROTONET
    # ---------------------------------------------------------
    protonet_category = ""

    match = re.search(
        r"\bProtoNet\s+Category\s*[:\-]?\s*"
        r"([A-Za-z0-9_\-/ ]+?)"
        r"\s*(?:\(\s*CONFIRMED\s*\)|\n|$)",
        text,
        re.IGNORECASE
    )

    if match:
        protonet_category = clean(match.group(1))

    # ---------------------------------------------------------
    # DECISION
    # ---------------------------------------------------------
    decision_match = re.search(
        r"\b(NON-CONFORMANCE|NONCONFORMANCE|CONFORMANCE|PASS|FAIL)\b",
        text,
        re.IGNORECASE
    )

    decision = ""

    if decision_match:
        decision = decision_match.group(1).upper()

        if decision == "NONCONFORMANCE":
            decision = "NON-CONFORMANCE"

    # ---------------------------------------------------------
    # SEVERITY
    # ---------------------------------------------------------
    severity = first_match(
        r"\bSeverity\s*:\s*([^\n]+)"
    )

    severity = clean(severity)

    # ---------------------------------------------------------
    # REQUIRED ACTION
    # ---------------------------------------------------------
    required_action = ""

    patterns = [
        r"\bRequired\s+Action\s*:\s*([^\n]+)",
        r"\bRequired\s+Action\s*\n\s*([^\n]+)",
        r"\bRequired\s+Action\s+([^\n]+)",
        r"\bAction\s*:\s*([^\n]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            required_action = match.group(1).strip()
            break

    required_action = clean(required_action)

    # Remove accidental next-section text
    required_action = re.split(
        r"\b(?:IPC-A-610G?|Standard Source|Class Requirement|"
        r"Acceptance Level|Electrical Impact|Description|"
        r"RAG Evidence|Root Cause|Audited by|Date)\b",
        required_action,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0].strip()

    # Guaranteed fallback for a non-conformance report
    if not required_action and decision == "NON-CONFORMANCE":
        required_action = (
            "HALT LOT - MANDATORY REWORK OR SCRAP"
        )

    # ---------------------------------------------------------
    # IPC CLAUSE
    # ---------------------------------------------------------
    ipc_clause = ""

    clause_patterns = [
        r"\bClause\s+([0-9]+(?:\.[0-9]+)*)",
        r"\bIPC-A-610[A-Z]?\s*\|\s*Clause\s+([0-9]+(?:\.[0-9]+)*)",
        r"\bSection\s+([0-9]+(?:\.[0-9]+)*)",
    ]

    for pattern in clause_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            ipc_clause = match.group(1).strip()
            break

    # ---------------------------------------------------------
    # ELECTRICAL IMPACT
    # ---------------------------------------------------------
    electrical_impact = first_match(
        r"\bElectrical\s+Impact\s*:\s*([^\n]+)"
    )

    electrical_impact = clean(electrical_impact)

    # ---------------------------------------------------------
    # DESCRIPTION
    # ---------------------------------------------------------
    description = first_match(
        r"\bDescription\s*:\s*([^\n]+)"
    )

    description = clean(description)

    # ---------------------------------------------------------
    # REWORK
    # ---------------------------------------------------------
    rework_procedure = first_match(
        r"\bProcedure\s+[0-9.]+\s*-\s*([^\n]+)"
    )

    if not rework_procedure:
        rework_procedure = first_match(
            r"\bCorrective\s+Rework\s+Procedure\s*:?\s*([^\n]+)"
        )

    rework_procedure = clean(rework_procedure)

    # ---------------------------------------------------------
    # ROOT CAUSES
    # ---------------------------------------------------------
    root_causes = []

    causes_match = re.search(
        r"\bRoot\s+Cause\s+Hypotheses\s*:\s*(.*?)(?=\n\s*(?:Audited by|Date:|$))",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if causes_match:

        causes_text = clean(
            causes_match.group(1)
        )

        root_causes = [
            clean(item)
            for item in re.split(
                r",\s*(?=[A-Z])",
                causes_text
            )
            if clean(item)
        ]

    # ---------------------------------------------------------
    # AUDITOR / DATE / PCB
    # ---------------------------------------------------------
    auditor = first_match(
        r"\bAudited\s+by\s*:\s*([^\n]+)"
    )

    audit_date = first_match(
        r"\bDate\s*:\s*([^\n]+)"
    )

    pcb_id = first_match(
        r"\b(PCB[1-9][0-9]*)\b"
    )

    # ---------------------------------------------------------
    # FINAL STRUCTURED REPORT
    # ---------------------------------------------------------

    # FINAL ROOT CAUSES EXTRACTION
    #
    # Preserve root-cause hypotheses from the uploaded inspection
    # certificate so the Standards Assistant can answer:
    # "What are the possible causes of this defect?"

    try:
        _root_cause_text = ""

        # Search the complete extracted report text.
        _source_for_causes = str(
            locals().get("text")
            or locals().get("raw_text")
            or locals().get("report_text")
            or ""
        )

        if not _source_for_causes:
            _source_for_causes = str(
                locals().get("content")
                or ""
            )

        _rc_match = re.search(
            r'Root\s*Cause(?:s)?\s*Hypotheses?\s*:\s*(.*?)(?=\n\s*\n|\n\s*(?:Audited by|Date|Inspection|IPC|Evidence|RAG Evidence)|\Z)',
            _source_for_causes,
            re.IGNORECASE | re.DOTALL
        )

        if _rc_match:
            _root_cause_text = _rc_match.group(1).strip()

        if _root_cause_text:

            _root_causes = [
                x.strip(" .;-")
                for x in re.split(
                    r',\s*(?=[A-Z])|\n+|\s*;\s*',
                    _root_cause_text
                )
                if x.strip(" .;-")
            ]

            # If comma splitting produced reasonable causes,
            # preserve them. Otherwise keep the complete text.
            if len(_root_causes) >= 2:

                report["root_causes"] = _root_causes

            else:

                report["root_causes"] = [
                    _root_cause_text
                ]

    except Exception as _root_cause_error:

        print(
            "[REPORT PARSER] Root cause extraction warning:",
            repr(_root_cause_error)
        )


    # FINAL ROOT CAUSES EXTRACTION
    # Extract directly from the original parser input so this does not
    # depend on local variable names such as text/raw_text/content.
    try:
        _source_for_root_causes = str(text or "")
        _rc_match = re.search(
            r"Root\\s*Cause(?:s)?\\s*Hypotheses?\\s*:\\s*(.*?)(?=\\n\\s*\\n|\\n\\s*(?:Audited by|Date|Inspection|IPC|Evidence|RAG Evidence)|\\Z)",
            _source_for_root_causes,
            re.IGNORECASE | re.DOTALL
        )
        if _rc_match:
            _rc_text = _rc_match.group(1).strip()
            if _rc_text:
                _root_causes = [
                    x.strip(" .;-")
                    for x in re.split(
                        r",\\s*(?=[A-Z])|\\n+|\\s*;\\s*",
                        _rc_text
                    )
                    if x.strip(" .;-")
                ]
                if len(_root_causes) >= 2:
                    report["root_causes"] = _root_causes
                else:
                    report["root_causes"] = [_rc_text]
    
    except Exception as _root_cause_error:
        print("Root-cause extraction warning:", _root_cause_error)


    # FINAL ROOT CAUSES EXTRACTION
    # Extract directly from the original parser input so this does not
    # depend on local variable names such as text/raw_text/content.
    try:
        _source_for_root_causes = str(text or "")
        _rc_match = re.search(
            r"Root\\s*Cause(?:s)?\\s*Hypotheses?\\s*:\\s*(.*?)(?=\\n\\s*\\n|\\n\\s*(?:Audited by|Date|Inspection|IPC|Evidence|RAG Evidence)|\\Z)",
            _source_for_root_causes,
            re.IGNORECASE | re.DOTALL
        )
        if _rc_match:
            _rc_text = _rc_match.group(1).strip()
            if _rc_text:
                _root_causes = [
                    x.strip(" .;-")
                    for x in re.split(
                        r",\\s*(?=[A-Z])|\\n+|\\s*;\\s*",
                        _rc_text
                    )
                    if x.strip(" .;-")
                ]
                if len(_root_causes) >= 2:
                    report["root_causes"] = _root_causes
                else:
                    report["root_causes"] = [_rc_text]
    
    except Exception as _root_cause_error:
        print("Root-cause extraction warning:", _root_cause_error)

    return {
        "report_id": report_id,
        "pcb_id": pcb_id,
        "serial": serial,
        "lot": lot,
        "line": line_name,

        "operating_class": operating_class,
        "standard": standard,
        "ipc_clause": ipc_clause,

        "defect": defect,
        "cbam_decision": cbam_decision,
        "cbam_confidence": cbam_confidence,
        "protonet_category": protonet_category,

        "decision": decision,
        "severity": severity,
        "required_action": required_action,

        "electrical_impact": electrical_impact,
        "description": description,
        "rework_procedure": rework_procedure,
        "root_causes": root_causes,

        "auditor": auditor,
        "audit_date": audit_date,
    }

    # =========================================================
    # ROOT CAUSE EXTRACTION
    # =========================================================
    try:
        _root_source = str(text or "")

        _root_match = re.search(
            r"Root\s*Cause(?:s)?\s*Hypotheses?\s*:\s*(.*?)(?=\n\s*\n|\n\s*(?:Audited by|Date|Inspection Report|IPC|Evidence|RAG Evidence)|\Z)",
            _root_source,
            re.IGNORECASE | re.DOTALL
        )

        if _root_match:
            _root_text = _root_match.group(1).strip()

            if _root_text:
                _root_items = [
                    item.strip(" .;-")
                    for item in re.split(
                        r",\s*(?=[A-Z])|\n+|\s*;\s*",
                        _root_text
                    )
                    if item.strip(" .;-")
                ]

                if _root_items:
                    report["root_causes"] = _root_items

    except Exception as _root_error:
        print("Root cause extraction warning:", _root_error)


def normalize_report_context(
    report
):

    if not isinstance(
        report,
        dict
    ):
        return None

    normalized = dict(
        report
    )

    normalized["defect"] = (
        _normalize_report_defect(
            normalized.get(
                "defect",
                ""
            )
        )
    )

    if not normalized.get(
        "protonet_category"
    ):

        normalized[
            "protonet_category"
        ] = normalized[
            "defect"
        ]

    return normalized


def detect_report_question_intent(question):
    """
    Determine whether a question should be answered from the
    active inspection report rather than generic Standards RAG.

    IMPORTANT:
    Acceptance / non-conformance questions must be detected BEFORE
    causes questions because phrases such as:
        "why is this solder bridge not acceptable?"
    contain a defect name but are asking for acceptance reasoning.
    """

    q = normalize_chat_text(question).lower().strip()

    # ---------------------------------------------------------
    # ACCEPTANCE / NON-CONFORMANCE
    # ---------------------------------------------------------
    acceptance_patterns = [
        "not acceptable",
        "not accepted",
        "unacceptable",
        "why is this not acceptable",
        "why is this not accepted",
        "why was this rejected",
        "why is this rejected",
        "why did this fail",
        "why does this fail",
        "why is this a non-conformance",
        "why is this non-conformance",
        "why is this nonconformance",
        "why did this become a non-conformance",
        "why did this become nonconformance",
        "is this acceptable",
        "is this accepted",
        "is this compliant",
        "is this non-compliant",
        "is this noncompliant",
        "can this pass",
        "should this pass",
        "would this pass",
        "does this pass",
        "pass or fail",
        "why cannot this pass",
        "why can't this pass",
        "why can this not pass",
        "why is the board rejected",
        "why was the board rejected",
    ]

    if any(pattern in q for pattern in acceptance_patterns):
        return "acceptance"

    # ---------------------------------------------------------
    # REPORT SUMMARY / FINDING
    # ---------------------------------------------------------
    summary_patterns = [
        "what was detected",
        "what defect was detected",
        "what defect was found",
        "what is the defect",
        "what did the inspection find",
        "what did the inspection detect",
        "what is the inspection result",
        "what is the inspection finding",
        "what is the finding",
        "why did inspection fail",
        "why did the inspection fail",
        "what happened to this board",
        "summarize this report",
        "summarize the report",
        "explain this report",
        "explain the inspection",
    ]

    if any(pattern in q for pattern in summary_patterns):
        return "summary"

    # ---------------------------------------------------------
    # REWORK / CORRECTION
    # ---------------------------------------------------------
    rework_patterns = [
        "how should this be corrected",
        "how should this be fixed",
        "how can this be corrected",
        "how can this be fixed",
        "how do i fix this",
        "how do i repair this",
        "how to fix this",
        "how to repair this",
        "how should this defect be corrected",
        "what rework is required",
        "what rework should be performed",
        "what corrective action is required",
        "what correction is required",
        "how should the board be repaired",
    ]

    if any(pattern in q for pattern in rework_patterns):
        return "rework"

    # ---------------------------------------------------------
    # CAUSES / ROOT CAUSES
    # ---------------------------------------------------------
    cause_patterns = [
        "what caused this",
        "what could have caused this",
        "what causes this",
        "what are the causes",
        "what could cause this",
        "why did this happen",
        "why does this happen",
        "why did this occur",
        "root cause",
        "root causes",
        "possible causes",
        "likely causes",
        "cause of this defect",
        "causes of this defect",
    ]

    if any(pattern in q for pattern in cause_patterns):
        return "causes"

    # ---------------------------------------------------------
    # CLASS COMPARISON
    # ---------------------------------------------------------
    class_patterns = [
        "class 1",
        "class 2",
        "class 3",
        "product class",
        "different class",
        "other class",
        "class comparison",
        "would this be acceptable for class",
        "acceptable for class",
    ]

    if any(pattern in q for pattern in class_patterns):
        return "classification"

    # ---------------------------------------------------------
    # EVIDENCE / STANDARD
    # ---------------------------------------------------------
    evidence_patterns = [
        "which standard",
        "what standard",
        "which clause",
        "what clause",
        "where in ipc",
        "ipc requirement",
        "acceptance criteria",
        "acceptance requirement",
        "standard requirement",
        "what does ipc say",
        "what does the standard say",
    ]

    if any(pattern in q for pattern in evidence_patterns):
        return "acceptance"

    return None

def _report_class_number(
    value
):

    if not value:
        return None

    match = re.search(
        r"class\s*([123])",
        str(value),
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def _extract_class_criterion_from_standard(
    standard,
    class_name
):

    if not isinstance(
        standard,
        dict
    ):
        return ""

    class_number = _report_class_number(
        class_name
    )

    if class_number is None:
        return ""

    possible_keys = [

        f"class_{class_number}_criteria",

        f"Class {class_number} Criteria",

        f"class{class_number}_criteria",

        f"class_{class_number}",

        f"Class {class_number}"
    ]

    for key in possible_keys:

        value = standard.get(
            key
        )

        if value:
            return str(
                value
            ).strip()

    return ""


def find_structured_standard_for_report(
    report,
    defect=None
):

    defect = (
        defect
        or report.get(
            "defect"
        )
    )

    if not defect:
        return None

    normalized = normalize_prediction_class(
        defect
    )

    # First use the project's structured KB lookup.
    try:

        standard = (
            get_standard_by_defect_type(
                normalized
            )
        )

        if standard:
            return standard

    except Exception:
        pass

    # Fallback: scan all standards.
    try:

        all_standards = get_all_standards()

        if isinstance(
            all_standards,
            dict
        ):

            iterable = (
                all_standards.values()
            )

        else:

            iterable = all_standards

        aliases = {
            normalized.lower(),
            normalized.replace(
                "_",
                " "
            ).lower()
        }

        for standard in iterable:

            if not isinstance(
                standard,
                dict
            ):
                continue

            haystack = " ".join(
                str(
                    standard.get(
                        key,
                        ""
                    )
                )
                for key in [
                    "defect_type",
                    "title",
                    "description",
                    "name"
                ]
            ).lower()

            if any(
                alias in haystack
                for alias in aliases
            ):

                return standard

    except Exception:
        pass

    return None


def _format_report_evidence(
    report,
    standard
):

    evidence = []

    standard_name = (
        report.get(
            "standard"
        )
        or "IPC-A-610"
    )

    clause = report.get(
        "ipc_clause"
    )

    defect = (
        report.get(
            "defect"
        )
        or "reported defect"
    )

    evidence.append({

        "source":
            standard_name,

        "section":
            clause or "Reported finding",

        "title":
            defect.replace(
                "_",
                " "
            ),

        "evidence":
            (
                report.get(
                    "description"
                )
                or report.get(
                    "acceptance_level"
                )
                or report.get(
                    "class_requirement"
                )
                or "The uploaded inspection report identifies this condition."
            ),

        "source_type":
            "inspection_report",

        "score":
            None
    })

    if standard:

        class_name = (
            report.get(
                "operating_class"
            )
            or "Class 3"
        )

        criterion = (
            _extract_class_criterion_from_standard(
                standard,
                class_name
            )
        )

        if criterion:

            evidence.append({

                "source":
                    standard.get(
                        "standard",
                        standard_name
                    ),

                "section":
                    standard.get(
                        "section",
                        clause
                        or ""
                    ),

                "title":
                    standard.get(
                        "title",
                        defect.replace(
                            "_",
                            " "
                        )
                    ),

                "evidence":
                    criterion,

                "source_type":
                    "structured_standards_kb",

                "score":
                    None
            })

    return evidence


def answer_report_grounded_question(
    question,
    report,
    operating_class="Class 3",
    rag=None
):

    report = normalize_report_context(
        report
    )

    if not report:

        return None

    intent = detect_report_question_intent(
        question
    )

    defect = (
        report.get(
            "defect"
        )
        or report.get(
            "protonet_category"
        )
        or "Unknown defect"
    )

    defect_display = defect.replace(
        "_",
        " "
    )

    report_class = (
        report.get(
            "operating_class"
        )
        or operating_class
        or "Class 3"
    )

    standard_name = (
        report.get(
            "standard"
        )
        or "IPC-A-610"
    )

    clause = report.get(
        "ipc_clause"
    )

    clause_text = (
        f" Clause {clause}"
        if clause
        else ""
    )

    standard = (
        find_structured_standard_for_report(
            report,
            defect
        )
    )

    evidence = _format_report_evidence(
        report,
        standard
    )

    # ========================================================
    # WHY NOT ACCEPTABLE
    # ========================================================

    if intent == "report_acceptance":

        criterion = (
            report.get(
                "acceptance_level"
            )
            or report.get(
                "class_requirement"
            )
        )

        if not criterion and standard:

            criterion = (
                _extract_class_criterion_from_standard(
                    standard,
                    report_class
                )
            )

        if report.get(
            "not_acceptable"
        ):

            answer = (
                f"It is not acceptable because the report "
                f"identified **{defect_display}** as a "
                f"non-conformance for {report_class}.\n\n"
            )

            if clause:

                answer += (
                    f"Under **{standard_name}{clause_text}**, "
                    f"this condition is treated as a defect "
                    f"and therefore does not meet the "
                    f"acceptance requirement.\n\n"
                )

            elif criterion:

                answer += (
                    f"The report's acceptance requirement is: "
                    f"**{criterion}**\n\n"
                )

            else:

                answer += (
                    "The uploaded report explicitly marks the "
                    "condition as NON-CONFORMANCE.\n\n"
                )

            if report.get(
                "electrical_impact"
            ):

                answer += (
                    f"**Why it matters:** "
                    f"{report['electrical_impact']}\n\n"
                )

            answer += (
                f"**Reported action:** "
                f"{report.get('required_action') or 'Rework or otherwise control the non-conforming unit.'}"
            )

        else:

            answer = (
                f"The uploaded report identifies "
                f"**{defect_display}**, but it does not contain "
                f"enough information to establish rejection "
                f"for {report_class}."
            )

        return {

            "answer":
                answer,

            "intent":
                intent,

            "report_used":
                True,

            "rag_used":
                False,

            "standard":
                standard_name,

            "clause":
                clause,

            "defect":
                defect,

            "operating_class":
                report_class,

            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "report_summary":
                report
        }

    # ========================================================
    # ROOT CAUSES
    # ========================================================

    if intent == "report_causes":

        causes = report.get(
            "root_causes",
            []
        )

        if causes:

            lines = []

            for index, cause in enumerate(
                causes,
                start=1
            ):

                lines.append(
                    f"{index}. {cause}"
                )

            answer = (
                f"The uploaded report gives these "
                f"root-cause hypotheses for "
                f"**{defect_display}**:\n\n"
                + "\n".join(lines)
            )

            answer += (
                "\n\nThese are reported **root-cause "
                "hypotheses**, not proof that each cause "
                "actually occurred on the board."
            )

        else:

            answer = (
                f"The uploaded report does not contain "
                f"specific root-cause hypotheses for "
                f"**{defect_display}**."
            )

        return {

            "answer":
                answer,

            "intent":
                intent,

            "report_used":
                True,

            "rag_used":
                False,

            "defect":
                defect,

            "operating_class":
                report_class,

            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "report_summary":
                report
        }

    # ========================================================
    # REWORK
    # ========================================================

    if intent == "report_rework":

        procedure = report.get(
            "rework_procedure"
        )

        if procedure:

            answer = (
                f"For **{defect_display}**, the uploaded "
                f"report specifies this rework guidance:\n\n"
                f"**{procedure}**"
            )

        else:

            answer = (
                f"The uploaded report does not contain a "
                f"specific rework procedure for "
                f"**{defect_display}**."
            )

        if report.get(
            "required_action"
        ):

            answer += (
                f"\n\n**Required action in the report:** "
                f"{report['required_action']}"
            )

        return {

            "answer":
                answer,

            "intent":
                intent,

            "report_used":
                True,

            "rag_used":
                False,

            "defect":
                defect,

            "operating_class":
                report_class,

            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "report_summary":
                report
        }

    # ========================================================
    # EVIDENCE
    # ========================================================

    if intent == "report_evidence":

        answer = (
            f"The report supports the **{defect_display}** "
            f"finding using the following evidence:\n\n"
        )

        if clause:

            answer += (
                f"• **{standard_name}{clause_text}** — "
                f"the report maps the finding to this clause.\n"
            )

        if report.get(
            "description"
        ):

            answer += (
                f"• **Finding description** — "
                f"{report['description']}\n"
            )

        if report.get(
            "acceptance_level"
        ):

            answer += (
                f"• **Acceptance level** — "
                f"{report['acceptance_level']}\n"
            )

        if report.get(
            "class_requirement"
        ):

            answer += (
                f"• **Class requirement** — "
                f"{report['class_requirement']}\n"
            )

        if report.get(
            "pdf_evidence_used"
        ):

            answer += (
                "• **PDF evidence** — the report records "
                "that standards evidence was retrieved.\n"
            )

        if report.get(
            "structured_kb_used"
        ):

            answer += (
                "• **Structured standards KB** — the report "
                "records that the structured standards knowledge "
                "base was used."
            )

        return {

            "answer":
                answer.strip(),

            "intent":
                intent,

            "report_used":
                True,

            "rag_used":
                False,

            "defect":
                defect,

            "operating_class":
                report_class,

            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "report_summary":
                report
        }

    # ========================================================
    # CLASS COMPARISON
    # ========================================================

    if intent == "report_class_compare":

        requested_class = _first_report_match(
            question,
            [
                r"class\s*([123])"
            ]
        )

        if requested_class:

            requested_class = (
                f"Class {requested_class}"
            )

        else:

            requested_class = report_class

        criterion = ""

        if standard:

            criterion = (
                _extract_class_criterion_from_standard(
                    standard,
                    requested_class
                )
            )

        if not criterion:

            report_acceptance = (
                report.get(
                    "acceptance_level",
                    ""
                )
            )

            if (
                requested_class.lower()
                == report_class.lower()
            ):

                criterion = report_acceptance

        if criterion:

            is_defect = (
                "defect"
                in criterion.lower()
            )

            if is_defect:

                answer = (
                    f"For **{requested_class}**, the "
                    f"criterion retrieved for "
                    f"**{defect_display}** is:\n\n"
                    f"**{criterion}**\n\n"
                    f"That means the condition is **not "
                    f"acceptable** for {requested_class}."
                )

            else:

                answer = (
                    f"For **{requested_class}**, the "
                    f"criterion retrieved for "
                    f"**{defect_display}** is:\n\n"
                    f"**{criterion}**"
                )

        else:

            answer = (
                f"I could not find a direct class-specific "
                f"criterion for **{defect_display}** and "
                f"{requested_class} in the available "
                f"structured standards data."
            )

        return {

            "answer":
                answer,

            "intent":
                intent,

            "report_used":
                True,

            "rag_used":
                False,

            "defect":
                defect,

            "operating_class":
                requested_class,

            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "report_summary":
                report
        }

    # ========================================================
    # REPORT SUMMARY
    # ========================================================

    if intent == "report_summary":

        answer = (
            f"### Inspection Report Summary\n\n"
            f"• **Finding:** {defect_display}\n"
            f"• **Decision:** {report.get('decision') or 'Not specified'}\n"
            f"• **Operating class:** {report_class}\n"
            f"• **Severity:** {report.get('severity') or 'Not specified'}\n"
        )

        if report.get(
            "standard"
        ):

            answer += (
                f"• **Standard:** {report['standard']}\n"
            )

        if clause:

            answer += (
                f"• **Clause:** {clause}\n"
            )

        if report.get(
            "required_action"
        ):

            answer += (
                f"• **Required action:** "
                f"{report['required_action']}\n"
            )

        if report.get(
            "cbam_confidence"
        ) is not None:

            answer += (
                f"• **CBAM confidence:** "
                f"{report['cbam_confidence']:.2f}%\n"
            )

        return {

            "answer":
                answer,

            "intent":
                intent,

            "report_used":
                True,

            "rag_used":
                False,

            "defect":
                defect,

            "operating_class":
                report_class,

            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "report_summary":
                report
        }

    # ========================================================
    # GENERAL REPORT QUESTION
    # ========================================================

    answer = (
        f"The uploaded inspection report identifies "
        f"**{defect_display}** for {report_class}."
    )

    if report.get(
        "decision"
    ):

        answer += (
            f"\n\n**Decision:** "
            f"{report['decision']}"
        )

    if clause:

        answer += (
            f"\n\n**Standards reference:** "
            f"{standard_name}{clause_text}"
        )

    if report.get(
        "description"
    ):

        answer += (
            f"\n\n**Finding:** "
            f"{report['description']}"
        )

    return {

        "answer":
            answer,

        "intent":
            intent,

        "report_used":
            True,

        "rag_used":
            False,

        "defect":
            defect,

        "operating_class":
            report_class,

        "evidence":
            evidence,

        "evidence_count":
            len(evidence),

        "report_summary":
            report
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def serve_index():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# ============================================================
# STATUS
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def get_status():

    return jsonify({

        "status":
            "online",

        "system":
            "ManufacturingRAG-QA",

        "version":
            "1.0.0",

        "device":
            str(device),

        "model": {

            "name":
                "PCBCBAMNet",

            "checkpoint":
                weights_path
                if weights_path is not None
                else None,

            "checkpoint_loaded":
                model_loaded,

            "classification_type":
                "binary",

            "classes":
                CBAM_CLASSES,

            "embedding_dimension":
                512
        },

        "cbam_classes":
            CBAM_CLASSES,

        "prototype_classes":
            BASE_CLASSES,

        "registered_prototypes":
            len(
                protonet.prototype_bank
            ),

        "standards_clauses_indexed":
            len(
                get_all_standards()
            ),

        "rag":
            get_rag_status()
    })


# ============================================================
# DEMO SAMPLES
# ============================================================

@app.route(
    "/api/samples",
    methods=["GET"]
)
def get_samples():

    manifest_path = os.path.join(
        DATA_DIR,
        "samples_manifest.json"
    )

    if os.path.exists(manifest_path):

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            return jsonify(
                json.load(f)
            )

    return jsonify({})


# ============================================================
# PRIMARY INSPECTION
# ============================================================

@app.route(
    "/api/inspect",
    methods=["POST"]
)
def inspect_image():

    try:

        req_data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        operating_class = (
            request.form.get(
                "operating_class"
            )
            or req_data.get(
                "operating_class",
                "Class 3"
            )
        )

        mode = (
            request.form.get(
                "mode"
            )
            or req_data.get(
                "mode",
                "hybrid"
            )
        )

        image_input = None

        if "file" in request.files:

            file = request.files["file"]

            image_input = file.read()

        elif "image_b64" in req_data:

            image_input = req_data[
                "image_b64"
            ]

        elif "sample_name" in req_data:

            sample_name = req_data[
                "sample_name"
            ]

            sample_path = os.path.join(
                DATA_DIR,
                f"sample_{sample_name}.png"
            )

            if os.path.exists(sample_path):

                image_input = sample_path

            else:

                return jsonify({

                    "error":
                        f"Sample not found: {sample_name}"

                }), 404

        else:

            return jsonify({

                "error":
                    "No image or sample provided."

            }), 400

        tensor_np, original_rgb = (
            preprocessor.preprocess_image(
                image_input,
                apply_enhancement=True
            )
        )

        input_tensor = (
            torch.from_numpy(
                tensor_np
            )
            .float()
            .to(device)
        )

        model_input = (
            input_tensor.unsqueeze(0)
        )

        # ====================================================
        # CBAM
        # ====================================================

        model.eval()

        with torch.no_grad():

            logits, embeddings = (
                model(model_input)
            )

            probs = (
                torch.softmax(
                    logits,
                    dim=1
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

            pred_idx = int(
                np.argmax(probs)
            )

            cbam_predicted_class = (
                CBAM_CLASSES[pred_idx]
            )

            cbam_confidence = float(
                probs[pred_idx]
            )

        cbam_probabilities = {

            CBAM_CLASSES[i]:
                float(probs[i])

            for i in range(
                len(CBAM_CLASSES)
            )
        }

        # ====================================================
        # ProtoNet
        # ====================================================

        protonet_result = (
            protonet.predict_few_shot(
                model_input,
                top_k=3
            )
        )

        proto_class = (
            protonet_result.get(
                "predicted_class"
            )
        )

        try:

            proto_confidence = float(
                protonet_result.get(
                    "confidence",
                    0.0
                )
            )

        except Exception:

            proto_confidence = 0.0

        # ====================================================
        # DECISIONS
        # ====================================================

        binary_decision = {

            "class_name":
                cbam_predicted_class,

            "confidence":
                cbam_confidence,

            "all_probabilities":
                cbam_probabilities,

            "classification_type":
                "normal_vs_anomaly",

            "decision_source":
                "CBAM"
        }

        category_decision = {

            "class_name":
                None,

            "confidence":
                proto_confidence,

            "decision_source":
                "ProtoNet",

            "confirmed":
                False,

            "status":
                "No reliable defect category"
        }

        if (
            cbam_predicted_class == "anomaly"
            and proto_class
            and proto_class != "Good_Assembly"
        ):

            category_decision = {

                "class_name":
                    normalize_prediction_class(
                        proto_class
                    ),

                "confidence":
                    proto_confidence,

                "decision_source":
                    "ProtoNet",

                "confirmed":
                    True,

                "status":
                    "Category identified"
            }

        elif cbam_predicted_class == "normal":

            category_decision = {

                "class_name":
                    "Good_Assembly",

                "confidence":
                    proto_confidence,

                "decision_source":
                    "CBAM + ProtoNet",

                "confirmed":
                    False,

                "status":
                    "Binary inspection is normal"
            }

        # ====================================================
        # GRAD-CAM
        # ====================================================

        cam_map, _, _ = (
            gradcam.generate_cam(
                model_input,
                target_class_idx=pred_idx
            )
        )

        overlay_rgb, heatmap_rgb = (
            GradCAM.overlay_heatmap(
                original_rgb,
                cam_map,
                alpha=0.55,
                colormap_type=cv2.COLORMAP_JET
            )
        )

        bounding_boxes = (
            GradCAM.extract_defect_bounding_boxes(
                cam_map,
                threshold_ratio=0.55
            )
        )

        defect_area_ratio = float(
            np.mean(
                cam_map > 0.4
            )
        )

        visual_findings = {

            "defect_area_ratio":
                defect_area_ratio,

            "bounding_boxes":
                bounding_boxes,

            "original_image_b64":
                PCBPreprocessor.encode_image_base64(
                    original_rgb
                ),

            "overlay_image_b64":
                PCBPreprocessor.encode_image_base64(
                    overlay_rgb
                ),

            "heatmap_image_b64":
                PCBPreprocessor.encode_image_base64(
                    heatmap_rgb
                )
        }

        # ====================================================
        # RAG / STANDARDS
        # ====================================================

        proto_confirmed = bool(
            category_decision.get(
                "confirmed",
                False
            )
        )

        confirmed_category = (
            category_decision.get(
                "class_name"
            )
            if proto_confirmed
            else None
        )

        if (
            cbam_predicted_class == "anomaly"
            and not proto_confirmed
        ):

            audit_finding = {

                "confidence":
                    cbam_confidence,

                "defect_type":
                    "Defect category not confidently identified",

                "predicted_class":
                    cbam_predicted_class,

                "operating_class":
                    operating_class,

                "finding":
                    (
                        "CBAM detected an anomaly with "
                        f"confidence "
                        f"{cbam_confidence:.3f}, but "
                        "ProtoNet did not confidently "
                        "identify a specific defect category."
                    ),

                "evidence_count":
                    0,

                "evidence_summary":
                    [],

                "retrieved_evidence":
                    [],

                "standard":
                    None,

                "knowledge_source":
                    "No confirmed standards mapping",

                "rag_query":
                    None,

                "grounding":
                    {
                        "llm_generated":
                            False,

                        "pdf_evidence_used":
                            False,

                        "structured_kb_used":
                            False
                    },

                "visual_findings":
                    visual_findings
            }

        elif (
            cbam_predicted_class == "anomaly"
            and proto_confirmed
            and confirmed_category
        ):

            rag = initialize_rag()

            if rag is not None:

                try:

                    audit_finding = (
                        rag.synthesize_audit_finding(

                            defect_type=
                                confirmed_category,

                            predicted_class=
                                confirmed_category,

                            confidence=
                                category_decision[
                                    "confidence"
                                ],

                            visual_findings=
                                visual_findings,

                            operating_class=
                                operating_class,

                            top_k=5
                        )
                    )

                    if not isinstance(
                        audit_finding,
                        dict
                    ):

                        audit_finding = {}

                    audit_finding.setdefault(
                        "defect_type",
                        confirmed_category
                    )

                    audit_finding.setdefault(
                        "predicted_class",
                        confirmed_category
                    )

                    audit_finding.setdefault(
                        "confidence",
                        category_decision[
                            "confidence"
                        ]
                    )

                    audit_finding.setdefault(
                        "operating_class",
                        operating_class
                    )

                    audit_finding.setdefault(
                        "visual_findings",
                        visual_findings
                    )

                except Exception as e:

                    print(
                        "[Server] RAG request failed:"
                    )

                    print(
                        f"         {e}"
                    )

                    audit_finding = {

                        "confidence":
                            category_decision[
                                "confidence"
                            ],

                        "defect_type":
                            confirmed_category,

                        "predicted_class":
                            confirmed_category,

                        "operating_class":
                            operating_class,

                        "finding":
                            (
                                f"ProtoNet identified "
                                f"{confirmed_category} "
                                f"with confidence "
                                f"{category_decision['confidence']:.3f}, "
                                "but RAG standards retrieval "
                                "was unavailable."
                            ),

                        "evidence_count":
                            0,

                        "evidence_summary":
                            [],

                        "retrieved_evidence":
                            [],

                        "standard":
                            None,

                        "knowledge_source":
                            "RAG unavailable",

                        "rag_query":
                            None,

                        "grounding":
                            {
                                "llm_generated":
                                    False,

                                "pdf_evidence_used":
                                    False,

                                "structured_kb_used":
                                    False
                            },

                        "visual_findings":
                            visual_findings
                    }

            else:

                audit_finding = {

                    "confidence":
                        category_decision[
                            "confidence"
                        ],

                    "defect_type":
                        confirmed_category,

                    "predicted_class":
                        confirmed_category,

                    "operating_class":
                        operating_class,

                    "finding":
                        (
                            f"ProtoNet identified "
                            f"{confirmed_category} "
                            f"with confidence "
                            f"{category_decision['confidence']:.3f}. "
                            "RAG standards retrieval is "
                            "currently unavailable."
                        ),

                    "evidence_count":
                        0,

                    "evidence_summary":
                        [],

                    "retrieved_evidence":
                        [],

                    "standard":
                        None,

                    "knowledge_source":
                        "RAG unavailable",

                    "rag_query":
                        None,

                    "grounding":
                        {
                            "llm_generated":
                                False,

                            "pdf_evidence_used":
                                False,

                            "structured_kb_used":
                                False
                        },

                    "visual_findings":
                        visual_findings
                }

        else:

            audit_finding = {

                "confidence":
                    cbam_confidence,

                "defect_type":
                    "Good_Assembly",

                "predicted_class":
                    cbam_predicted_class,

                "operating_class":
                    operating_class,

                "finding":
                    (
                        "CBAM classified the assembly "
                        "as normal with confidence "
                        f"{cbam_confidence:.3f}."
                    ),

                "evidence_count":
                    0,

                "evidence_summary":
                    [],

                "retrieved_evidence":
                    [],

                "standard":
                    None,

                "knowledge_source":
                    "No defect standards mapping required",

                "rag_query":
                    None,

                "grounding":
                    {
                        "llm_generated":
                            False,

                        "pdf_evidence_used":
                            False,

                        "structured_kb_used":
                            False
                    },

                "visual_findings":
                    visual_findings
            }

        # ====================================================
        # AUDIT REPORT
        # ====================================================

        inspection_payload = {

            "audit_finding":
                audit_finding,

            "visual_evidence":
                visual_findings,

            "binary_decision":
                binary_decision,

            "category_decision":
                category_decision
        }

        full_report = (
            audit_reporter.generate_report(
                inspection_payload
            )
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "status":
                "success",

            "prediction": {

                "class_name":
                    cbam_predicted_class,

                "confidence":
                    cbam_confidence,

                "decision_source":
                    "CBAM",

                "operating_class":
                    operating_class,

                "binary_decision":
                    binary_decision,

                "category_decision":
                    category_decision,

                "cbam_prediction": {

                    "class_name":
                        cbam_predicted_class,

                    "confidence":
                        cbam_confidence,

                    "classification_type":
                        "normal_vs_anomaly",

                    "all_probabilities":
                        cbam_probabilities
                },

                "protonet_prediction":
                    protonet_result
            },

            "visual_saliency": {

                "original_b64":
                    visual_findings[
                        "original_image_b64"
                    ],

                "overlay_b64":
                    visual_findings[
                        "overlay_image_b64"
                    ],

                "heatmap_b64":
                    visual_findings[
                        "heatmap_image_b64"
                    ],

                "bounding_boxes":
                    bounding_boxes,

                "defect_area_ratio":
                    defect_area_ratio
            },

            "rag": {

                "knowledge_source":
                    audit_finding.get(
                        "knowledge_source"
                    ),

                "query":
                    audit_finding.get(
                        "rag_query"
                    ),

                "evidence_count":
                    audit_finding.get(
                        "evidence_count",
                        0
                    ),

                "evidence":
                    audit_finding.get(
                        "evidence_summary",
                        []
                    )
            },

            "audit_finding":
                audit_finding,

            "audit_report":
                full_report
        }

        return jsonify(
            response
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# FEW-SHOT REGISTER
# ============================================================

@app.route(
    "/api/few-shot/register",
    methods=["POST"]
)
def register_prototype():

    try:

        defect_name = request.form.get(
            "defect_name"
        )

        description = request.form.get(
            "description",
            "Novel custom PCB defect"
        )

        ipc_standard_ref = request.form.get(
            "ipc_standard_ref",
            "IPC-A-610 Custom"
        )

        severity_baseline = request.form.get(
            "severity_baseline",
            "Major"
        )

        if not defect_name:

            return jsonify({

                "error":
                    "defect_name is required"

            }), 400

        files = request.files.getlist(
            "images"
        )

        if not files:

            return jsonify({

                "error":
                    "At least 1 support image is required"

            }), 400

        tensors = []

        for f in files:

            img_bytes = f.read()

            t_np, _ = (
                preprocessor.preprocess_image(
                    img_bytes
                )
            )

            tensors.append(
                torch.from_numpy(
                    t_np
                ).float()
            )

        support_batch = torch.stack(
            tensors
        )

        result = (
            protonet.register_novel_defect(

                defect_name=
                    defect_name,

                support_tensors=
                    support_batch,

                description=
                    description,

                ipc_standard_ref=
                    ipc_standard_ref,

                severity_baseline=
                    severity_baseline
            )
        )

        return jsonify(
            result
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# LIST PROTOTYPES
# ============================================================

@app.route(
    "/api/few-shot/prototypes",
    methods=["GET"]
)
def list_prototypes():

    return jsonify({

        "prototypes":
            protonet.list_active_prototypes(),

        "total":
            len(
                protonet.prototype_bank
            )
    })


# ============================================================
# DELETE PROTOTYPE
# ============================================================

@app.route(
    "/api/few-shot/prototypes/<defect_name>",
    methods=["DELETE"]
)
def delete_prototype(defect_name):

    success = (
        protonet.delete_prototype(
            defect_name
        )
    )

    return jsonify({

        "success":
            success,

        "defect_name":
            defect_name
    })


# ============================================================
# CONVERSATIONAL STANDARDS ASSISTANT
# ============================================================

def normalize_chat_text(text):

    if text is None:
        return ""

    return " ".join(
        str(text).strip().lower().split()
    )


def detect_chat_intent(question):

    text = normalize_chat_text(
        question
    )

    if not text:
        return "empty"

    greeting_phrases = {

        "hello",
        "hi",
        "hey",
        "hello there",
        "hi there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "good day"
    }

    if text in greeting_phrases:
        return "greeting"

    thanks_phrases = {

        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
        "thx",
        "ty"
    }

    if text in thanks_phrases:
        return "thanks"

    goodbye_phrases = {

        "bye",
        "goodbye",
        "see you",
        "see you later",
        "talk to you later"
    }

    if text in goodbye_phrases:
        return "goodbye"

    identity_phrases = {

        "who are you",
        "what are you",
        "what is this",
        "what is this assistant",
        "who is this"
    }

    if text in identity_phrases:
        return "identity"

    help_phrases = {

        "help",
        "what can you do",
        "what can you help with",
        "how can you help me",
        "what do you do"
    }

    if text in help_phrases:
        return "help"

    casual_phrases = {

        "okay",
        "ok",
        "cool",
        "great",
        "nice",
        "got it",
        "understood"
    }

    if text in casual_phrases:
        return "casual"

    vague_patterns = [

        "is this defect okay",
        "is this defect acceptable",
        "is this okay",
        "is this acceptable",
        "is this good",
        "can this pass",
        "will this pass",
        "should this pass",
        "is it okay",
        "is it acceptable",
        "is it good"
    ]

    if any(
        phrase in text
        for phrase in vague_patterns
    ):
        return "clarification"

    standards_keywords = [

        "ipc",
        "ipc-a-610",
        "standard",
        "standards",
        "acceptance",
        "acceptable",
        "acceptability",
        "requirement",
        "requirements",
        "criteria",
        "inspection",
        "inspect",
        "defect",
        "defects",
        "solder",
        "soldering",
        "solder bridge",
        "solder ball",
        "tombstone",
        "tombstoning",
        "missing component",
        "misaligned",
        "component",
        "assembly",
        "pcb",
        "pcba",
        "class 1",
        "class 2",
        "class 3",
        "workmanship",
        "rework",
        "clearance",
        "lead",
        "pad",
        "termination",
        "void",
        "voiding",
        "wetting",
        "fillet",
        "flux",
        "bga",
        "qfp",
        "qfn",
        "through-hole",
        "surface mount",
        "smt"
    ]

    if any(
        keyword in text
        for keyword in standards_keywords
    ):
        return "standards"

    question_words = [

        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "which ",
        "can ",
        "should ",
        "does ",
        "do ",
        "is ",
        "are "
    ]

    if any(
        text.startswith(word)
        for word in question_words
    ):
        return "standards"

    return "standards"


def conversational_response(intent):

    responses = {

        "greeting":
            (
                "Hello! 👋 Welcome to the Manufacturing Standards "
                "Assistant.\n\n"
                "I can help you with PCB and PCBA quality questions, "
                "IPC-A-610 requirements, defect acceptance criteria, "
                "inspection findings, and evidence from the available "
                "manufacturer standards.\n\n"
                "What would you like to check?"
            ),

        "thanks":
            (
                "You're welcome! 😊\n\n"
                "If you have another PCB defect, inspection result, "
                "or standards question, just ask me."
            ),

        "goodbye":
            (
                "You're welcome! Goodbye 👋\n\n"
                "I'll be here whenever you need help with PCB "
                "inspection or manufacturing standards."
            ),

        "identity":
            (
                "I'm the Manufacturing Standards Assistant, part of "
                "the ManufacturingRAG-QA system.\n\n"
                "I help interpret PCB/PCBA inspection questions using "
                "the available manufacturing standards and retrieved "
                "document evidence."
            ),

        "help":
            (
                "I can help with:\n\n"
                "• PCB/PCBA defect definitions\n"
                "• IPC-A-610 requirements\n"
                "• Defect acceptance and rejection criteria\n"
                "• Soldering and component-placement issues\n"
                "• Inspection findings\n"
                "• Manufacturing workmanship questions\n"
                "• Uploaded inspection report analysis\n"
                "• Evidence retrieved from the available standards"
            ),

        "casual":
            (
                "Got it. 👍\n\n"
                "Whenever you're ready, ask me a PCB or "
                "manufacturing standards question."
            ),

        "clarification":
            (
                "I can help determine whether a condition is "
                "acceptable, but I need to know which defect or "
                "condition you're referring to."
            )
    }

    return responses.get(
        intent,
        "How can I help you with your PCB manufacturing standards question?"
    )


def build_conversation_context(
    history,
    current_question
):

    current_question = str(
        current_question or ""
    ).strip()

    if not current_question:
        return ""

    normalized = normalize_chat_text(
        current_question
    )

    standalone_indicators = [

        "solder bridge",
        "solder bridging",
        "solder ball",
        "solder balls",
        "tombstone",
        "tombstoning",
        "missing component",
        "component missing",
        "component misalignment",
        "misaligned component",
        "insufficient solder",
        "poor wetting",
        "cold solder",
        "voiding",

        "ipc",
        "ipc-a-610",
        "class 1",
        "class 2",
        "class 3",

        "what is ",
        "what are ",
        "what causes ",
        "why does ",
        "why is ",
        "how do ",
        "how does ",
        "how to ",
        "what are the causes"
    ]

    has_explicit_topic = any(
        indicator in normalized
        for indicator in standalone_indicators
    )

    if has_explicit_topic:
        return current_question

    followup_phrases = [

        "is that acceptable",
        "is that okay",
        "is that allowed",
        "can that pass",
        "will that pass",
        "should that pass",
        "what causes it",
        "why does it happen",
        "how do i fix it",
        "how can i fix it",
        "how do i repair it",
        "how can i repair it",
        "what about class 1",
        "what about class 2",
        "what about class 3",
        "what about acceptance",
        "what about rework",
        "what about repair",
        "tell me more",
        "explain that",
        "what do you mean",
        "why",
        "how"
    ]

    is_followup = (
        normalized in followup_phrases
        or any(
            normalized.startswith(
                phrase
            )
            for phrase in followup_phrases
        )
    )

    if not is_followup:
        return current_question

    if not isinstance(
        history,
        list
    ):
        return current_question

    recent_history = history[-6:]

    context_lines = []

    for message in recent_history:

        if not isinstance(
            message,
            dict
        ):
            continue

        role = str(
            message.get(
                "role",
                ""
            )
        ).strip().lower()

        content = str(
            message.get(
                "content",
                ""
            )
        ).strip()

        if not content:
            continue

        if role == "user":

            context_lines.append(
                f"User: {content}"
            )

        elif role == "assistant":

            context_lines.append(
                f"Assistant: {content}"
            )

    if not context_lines:
        return current_question

    return (
        "Use the following conversation only to resolve "
        "the reference in the current follow-up question.\n\n"
        + "\n".join(context_lines)
        + "\n\nCurrent user question:\n"
        + current_question
    )


# ============================================================
# INSPECTION REPORT UPLOAD
# ============================================================

@app.route(
    "/api/rag/report",
    methods=["POST"]
)
def upload_inspection_report():

    try:

        if "file" not in request.files:

            return jsonify({

                "error":
                    "No inspection report file was uploaded."

            }), 400

        uploaded_file = request.files[
            "file"
        ]

        filename = (
            uploaded_file.filename
            or "inspection_report"
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in {
            ".pdf",
            ".txt",
            ".html",
            ".htm",
            ".md"
        }:

            return jsonify({

                "error":
                    (
                        "Unsupported report format. "
                        "Please upload PDF, TXT, HTML, or Markdown."
                    )

            }), 400

        file_bytes = uploaded_file.read()

        if not file_bytes:

            return jsonify({

                "error":
                    "The uploaded inspection report is empty."

            }), 400

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        if extension == ".pdf":

            report_text = extract_pdf_text(
                file_bytes
            )

        else:

            report_text = file_bytes.decode(
                "utf-8",
                errors="ignore"
            )

            # Basic HTML cleanup
            report_text = re.sub(
                r"<br\s*/?>",
                "\n",
                report_text,
                flags=re.IGNORECASE
            )

            report_text = re.sub(
                r"<[^>]+>",
                " ",
                report_text
            )

        if not report_text.strip():

            return jsonify({

                "error":
                    (
                        "The report was opened successfully, "
                        "but no readable text could be extracted."
                    )

            }), 422

        # ----------------------------------------------------
        # Parse
        # ----------------------------------------------------

        report = parse_inspection_report_text(
            report_text
        )
        # FINAL FIX: remember the active uploaded inspection report
        ACTIVE_REPORT_STATE["report"] = report

        # ----------------------------------------------------
        # Report quality check
        # ----------------------------------------------------

        useful_fields = 0

        for key in [
            "standard",
            "operating_class",
            "decision",
            "defect",
            "severity",
            "ipc_clause"
        ]:

            if report.get(key):
                useful_fields += 1

        if useful_fields == 0:

            return jsonify({

                "error":
                    (
                        "The uploaded document does not appear "
                        "to be a ManufacturingRAG-QA inspection "
                        "report. No inspection fields could be "
                        "identified."
                    )

            }), 422

        print(
            "[Report] Inspection report uploaded:"
        )

        print(
            f"         File: {filename}"
        )

        print(
            f"         Defect: {report.get('defect')}"
        )

        print(
            f"         Class: {report.get('operating_class')}"
        )

        print(
            f"         Decision: {report.get('decision')}"
        )

        print(
            f"         Clause: {report.get('ipc_clause')}"
        )

        return jsonify({

            "status":
                "success",

            "message":
                "Inspection report analyzed successfully.",

            "filename":
                filename,

            "report":
                report
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                str(e),

            "message":
                (
                    "Failed to analyze the inspection report."
                )

        }), 500


# ============================================================
# RAG QUERY / CONVERSATIONAL STANDARDS ASSISTANT
# ============================================================



@app.route("/api/rag/query", methods=["POST"])

def query_standards():
    data = request.get_json(silent=True) or {}

    question = str(data.get("question", "")).strip()
    report = data.get("inspection_report")

    # Fall back to server-side active report
    if not isinstance(report, dict):
        cached = globals().get("_ACTIVE_REPORT_CACHE")
        if isinstance(cached, dict):
            report = cached.get("report", cached)

    q = question.lower()

    # =========================================================
    # DIRECT SOLDER BRIDGE CAUSE ANSWER
    # =========================================================
    if "solder bridge" in q or "solder bridging" in q:

        if any(word in q for word in [
            "cause", "causes", "why", "reason", "reasons"
        ]):

            report_causes = []

            if isinstance(report, dict):
                report_causes = (
                    report.get("root_causes")
                    or report.get("root_cause_hypotheses")
                    or []
                )

            if report_causes:
                answer = (
                    "**Possible causes of Solder Bridging:**\\n\\n"
                    + "\\n".join(
                        f"- {str(c).strip()}"
                        for c in report_causes
                        if str(c).strip()
                    )
                )
            else:
                answer = (
                    "**Possible causes of Solder Bridging include:**\\n\\n"
                    "- Excessive solder paste volume due to stencil aperture thickness or wear.\\n"
                    "- Improper stencil-to-board print alignment.\\n"
                    "- Incorrect reflow peak temperature or excessive solder paste slump.\\n"
                    "- Insufficient solder mask dam between fine-pitch IC leads.\\n\\n"
                    "**Process factors to check:** stencil condition and alignment, solder-paste deposition, reflow profile, and solder-mask spacing."
                )

            return jsonify({
                "answer": answer,
                "response": answer,
                "source": "inspection_report" if report_causes else "standards_knowledge",
                "report_context_used": bool(report),
                "intent": "causes"
            })

    # =========================================================
    # NORMAL RAG FALLBACK
    # =========================================================
    try:
        engine = get_or_create_rag_engine()

        if engine is None:
            return jsonify({
                "answer": "Standards RAG engine is currently unavailable.",
                "response": "Standards RAG engine is currently unavailable."
            }), 503

        result = engine.answer_standards_query(
            question,
            operating_class=data.get("operating_class", "Class 3")
        )

        return jsonify({
            "answer": result,
            "response": result
        })

    except Exception as e:
        return jsonify({
            "answer": f"Unable to answer standards query: {str(e)}",
            "response": f"Unable to answer standards query: {str(e)}"
        }), 500

@app.route(
    "/api/standards",
    methods=["GET"]
)
def list_standards():

    return jsonify(
        get_all_standards()
    )


# ============================================================
# AUDIT CERTIFICATE
# ============================================================

@app.route(
    "/api/audit/certificate",
    methods=["POST"]
)
def get_certificate_html():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        report = data.get(
            "report"
        )

        if not report:

            return jsonify({

                "error":
                    "Report data required"

            }), 400

        html = (
            audit_reporter.render_html_report(
                report
            )
        )

        return (
            html,
            200,
            {
                "Content-Type":
                    "text/html; charset=utf-8"
            }
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                str(e),

            "type":
                type(e).__name__

        }), 500


# ============================================================
# DOWNLOAD AUDIT CERTIFICATE AS PDF
# ============================================================

def _pdf_escape(value):
    """
    Escape text for a basic PDF text stream.
    Keeps this endpoint dependency-light.
    """
    value = str(value or "")
    value = value.replace("\\", "\\\\")
    value = value.replace("(", "\\(")
    value = value.replace(")", "\\)")
    return value


def _flatten_report_for_pdf(report, prefix=""):
    """
    Convert nested inspection report data into readable
    key/value lines.
    """
    lines = []

    if not isinstance(report, dict):
        return [str(report)]

    for key, value in report.items():

        label = str(key).replace("_", " ").title()

        if isinstance(value, dict):
            lines.append(f"{prefix}{label}:")
            lines.extend(
                _flatten_report_for_pdf(
                    value,
                    prefix + "  "
                )
            )

        elif isinstance(value, list):

            if not value:
                lines.append(
                    f"{prefix}{label}: None"
                )
                continue

            lines.append(
                f"{prefix}{label}:"
            )

            for item in value:

                if isinstance(item, dict):
                    lines.extend(
                        _flatten_report_for_pdf(
                            item,
                            prefix + "  "
                        )
                    )

                else:
                    lines.append(
                        f"{prefix}  - {item}"
                    )

        else:
            lines.append(
                f"{prefix}{label}: {value}"
            )

    return lines


def _build_basic_pdf(title, report):
    """
    Generate a valid text PDF without requiring reportlab.

    This is intentionally dependency-light so certificate
    download works even when reportlab is not installed.
    """

    raw_lines = [
        title,
        "",
        "ManufacturingRAG-QA",
        "Automated PCB Inspection Certificate",
        ""
    ]

    raw_lines.extend(
        _flatten_report_for_pdf(report)
    )

    # --------------------------------------------------------
    # Wrap long lines
    # --------------------------------------------------------

    wrapped = []

    for line in raw_lines:

        line = str(line)

        if not line:
            wrapped.append("")
            continue

        while len(line) > 92:

            cut = line.rfind(" ", 0, 92)

            if cut <= 0:
                cut = 92

            wrapped.append(
                line[:cut]
            )

            line = line[cut:].lstrip()

        wrapped.append(line)

    # --------------------------------------------------------
    # PDF pages
    # --------------------------------------------------------

    pages = []

    page_lines = []
    max_lines = 48

    for line in wrapped:

        if len(page_lines) >= max_lines:

            pages.append(page_lines)
            page_lines = []

        page_lines.append(line)

    if page_lines or not pages:
        pages.append(page_lines)

    # --------------------------------------------------------
    # Build PDF objects
    # --------------------------------------------------------

    objects = []

    # 1 Catalog
    objects.append(
        "<< /Type /Catalog /Pages 2 0 R >>"
    )

    # 2 Pages placeholder
    objects.append("")

    page_ids = []
    content_ids = []

    for page_lines in pages:

        page_obj_id = len(objects) + 1

        content_obj_id = page_obj_id + 1

        page_ids.append(page_obj_id)
        content_ids.append(content_obj_id)

        objects.append(
            f"<< /Type /Page "
            f"/Parent 2 0 R "
            f"/MediaBox [0 0 595 842] "
            f"/Resources << "
            f"/Font << /F1 {content_obj_id + 1} 0 R >> "
            f">> "
            f"/Contents {content_obj_id} 0 R >>"
        )

        commands = []

        commands.append(
            "BT"
        )

        commands.append(
            "/F1 10 Tf"
        )

        commands.append(
            "40 805 Td"
        )

        first = True

        for line in page_lines:

            if first:
                first = False
            else:
                commands.append(
                    "0 -15 Td"
                )

            commands.append(
                f"({_pdf_escape(line)}) Tj"
            )

        commands.append(
            "ET"
        )

        content = "\n".join(commands)

        objects.append(
            f"<< /Length {len(content.encode('latin-1', errors='replace'))} >>\n"
            f"stream\n"
            f"{content}\n"
            f"endstream"
        )

        # Font object
        objects.append(
            "<< /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >>"
        )

    # Fix Pages object
    kids = " ".join(
        f"{pid} 0 R"
        for pid in page_ids
    )

    objects[1] = (
        f"<< /Type /Pages "
        f"/Kids [{kids}] "
        f"/Count {len(page_ids)} >>"
    )

    # --------------------------------------------------------
    # Serialize
    # --------------------------------------------------------

    pdf = bytearray()

    pdf.extend(
        b"%PDF-1.4\n"
    )

    offsets = [0]

    for index, obj in enumerate(objects, start=1):

        offsets.append(
            len(pdf)
        )

        pdf.extend(
            f"{index} 0 obj\n".encode(
                "latin-1"
            )
        )

        pdf.extend(
            obj.encode(
                "latin-1",
                errors="replace"
            )
        )

        pdf.extend(
            b"\nendobj\n"
        )

    xref_position = len(pdf)

    pdf.extend(
        f"xref\n0 {len(objects) + 1}\n".encode(
            "latin-1"
        )
    )

    pdf.extend(
        b"0000000000 65535 f \n"
    )

    for offset in offsets[1:]:

        pdf.extend(
            f"{offset:010d} 00000 n \n".encode(
                "latin-1"
            )
        )

    pdf.extend(
        (
            f"trailer\n"
            f"<< /Size {len(objects) + 1} "
            f"/Root 1 0 R >>\n"
            f"startxref\n"
            f"{xref_position}\n"
            f"%%EOF"
        ).encode(
            "latin-1"
        )
    )

    return bytes(pdf)


@app.route(
    "/api/audit/certificate/pdf",
    methods=["POST"]
)
def download_certificate_pdf():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        report = data.get("report")

        if not report:
            return jsonify({
                "error":
                    "Report data required"
            }), 400

        report_id = (
            report.get("report_id")
            or report.get("id")
            or "inspection"
        )

        filename = (
            "ManufacturingRAG-QA_"
            f"Certificate_{report_id}.pdf"
        )

        pdf_bytes = _build_basic_pdf(
            "ManufacturingRAG-QA Inspection Certificate",
            report
        )

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500



# ============================================================
# ANALYTICS
# ============================================================

@app.route(
    "/api/analytics",
    methods=["GET"]
)
def get_analytics():

    return jsonify({

        "total_inspected":
            1420,

        "first_pass_yield":
            94.2,

        "critical_defects":
            18,

        "reworked_units":
            64,

        "defect_pareto": [

            {
                "defect":
                    "Solder Bridge",

                "count":
                    28,

                "percentage":
                    34.1
            },

            {
                "defect":
                    "Tombstoning",

                "count":
                    22,

                "percentage":
                    26.8
            },

            {
                "defect":
                    "Component Misaligned",

                "count":
                    16,

                "percentage":
                    19.5
            },

            {
                "defect":
                    "Missing Component",

                "count":
                    10,

                "percentage":
                    12.2
            },

            {
                "defect":
                    "Solder Ball",

                "count":
                    6,

                "percentage":
                    7.4
            }
        ]
    })


# ============================================================
# SERVER START
# ============================================================


# ===== FINAL ACTIVE REPORT INTERCEPTOR =====

_ACTIVE_REPORT_CACHE = {
    "report": None
}


@app.after_request
def _capture_uploaded_inspection_report(response):
    """
    Store the structured inspection report returned by
    /api/rag/report so subsequent chat questions can use it.
    """
    try:
        if request.path == "/api/rag/report":
            payload = response.get_json(silent=True)

            if isinstance(payload, dict):
                report = payload.get("report")

                if isinstance(report, dict) and report:
                    _ACTIVE_REPORT_CACHE["report"] = report

                    print(
                        "[REPORT CACHE] Active report stored:",
                        report.get("defect"),
                        report.get("standard"),
                        report.get("ipc_clause")
                    )

    except Exception as exc:
        print(
            "[REPORT CACHE] Error:",
            repr(exc)
        )

    return response


def _report_chat_intent(question):
    """
    Determine whether a question should be answered from the
    active inspection report.
    """

    q = str(question or "").strip().lower()

    # Acceptance / rejection
    if any(x in q for x in [
        "not acceptable",
        "not accepted",
        "unacceptable",
        "why is it rejected",
        "why is this rejected",
        "why rejected",
        "why does it fail",
        "why is it failing",
        "pass or fail",
        "acceptable",
        "accepted",
        "rejected",
        "reject",
        "compliant"
    ]):
        return "report_acceptance"

    # Causes / root causes
    if any(x in q for x in [
        "possible causes",
        "possible cause",
        "root cause",
        "root causes",
        "what causes",
        "what caused",
        "causes of",
        "cause of",
        "why did this happen",
        "why does this happen",
        "why is this happening",
        "why did the defect occur",
        "reason for this defect",
        "reasons for this defect"
    ]):
        return "report_causes"

    # Corrective action / rework
    if any(x in q for x in [
        "how should this defect be corrected",
        "how should this be corrected",
        "how can this be corrected",
        "how to correct",
        "how do i correct",
        "how to fix",
        "how do i fix",
        "how can i fix",
        "how to repair",
        "how should it be repaired",
        "repair this",
        "rework",
        "corrective action",
        "correction procedure"
    ]):
        return "report_rework"

    # Standard / clause
    if any(x in q for x in [
        "which ipc requirement",
        "which ipc clause",
        "what ipc requirement",
        "what ipc clause",
        "which standard",
        "what standard",
        "which requirement",
        "what requirement does this violate",
        "what clause does this violate",
        "what clause applies"
    ]):
        return "report_standard"

    # Risk / impact
    if any(x in q for x in [
        "what is the risk",
        "what are the risks",
        "risk of this defect",
        "risks of this defect",
        "what impact",
        "what is the impact",
        "electrical impact",
        "why is this dangerous",
        "why is this a problem",
        "what could happen"
    ]):
        return "report_risk"

    # Product class
    if any(x in q for x in [
        "class 1",
        "class 2",
        "class 3",
        "product class",
        "would this be acceptable",
        "would this pass"
    ]):
        return "report_acceptance"

    # General questions explicitly referring to the active defect
    if any(x in q for x in [
        "this defect",
        "this issue",
        "this finding",
        "this solder bridge",
        "this solder bridging",
        "the defect",
        "the finding",
        "the solder bridge",
        "the solder bridging"
    ]):
        return "report_general"

    return None


def _answer_from_active_report(question, report):
    """
    Generate a report-grounded answer WITHOUT calling the generic
    Standards RAG engine.
    """

    intent = _report_chat_intent(question)

    defect = (
        report.get("defect")
        or report.get("reported_finding")
        or report.get("protonet_category")
        or "the detected defect"
    )

    defect_display = str(defect).replace("_", " ")
    defect_display = " ".join(
        word.capitalize()
        for word in defect_display.split()
    )

    standard = (
        report.get("standard")
        or "the reported standard"
    )

    clause = (
        report.get("ipc_clause")
        or report.get("clause")
        or ""
    )

    operating_class = (
        report.get("operating_class")
        or "the configured operating class"
    )

    severity = (
        report.get("severity")
        or ""
    )

    electrical_impact = (
        report.get("electrical_impact")
        or ""
    )

    required_action = (
        report.get("required_action")
        or report.get("action")
        or ""
    )

    class_requirement = (
        report.get("class_requirement")
        or report.get("acceptance_level")
        or ""
    )

    causes = report.get("root_causes") or []

    if isinstance(causes, str):
        causes = [
            x.strip()
            for x in re.split(r"[,;\n]+", causes)
            if x.strip()
        ]

    rework = (
        report.get("rework_procedure")
        or report.get("corrective_action")
        or report.get("repair_procedure")
        or ""
    )

    # --------------------------------------------------------
    # ACCEPTANCE
    # --------------------------------------------------------
    if intent == "report_acceptance":

        answer = (
            f"**{defect_display} is not acceptable because the "
            f"active inspection report identifies it as a "
            f"NON-CONFORMANCE.**"
        )

        if standard and clause:
            answer += (
                f"\n\nUnder **{standard}, Clause {clause}**, the "
                f"reported condition is treated as a defect for "
                f"**{operating_class}**."
            )

        if class_requirement:
            answer += (
                f"\n\n**Acceptance requirement:** "
                f"{class_requirement}"
            )

        if electrical_impact:
            answer += (
                f"\n\n**Why it matters:** "
                f"{electrical_impact}"
            )

        if severity:
            answer += (
                f"\n\n**Severity:** {severity}"
            )

        if required_action:
            answer += (
                f"\n\n**Required action:** "
                f"{required_action}"
            )

        return answer

    # --------------------------------------------------------
    # CAUSES
    # --------------------------------------------------------
    if intent == "report_causes":

        if causes:
            answer = (
                f"Based on the active inspection report, the "
                f"possible causes of **{defect_display}** are:"
            )

            for cause in causes:
                answer += f"\n\n- {cause}"

            return answer

        return (
            f"The active report identifies **{defect_display}**, "
            f"but it does not contain recorded root-cause "
            f"hypotheses."
        )

    # --------------------------------------------------------
    # REWORK
    # --------------------------------------------------------
    if intent == "report_rework":

        if rework:
            answer = (
                f"For the reported **{defect_display}**, the "
                f"inspection report specifies the following "
                f"corrective/rework procedure:\n\n"
                f"**{rework}**"
            )

            if required_action:
                answer += (
                    f"\n\n**Required action:** "
                    f"{required_action}"
                )

            return answer

        if required_action:
            return (
                f"The report does not contain a detailed rework "
                f"procedure, but the required action is:\n\n"
                f"**{required_action}**"
            )

        return (
            f"The active report does not contain a specific "
            f"rework procedure for {defect_display}."
        )

    # --------------------------------------------------------
    # STANDARD
    # --------------------------------------------------------
    if intent == "report_standard":

        if standard and clause:
            return (
                f"The active inspection report grounds this "
                f"finding in **{standard}, Clause {clause}**.\n\n"
                f"The reported defect is **{defect_display}** "
                f"for **{operating_class}**."
            )

        if standard:
            return (
                f"The active inspection report identifies "
                f"**{standard}** as the applicable standard for "
                f"the reported {defect_display}."
            )

        return (
            f"The active report identifies {defect_display}, "
            f"but no applicable standard was recorded."
        )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------
    if intent == "report_risk":

        if electrical_impact:
            return (
                f"The reported risk of **{defect_display}** is:\n\n"
                f"**{electrical_impact}**"
            )

        return (
            f"The active report identifies {defect_display} as "
            f"a {severity or 'non-conforming'} condition, but "
            f"does not contain a specific electrical-impact "
            f"description."
        )

    # --------------------------------------------------------
    # GENERAL ACTIVE-REPORT QUESTION
    # --------------------------------------------------------
    return (
        f"The active inspection report identifies "
        f"**{defect_display}** as a **{report.get('decision', 'reported finding')}** "
        f"under **{standard}** for **{operating_class}**."
    )


@app.before_request
def _intercept_active_report_questions():
    """
    Intercept ALL supported report-grounded questions before
    the existing /api/rag/query implementation.
    """

    if request.path != "/api/rag/query":
        return None

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return None

        question = str(
            payload.get("question") or ""
        ).strip()

        if not question:
            return None

        # Prefer the report sent by the frontend.
        report = payload.get("inspection_report")

        # Fall back to the server-side cached report.
        if not isinstance(report, dict) or not report:
            report = _ACTIVE_REPORT_CACHE.get("report")

        if not isinstance(report, dict) or not report:
            return None

        intent = _report_chat_intent(question)

        # Only intercept questions that clearly concern the
        # active inspection report.
        if intent is None:
            return None

        answer = _answer_from_active_report(
            question,
            report
        )

        return jsonify({
            "answer": answer,
            "intent": intent,
            "report_used": True,
            "rag_used": False,
            "standard": report.get("standard"),
            "clause": (
                report.get("ipc_clause")
                or report.get("clause")
            ),
            "defect": report.get("defect"),
            "operating_class": report.get(
                "operating_class"
            ),
            "severity": report.get("severity"),
            "report_summary": report,
            "source": "active_inspection_report"
        })

    except Exception as exc:

        print(
            "[REPORT CHAT INTERCEPTOR ERROR]",
            repr(exc)
        )

        return jsonify({
            "answer": (
                "The active inspection report could not "
                "be processed."
            ),
            "report_used": True,
            "rag_used": False,
            "error": str(exc)
        }), 500


# ===== END FINAL ACTIVE REPORT INTERCEPTOR =====


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "\n======================================================="
    )

    print(
        " ManufacturingRAG-QA Server"
    )

    print(
        f" Running at http://localhost:{port}"
    )

    print(
        " RAG: Lazy initialization enabled"
    )

    print(
        " RAG cache: D: project hf_cache"
    )

    print(
        " RAG model: all-MiniLM-L6-v2"
    )

    print(
        " Inspection Report Analysis: ENABLED"
    )

    print(
        " Report-aware Standards Chat: ENABLED"
    )

    print(
        "=======================================================\n"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )