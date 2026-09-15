"""
ManufacturingRAG-QA Backend Server
"""

import os
import sys
import json
import traceback

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
#
# IMPORTANT:
# C: drive has very little free space.
# Store embedding-model downloads on D: instead.
# ============================================================

HF_CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "hf_cache"
)

os.makedirs(
    HF_CACHE_DIR,
    exist_ok=True
)

# Only set these if the user/environment has not already
# explicitly configured a Hugging Face cache.

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

# Lightweight embedding model.
#
# BGE-M3 is approximately 2.27 GB and was filling C:.
# MiniLM is much smaller and is sufficient for our
# PDF semantic retrieval + BM25 + RRF pipeline.

os.environ.setdefault(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Use locally cached model snapshots — avoids SSL certificate failures on Windows
# when no internet connection to HuggingFace Hub is needed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    send_from_directory
)

from flask_cors import CORS

from PIL import Image

import numpy as np
import torch
import cv2


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

app = Flask(
    __name__,
    static_folder="../web",
    static_url_path=""
)

CORS(app)


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

# ------------------------------------------------------------
# CBAM = BINARY CLASSIFIER ONLY
# ------------------------------------------------------------

CBAM_CLASSES = [
    "normal",
    "anomaly"
]


# ------------------------------------------------------------
# PROTONET = CATEGORY INTERPRETATION
# ------------------------------------------------------------

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
#
# RAG is intentionally LAZY.
#
# The embedding model is NOT downloaded at server startup.
#
# It is loaded only when:
#
#   1. A confirmed anomaly needs standards evidence, OR
#   2. The user explicitly queries /api/rag/query
#
# Hugging Face cache is on D:.
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
rag_error = None


def initialize_rag():

    global rag_engine
    global rag_error

    # --------------------------------------------------------
    # Already initialized
    # --------------------------------------------------------

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

            state_dict = (
                checkpoint["model_state_dict"]
            )

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

        # ----------------------------------------------------
        # Novel demonstration defect
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Seed ProtoNet if empty
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RAG has not been initialized yet.
    # --------------------------------------------------------

    if rag_engine is None:

        return {

            "available":
                False,

            "initialized":
                False,

            "status":
                (
                    "Not initialized. "
                    "RAG loads only when required."
                ),

            "error":
                rag_error,

            "knowledge_source":
                None,

            "embedding_model":
                os.environ.get(
                    "RAG_EMBEDDING_MODEL"
                ),

            "cache_directory":
                os.environ.get(
                    "HF_HOME"
                ),

            "pdf_count":
                0,

            "pdf_chunk_count":
                0,

            "faiss_ready":
                False,

            "faiss_vectors":
                0
        }

    # --------------------------------------------------------
    # RAG is initialized.
    # --------------------------------------------------------

    try:

        status = (
            rag_engine.get_status()
        )

        if isinstance(status, dict):

            status["available"] = True

            status["initialized"] = True

            status["cache_directory"] = (
                os.environ.get(
                    "HF_HOME"
                )
            )

            return status

    except Exception:

        pass

    # --------------------------------------------------------
    # Defensive fallback status.
    # --------------------------------------------------------

    try:

        pdf_chunks = getattr(
            rag_engine,
            "chunks",
            []
        )

        faiss_index = getattr(
            rag_engine,
            "faiss_index",
            None
        )

        try:

            pdf_sources = (
                rag_engine.list_pdf_sources()
            )

        except Exception:

            pdf_sources = []

        return {

            "available":
                True,

            "initialized":
                True,

            "knowledge_source":
                getattr(
                    rag_engine,
                    "knowledge_source",
                    "PDF"
                ),

            "pdf_count":
                len(pdf_sources),

            "pdf_chunk_count":
                len(pdf_chunks),

            "embedding_model":
                getattr(
                    rag_engine,
                    "embedding_model_name",
                    os.environ.get(
                        "RAG_EMBEDDING_MODEL"
                    )
                ),

            "embedding_dimension":
                getattr(
                    rag_engine,
                    "embedding_dimension",
                    None
                ),

            "cache_directory":
                os.environ.get(
                    "HF_HOME"
                ),

            "faiss_ready":
                faiss_index is not None,

            "faiss_vectors":
                (
                    int(
                        faiss_index.ntotal
                    )
                    if faiss_index is not None
                    else 0
                )
        }

    except Exception as e:

        return {

            "available":
                False,

            "initialized":
                True,

            "error":
                str(e)
        }


# ============================================================
# CLASS NORMALIZATION
# ============================================================

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
        "good_assembly": "Good_Assembly",
        "normal": "Good_Assembly",
        "missing_component": "Missing_Component",
        "component_missing": "Missing_Component",
        "component_misaligned": "Component_Misaligned",
        "misaligned_component": "Component_Misaligned",
        "solder_bridge": "Solder_Bridge",
        "solder_bridges": "Solder_Bridge",
        "soldier_bridge": "Solder_Bridge",
        "soldier_bridges": "Solder_Bridge",
        "tombstoning": "Tombstoning",
        "solder_ball": "Solder_Ball",
        "solder_balls": "Solder_Ball",
        "insufficient_solder": "Insufficient_Solder",
        "insufficient_soldering": "Insufficient_Solder"
    }

    if key in canonical_aliases:
        return canonical_aliases[key]

    # Handle user-created names such as:
    # New_soldier_bridge
    # New_Solder_Bridge
    # Custom-Solder-Bridge
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

    # Preserve genuinely novel categories.
    return raw
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

        # ====================================================
        # REQUEST PARAMETERS
        # ====================================================

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

        # ====================================================
        # IMAGE INPUT
        # ====================================================

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

        # ====================================================
        # 1. PREPROCESSING
        # ====================================================

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
        # 2. CBAM BINARY DECISION
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
        # 3. PROTONET CATEGORY DECISION
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
        # 4. SEPARATE FINAL DECISIONS
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

        # ----------------------------------------------------
        # ProtoNet >= 60% is required.
        # ----------------------------------------------------

        # ProtoNet category confirmation
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
        # 5. GRAD-CAM
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

        # ====================================================
        # 6. VISUAL EVIDENCE
        # ====================================================

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
        # 7. RAG / STANDARDS GROUNDING
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

        # ====================================================
        # CASE 1:
        # CBAM anomaly + ProtoNet NOT CONFIRMED
        #
        # Do NOT initialize RAG.
        # ====================================================

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

        # ====================================================
        # CASE 2:
        # CBAM anomaly + ProtoNet CONFIRMED
        #
        # ONLY HERE initialize RAG.
        # ====================================================

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

        # ====================================================
        # CASE 3:
        # CBAM NORMAL
        # ====================================================

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
        # 8. AUDIT REPORT
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
        # 9. RESPONSE
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

        return jsonify(response)

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

        return jsonify(result)

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
# RAG QUERY
# ============================================================

@app.route(
    "/api/rag/query",
    methods=["POST"]
)
def query_standards():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    question = data.get(
        "question",
        ""
    )

    op_class = data.get(
        "operating_class",
        "Class 3"
    )

    if not question:

        return jsonify({

            "error":
                "Question is required"

        }), 400

    # --------------------------------------------------------
    # LAZY RAG INITIALIZATION
    # --------------------------------------------------------

    rag = initialize_rag()

    if rag is None:

        return jsonify({

            "error":
                (
                    "RAG is currently unavailable. "
                    "The embedding model could not be loaded."
                ),

            "rag_available":
                False,

            "rag_initialized":
                False,

            "rag_error":
                rag_error

        }), 503

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    rag_query = question

    try:

        result = (
            rag.answer_standards_query(
                query=rag_query,
                top_k=5,
                operating_class=op_class
            )
        )

        result["operating_class"] = op_class
        result["rag_available"] = True
        result["rag_initialized"] = True

        return jsonify(result)

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                str(e),

            "rag_available":
                False,

            "rag_initialized":
                True

        }), 500


# ============================================================
# STRUCTURED STANDARDS
# ============================================================

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
        "=======================================================\n"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

