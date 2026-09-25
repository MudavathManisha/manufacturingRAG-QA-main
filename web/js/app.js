/* ============================================================
   ManufacturingRAG-QA
   Frontend Controller
   Matches current server/app.py
   ============================================================ */

"use strict";

/* ------------------------------------------------------------
   API ROUTES
   ------------------------------------------------------------ */

const API = {
    status: "/api/status",
    inspect: "/api/inspect",

    // Continual learning
    registerPrototype: "/api/few-shot/register",
    prototypes: "/api/few-shot/prototypes",

    // Standards / RAG
    standards: "/api/standards",
    ragQuery: "/api/rag/query",

    // Audit certificate
    certificate: "/api/audit/certificate"
};


/* ------------------------------------------------------------
   GLOBAL STATE
   ------------------------------------------------------------ */

let currentFile = null;
let currentInspection = null;
let currentReport = null;
let inspectionReportContext = null;

let originalImage = null;
let overlayImage = null;
let heatmapImage = null;

let canvasContext = null;

let reportHistory = [];

const REPORT_STORAGE_KEY = "manufacturingrag_audit_reports";

// Continual-learning global state & persistence
const LEARNED_DEFECTS_STORAGE_KEY = "manufacturingrag_learned_defects_history";
let prototypeNames = [];          // known defect class names (loaded from backend)
let learnedDefectsHistory = [];   // persisted newly auto-learned defect objects

/* Restore learned defects from localStorage */
function restoreLearnedDefects() {
    try {
        const stored = localStorage.getItem(LEARNED_DEFECTS_STORAGE_KEY);
        if (stored) {
            learnedDefectsHistory = JSON.parse(stored);
        } else {
            learnedDefectsHistory = [];
        }
    } catch (e) {
        console.error("Failed to parse learned defects from storage:", e);
        learnedDefectsHistory = [];
    }

    const container = document.getElementById("newlyLearnedContainer");
    const emptyEl  = document.getElementById("newlyLearnedEmpty");
    if (!container) return;

    if (learnedDefectsHistory.length > 0) {
        if (emptyEl) emptyEl.style.display = "none";
        // Clear existing dynamically generated cards
        container.querySelectorAll(".learned-defect-card").forEach(el => el.remove());
        // Render in reverse order so latest is on top
        learnedDefectsHistory.slice().reverse().forEach(item => {
            renderLearnedCardElement(item, false);
        });
    } else {
        if (emptyEl) emptyEl.style.display = "block";
    }

    updateLearnedDefectsCounter();
}

function updateLearnedDefectsCounter() {
    const countEl = document.getElementById("clNewlyLearnedCount");
    if (countEl) countEl.textContent = learnedDefectsHistory.length;
}

/* Clear learned defects history */
function clearLearnedDefectsHistory() {
    learnedDefectsHistory = [];
    localStorage.removeItem(LEARNED_DEFECTS_STORAGE_KEY);
    const container = document.getElementById("newlyLearnedContainer");
    if (container) {
        container.querySelectorAll(".learned-defect-card").forEach(el => el.remove());
    }
    const emptyEl = document.getElementById("newlyLearnedEmpty");
    if (emptyEl) emptyEl.style.display = "block";
    updateLearnedDefectsCounter();
    showToast("Learned defect history cleared.", "info");
}

/* Render a single card element */
function renderLearnedCardElement(defectItem, insertTop = true) {
    const container = document.getElementById("newlyLearnedContainer");
    const emptyEl  = document.getElementById("newlyLearnedEmpty");
    if (!container) return;
    if (emptyEl) emptyEl.style.display = "none";

    const standardHtml = (defectItem.standard && defectItem.standard !== "—" && defectItem.standard !== "None") 
        ? `<span><b>Standard:</b> ${defectItem.standard}</span>` 
        : "";

    const card = document.createElement("div");
    card.className = "prototype-card learned-defect-card";
    card.style.cssText = "border-left:4px solid #22c55e; margin-bottom:0.75rem;";
    card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
            <div>
                <strong style="font-size:1rem;">${defectItem.name}</strong>
                <span style="margin-left:0.5rem;background:#22c55e;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem;">Auto-Learned</span>
            </div>
            <span style="font-size:0.75rem;color:var(--text-muted);">Detected at ${defectItem.timestamp || "Recently"}</span>
        </div>
        <div style="display:flex;gap:1.5rem;margin-top:0.5rem;font-size:0.85rem;flex-wrap:wrap;">
            <span><b>Confidence:</b> ${(defectItem.confidence * 100).toFixed(1)}%</span>
            ${standardHtml}
        </div>
        <p style="margin:0.5rem 0 0;font-size:0.82rem;color:var(--text-muted);">
            This defect pattern has been committed to prototype memory. Future inspections of
            the same defect pattern will be recognized without retraining.
        </p>`;

    if (insertTop && container.firstChild) {
        container.insertBefore(card, container.firstChild);
    } else {
        container.appendChild(card);
    }
}

/* Load known defect prototypes from the backend */
async function loadPrototypes() {
    try {
        const response = await fetch(API.prototypes);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (Array.isArray(data.prototypes)) {
            prototypeNames = data.prototypes.map(p => p.name || p.class_name || p);
        } else if (Array.isArray(data)) {
            prototypeNames = data.map(p => p.name || p.class_name || p);
        } else {
            prototypeNames = [];
        }
        // Update the count badge in Tab 2
        const countEl = document.getElementById("clActivePrototypesCount");
        if (countEl) countEl.textContent = prototypeNames.length;
        const badgeEl = document.getElementById("clMemoryBadge");
        if (badgeEl) badgeEl.textContent = `${prototypeNames.length} Prototypes`;
    } catch (e) {
        console.error("Error loading prototypes:", e);
    }
}

/* Append a newly learned defect card to the Tab 2 panel and persist it */
function addLearnedDefectCard(defectName, confidence, reportData) {
    const severity  = reportData?.audit_finding?.severity || reportData?.audit_finding?.audit_severity || "Unknown";
    const standard  = reportData?.audit_finding?.applicable_standard || "—";
    const timestamp = new Date().toLocaleTimeString();

    const defectItem = {
        id: Date.now(),
        name: defectName,
        confidence: confidence,
        severity: severity,
        standard: standard,
        timestamp: timestamp
    };

    // Add to history and persist
    learnedDefectsHistory.push(defectItem);
    try {
        localStorage.setItem(LEARNED_DEFECTS_STORAGE_KEY, JSON.stringify(learnedDefectsHistory));
    } catch (e) {
        console.error("Failed to save to localStorage:", e);
    }

    renderLearnedCardElement(defectItem, true);
    updateLearnedDefectsCounter();
}

/* Auto-register unknown/low-confidence defect and populate Tab 2 */
async function handleUnknownDefect(fallbackClass, fallbackConf, reportData) {
    const protoPrediction = reportData?.prediction?.protonet_prediction || {};
    let predictedClass = protoPrediction.predicted_class || fallbackClass;
    const confidence = protoPrediction.confidence !== undefined ? protoPrediction.confidence : fallbackConf;

    // Normal PCBs don't need continual learning
    const pLow = (predictedClass || "").toLowerCase().trim();
    const cbamLow = (reportData?.prediction?.class_name || "").toLowerCase().trim();
    if (pLow === "good_assembly" || pLow === "normal" || pLow === "good" || pLow.includes("normal")) return;
    if (cbamLow === "normal" || cbamLow === "good_assembly" || cbamLow === "good") return;

    // High confidence indicates an already recognized known defect
    if (confidence >= 0.6) return;


    // Normalize helper for fuzzy matching
    const norm = str => (str || "").toLowerCase().replace(/[^a-z0-9]/g, "");

    // Generate a distinct and descriptive name for this new defect category
    const patternSeq = learnedDefectsHistory.length + 1;
    let registeredName = "";

    // Clean up base candidate name
    let cleanBase = (predictedClass || "Anomaly").replace(/_/g, " ").trim();
    if (cleanBase.toLowerCase().startsWith("new ")) {
        cleanBase = cleanBase.substring(4).trim();
    }

    // Check if this class is already known in pre-trained memory or past learned defects
    const isAlreadyKnown = prototypeNames.some(p => norm(p) === norm(predictedClass)) ||
                          learnedDefectsHistory.some(d => norm(d.name) === norm(predictedClass)) ||
                          norm(predictedClass) === "anomaly";

    if (isAlreadyKnown) {
        // Because the model had low confidence, it's NOT the exact known class, but a novel defect pattern!
        if (norm(cleanBase) === "anomaly" || norm(cleanBase) === "unknown") {
            registeredName = `Novel PCB Defect (Pattern #${patternSeq})`;
        } else {
            registeredName = `Novel ${cleanBase} (Pattern #${patternSeq})`;
        }
    } else {
        registeredName = cleanBase;
    }

    // Check if this exact name was already learned in this storage
    if (learnedDefectsHistory.some(d => norm(d.name) === norm(registeredName))) {
        showToast(`ℹ️ Defect pattern "${registeredName}" is already recorded.`, "info");
        return;
    }

    // Show prominent in-page toast
    showToast(`🔍 Novel defect pattern detected: "${registeredName}" (${(confidence * 100).toFixed(0)}% conf). Auto-registering…`, "info");

    // Browser notification (if permission granted)
    if ("Notification" in window && Notification.permission === "granted") {
        new Notification("ManufacturingRAG-QA — New Defect Detected", {
            body: `"${registeredName}" auto-registered into prototype memory.`
        });
    }

    try {
        const formData = new FormData();
        formData.append("defect_name", registeredName);
        formData.append("description", "Automatically learned defect pattern from inspection.");
        formData.append("ipc_standard_ref", "");
        formData.append("severity_baseline", "");
        
        // Attach support image
        if (currentFile) {
            formData.append("images", currentFile);
        }

        const response = await fetch(API.registerPrototype, { method: "POST", body: formData });
        const result   = await response.json();
        if (!response.ok) throw new Error(result.error || "Registration failed");

        // Refresh prototype list
        await loadPrototypes();
        if (!prototypeNames.includes(registeredName)) {
            prototypeNames.push(registeredName);
        }

        showToast(`✅ "${registeredName}" added to prototype memory. Check Tab 2.`, "success");
        addLearnedDefectCard(registeredName, confidence, reportData);

    } catch (e) {
        console.error("Auto-registration error:", e);
        showToast(`⚠️ Stored "${registeredName}" in session memory.`, "warning");
        addLearnedDefectCard(registeredName, confidence, reportData);
    }
}





/* ------------------------------------------------------------
   DOM HELPERS
   ------------------------------------------------------------ */

function $(id) {
    return document.getElementById(id);
}

function showElement(id) {
    const el = $(id);
    if (el) {
        el.style.display = "";
    }
}

function hideElement(id) {
    const el = $(id);
    if (el) {
        el.style.display = "none";
    }
}

function setText(id, value) {
    const el = $(id);
    if (el) {
        el.textContent = value ?? "";
    }
}

function escapeHtml(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* ------------------------------------------------------------
   TOAST
   ------------------------------------------------------------ */

function showToast(message, type = "info") {
    const container = $("toastContainer");

    if (!container) {
        console.log(`[${type}] ${message}`);
        return;
    }

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("toast-hide");

        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3500);
}


/* ------------------------------------------------------------
   LOADING
   ------------------------------------------------------------ */

function showLoading(message = "Processing...") {
    const overlay = $("loadingOverlay");

    if (!overlay) {
        return;
    }

    setText("loadingText", message);
    overlay.style.display = "flex";
}

function hideLoading() {
    const overlay = $("loadingOverlay");

    if (overlay) {
        overlay.style.display = "none";
    }
}


/* ------------------------------------------------------------
   NAVIGATION
   ------------------------------------------------------------ */

function initNavigation() {
    const navButtons = document.querySelectorAll("[data-tab]");

    navButtons.forEach(button => {
        button.addEventListener("click", () => {
            const tabName = button.dataset.tab; const sectionId = `tab-${tabName}`;

            if (!sectionId) {
                return;
            }

            document.querySelectorAll(".page-section").forEach(section => {
                section.classList.remove("active");
            });

            const target = $(sectionId);

            if (target) {
                target.classList.add("active");
            }

            navButtons.forEach(btn => {
                btn.classList.remove("active");
            });

            button.classList.add("active");

            if (sectionId === "tab-fewshot") {
                loadPrototypes();
            }

            if (sectionId === "tab-analytics") {
                loadReports();
            }
        });
    });
}


/* ------------------------------------------------------------
   SYSTEM STATUS
   ------------------------------------------------------------ */

async function loadSystemStatus() {
    try {
        const response = await fetch(API.status);

        if (!response.ok) {
            throw new Error(`Status request failed: ${response.status}`);
        }

        const data = await response.json();

        updateSystemStatus(data);

    } catch (error) {
        console.error("System status error:", error);

        const statusElement = $("systemStatusText");

        if (statusElement) {
            statusElement.textContent = "System Offline";
            statusElement.classList.add("status-error");
        }
    }
}

function updateSystemStatus(data) {
    const statusElement = $("systemStatusText");

    if (!statusElement) {
        return;
    }

    const checkpointLoaded = data.checkpoint_loaded === true || data.model_loaded === true || (data.model && data.model.checkpoint_loaded === true);

    if (checkpointLoaded) {
        statusElement.textContent = "System Ready";
        statusElement.classList.remove("status-error");
        statusElement.classList.add("status-success");
    } else {
        statusElement.textContent = "Model Not Loaded";
        statusElement.classList.remove("status-success");
        statusElement.classList.add("status-error");
    }
}


/* ------------------------------------------------------------
   FILE UPLOAD
   ------------------------------------------------------------ */

function initFileUpload() {
    const fileInput = $("fileInput");
    const dropzone = $("imageDropzone");

    if (!fileInput) {
        return;
    }

    fileInput.addEventListener("change", event => {
        const files = event.target.files;

        if (!files || !files.length) {
            return;
        }

        handleInspectionFile(files[0]);
    });

    if (!dropzone) {
        return;
    }

    dropzone.addEventListener("click", event => {
        if (event.target !== fileInput) {
            fileInput.click();
        }
    });

    dropzone.addEventListener("dragover", event => {
        event.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", event => {
        event.preventDefault();

        dropzone.classList.remove("drag-over");

        const files = event.dataTransfer.files;

        if (!files || !files.length) {
            return;
        }

        handleInspectionFile(files[0]);
    });
}

function handleInspectionFile(file) {
    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {
        showToast("Please select an image file.", "error");
        return;
    }

    currentFile = file;

    setText("selectedFileName", file.name);

    const reader = new FileReader();

    reader.onload = event => {
        displayPreview(event.target.result);
    };

    reader.readAsDataURL(file);

    const inspectButton = $("inspectButton");

    if (inspectButton) {
        inspectButton.disabled = false;
    }
}


/* ------------------------------------------------------------
   IMAGE PREVIEW
   ------------------------------------------------------------ */

function displayPreview(dataUrl) {
    const canvas = $("inspectionCanvas");
    const placeholder = $("imagePlaceholder");

    if (!canvas) {
        return;
    }

    const img = new Image();

    img.onload = () => {
        originalImage = img;

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        canvasContext = canvas.getContext("2d");

        canvasContext.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        canvasContext.drawImage(
            img,
            0,
            0,
            canvas.width,
            canvas.height
        );

        if (placeholder) {
            placeholder.style.display = "none";
        }

        canvas.style.display = "block";
    };

    img.src = dataUrl;
}


/* ------------------------------------------------------------
   INSPECTION
   ------------------------------------------------------------ */

function initInspection() {
    const button = $("inspectButton");

    if (!button) {
        return;
    }

    button.addEventListener("click", inspectCurrentFile);
}

async function inspectCurrentFile() {
    if (!currentFile) {
        showToast("Please select a PCB image first.", "error");
        return;
    }

    const formData = new FormData();

    formData.append("file", currentFile);

    const operatingClass =
        $("operatingClass")?.value || "Class 3";

    formData.append("operating_class", operatingClass);
    formData.append("mode", "hybrid");

    showLoading("Analyzing PCB image...");

    try {
        const response = await fetch(API.inspect, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                `Inspection failed: ${response.status}`
            );
        }

        currentInspection = data;
        currentReport = data.audit_report || null;

        renderInspectionResult(data);

        if (currentReport) {
            saveReportToHistory(currentReport);
        }

        showToast(
            "Inspection completed successfully.",
            "success"
        );

    } catch (error) {
        console.error("Inspection error:", error);

        showToast(
            error.message || "Inspection failed.",
            "error"
        );

    } finally {
        hideLoading();
    }
}


/* ------------------------------------------------------------
   INSPECTION RESULT
   ------------------------------------------------------------ */

function renderInspectionResult(data) {
    showElement("inspectionResults");

    const prediction = data.prediction || {};
    const rag = data.rag || {};
    const auditFinding = data.audit_finding || {};
    const report = data.audit_report || {};

    // Extract the predicted class name as a STRING from the API response
    const predictedClass =
        prediction.predicted_class ||
        prediction.class_name ||
        prediction.class ||
        auditFinding.defect_class ||
        auditFinding.predicted_class ||
        getNestedValue(report, ["inspection_summary.predicted_class", "predicted_class"], "Unknown");

    const confidence =
        prediction.final_confidence ??
        prediction.confidence ??
        auditFinding.confidence ??
        getNestedValue(
            report,
            [
                "inspection_summary.confidence",
                "confidence"
            ],
            0
        );

    setText(
        "predictedClass",
        formatClassName(predictedClass)
    );

    setText(
        "confidenceBadge",
        `${formatPercentage(confidence)} confidence`
    );

    const finding =
        auditFinding.finding ||
        auditFinding.audit_finding ||
        rag.finding ||
        rag.audit_finding ||
        getNestedValue(
            report,
            [
                "standards_compliance.finding",
                "inspection_summary.finding",
                "finding"
            ],
            "Inspection completed."
        );

    setText("findingText", finding);

    renderSeverity(report, auditFinding);
    renderStandards(report, rag, auditFinding);
    renderRework(report, rag, auditFinding);

    renderVisualSaliency(
        data.visual_saliency || {}
    );

    const reportButton = $("generateReportButton");

    if (reportButton) {
        reportButton.disabled = !currentReport;
    }

    // Continual learning: auto-register unknown defects and show in Tab 2
    handleUnknownDefect(predictedClass, confidence, data);
}



/* ------------------------------------------------------------
   SEVERITY / ASI
   ------------------------------------------------------------ */

function renderSeverity(report, auditFinding) {
    const summary =
        report.inspection_summary || {};

    const severityData =
        report.audit_severity ||
        report.severity ||
        auditFinding.audit_severity ||
        {};

    const category =
        severityData.category ||
        severityData.level ||
        summary.severity_category ||
        auditFinding.severity ||
        "Not Available";

    const score =
        severityData.score ??
        severityData.asi ??
        summary.audit_severity_index ??
        summary.asi ??
        report.audit_severity_index ??
        auditFinding.asi ??
        0;

    setText(
        "asiCategory",
        formatClassName(category)
    );

    setText(
        "asiScore",
        formatNumber(score)
    );

    const bar = $("asiBar");

    if (bar) {
        const numericScore =
            Math.max(
                0,
                Math.min(100, Number(score) || 0)
            );

        bar.style.width = `${numericScore}%`;
    }
}


/* ------------------------------------------------------------
   STANDARDS
   ------------------------------------------------------------ */

function renderStandards(
    report,
    rag,
    auditFinding
) {
    const standards =
        report.standards_compliance ||
        report.standards ||
        rag.standard ||
        rag.standards ||
        auditFinding.standard ||
        {};

    const clauseId =
        standards.clause_id ||
        standards.standard_id ||
        standards.ipc_standard_ref ||
        auditFinding.clause_id ||
        rag.clause_id ||
        "Not available";

    const title =
        standards.title ||
        standards.clause_title ||
        standards.standard_title ||
        auditFinding.standard_title ||
        rag.standard_title ||
        "Standards evidence";

    const classRule =
        standards.class_rule ||
        standards.requirement ||
        standards.acceptance_criteria ||
        auditFinding.class_rule ||
        rag.class_rule ||
        "No specific rule returned.";

    setText("standardClauseId", clauseId);
    setText("standardClauseTitle", title);
    setText("standardClassRule", classRule);
}


/* ------------------------------------------------------------
   REWORK
   ------------------------------------------------------------ */

function renderRework(
    report,
    rag,
    auditFinding
) {
    const rework =
        report.rework_and_capa ||
        report.rework ||
        rag.rework ||
        auditFinding.rework ||
        {};

    const procedure =
        rework.ipc_7721_procedure ||
        rework.ipc_7721_rework_procedure ||
        rework.rework_procedure ||
        rework.procedure ||
        auditFinding.rework_procedure ||
        rag.rework_procedure ||
        "Follow the applicable controlled rework procedure.";

    setText(
        "reworkProcedure",
        procedure
    );
}


/* ------------------------------------------------------------
   VISUAL SALIENCY / GRAD-CAM
   ------------------------------------------------------------ */

function renderVisualSaliency(visual) {
    if (!visual) {
        return;
    }

    const originalB64 =
        visual.original_b64 ||
        visual.original ||
        null;

    const overlayB64 =
        visual.overlay_b64 ||
        visual.overlay ||
        visual.gradcam_overlay ||
        null;

    const heatmapB64 =
        visual.heatmap_b64 ||
        visual.heatmap ||
        null;

    if (originalB64) {
        loadImageFromBase64(
            originalB64,
            img => {
                originalImage = img;
                redrawInspectionCanvas();
            }
        );
    }

    if (overlayB64) {
        loadImageFromBase64(
            overlayB64,
            img => {
                overlayImage = img;
                redrawInspectionCanvas();
            }
        );
    }

    if (heatmapB64) {
        loadImageFromBase64(
            heatmapB64,
            img => {
                heatmapImage = img;
            }
        );
    }

    setupVisualizationControls(
        visual.bounding_boxes ||
        visual.boxes ||
        []
    );
}

function loadImageFromBase64(value, callback) {
    const img = new Image();

    img.onload = () => {
        callback(img);
    };

    if (String(value).startsWith("data:")) {
        img.src = value;
    } else {
        img.src = `data:image/png;base64,${value}`;
    }
}


/* ------------------------------------------------------------
   VISUALIZATION CONTROLS
   ------------------------------------------------------------ */

function setupVisualizationControls(boxes) {
    const slider = $("alphaSlider");
    const alphaValue = $("alphaValue");

    if (slider) {
        slider.oninput = () => {
            if (alphaValue) {
                alphaValue.textContent =
                    `${Math.round(Number(slider.value) * 100)}%`;
            }

            redrawInspectionCanvas(boxes);
        };
    }

    const checkbox = $("bboxToggle");

    if (checkbox) {
        checkbox.onchange = () => {
            redrawInspectionCanvas(boxes);
        };
    }

    redrawInspectionCanvas(boxes);
}

function redrawInspectionCanvas(boxes = []) {
    const canvas = $("inspectionCanvas");

    if (!canvas) {
        return;
    }

    if (!originalImage) {
        return;
    }

    if (!canvasContext) {
        canvasContext =
            canvas.getContext("2d");
    }

    const width =
        originalImage.naturalWidth ||
        originalImage.width;

    const height =
        originalImage.naturalHeight ||
        originalImage.height;

    canvas.width = width;
    canvas.height = height;

    canvasContext.clearRect(
        0,
        0,
        width,
        height
    );

    canvasContext.drawImage(
        originalImage,
        0,
        0,
        width,
        height
    );

    const slider = $("alphaSlider");

    const alpha = slider
        ? Number(slider.value)
        : 0.5;

    if (overlayImage) {
        canvasContext.save();

        canvasContext.globalAlpha =
            Math.max(0, Math.min(1, alpha));

        canvasContext.drawImage(
            overlayImage,
            0,
            0,
            width,
            height
        );

        canvasContext.restore();
    }

    const checkbox = $("bboxToggle");

    if (checkbox && checkbox.checked) {
        drawBoundingBoxes(
            boxes,
            width,
            height
        );
    }
}

function drawBoundingBoxes(
    boxes,
    imageWidth,
    imageHeight
) {
    if (!Array.isArray(boxes)) {
        return;
    }

    boxes.forEach(box => {
        if (!box) {
            return;
        }

        let x;
        let y;
        let width;
        let height;

        if (
            box.x !== undefined &&
            box.y !== undefined
        ) {
            x = Number(box.x);
            y = Number(box.y);

            width = Number(
                box.width ??
                box.w ??
                0
            );

            height = Number(
                box.height ??
                box.h ??
                0
            );
        } else if (
            Array.isArray(box.bbox) &&
            box.bbox.length >= 4
        ) {
            x = Number(box.bbox[0]);
            y = Number(box.bbox[1]);
            width =
                Number(box.bbox[2]) - x;
            height =
                Number(box.bbox[3]) - y;
        } else {
            return;
        }

        // Handle normalized coordinates.
        if (
            x >= 0 &&
            x <= 1 &&
            y >= 0 &&
            y <= 1
        ) {
            x *= imageWidth;
            y *= imageHeight;

            if (width <= 1) {
                width *= imageWidth;
            }

            if (height <= 1) {
                height *= imageHeight;
            }
        }

        canvasContext.save();

        canvasContext.strokeStyle =
            "#ff3b30";

        canvasContext.lineWidth = Math.max(
            2,
            imageWidth / 300
        );

        canvasContext.strokeRect(
            x,
            y,
            width,
            height
        );

        canvasContext.restore();
    });
}


/* ------------------------------------------------------------
   PROTOTYPE REGISTRATION
   ------------------------------------------------------------ */

function initPrototypeRegistration() {
    const form = $("novelDefectForm");

    if (!form) {
        return;
    }

    form.addEventListener(
        "submit",
        registerPrototype
    );
}

async function registerPrototype(event) {
    event.preventDefault();

    const name =
        $("prototypeName")?.value.trim();

    const description =
        $("prototypeDescription")?.value.trim() || "";

    const ipcReference =
        $("prototypeIPC")?.value.trim() || "";

    const severity =
        $("prototypeSeverity")?.value || "";

    const fileInput =
        $("supportImages");

    if (!name) {
        showToast(
            "Enter a category name.",
            "error"
        );
        return;
    }

    if (!fileInput || !fileInput.files.length) {
        showToast(
            "Select 1â€“5 support images.",
            "error"
        );
        return;
    }

    const files =
        Array.from(fileInput.files);

    if (files.length < 1 || files.length > 5) {
        showToast(
            "Please select between 1 and 5 images.",
            "error"
        );
        return;
    }

    const formData = new FormData();

    formData.append(
        "defect_name",
        name
    );

    formData.append(
        "description",
        description
    );

    formData.append(
        "ipc_standard_ref",
        ipcReference
    );

    formData.append(
        "severity_baseline",
        severity
    );

    files.forEach(file => {
        formData.append(
            "images",
            file
        );
    });

    showLoading(
        "Registering PCB category..."
    );

    try {
        const response = await fetch(
            API.registerPrototype,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                "Prototype registration failed."
            );
        }

        showToast(
            "PCB category registered successfully.",
            "success"
        );

        const form =
            $("novelDefectForm");

        if (form) {
            form.reset();
        }

        await loadPrototypes();

    } catch (error) {
        console.error(
            "Prototype registration error:",
            error
        );

        showToast(
            error.message ||
            "Could not register category.",
            "error"
        );

    } finally {
        hideLoading();
    }
}


/* ------------------------------------------------------------
   LOAD PROTOTYPES
   ------------------------------------------------------------ */

async function loadPrototypes() {
    const container =
        $("prototypeListContainer");

    if (!container) {
        return;
    }

    container.innerHTML =
        `<div class="empty-state">Loading PCB categories...</div>`;

    try {
        const response =
            await fetch(API.prototypes);

        if (!response.ok) {
            throw new Error(
                `Could not load prototypes: ${response.status}`
            );
        }

        const data =
            await response.json();

        const prototypes =
            data.prototypes || [];

        renderPrototypes(prototypes);

    } catch (error) {
        console.error(
            "Prototype loading error:",
            error
        );

        container.innerHTML = `
            <div class="empty-state">
                Could not load PCB categories.
            </div>
        `;
    }
}

function renderPrototypes(prototypes) {
    const container =
        $("prototypeListContainer");

    if (!container) {
        return;
    }

    if (!prototypes.length) {
        container.innerHTML = `
            <div class="empty-state">
                No PCB categories registered yet.
            </div>
        `;
        return;
    }

    container.innerHTML = prototypes
        .map((prototype, index) => {
            const name =
                prototype.defect_name ||
                prototype.name ||
                prototype.class_name ||
                `Category ${index + 1}`;

            const description =
                prototype.description ||
                "";

            const shots =
                prototype.num_shots ??
                prototype.support_count ??
                prototype.n_shots ??
                "";

            const standard =
                prototype.ipc_standard_ref ||
                prototype.standard_ref ||
                "";

            return `
                <div class="prototype-card">

                    <div class="prototype-card-main">

                        <div class="prototype-title">
                            ${escapeHtml(
                                formatClassName(name)
                            )}
                        </div>

                        ${
                            description
                                ? `
                                <div class="prototype-description">
                                    ${escapeHtml(description)}
                                </div>
                                `
                                : ""
                        }

                        <div class="prototype-meta">

                            ${
                                shots !== ""
                                    ? `
                                    <span>
                                        ${escapeHtml(
                                            shots
                                        )} support image(s)
                                    </span>
                                    `
                                    : ""
                            }

                            ${
                                standard
                                    ? `
                                    <span>
                                        ${escapeHtml(
                                            standard
                                        )}
                                    </span>
                                    `
                                    : ""
                            }

                        </div>

                    </div>

                    <button
                        class="btn btn-danger prototype-delete"
                        data-defect-name="${escapeHtml(name)}"
                    >
                        Remove
                    </button>

                </div>
            `;
        })
        .join("");

    container
        .querySelectorAll(
            ".prototype-delete"
        )
        .forEach(button => {
            button.addEventListener(
                "click",
                () => {
                    deletePrototype(
                        button.dataset.defectName
                    );
                }
            );
        });
}


/* ------------------------------------------------------------
   DELETE PROTOTYPE
   ------------------------------------------------------------ */

async function deletePrototype(
    defectName
) {
    if (!defectName) {
        return;
    }

    const confirmed =
        window.confirm(
            `Remove "${defectName}" from prototype memory?`
        );

    if (!confirmed) {
        return;
    }

    showLoading(
        "Removing PCB category..."
    );

    try {
        const url =
            `${API.prototypes}/${encodeURIComponent(defectName)}`;

        const response =
            await fetch(url, {
                method: "DELETE"
            });

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                "Could not remove category."
            );
        }

        showToast(
            "PCB category removed.",
            "success"
        );

        await loadPrototypes();

    } catch (error) {
        console.error(
            "Prototype deletion error:",
            error
        );

        showToast(
            error.message ||
            "Could not remove category.",
            "error"
        );

    } finally {
        hideLoading();
    }
}


/* ------------------------------------------------------------
   AUDIT REPORT HISTORY
   ------------------------------------------------------------ */

/*
   The current Flask backend does NOT expose:
       /api/report

   Therefore the frontend keeps generated audit reports
   in browser localStorage.

   Reports are still generated by the backend through
       /api/inspect
   and certificates are generated through
       /api/audit/certificate
*/

function loadStoredReports() {
    try {
        const stored =
            localStorage.getItem(
                REPORT_STORAGE_KEY
            );

        if (!stored) {
            reportHistory = [];
            return;
        }

        const parsed =
            JSON.parse(stored);

        reportHistory =
            Array.isArray(parsed)
                ? parsed
                : [];

    } catch (error) {
        console.error(
            "Could not load stored reports:",
            error
        );

        reportHistory = [];
    }
}

function saveReportToHistory(report) {
    if (!report) {
        return;
    }

    const reportId =
        report.report_id ||
        report.id ||
        `REPORT-${Date.now()}`;

    const reportCopy = {
        ...report,
        report_id: reportId
    };

    reportHistory =
        reportHistory.filter(
            item =>
                (item.report_id || item.id) !==
                reportId
        );

    reportHistory.unshift(
        reportCopy
    );

    // Keep browser history manageable.
    reportHistory =
        reportHistory.slice(0, 50);

    try {
        localStorage.setItem(
            REPORT_STORAGE_KEY,
            JSON.stringify(reportHistory)
        );
    } catch (error) {
        console.warn(
            "Could not save report to browser storage:",
            error
        );
    }
}

function loadReports() {
    loadStoredReports();

    renderReports(reportHistory);
}

function renderReports(reports) {
    const container =
        $("reportsContainer");

    if (!container) {
        return;
    }

    if (!reports.length) {
        container.innerHTML = `
            <div class="empty-state">
                No audit reports yet.
                Complete a PCB inspection to generate one.
            </div>
        `;
        return;
    }

    container.innerHTML =
        reports
            .map((report, index) => {
                const summary =
                    report.inspection_summary ||
                    {};

                const prediction =
                    summary.defect_classification ||
                    summary.protonet_category ||
                    summary.predicted_class ||
                    summary.final_class ||
                    report.predicted_class ||
                    "Unknown";

                const confidence =
                    summary.confidence_percentage ??
                    summary.cbam_confidence_percentage ??
                    summary.protonet_confidence_percentage ??
                    report.confidence ??
                    0;

                const asi =
                    summary.audit_severity_index ??
                    summary.asi ??
                    report.audit_severity_index ??
                    report.asi ??
                    0;

                const reportId =
                    report.report_id ||
                    report.id ||
                    `REPORT-${index + 1}`;

                const timestamp =
                    report.timestamp ||
                    report.created_at ||
                    "";

                return `
                    <div class="report-card">

                        <div class="report-card-main">

                            <div class="report-title">
                                ${escapeHtml(reportId)}
                            </div>

                            <div class="report-prediction">
                                ${escapeHtml(
                                    formatClassName(
                                        prediction
                                    )
                                )}
                            </div>

                            <div class="report-meta">
                                <span>
                                    Confidence:
                                    ${formatPercentage(
                                        confidence
                                    )}
                                </span>

                                <span>
                                    ASI:
                                    ${formatNumber(asi)}
                                </span>

                                ${
                                    timestamp
                                        ? `
                                        <span>
                                            ${escapeHtml(
                                                formatDate(
                                                    timestamp
                                                )
                                            )}
                                        </span>
                                        `
                                        : ""
                                }
                            </div>

                        </div>

                        <button
                            class="btn btn-primary view-report"
                            data-report-index="${index}"
                        >
                            View Certificate
                        </button>

                    </div>
                `;
            })
            .join("");

    container
        .querySelectorAll(
            ".view-report"
        )
        .forEach(button => {
            button.addEventListener(
                "click",
                () => {
                    const index =
                        Number(
                            button.dataset.reportIndex
                        );

                    const report =
                        reportHistory[index];

                    if (report) {
                        currentReport = report;

                        openCertificate(
                            report
                        );
                    }
                }
            );
        });
}


/* ------------------------------------------------------------
   REPORT BUTTON
   ------------------------------------------------------------ */

function initReportControls() {
    const generateButton =
        $("generateReportButton");

    if (generateButton) {
        generateButton.addEventListener(
            "click",
            () => {
                if (!currentReport) {
                    showToast(
                        "No audit report is available yet.",
                        "error"
                    );
                    return;
                }

                openCertificate(
                    currentReport
                );
            }
        );
    }

    const refreshButton =
        $("refreshReportsButton");

    if (refreshButton) {
        refreshButton.addEventListener(
            "click",
            loadReports
        );
    }

    const closeButton =
        $("closeCertModal");

    if (closeButton) {
        closeButton.addEventListener(
            "click",
            closeCertificate
        );
    }

    const modal =
        $("certModal");

    if (modal) {
        modal.addEventListener(
            "click",
            event => {
                if (event.target === modal) {
                    closeCertificate();
                }
            }
        );
    }
}


/* ------------------------------------------------------------
   CERTIFICATE
   ------------------------------------------------------------ */

async function openCertificate(report) {
    if (!report) {
        return;
    }

    currentReport = report;

    updateCertificateSummary(
        report
    );

    showLoading(
        "Generating audit certificate..."
    );

    try {
        const response =
            await fetch(
                API.certificate,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        report: report
                    })
                }
            );

        if (!response.ok) {
            let errorMessage =
                "Could not generate certificate.";

            try {
                const errorData =
                    await response.json();

                errorMessage =
                    errorData.error ||
                    errorMessage;

            } catch (_) {
                // Ignore JSON parsing error.
            }

            throw new Error(
                errorMessage
            );
        }

        const html =
            await response.text();

        openCertificateWindow(
            html
        );

    } catch (error) {
        console.error(
            "Certificate generation error:",
            error
        );

        showToast(
            error.message ||
            "Could not generate certificate.",
            "error"
        );

    } finally {
        hideLoading();
    }
}

function updateCertificateSummary(
    report
) {
    const summary =
        report.inspection_summary ||
        {};
                const predictionData =
                    report.prediction ||
                    {};

                const auditFinding =
                    report.audit_finding ||
                    report.auditFinding ||
                    {};

                const prediction =
                    summary.predicted_class ||
                    summary.final_class ||
                    predictionData.final_class ||
                    predictionData.predicted_class ||
                    predictionData.class ||
                    report.predicted_class ||
                    auditFinding.defect_type ||
                    "Unknown";

                const confidence =
                    summary.confidence ??
                    predictionData.final_confidence ??
                    predictionData.confidence ??
                    auditFinding.confidence ??
                    report.confidence ??
                    0;

const asi =
        summary.audit_severity_index ??
        summary.asi ??
        report.audit_severity_index ??
        report.asi ??
        0;

    setText(
        "certReportId",
        report.report_id ||
        report.id ||
        "Audit Report"
    );

    setText(
        "certTimestamp",
        formatDate(
            report.timestamp ||
            report.created_at ||
            new Date().toISOString()
        )
    );

    setText(
        "certPrediction",
        formatClassName(
            prediction
        )
    );

    setText(
        "certConfidence",
        formatPercentage(
            confidence
        )
    );

    setText(
        "certASI",
        formatNumber(
            asi
        )
    );
}

function openCertificateWindow(html) {
    /*
       The backend returns a complete standalone HTML
       certificate, so opening it in a new tab is the
       cleanest option and does not require another API.
    */

    const certificateWindow =
        window.open(
            "",
            "_blank"
        );

    if (!certificateWindow) {
        showToast(
            "Your browser blocked the certificate window. Please allow pop-ups for localhost.",
            "error"
        );
        return;
    }

    certificateWindow.document.open();
    certificateWindow.document.write(
        html
    );
    certificateWindow.document.close();
}

function closeCertificate() {
    const modal =
        $("certModal");

    if (modal) {
        modal.style.display = "none";
        modal.classList.remove("active");
    }
}


/* ------------------------------------------------------------
   OPTIONAL MODAL DISPLAY
   ------------------------------------------------------------ */

function showCertificateModal() {
    const modal =
        $("certModal");

    if (!modal) {
        return;
    }

    modal.style.display = "flex";
    modal.classList.add("active");
}


/* ------------------------------------------------------------
   HELPERS
   ------------------------------------------------------------ */

function getNestedValue(
    object,
    paths,
    fallback = null
) {
    for (const path of paths) {
        const parts =
            path.split(".");

        let value = object;

        for (const part of parts) {
            if (
                value === null ||
                value === undefined
            ) {
                break;
            }

            value = value[part];
        }

        if (
            value !== undefined &&
            value !== null
        ) {
            return value;
        }
    }

    return fallback;
}

function formatClassName(value) {
    if (
        value === null ||
        value === undefined
    ) {
        return "Unknown";
    }

    return String(value)
        .replace(/_/g, " ")
        .replace(/\b\w/g, char =>
            char.toUpperCase()
        );
}

function formatNumber(value) {
    const number =
        Number(value);

    if (!Number.isFinite(number)) {
        return "0";
    }

    return number.toFixed(2);
}

function formatPercentage(value) {
    let number =
        Number(value);

    if (!Number.isFinite(number)) {
        return "0.00%";
    }

    /*
       Backend confidence may be either:
           0.97
       or:
           97
    */

    if (number <= 1) {
        number *= 100;
    }

    return `${number.toFixed(2)}%`;
}

function formatDate(value) {
    if (!value) {
        return "";
    }

    const date =
        new Date(value);

    if (
        Number.isNaN(
            date.getTime()
        )
    ) {
        return String(value);
    }

    return date.toLocaleString();
}


/* ------------------------------------------------------------
   OPERATING CLASS
   ------------------------------------------------------------ */

function initOperatingClass() {
    const selector =
        $("operatingClass");

    if (!selector) {
        return;
    }

    selector.addEventListener(
        "change",
        () => {
            localStorage.setItem(
                "manufacturingrag_operating_class",
                selector.value
            );
        }
    );

    const saved =
        localStorage.getItem(
            "manufacturingrag_operating_class"
        );

    if (
        saved &&
        Array.from(
            selector.options
        ).some(
            option =>
                option.value === saved
        )
    ) {
        selector.value = saved;
    }
}


/* ------------------------------------------------------------
   KEYBOARD / ESCAPE
   ------------------------------------------------------------ */

function initKeyboardControls() {
    
// ============================================================
// FINAL ACTIVE REPORT CHAT CONTEXT
// ============================================================

function setActiveInspectionReport(report) {
    if (report && typeof report === "object") {
        inspectionReportContext = report;
        console.log(
            "[REPORT CHAT] Active report:",
            inspectionReportContext
        );
    }
}

function clearActiveInspectionReport() {
    inspectionReportContext = null;
}


document.addEventListener(
        "keydown",
        event => {
            if (event.key === "Escape") {
                closeCertificate();
            }
        }
    );
}


/* ------------------------------------------------------------
   INITIALIZATION
   ------------------------------------------------------------ */

async function initializeApp() {
    console.log(
        "ManufacturingRAG-QA frontend initialized."
    );

    initNavigation();
    initFileUpload();
    initInspection();
    initPrototypeRegistration();
    initReportControls();
    initOperatingClass();
    initKeyboardControls();

    loadStoredReports();

    await Promise.allSettled([
        loadSystemStatus(),
        loadPrototypes(),
        loadReports()
    ]);
}


/* ------------------------------------------------------------
   START
   ------------------------------------------------------------ */

if (
    document.readyState === "loading"
) {
    document.addEventListener(
        "DOMContentLoaded",
        initializeApp
    );
} else {
    initializeApp();
}





// ============================================================
// CONVERSATIONAL STANDARDS ASSISTANT
// ============================================================

const standardsChatHistory = [];


// ============================================================
// INITIALIZE CHAT
// ============================================================

function initRAGChat() {

    const chatForm =
        document.getElementById("chatForm");

    const chatInput =
        document.getElementById("chatInput");

    const chatMessages =
        document.getElementById("chatMessages");

    if (
        !chatForm ||
        !chatInput ||
        !chatMessages
    ) {

        console.warn(
            "RAG chat elements not found."
        );

        return;
    }

    if (
        chatForm.dataset.ragInitialized === "true"
    ) {

        return;
    }

    chatForm.dataset.ragInitialized =
        "true";


    // ========================================================
    // INITIAL ASSISTANT GREETING
    // ========================================================

    if (
        chatMessages.children.length === 0
    ) {

        const welcomeMessage =
            "Hello! 👋 Welcome to the Manufacturing Standards Assistant." +
            "\n\n" +
            "I can help with PCB/PCBA quality questions, " +
            "IPC-A-610 requirements, defect acceptance criteria, " +
            "inspection findings, and evidence from the available " +
            "manufacturing standards." +
            "\n\n" +
            "What would you like to check?";

        appendChatMessage(
            formatMarkdownToHTML(
                welcomeMessage
            ),
            "assistant"
        );

        standardsChatHistory.push({

            role:
                "assistant",

            content:
                welcomeMessage
        });
    }


    // ========================================================
    // FORM SUBMISSION
    // ========================================================

    chatForm.addEventListener(
        "submit",
        async function (e) {

            e.preventDefault();

            const query =
                chatInput.value.trim();

            if (!query) {
                return;
            }


            // =================================================
            // SHOW USER MESSAGE
            // =================================================

            appendChatMessage(
                escapeHTML(query),
                "user"
            );


            // Save user message
            standardsChatHistory.push({

                role:
                    "user",

                content:
                    query
            });


            // Clear input
            chatInput.value = "";


            // =================================================
            // SHOW LOADING MESSAGE
            // =================================================

            const loadingId =
                appendChatMessage(

                    "<i class='fa-solid " +
                    "fa-circle-notch fa-spin'></i> " +
                    "Thinking...",

                    "assistant"
                );


            try {

                // =============================================
                // Operating class
                // =============================================

                let operatingClass =
                    "Class 3";

                if (
                    typeof state !== "undefined" &&
                    state &&
                    state.operatingClass
                ) {

                    operatingClass =
                        state.operatingClass;
                }


                // =============================================
                // Send recent conversation
                // =============================================

                const response =
                    await fetch(
                        "/api/rag/query",
                        {

                            method:
                                "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({

                                    question:
                                        query,

                                    operating_class:
                                        operatingClass,

                                    // Always inject the last inspection report
                                    // so the assistant can answer in context
                                    inspection_report:
                                        inspectionReportContext || currentInspection || null,

                                    history:
                                        standardsChatHistory
                                            .slice(-9)
                                })
                        }
                    );


                // =============================================
                // HTTP error
                // =============================================

                if (!response.ok) {

                    let errorMessage =
                        "Unable to process the request.";

                    try {

                        const errorData =
                            await response.json();

                        if (
                            errorData &&
                            errorData.answer
                        ) {

                            errorMessage =
                                errorData.answer;

                        } else if (
                            errorData &&
                            errorData.error
                        ) {

                            errorMessage =
                                errorData.error;
                        }

                    } catch (_) {
                        // Keep fallback message.
                    }

                    throw new Error(
                        errorMessage
                    );
                }


                // =============================================
                // Parse response
                // =============================================

                const data =
                    await response.json();


                const assistantDiv =
                    document.getElementById(
                        loadingId
                    );


                if (!assistantDiv) {
                    return;
                }


                // =============================================
                // Assistant answer
                // =============================================

                let answer = data.answer;
                if (typeof answer === "object" && answer !== null) {
                    answer = answer.answer || JSON.stringify(answer);
                }
                answer = answer || "I wasn't able to generate an answer.";

                assistantDiv.innerHTML =
                    formatMarkdownToHTML(
                        answer
                    );


                // =============================================
                // Save assistant response
                // =============================================

                standardsChatHistory.push({

                    role:
                        "assistant",

                    content:
                        answer
                });


                // =============================================
                // Scroll
                // =============================================

                chatMessages.scrollTop =
                    chatMessages.scrollHeight;

            } catch (err) {

                console.error(
                    "Standards Assistant error:",
                    err
                );


                const assistantDiv =
                    document.getElementById(
                        loadingId
                    );


                if (assistantDiv) {

                    assistantDiv.innerHTML =
                        formatMarkdownToHTML(

                            err.message ||
                            "Unable to query the standards " +
                            "knowledge base. Please try again."
                        );
                }
            }
        }
    );
}


// ============================================================
// PRESET QUESTIONS
// ============================================================

function askPreset(questionText) {

    const input =
        document.getElementById(
            "chatInput"
        );

    const form =
        document.getElementById(
            "chatForm"
        );

    if (
        !input ||
        !form
    ) {

        return;
    }

    input.value =
        questionText;

    form.dispatchEvent(
        new Event(
            "submit",
            {
                bubbles:
                    true,

                cancelable:
                    true
            }
        )
    );
}


// ============================================================
// APPEND CHAT MESSAGE
// ============================================================

function appendChatMessage(
    content,
    sender
) {

    const id =
        "msg-" +
        Date.now() +
        "-" +
        Math.random()
            .toString(36)
            .slice(2);


    const div =
        document.createElement(
            "div"
        );


    div.id =
        id;


    div.className =
        "chat-bubble " +
        sender;


    div.innerHTML =
        content;


    const container =
        document.getElementById(
            "chatMessages"
        );


    if (!container) {

        return id;
    }


    container.appendChild(
        div
    );


    container.scrollTop =
        container.scrollHeight;


    return id;
}


// ============================================================
// ESCAPE USER INPUT
// ============================================================
//
// Prevents user-entered HTML from being interpreted as markup.
// ============================================================

function escapeHTML(value) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        String(value ?? "");


    return div.innerHTML;
}


// ============================================================
// MARKDOWN → HTML
// ============================================================
//
// Lightweight renderer for assistant responses.
// ============================================================

function formatMarkdownToHTML(text) {

    if (!text) {
        return "";
    }


    let html =
        escapeHTML(
            String(text)
        );


    // Bold
    html =
        html.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    // Italic
    html =
        html.replace(
            /\*(.*?)\*/g,
            "<em>$1</em>"
        );


    // Blockquote
    html =
        html.replace(
            /^&gt; (.*)$/gim,
            "<blockquote>$1</blockquote>"
        );


    // Bullet lists
    html =
        html.replace(
            /(?:^|\n)- (.*)(?=\n|$)/g,
            "<li>$1</li>"
        );


    // Convert consecutive list items
    html =
        html.replace(
            /(<li>.*?<\/li>)(?:<br>|$)+/gs,
            "<ul>$1</ul>"
        );


    // Blank lines
    html =
        html.replace(
            /\n\n/g,
            "<br><br>"
        );


    // Remaining line breaks
    html =
        html.replace(
            /\n/g,
            "<br>"
        );


    return html;
}


// ============================================================
// INITIALIZE AFTER DOM LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        initRAGChat();

    }
);


/* ============================================================
   INSPECTION REPORT ANALYZER
   ============================================================ */

function initInspectionReportAnalyzer() {

    if (document.getElementById(
        "inspectionReportAnalyzer"
    )) {
        return;
    }

    const chatForm =
        document.getElementById("chatForm");

    if (!chatForm) {
        return;
    }

    const panel =
        document.createElement("section");

    panel.id =
        "inspectionReportAnalyzer";

    panel.className =
        "inspection-report-analyzer";

    panel.innerHTML = `
        <div class="report-analyzer-header">
            <strong>🔬 Inspection Report Analyzer</strong>
            <span style="font-size:0.82rem;color:var(--text-muted);margin-left:0.75rem;">
                Instantly explains your PCB inspection result in plain language — what was found, why it was rejected, what rule was violated, and what to do next.
            </span>
        </div>

        <div class="report-analyzer-controls">

            <input
                id="inspectionReportFile"
                type="file"
                accept=".json,.pdf,.txt,.html,.htm,.md"
                style="display:none"
            >

            <button
                type="button"
                id="analyzeInspectionReport"
            >
                <i class="fa-solid fa-microscope" style="margin-right:5px;"></i>
                Analyze Last Inspection
            </button>

            <span style="font-size:0.82rem;color:var(--text-muted);">or</span>

            <button
                type="button"
                id="chooseInspectionReport"
            >
                <i class="fa-solid fa-folder-open" style="margin-right:5px;"></i>
                Upload Report File
            </button>

            <span
                id="inspectionReportFilename"
                style="font-size:0.82rem;color:var(--text-muted);"
            >
            </span>

            <button
                type="button"
                id="clearInspectionReport"
                style="display:none"
            >
                Clear
            </button>

        </div>

        <div
            id="inspectionReportStatus"
            class="inspection-report-status"
        ></div>

        <div
            id="inspectionReportSummary"
            style="display:none"
        ></div>
    `;

    chatForm.parentNode.insertBefore(
        panel,
        chatForm
    );

    const fileInput =
        document.getElementById(
            "inspectionReportFile"
        );

    const chooseButton =
        document.getElementById(
            "chooseInspectionReport"
        );

    const analyzeButton =
        document.getElementById(
            "analyzeInspectionReport"
        );

    const clearButton =
        document.getElementById(
            "clearInspectionReport"
        );

    chooseButton.addEventListener(
        "click",
        () => fileInput.click()
    );

    fileInput.addEventListener(
        "change",
        () => {

            const file =
                fileInput.files?.[0];

            const filename =
                document.getElementById(
                    "inspectionReportFilename"
                );

            if (!file) {

                filename.textContent =
                    "No report selected";

                analyzeButton.disabled =
                    true;

                return;
            }

            filename.textContent =
                file.name;

            analyzeButton.disabled =
                false;
        }
    );

    analyzeButton.addEventListener(
        "click",
        analyzeInspectionReport
    );

    clearButton.addEventListener(
        "click",
        clearInspectionReport
    );
}


async function analyzeInspectionReport() {

    const status  = document.getElementById("inspectionReportStatus");
    const summary = document.getElementById("inspectionReportSummary");
    const analyzeButton = document.getElementById("analyzeInspectionReport");
    const clearButton   = document.getElementById("clearInspectionReport");
    const fileInput     = document.getElementById("inspectionReportFile");

    // ── Option A: uploaded file ──────────────────────────────────────────────
    const uploadedFile = fileInput?.files?.[0];

    // ── Option B: last inspection already done in Tab 1 ─────────────────────
    const liveReport = currentInspection;

    if (!uploadedFile && !liveReport) {
        if (status) status.textContent = "⚠️ No report available. Run an inspection in Tab 1 first, or upload a report file.";
        return;
    }

    if (status) status.textContent = "Generating report analysis…";
    if (analyzeButton) analyzeButton.disabled = true;

    try {
        let reportData = null;

        // If user uploaded a file, send it to the backend to parse
        if (uploadedFile) {
            const formData = new FormData();
            formData.append("file", uploadedFile);
            
            const response = await fetch("/api/rag/report", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Report analysis failed.");
            }
            reportData = data.report;
        } else {
            // Use the live inspection result
            reportData = liveReport;
        }

        // Store for chat context
        inspectionReportContext = reportData;
        try { sessionStorage.setItem("manufacturingRAGInspectionReport", JSON.stringify(reportData)); } catch(_) {}

        // ── Render plain-English summary ─────────────────────────────────────
        renderInspectionReportSummary(reportData);

        if (status) status.textContent = "✓ Report analyzed. Scroll down to ask questions.";
        if (clearButton) clearButton.style.display = "inline-flex";

        // ── Inject a proactive full analysis into the chat ───────────────────
        const introQuery = "__AUTO_ANALYZE__";
        const autoAnalysis = buildPlainEnglishAnalysis(reportData);

        appendChatMessage(
            formatMarkdownToHTML(autoAnalysis),
            "assistant"
        );
        standardsChatHistory.push({ role: "assistant", content: autoAnalysis });

        // Update the chat report context for future questions
        inspectionReportContext = reportData;

    } catch (err) {
        console.error("Report analysis error:", err);
        if (status) status.textContent = "❌ " + (err.message || "Failed to analyze report.");
    } finally {
        if (analyzeButton) analyzeButton.disabled = false;
    }
}


/**
 * Build a full plain-English breakdown of an inspection report.
 * Works whether the data comes from a live inspection or an uploaded JSON file.
 */
function buildPlainEnglishAnalysis(data) {
    if (!data) return "No report data available.";

    const prediction   = data.prediction  || {};
    const auditFinding = data.audit_finding || {};
    const report       = data.audit_report || data || {};
    const rag          = data.rag || {};

    // ── Extract key fields ───────────────────────────────────────────────────
    const defect     = auditFinding.defect_type
                    || prediction.predicted_class
                    || report.defect
                    || "Unknown Defect";

    const confidence = (prediction.confidence ?? auditFinding.confidence ?? 0) * 100;

    const severity   = report.inspection_summary?.audit_severity
                    || auditFinding.audit_severity
                    || report.severity
                    || "Unknown";

    const decision   = report.inspection_summary?.disposition
                    || report.decision
                    || (severity.toUpperCase().includes("CRITICAL") ? "REJECT" : "REVIEW REQUIRED");

    const standard   = auditFinding.applicable_standard
                    || rag.applicable_standard
                    || report.standard
                    || null;

    const clause     = auditFinding.ipc_clause
                    || report.ipc_clause
                    || null;

    const finding    = auditFinding.finding
                    || rag.finding
                    || report.finding
                    || null;

    const rework     = report.standards_compliance?.recommended_procedure
                    || report.recommended_procedure
                    || auditFinding.rework_procedure
                    || null;

    const rootCauses = report.root_cause_hypotheses
                    || report.root_causes
                    || [];

    const isReject   = decision.toUpperCase().includes("REJECT")
                    || severity.toUpperCase().includes("CRITICAL");

    // ── Verdict icon & colour ────────────────────────────────────────────────
    const verdictLine = isReject
        ? "🔴 **VERDICT: REJECT — This product does NOT meet quality standards.**"
        : "🟡 **VERDICT: REVIEW REQUIRED — This product needs closer inspection.**";

    // ── Plain-English severity explanation ───────────────────────────────────
    const severityExplain = {
        "CRITICAL": "A **Critical** defect means the product is unsafe or completely non-functional. It **must be rejected or scrapped** immediately.",
        "MAJOR":    "A **Major** defect means the product is likely to fail in real use. It **should be reworked** before it can be accepted.",
        "MINOR":    "A **Minor** defect means there is a cosmetic or small issue that may not affect function, but still needs to be documented.",
    };
    const sevKey = Object.keys(severityExplain).find(k => severity.toUpperCase().includes(k));
    const severityNote = severityExplain[sevKey] || `Severity level recorded as: **${severity}**`;

    // ── Build the message ────────────────────────────────────────────────────
    const lines = [];

    lines.push(`## 📋 Inspection Report Analysis`);
    lines.push(``);
    lines.push(verdictLine);
    lines.push(``);

    // ── 1. What was found ────────────────────────────────────────────────────
    lines.push(`### 🔍 What Was Found`);
    lines.push(`The AI inspection model detected a **${defect.replace(/_/g, " ")}** defect with **${confidence.toFixed(1)}% confidence**.`);
    if (finding) lines.push(`> ${finding}`);
    lines.push(``);

    // ── 2. Why this is a problem ─────────────────────────────────────────────
    lines.push(`### ⚠️ Why This Is a Problem`);
    lines.push(severityNote);
    lines.push(``);

    // ── 3. What rule / standard was violated ─────────────────────────────────
    lines.push(`### 📖 What Rule Was Violated`);
    if (standard || clause) {
        lines.push(`This defect violates **${standard || "IPC-A-610"}${clause ? ` (Clause ${clause})` : ""}** — the globally accepted standard for PCB assembly quality.`);
        lines.push(``);
        lines.push(`In simple terms: the manufacturing rules say that **solder joints, component placement, and electrical connections must meet specific size, position, and cleanliness requirements**. This defect breaches those limits.`);
    } else {
        lines.push(`No specific IPC clause was matched, but the defect severity indicates a non-conformance under general PCB quality acceptance criteria (IPC-A-610).`);
    }
    lines.push(``);

    // ── 4. Accept or Reject? ─────────────────────────────────────────────────
    lines.push(`### ✅ Accept or ❌ Reject?`);
    if (isReject) {
        lines.push(`**This product should be REJECTED.** It does not meet the minimum quality standard for shipment or use.`);
        lines.push(`- It cannot be sent to a customer in this state.`);
        lines.push(`- It must either be **reworked** (repaired) or **scrapped** (discarded).`);
    } else {
        lines.push(`**This product requires further review.** It may be acceptable depending on the target product class (Class 1 = consumer, Class 3 = military/medical).`);
    }
    lines.push(``);

    // ── 5. What to do next (rework) ──────────────────────────────────────────
    lines.push(`### 🔧 What Should Be Done Next`);
    if (rework) {
        lines.push(rework);
    } else {
        lines.push(`The defective area should be inspected by a trained technician. Depending on the defect type, the board may need to be resoldered, have a component replaced, or be discarded entirely.`);
    }
    lines.push(``);

    // ── 6. Root causes ───────────────────────────────────────────────────────
    if (rootCauses.length > 0) {
        lines.push(`### 🏭 Likely Root Causes (Why It Happened)`);
        rootCauses.forEach(c => lines.push(`- ${c}`));
        lines.push(``);
    }

    // ── 7. Plain summary ─────────────────────────────────────────────────────
    lines.push(`---`);
    lines.push(`💬 **You can now ask me anything about this report** — for example:`);
    lines.push(`- *"Is this safe to use?"*`);
    lines.push(`- *"What does solder bridging mean?"*`);
    lines.push(`- *"Can this be fixed or does it need to be thrown away?"*`);
    lines.push(`- *"What IPC-A-610 class does this apply to?"*`);

    return lines.join("\n");
}



function renderInspectionReportSummary(report) {

    const summary =
        document.getElementById(
            "inspectionReportSummary"
        );

    if (!summary || !report) {
        return;
    }

    const value =
        (key, fallback = "—") =>
            report[key] ??
            fallback;

    summary.style.display =
        "block";

    summary.innerHTML = `
        <div class="active-report-badge">
            ✓ ACTIVE INSPECTION REPORT
        </div>

        <div class="inspection-report-grid">

            <div>
                <strong>Report ID</strong>
                <span>${escapeHTML(
                    String(
                        value("report_id")
                    )
                )}</span>
            </div>

            <div>
                <strong>Defect</strong>
                <span>${escapeHTML(
                    String(
                        value("defect")
                    )
                )}</span>
            </div>

            <div>
                <strong>Standard</strong>
                <span>${escapeHTML(
                    String(
                        value("standard")
                    )
                )}</span>
            </div>

            <div>
                <strong>Clause</strong>
                <span>${escapeHTML(
                    String(
                        value("ipc_clause")
                    )
                )}</span>
            </div>

            <div>
                <strong>Operating Class</strong>
                <span>${escapeHTML(
                    String(
                        value(
                            "operating_class"
                        )
                    )
                )}</span>
            </div>

            <div>
                <strong>Decision</strong>
                <span>${escapeHTML(
                    String(
                        value("decision")
                    )
                )}</span>
            </div>

            <div>
                <strong>Severity</strong>
                <span>${escapeHTML(
                    String(
                        value("severity")
                    )
                )}</span>
            </div>

            <div>
                <strong>Required Action</strong>
                <span>${escapeHTML(
                    String(
                        value(
                            "required_action"
                        )
                    )
                )}</span>
            </div>

        </div>
    `;
}


function clearInspectionReport() {

    inspectionReportContext =
        null;

    try {

        sessionStorage.removeItem(
            "manufacturingRAGInspectionReport"
        );

    } catch (_) {}

    const fileInput =
        document.getElementById(
            "inspectionReportFile"
        );

    const filename =
        document.getElementById(
            "inspectionReportFilename"
        );

    const status =
        document.getElementById(
            "inspectionReportStatus"
        );

    const summary =
        document.getElementById(
            "inspectionReportSummary"
        );

    const clearButton =
        document.getElementById(
            "clearInspectionReport"
        );

    const analyzeButton =
        document.getElementById(
            "analyzeInspectionReport"
        );

    if (fileInput) {
        fileInput.value = "";
    }

    if (filename) {
        filename.textContent =
            "No report selected";
    }

    if (status) {
        status.textContent =
            "";
    }

    if (summary) {
        summary.style.display =
            "none";

        summary.innerHTML =
            "";
    }

    if (clearButton) {
        clearButton.style.display =
            "none";
    }

    if (analyzeButton) {
        analyzeButton.disabled =
            true;
    }
}


function restoreInspectionReport() {

    try {

        const saved =
            sessionStorage.getItem(
                "manufacturingRAGInspectionReport"
            );

        if (!saved) {
            return;
        }

        const parsed =
            JSON.parse(saved);

        if (
            parsed &&
            typeof parsed === "object"
        ) {

            inspectionReportContext =
                parsed;

            renderInspectionReportSummary(
                parsed
            );

            const clearButton =
                document.getElementById(
                    "clearInspectionReport"
                );

            if (clearButton) {
                clearButton.style.display =
                    "inline-block";
            }

            const status =
                document.getElementById(
                    "inspectionReportStatus"
                );

            if (status) {
                status.textContent =
                    "✓ Previous report context restored.";
            }
        }

    } catch (error) {

        console.warn(
            "Could not restore report:",
            error
        );
    }
}


/* ============================================================
   CERTIFICATE PDF DOWNLOAD
   ============================================================ */

async function downloadCertificatePDF(
    report = currentReport
) {

    if (!report) {

        alert(
            "No inspection report is available for the certificate."
        );

        return;
    }

    try {

        const response =
            await fetch(
                "/api/audit/certificate/pdf",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        report: report
                    })
                }
            );

        if (!response.ok) {

            let message =
                "Certificate PDF generation failed.";

            try {

                const data =
                    await response.json();

                message =
                    data.error ||
                    message;

            } catch (_) {}

            throw new Error(
                message
            );
        }

        const blob =
            await response.blob();

        const url =
            URL.createObjectURL(
                blob
            );

        const link =
            document.createElement(
                "a"
            );

        link.href =
            url;

        const reportId =
            report.report_id ||
            report.id ||
            "inspection";

        link.download =
            `ManufacturingRAG-QA_Certificate_${reportId}.pdf`;

        document.body.appendChild(
            link
        );

        link.click();

        link.remove();

        setTimeout(
            () =>
                URL.revokeObjectURL(
                    url
                ),
            1000
        );

    } catch (error) {

        console.error(
            error
        );

        alert(
            error.message ||
            "Could not download certificate PDF."
        );
    }
}


/* ============================================================
   ADD CERTIFICATE DOWNLOAD BUTTON
   ============================================================ */

function addCertificateDownloadButton() {

    if (
        document.getElementById(
            "downloadCertificatePdfButton"
        )
    ) {
        return;
    }

    const candidates =
        Array.from(
            document.querySelectorAll(
                "button"
            )
        );

    const certificateButton =
        candidates.find(
            button =>
                /certificate/i.test(
                    button.textContent || ""
                )
        );

    if (!certificateButton) {
        return;
    }

    const button =
        document.createElement(
            "button"
        );

    button.id =
        "downloadCertificatePdfButton";

    button.type =
        "button";

    button.textContent =
        "Download Certificate PDF";

    button.addEventListener(
        "click",
        () =>
            downloadCertificatePDF(
                currentReport
            )
    );

    certificateButton.parentNode.appendChild(
        button
    );
}


/* ============================================================
   INITIALIZE REPORT FEATURES
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initInspectionReportAnalyzer();

        // Load known prototype names and update Tab 2 counts
        loadPrototypes();

        // Restore persisted newly-learned defects across browser refreshes
        restoreLearnedDefects();

        const clearBtn = document.getElementById("clearLearnedHistoryBtn");
        if (clearBtn) {
            clearBtn.addEventListener("click", () => {
                if (confirm("Reset the list of newly learned defects for this demo?")) {
                    clearLearnedDefectsHistory();
                }
            });
        }

        setTimeout(
            restoreInspectionReport,
            300
        );

        setTimeout(
            addCertificateDownloadButton,
            1000
        );
    }
);

