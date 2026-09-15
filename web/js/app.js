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

let originalImage = null;
let overlayImage = null;
let heatmapImage = null;

let canvasContext = null;

let reportHistory = [];

const REPORT_STORAGE_KEY = "manufacturingrag_audit_reports";


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

    const predictedClass =
        prediction.final_class ||
        prediction.predicted_class ||
        prediction.class ||
        auditFinding.defect_type ||
        "Unknown";

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





/* =========================================================
   STANDARDS RAG CHAT - RESTORED
   ========================================================= */

function initRAGChat() {
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatMessages = document.getElementById("chatMessages");

    if (!chatForm || !chatInput || !chatMessages) {
        console.warn("RAG chat elements not found.");
        return;
    }

    if (chatForm.dataset.ragInitialized === "true") return;
    chatForm.dataset.ragInitialized = "true";

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const query = chatInput.value.trim();
        if (!query) return;

        appendChatMessage(query, "user");
        chatInput.value = "";

        const loadingId = appendChatMessage(
            "<i class='fa-solid fa-circle-notch fa-spin'></i> Retrieving standard clauses and synthesizing analysis...",
            "assistant"
        );

        try {
            const res = await fetch("/api/rag/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: query,
                    operating_class:
                        typeof state !== "undefined"
                            ? state.operatingClass
                            : "Class 3"
                })
            });

            if (!res.ok) {
                throw new Error("HTTP " + res.status);
            }

            const data = await res.json();

            const assistantDiv =
                document.getElementById(loadingId);

            if (assistantDiv) {
                assistantDiv.innerHTML =
                    formatMarkdownToHTML(
                        data.answer ||
                        "No answer was returned by the standards engine."
                    );
            }

            chatMessages.scrollTop =
                chatMessages.scrollHeight;

        } catch (err) {
            console.error("RAG chat error:", err);

            const assistantDiv =
                document.getElementById(loadingId);

            if (assistantDiv) {
                assistantDiv.textContent =
                    "Unable to query the standards database. Please try again.";
            }
        }
    });
}

function askPreset(questionText) {
    const input = document.getElementById("chatInput");
    const form = document.getElementById("chatForm");

    if (!input || !form) return;

    input.value = questionText;
    form.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true })
    );
}

function appendChatMessage(content, sender) {
    const id =
        "msg-" +
        Date.now() +
        "-" +
        Math.random().toString(36).slice(2);

    const div = document.createElement("div");

    div.id = id;
    div.className = "chat-bubble " + sender;
    div.innerHTML = content;

    const container =
        document.getElementById("chatMessages");

    if (!container) return id;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;

    return id;
}

function formatMarkdownToHTML(text) {
    if (!text) return "";

    return String(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(
            /^> (.*$)/gim,
            "<blockquote>$1</blockquote>"
        )
        .replace(
            /^- (.*$)/gim,
            "<li>$1</li>"
        )
        .replace(/\n\n/g, "<br><br>")
        .replace(/\n/g, "<br>");
}

/* Initialize after DOM is ready */
document.addEventListener("DOMContentLoaded", function () {
    initRAGChat();
});
