
"""
Quality Audit Report Generator and Severity Scoring Engine.

Synthesizes inspection findings, Grad-CAM visual evidence,
IPC-A-610 clause citations, retrieved RAG evidence,
and calculates the composite Audit Severity Index (ASI).

IMPORTANT SAFETY / REPORTING RULES
----------------------------------
1. CBAM is responsible for the binary decision:
       normal / anomaly

2. ProtoNet is responsible for specific defect identification.

3. A specific IPC-A-610 clause may ONLY be assigned when
   ProtoNet has confidently confirmed a defect category.

4. An anomaly without a confirmed ProtoNet category is reported as:
       UNCONFIRMED ANOMALY

   It must NOT automatically become:
       MAJOR DEFECT

5. An unconfirmed anomaly must NOT inherit:
       - stale IPC clauses
       - default defect mappings
       - defect-specific rework procedures
       - defect-specific root causes
       - defect-specific electrical impact

6. The governing audit standard (IPC-A-610G) may still appear
   in the certificate header. This does NOT mean a specific
   clause has been confirmed.

7. ASI represents audit severity, not merely model confidence.
   Therefore, an unconfirmed anomaly is intentionally capped
   below the confirmed-defect major threshold unless an
   independent risk factor is explicitly supplied.
"""

import os
import json
import uuid
from datetime import datetime
from html import escape
from typing import Dict, List, Any, Optional


class AuditReporter:
    """
    Generates standardized electronics quality audit certificates
    and reports from CBAM/ProtoNet + RAG inspection findings.
    """

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    CONFIRMED_DEFECT_THRESHOLD = 45.0
    UNCONFIRMED_ANOMALY_MAX_ASI = 39.9

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # SEVERITY SCORING
    # ------------------------------------------------------------------

    def calculate_audit_severity_index(
        self,
        confidence: float,
        risk_factor: float,
        operating_class: str,
        defect_area_ratio: float = 0.05,
        category_confirmed: bool = True,
        cbam_is_anomaly: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates composite Audit Severity Index (ASI) [0.0 - 100.0].

        For a CONFIRMED defect:

            ASI =
                0.35 * confidence
              + 0.40 * risk_factor
              + 0.15 * class_weight
              + 0.10 * normalized_defect_area

        The result is scaled to 0-100.

        For an UNCONFIRMED anomaly, severity is intentionally treated
        differently. High CBAM confidence alone does not prove a
        specific defect. Therefore an unconfirmed anomaly cannot
        automatically cross the confirmed-defect major threshold.

        This prevents a result such as:

            CBAM anomaly = 86%
            ProtoNet = 15%
            risk = 0

        from being incorrectly reported as:

            MAJOR DEFECT
        """

        class_weights = {
            "class 1": 0.50,
            "class 2": 0.75,
            "class 3": 1.00
        }

        operating_class_clean = str(
            operating_class or "Class 3"
        ).strip().lower()

        c_weight = class_weights.get(
            operating_class_clean,
            1.00
        )

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        try:
            risk_factor = float(risk_factor)
        except (TypeError, ValueError):
            risk_factor = 0.0

        try:
            defect_area_ratio = float(defect_area_ratio)
        except (TypeError, ValueError):
            defect_area_ratio = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        risk_factor = max(
            0.0,
            min(1.0, risk_factor)
        )

        defect_area_ratio = max(
            0.0,
            min(1.0, defect_area_ratio)
        )

        normalized_area = min(
            defect_area_ratio * 10.0,
            1.0
        )

        # --------------------------------------------------------------
        # CONFIRMED DEFECT
        # --------------------------------------------------------------

        if category_confirmed:

            asi = (
                0.35 * confidence
                + 0.40 * risk_factor
                + 0.15 * c_weight
                + 0.10 * normalized_area
            ) * 100.0

            asi = round(
                min(100.0, max(0.0, asi)),
                1
            )

            if asi >= 75.0:

                level = "CRITICAL NON-CONFORMANCE"

                color = "#ef4444"

                action = (
                    "HALT LOT - MANDATORY REWORK OR SCRAP"
                )

            elif asi >= self.CONFIRMED_DEFECT_THRESHOLD:

                level = "MAJOR DEFECT"

                color = "#f97316"

                action = (
                    "CONTAIN LOT & PERFORM REWORK "
                    "(IPC-7721)"
                )

            elif asi >= 20.0:

                level = "MINOR / PROCESS INDICATOR"

                color = "#eab308"

                action = (
                    "MONITOR LINE SPC - "
                    "NOTIFY SMT OPERATOR"
                )

            else:

                level = "TARGET / ACCEPTABLE"

                color = "#22c55e"

                action = (
                    "PASS TO NEXT ASSEMBLY STAGE"
                )

            return {
                "score": asi,
                "level": level,
                "color_code": color,
                "recommended_action": action
            }

        # --------------------------------------------------------------
        # UNCONFIRMED ANOMALY
        # --------------------------------------------------------------
        #
        # Important:
        # An anomaly is NOT equivalent to a confirmed defect.
        #
        # We retain a meaningful ASI, but explicitly prevent it from
        # entering the confirmed-defect severity bands.
        # --------------------------------------------------------------

        if cbam_is_anomaly:

            # Confidence contributes to investigation priority,
            # but does not establish defect severity.
            investigation_score = (
                0.55 * confidence
                + 0.20 * risk_factor
                + 0.15 * c_weight
                + 0.10 * normalized_area
            ) * 100.0

            asi = round(
                min(
                    self.UNCONFIRMED_ANOMALY_MAX_ASI,
                    max(0.0, investigation_score)
                ),
                1
            )

            level = "UNCONFIRMED ANOMALY"

            color = "#eab308"

            action = (
                "CONTAIN / HOLD FOR DEFECT IDENTIFICATION"
            )

            return {
                "score": asi,
                "level": level,
                "color_code": color,
                "recommended_action": action
            }

        # --------------------------------------------------------------
        # NORMAL / NON-DEFECT CASE
        # --------------------------------------------------------------

        # Higher confidence that the assembly is normal should
        # correspond to lower residual audit risk.
        asi = round(
            min(
                20.0,
                max(
                    0.0,
                    (1.0 - confidence) * 20.0
                )
            ),
            1
        )

        return {
            "score": asi,
            "level": "TARGET / ACCEPTABLE",
            "color_code": "#22c55e",
            "recommended_action": (
                "PASS TO NEXT ASSEMBLY STAGE"
            )
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _first_non_empty(*values, default=""):
        """
        Returns the first non-empty value.
        """

        for value in values:

            if value is None:
                continue

            if isinstance(value, str):

                if value.strip():
                    return value.strip()

            elif value:
                return value

        return default

    @staticmethod
    def _normalise_list(value) -> List[str]:
        """
        Ensures root-cause information is always represented as a list.
        """

        if value is None:
            return []

        if isinstance(value, list):
            return [
                str(v)
                for v in value
                if str(v).strip()
            ]

        if isinstance(value, tuple):
            return [
                str(v)
                for v in value
                if str(v).strip()
            ]

        if isinstance(value, str):

            if not value.strip():
                return []

            return [value.strip()]

        return [str(value)]

    @staticmethod
    def _extract_standard_data(
        finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract standards information from the RAG finding.

        Missing standards data is treated as NO CONFIRMED
        STANDARD MAPPING.
        """

        standard = finding.get(
            "standard",
            {}
        )

        if not isinstance(standard, dict):
            standard = {}

        return standard

    @staticmethod
    def _extract_rag_evidence(
        finding: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extracts PDF retrieval evidence from the RAG finding.
        """

        evidence = finding.get(
            "retrieved_evidence",
            []
        )

        if not isinstance(evidence, list):
            return []

        cleaned = []

        for item in evidence:

            if not isinstance(item, dict):
                continue

            cleaned.append({
                "source": item.get(
                    "source",
                    ""
                ),

                "page": item.get(
                    "page",
                    ""
                ),

                "text": item.get(
                    "text",
                    ""
                ),

                "semantic_score": item.get(
                    "semantic_score",
                    item.get(
                        "score",
                        None
                    )
                ),

                "rank": item.get(
                    "rank",
                    None
                )
            })

        return cleaned

    # ------------------------------------------------------------------
    # REPORT GENERATION
    # ------------------------------------------------------------------

    def generate_report(
        self,
        inspection_result: Dict[str, Any],
        board_serial: Optional[str] = None,
        line_id: str = "SMT-LINE-04",
        operator_id: str = "QE-AUDITOR-88",
        lot_id: str = "LOT-2026-AUG-884"
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive quality inspection audit certificate.

        Decision hierarchy:

        CBAM
            -> normal / anomaly

        ProtoNet
            -> specific defect category only when confirmed

        RAG
            -> PDF evidence when available

        Structured KB
            -> only used for a confirmed defect category

        Safety rule:

        CBAM anomaly + ProtoNet unconfirmed
            -> UNCONFIRMED ANOMALY
            -> no specific IPC clause
            -> no defect-specific rework
            -> no defect-specific root cause
        """

        # --------------------------------------------------------------
        # REPORT IDENTIFIERS
        # --------------------------------------------------------------

        report_id = (
            f"AUD-"
            f"{datetime.now().strftime('%Y%m%d')}-"
            f"{str(uuid.uuid4())[:8].upper()}"
        )

        timestamp = datetime.now().isoformat()

        board_sn = (
            board_serial
            or f"PCB-SN-{str(uuid.uuid4())[:8].upper()}"
        )

        # --------------------------------------------------------------
        # RAW INSPECTION OBJECTS
        # --------------------------------------------------------------

        finding = inspection_result.get(
            "audit_finding",
            {}
        )

        if not isinstance(finding, dict):
            finding = {}

        visual = inspection_result.get(
            "visual_evidence",
            {}
        )

        if not isinstance(visual, dict):
            visual = {}

        binary_decision = inspection_result.get(
            "binary_decision",
            {}
        )

        if not isinstance(binary_decision, dict):
            binary_decision = {}

        category_decision = inspection_result.get(
            "category_decision",
            {}
        )

        if not isinstance(category_decision, dict):
            category_decision = {}

        # --------------------------------------------------------------
        # CBAM DECISION
        # --------------------------------------------------------------

        cbam_classification = self._first_non_empty(
            binary_decision.get("class_name"),
            finding.get("predicted_class"),
            finding.get("cbam_class"),
            finding.get("defect_type"),
            default="Unknown"
        )

        cbam_classification = str(
            cbam_classification
        ).strip()

        cbam_confidence = binary_decision.get(
            "confidence",
            finding.get(
                "confidence",
                0.95
            )
        )

        try:
            cbam_confidence = float(
                cbam_confidence
            )
        except (
            TypeError,
            ValueError
        ):
            cbam_confidence = 0.95

        cbam_confidence = max(
            0.0,
            min(
                1.0,
                cbam_confidence
            )
        )

        # --------------------------------------------------------------
        # PROTONET DECISION
        # --------------------------------------------------------------

        proto_category = category_decision.get(
            "class_name"
        )

        proto_confirmed = bool(
            category_decision.get(
                "confirmed",
                False
            )
        )

        proto_confidence = category_decision.get(
            "confidence",
            0.0
        )

        try:
            proto_confidence = float(
                proto_confidence
            )
        except (
            TypeError,
            ValueError
        ):
            proto_confidence = 0.0

        proto_confidence = max(
            0.0,
            min(
                1.0,
                proto_confidence
            )
        )

        if proto_category is not None:

            proto_category = str(
                proto_category
            ).strip()

            if not proto_category:
                proto_category = None

        # --------------------------------------------------------------
        # NORMALISE CLASSIFICATION
        # --------------------------------------------------------------

        cbam_lower = cbam_classification.lower()

        cbam_is_anomaly = cbam_lower in {
            "anomaly",
            "abnormal",
            "defect",
            "defective"
        }

        cbam_is_normal = cbam_lower in {
            "normal",
            "good",
            "pass",
            "ok"
        }

        # --------------------------------------------------------------
        # CONFIRMED CATEGORY
        # --------------------------------------------------------------

        category_confirmed = (
            proto_confirmed
            and bool(proto_category)
        )

        category_unconfirmed = (
            cbam_is_anomaly
            and not category_confirmed
        )

        # --------------------------------------------------------------
        # FINAL REPORTED DEFECT
        # --------------------------------------------------------------

        if cbam_is_normal:

            defect_type = "Good_Assembly"

            finding_status = (
                "CBAM classified the assembly as normal."
            )

        elif cbam_is_anomaly and category_confirmed:

            defect_type = proto_category

            finding_status = (
                "CBAM detected an anomaly and "
                "ProtoNet confirmed the defect category."
            )

        elif cbam_is_anomaly:

            defect_type = (
                "Defect category not confidently identified"
            )

            finding_status = (
                "CBAM detected an anomaly, but "
                "ProtoNet did not confidently confirm "
                "a specific defect category."
            )

        else:

            if category_confirmed:

                defect_type = proto_category

                finding_status = (
                    "ProtoNet confirmed a defect category; "
                    "manual review of the binary classification "
                    "is recommended."
                )

            else:

                defect_type = self._first_non_empty(
                    finding.get("defect_type"),
                    default="Unknown"
                )

                finding_status = (
                    "Inspection result requires review."
                )

        # --------------------------------------------------------------
        # STANDARD / OPERATING CLASS
        # --------------------------------------------------------------

        standard = self._extract_standard_data(
            finding
        )

        operating_class = self._first_non_empty(
            finding.get(
                "operating_class"
            ),

            standard.get(
                "operating_class"
            ),

            default="Class 3"
        )

        # --------------------------------------------------------------
        # DEFECT AREA
        # --------------------------------------------------------------

        try:

            defect_area_ratio = float(
                visual.get(
                    "defect_area_ratio",
                    finding.get(
                        "defect_area_ratio",
                        0.04
                    )
                )
            )

        except (
            TypeError,
            ValueError
        ):

            defect_area_ratio = 0.04

        defect_area_ratio = max(
            0.0,
            min(
                1.0,
                defect_area_ratio
            )
        )

        # --------------------------------------------------------------
        # RISK FACTOR
        # --------------------------------------------------------------

        structured_risk = standard.get(
            "risk_factor"
        )

        if structured_risk is not None:

            try:

                risk_factor = float(
                    structured_risk
                )

            except (
                TypeError,
                ValueError
            ):

                risk_factor = 0.0

        elif category_unconfirmed:

            # No confirmed defect = do not infer defect risk.
            risk_factor = 0.0

        else:

            defect_lower = str(
                defect_type
            ).lower()

            if (
                "bridge" in defect_lower
                or "missing" in defect_lower
            ):

                risk_factor = 0.95

            elif "tombstone" in defect_lower:

                risk_factor = 0.90

            elif "insufficient" in defect_lower:

                risk_factor = 0.75

            elif (
                "ball" in defect_lower
                or "misaligned" in defect_lower
            ):

                risk_factor = 0.60

            else:

                risk_factor = 0.0

        risk_factor = max(
            0.0,
            min(
                1.0,
                risk_factor
            )
        )

        # --------------------------------------------------------------
        # ASI
        # --------------------------------------------------------------

        severity_meta = (
            self.calculate_audit_severity_index(
                confidence=cbam_confidence,
                risk_factor=risk_factor,
                operating_class=operating_class,
                defect_area_ratio=defect_area_ratio,
                category_confirmed=category_confirmed,
                cbam_is_anomaly=cbam_is_anomaly
            )
        )

        # --------------------------------------------------------------
        # FINAL SEVERITY LABEL
        # --------------------------------------------------------------

        if category_unconfirmed:

            severity_level = (
                "UNCONFIRMED ANOMALY"
            )

        else:

            severity_level = self._first_non_empty(
                standard.get(
                    "severity_level"
                ),

                finding.get(
                    "severity_level"
                ),

                default=severity_meta[
                    "level"
                ]
            )

        # --------------------------------------------------------------
        # RAG EVIDENCE
        # --------------------------------------------------------------

        retrieved_evidence = (
            self._extract_rag_evidence(
                finding
            )
        )

        evidence_summary = finding.get(
            "evidence_summary",
            ""
        )

        if not isinstance(
            evidence_summary,
            str
        ):
            evidence_summary = ""

        knowledge_source = finding.get(
            "knowledge_source",
            ""
        )

        if not knowledge_source:

            knowledge_source = (
                "No confirmed standards mapping"
                if category_unconfirmed
                else "Not specified"
            )

        grounding = finding.get(
            "grounding",
            {}
        )

        if not isinstance(
            grounding,
            dict
        ):
            grounding = {}

        pdf_evidence_used = bool(
            grounding.get(
                "pdf_evidence_used",
                bool(retrieved_evidence)
            )
        )

        structured_kb_used = bool(
            grounding.get(
                "structured_kb_used",
                bool(standard)
            )
        )

        llm_generated = bool(
            grounding.get(
                "llm_generated",
                False
            )
        )

        # --------------------------------------------------------------
        # SAFETY:
        # UNCONFIRMED ANOMALIES MUST NOT USE STALE STANDARDS DATA
        # --------------------------------------------------------------

        if category_unconfirmed:

            standard = {}

            standard_id = (
                "Not determined"
            )

            section = (
                "Not determined"
            )

            clause_title = (
                "No specific defect category confirmed"
            )

            standard_description = (
                "No specific IPC-A-610 clause can be assigned "
                "because CBAM detected an anomaly but ProtoNet "
                "did not confidently confirm a defect category."
            )

            applicable_rule = (
                "Not specified because no specific defect "
                "category was confirmed."
            )

            acceptance_level = (
                "Not specified because no specific defect "
                "category was confirmed."
            )

            electrical_impact = (
                "Not determined."
            )

            standard_source = (
                "No confirmed standards mapping"
            )

            # Prevent stale RAG / KB flags from claiming that
            # a specific standards mapping was used.
            pdf_evidence_used = False
            structured_kb_used = False

            # For an unconfirmed anomaly, there is no valid
            # defect-specific standards evidence.
            knowledge_source = (
                "No confirmed standards mapping"
            )

        else:

            # ----------------------------------------------------------
            # CONFIRMED DEFECT STANDARDS PATH
            # ----------------------------------------------------------

            standard_id = self._first_non_empty(
                finding.get(
                    "clause_id"
                ),

                standard.get(
                    "id"
                ),

                default="Not determined"
            )

            section = self._first_non_empty(
                finding.get(
                    "section"
                ),

                standard.get(
                    "section"
                ),

                default="Not determined"
            )

            clause_title = self._first_non_empty(
                finding.get(
                    "clause_title"
                ),

                standard.get(
                    "title"
                ),

                default="No specific clause identified"
            )

            applicable_rule = self._first_non_empty(
                standard.get(
                    "acceptance_level"
                ),

                finding.get(
                    "applicable_criteria"
                ),

                finding.get(
                    "acceptance_level"
                ),

                finding.get(
                    "class_requirement"
                ),

                default="Not specified in retrieved evidence."
            )

            electrical_impact = self._first_non_empty(
                standard.get(
                    "electrical_impact"
                ),

                finding.get(
                    "electrical_impact"
                ),

                default="Not specified."
            )

            standard_description = self._first_non_empty(
                standard.get(
                    "description"
                ),

                finding.get(
                    "description"
                ),

                default=""
            )

            acceptance_level = self._first_non_empty(
                standard.get(
                    "acceptance_level"
                ),

                finding.get(
                    "acceptance_level"
                ),

                default=applicable_rule
            )

            if pdf_evidence_used:

                standard_source = (
                    "RAG PDF Evidence"
                )

            elif structured_kb_used:

                standard_source = (
                    "Structured Standards KB"
                )

            else:

                standard_source = (
                    "No confirmed standards mapping"
                )

        # --------------------------------------------------------------
        # GOVERNING STANDARD
        # --------------------------------------------------------------

        standard_reference = self._first_non_empty(
            finding.get(
                "target_standard"
            ),

            standard.get(
                "standard"
            ),

            default="IPC-A-610G"
        )

        # --------------------------------------------------------------
        # REWORK PROCEDURE
        # --------------------------------------------------------------

        if category_unconfirmed:

            rework_procedure = (
                "Specific IPC-7711/7721 rework procedure "
                "cannot be selected until the defect category "
                "is confirmed."
            )

        else:

            rework_procedure = self._first_non_empty(
                standard.get(
                    "rework_procedure"
                ),

                standard.get(
                    "ipc_7721_rework_procedure"
                ),

                finding.get(
                    "ipc_7721_rework"
                ),

                finding.get(
                    "rework_procedure"
                ),

                default=(
                    "No defect-specific rework procedure "
                    "was retrieved."
                )
            )

        # --------------------------------------------------------------
        # ROOT CAUSES
        # --------------------------------------------------------------

        if category_unconfirmed:

            root_causes = []

        else:

            root_causes = self._normalise_list(
                self._first_non_empty(
                    standard.get(
                        "root_cause_factors"
                    ),

                    standard.get(
                        "root_cause_hypotheses"
                    ),

                    finding.get(
                        "root_cause_hypotheses"
                    ),

                    finding.get(
                        "root_cause_factors"
                    ),

                    default=[]
                )
            )

        # --------------------------------------------------------------
        # CONTAINMENT ACTION
        # --------------------------------------------------------------

        if category_unconfirmed:

            containment_action = (
                "Hold board and contain affected lot pending "
                "defect identification and engineering review."
            )

        elif severity_meta["score"] > 40:

            containment_action = (
                "Tag board with Red Reject traveler; "
                "isolate upstream lot."
            )

        else:

            containment_action = (
                "Green Pass traveler."
            )

        # --------------------------------------------------------------
        # REPORT
        # --------------------------------------------------------------

        report = {

            "report_id": report_id,

            "timestamp": timestamp,

            "metadata": {

                "board_serial": board_sn,

                "lot_id": lot_id,

                "line_id": line_id,

                "operator_id": operator_id,

                "operating_class": operating_class,

                "standard_reference":
                    standard_reference
            },

            # ----------------------------------------------------------
            # INSPECTION SUMMARY
            # ----------------------------------------------------------

            "inspection_summary": {

                "defect_classification":
                    defect_type,

                "cbam_classification":
                    cbam_classification,

                "cbam_confidence_percentage":
                    round(
                        cbam_confidence * 100,
                        2
                    ),

                "protonet_category":
                    (
                        proto_category
                        if category_confirmed
                        else None
                    ),

                "protonet_confidence_percentage_REMOVED":
                    round(
                        proto_confidence * 100,
                        2
                    ),

                "protonet_category_confirmed":
                    category_confirmed,

                "finding_status":
                    finding_status,

                "confidence_percentage":
                    round(
                        cbam_confidence * 100,
                        2
                    ),

                "disposition":
                    (
                        "NON-CONFORMANCE"
                        if cbam_is_anomaly
                        else finding.get(
                            "disposition",
                            "CONFORMANCE"
                        )
                    ),

                "audit_severity_index":
                    severity_meta[
                        "score"
                    ],

                "severity_category":
                    severity_meta[
                        "level"
                    ],

                "severity_level":
                    severity_level,

                "risk_factor":
                    round(
                        risk_factor,
                        3
                    ),

                "defect_area_ratio":
                    round(
                        defect_area_ratio,
                        4
                    ),

                "action_required":
                    severity_meta[
                        "recommended_action"
                    ]
            },

            # ----------------------------------------------------------
            # STANDARDS COMPLIANCE
            # ----------------------------------------------------------

            "standards_compliance": {

                "clause_id":
                    standard_id,

                "section":
                    section,

                "clause_title":
                    clause_title,

                "standard":
                    standard_reference,

                "standard_source":
                    standard_source,

                "applicable_rule":
                    applicable_rule,

                "class_requirement":
                    acceptance_level,

                "acceptance_level":
                    acceptance_level,

                "electrical_impact":
                    electrical_impact,

                "severity_level":
                    severity_level,

                "risk_factor":
                    risk_factor,

                "description":
                    standard_description
            },

            # ----------------------------------------------------------
            # RAG GROUNDING
            # ----------------------------------------------------------

            "rag_grounding": {

                "knowledge_source":
                    knowledge_source,

                "standard_source":
                    standard_source,

                "pdf_evidence_used":
                    pdf_evidence_used,

                "structured_kb_used":
                    structured_kb_used,

                "llm_generated":
                    llm_generated,

                "evidence_count":
                    len(
                        retrieved_evidence
                    ),

                "evidence_summary":
                    evidence_summary,

                "retrieved_evidence":
                    retrieved_evidence
            },

            # ----------------------------------------------------------
            # REWORK / CAPA
            # ----------------------------------------------------------

            "rework_and_capa": {

                "ipc_7721_procedure":
                    rework_procedure,

                "root_cause_factors":
                    root_causes,

                "containment_action":
                    containment_action
            },

            # ----------------------------------------------------------
            # VISUAL LOCALIZATION
            # ----------------------------------------------------------

            "visual_localization": {

                "bounding_boxes":
                    visual.get(
                        "bounding_boxes",
                        []
                    ),

                "heatmap_data_uri":
                    visual.get(
                        "overlay_image_b64",
                        ""
                    ),

                "original_data_uri":
                    visual.get(
                        "original_image_b64",
                        ""
                    )
            }
        }

        # --------------------------------------------------------------
        # SAVE JSON
        # --------------------------------------------------------------

        if self.log_dir:

            file_path = os.path.join(
                self.log_dir,
                f"{report_id}.json"
            )

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    report,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        return report

    # ------------------------------------------------------------------
    # HTML CERTIFICATE
    # ------------------------------------------------------------------

    def render_html_report(
        self,
        report: Dict[str, Any]
    ) -> str:
        """
        Renders a self-contained printable/exportable HTML certificate.
        """

        meta = report[
            "metadata"
        ]

        summary = report[
            "inspection_summary"
        ]

        st = report[
            "standards_compliance"
        ]

        rag = report.get(
            "rag_grounding",
            {}
        )

        rw = report[
            "rework_and_capa"
        ]

        vis = report[
            "visual_localization"
        ]

        defect_type = summary.get(
            "defect_classification",
            "Unknown"
        )

        cbam_classification = summary.get(
            "cbam_classification",
            "Unknown"
        )

        cbam_confidence = summary.get(
            "cbam_confidence_percentage",
            summary.get(
                "confidence_percentage",
                0
            )
        )

        proto_category = summary.get(
            "protonet_category"
        )

        proto_confidence = summary.get(
            "protonet_confidence_percentage_REMOVED",
            0
        )

        proto_confirmed = summary.get(
            "protonet_category_confirmed",
            False
        )

        finding_status = summary.get(
            "finding_status",
            ""
        )

        severity_score = float(
            summary.get(
                "audit_severity_index",
                0.0
            )
        )

        severity_category = summary.get(
            "severity_category",
            "UNKNOWN"
        )

        # --------------------------------------------------------------
        # DISPLAY VALUES
        # --------------------------------------------------------------

        if proto_confirmed and proto_category:

            proto_display = (
                f"{escape(str(proto_category))} "
                f"(CONFIRMED)"
            )

        else:

            proto_display = (
                "Not confidently identified"
            )

        cbam_is_anomaly = (
            str(
                cbam_classification
            ).lower()
            in {
                "anomaly",
                "abnormal",
                "defect",
                "defective"
            }
        )

        cbam_is_normal = (
            str(
                cbam_classification
            ).lower()
            in {
                "normal",
                "good",
                "pass",
                "ok"
            }
        )

        if cbam_is_normal:

            display_finding = (
                "Good Assembly"
            )

        elif proto_confirmed and proto_category:

            display_finding = escape(
                str(proto_category)
            )

        elif cbam_is_anomaly:

            display_finding = (
                "Anomaly — defect category "
                "not confidently identified"
            )

        else:

            display_finding = escape(
                str(defect_type)
            )

        is_confirmed_defect = (
            proto_confirmed
            and bool(proto_category)
        )

        is_unconfirmed_anomaly = (
            cbam_is_anomaly
            and not is_confirmed_defect
        )

        # --------------------------------------------------------------
        # HEADER COLOR
        # --------------------------------------------------------------

        if is_unconfirmed_anomaly:

            header_color = "#eab308"

        elif severity_score >= 75:

            header_color = "#ef4444"

        elif severity_score >= 45:

            header_color = "#f97316"

        elif severity_score >= 20:

            header_color = "#eab308"

        else:

            header_color = "#22c55e"

        # --------------------------------------------------------------
        # IPC DISPLAY
        # --------------------------------------------------------------

        standards_source = st.get(
            "standard_source",
            "Not specified"
        )

        if is_unconfirmed_anomaly:

            ipc_display = (
                "No specific IPC-A-610 clause confirmed"
            )

        elif (
            standards_source
            == "No confirmed standards mapping"
        ):

            ipc_display = (
                "No specific IPC-A-610 clause confirmed"
            )

        else:

            ipc_display = (
                f"{escape(str(st.get('standard', 'IPC-A-610G')))}"
                f" | Clause "
                f"{escape(str(st.get('section', 'Not determined')))}"
                f" - "
                f"{escape(str(st.get('clause_title', 'Not determined')))}"
            )

        # --------------------------------------------------------------
        # RAG EVIDENCE HTML
        # --------------------------------------------------------------

        evidence_items = rag.get(
            "retrieved_evidence",
            []
        )

        evidence_html = ""

        if evidence_items:
            for idx, evidence in enumerate(evidence_items, 1):
                source = escape(str(evidence.get("source", "")))
                page = evidence.get("page", "N/A")
                score = evidence.get("semantic_score", evidence.get("score"))
        
                if isinstance(score, (int, float)):
                    score_text = f"{score:.3f}"
                else:
                    score_text = "N/A"
        
                raw_text = str(evidence.get("text", "")).strip()
                excerpt = " ".join(raw_text.split())
        
                if len(excerpt) > 450:
                    excerpt = excerpt[:450].rsplit(" ", 1)[0] + "..."
        
                excerpt = escape(excerpt)
        
                evidence_html += f"""
<div class="evidence-item">
    <div class="evidence-header">
        <strong>Evidence {idx}</strong>
        <span>
            {source}
            |
            Page {page}
            |
            Score: {score_text}
        </span>
    </div>
    <div class="evidence-text">
        <strong>Evidence:</strong>
        {excerpt}
    </div>
</div>
"""


        else:

            evidence_html = """
            <div class="empty-evidence">
                No PDF evidence was retrieved for this finding.
            </div>
            """

        # --------------------------------------------------------------
        # RAG EXPLANATION
        # --------------------------------------------------------------

        pdf_used = bool(
            rag.get(
                "pdf_evidence_used",
                False
            )
        )

        structured_used = bool(
            rag.get(
                "structured_kb_used",
                False
            )
        )

        if is_unconfirmed_anomaly:

            grounding_notice = (
                "No specific IPC-A-610 clause was assigned "
                "because CBAM detected an anomaly but ProtoNet "
                "did not confidently confirm a defect category. "
                "No default defect mapping has been applied."
            )

        elif (
            standards_source
            == "No confirmed standards mapping"
        ):

            grounding_notice = (
                "No confirmed standards mapping is available "
                "for this finding."
            )

        elif pdf_used:

            grounding_notice = (
                "IPC information below is grounded "
                "in retrieved PDF evidence."
            )

        elif structured_used:

            grounding_notice = (
                "PDF evidence was not retrieved. "
                "The displayed IPC information comes "
                "from the structured standards knowledge base."
            )

        else:

            grounding_notice = (
                "No confirmed standards evidence is available "
                "for this finding."
            )

        grounding_notice = escape(
            grounding_notice
        )

        # --------------------------------------------------------------
        # ROOT CAUSES
        # --------------------------------------------------------------

        if rw.get(
            "root_cause_factors"
        ):

            root_cause_text = ", ".join(
                str(x)
                for x in rw.get(
                    "root_cause_factors",
                    []
                )
            )

        else:

            root_cause_text = (
                "No root-cause hypotheses recorded."
            )

        root_cause_text = escape(
            root_cause_text
        )

        # --------------------------------------------------------------
        # SAFE HTML VALUES
        # --------------------------------------------------------------

        report_id = escape(
            str(report.get("report_id", ""))
        )

        report_timestamp = escape(
            str(report.get("timestamp", ""))
        )

        board_serial = escape(
            str(meta.get("board_serial", ""))
        )

        lot_id = escape(
            str(meta.get("lot_id", ""))
        )

        line_id = escape(
            str(meta.get("line_id", ""))
        )

        operator_id = escape(
            str(meta.get("operator_id", ""))
        )

        standard_reference = escape(
            str(meta.get("standard_reference", "IPC-A-610G"))
        )

        operating_class = escape(
            str(meta.get("operating_class", "Class 3"))
        )

        disposition = escape(
            str(summary.get(
                "disposition",
                "NON-CONFORMANCE"
            ))
        )

        finding_status = escape(
            str(finding_status)
        )

        defect_type_display = escape(
            str(defect_type)
        )

        severity_level_display = escape(
            str(
                summary.get(
                    "severity_level",
                    severity_category
                )
            )
        )

        action_required = escape(
            str(
                summary.get(
                    "action_required",
                    ""
                )
            )
        )

        standard_source_display = escape(
            str(
                st.get(
                    "standard_source",
                    "Not specified"
                )
            )
        )

        class_requirement = escape(
            str(
                st.get(
                    "class_requirement",
                    "Not specified."
                )
            )
        )

        acceptance_level = escape(
            str(
                st.get(
                    "acceptance_level",
                    "Not specified."
                )
            )
        )

        electrical_impact = escape(
            str(
                st.get(
                    "electrical_impact",
                    "Not determined."
                )
            )

        )

        description = escape(
            str(
                st.get(
                    "description",
                    ""
                )
            )
        )

        knowledge_source = escape(
            str(
                rag.get(
                    "knowledge_source",
                    "Not specified"
                )
            )
        )

        evidence_count = len(
            rag.get(
                "retrieved_evidence",
                []
            )
        )

        retrieval_summary = (
            f"{evidence_count} supporting PDF source(s) retrieved. "
            "Full passages are retained in the audit record."
        )

        rework_procedure = escape(
            str(
                rw.get(
                    "ipc_7721_procedure",
                    ""
                )
            )
        )

        original_image = escape(
            str(
                vis.get(
                    "original_data_uri",
                    ""
                )
            ),
            quote=True
        )

        heatmap_image = escape(
            str(
                vis.get(
                    "heatmap_data_uri",
                    ""
                )
            ),
            quote=True
        )

        # --------------------------------------------------------------
        # HTML
        # --------------------------------------------------------------

        html = f"""<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <title>
        Quality Audit Report - {report_id}
    </title>

    <style>

        body {{
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                sans-serif;

            margin: 40px;

            background: #0f172a;

            color: #f8fafc;
        }}

        .certificate {{
            max-width: 950px;

            margin: 0 auto;

            background: #1e293b;

            border: 1px solid #334155;

            border-radius: 12px;

            padding: 32px;

            box-shadow:
                0 10px 25px rgba(0,0,0,0.5);
        }}

        .header {{
            display: flex;

            justify-content:
                space-between;

            align-items:
                center;

            border-bottom:
                2px solid {header_color};

            padding-bottom: 16px;

            margin-bottom: 24px;
        }}

        .title {{
            font-size: 24px;

            font-weight: 700;

            color: #f8fafc;
        }}

        .badge {{
            background:
                {header_color};

            color: #fff;

            padding: 6px 14px;

            border-radius: 9999px;

            font-size: 13px;

            font-weight: 700;

            text-transform:
                uppercase;
        }}

        .grid {{
            display: grid;

            grid-template-columns:
                repeat(2, 1fr);

            gap: 16px;

            margin-bottom: 24px;
        }}

        .card {{
            background: #0f172a;

            border: 1px solid #334155;

            border-radius: 8px;

            padding: 16px;
        }}

        .label {{
            font-size: 11px;

            text-transform:
                uppercase;

            color: #94a3b8;

            letter-spacing:
                0.05em;

            margin-bottom: 4px;
        }}

        .val {{
            font-size: 15px;

            font-weight: 600;

            color: #f8fafc;
        }}

        .detail {{
            font-size: 13px;

            color: #cbd5e1;

            line-height: 1.55;
        }}

        .severity-gauge {{
            background: #334155;

            border-radius: 9999px;

            height: 10px;

            width: 100%;

            overflow: hidden;

            margin-top: 8px;
        }}

        .severity-bar {{
            height: 100%;

            width:
                {max(0.0, min(100.0, severity_score))}%;

            background:
                {header_color};
        }}

        .images {{
            display: flex;

            gap: 16px;

            margin-bottom: 24px;
        }}

        .img-container {{
            flex: 1;

            background: #0f172a;

            border:
                1px solid #334155;

            border-radius: 8px;

            padding: 8px;

            text-align: center;
        }}

        .img-container img {{
            max-width: 100%;

            height: auto;

            border-radius: 4px;
        }}

        .rework-box {{
            background:
                rgba(59, 130, 246, 0.1);

            border:
                1px solid #3b82f6;

            border-radius: 8px;

            padding: 16px;

            margin-bottom: 24px;
        }}

        .rag-box {{
            background:
                rgba(16, 185, 129, 0.08);

            border:
                1px solid #10b981;

            border-radius: 8px;

            padding: 16px;

            margin-bottom: 24px;
        }}

        .evidence-item {{
            background: #0f172a;

            border:
                1px solid #334155;

            border-radius: 8px;

            padding: 12px;

            margin-top: 10px;
        }}

        .evidence-header {{
            display: flex;

            justify-content:
                space-between;

            gap: 12px;

            font-size: 12px;

            color: #94a3b8;

            margin-bottom: 8px;
        }}

        .evidence-header strong {{
            color: #6ee7b7;
        }}

        .evidence-text {{
            font-size: 13px;

            line-height: 1.55;

            color: #cbd5e1;
        }}

        .empty-evidence {{
            color: #94a3b8;

            font-size: 13px;

            padding: 8px 0;
        }}

        .footer {{
            display: flex;

            justify-content:
                space-between;

            border-top:
                1px solid #334155;

            padding-top: 16px;

            font-size: 12px;

            color: #64748b;
        }}

        .model-box {{
            margin-top: 14px;

            padding: 12px;

            border:
                1px solid #334155;

            border-radius: 8px;

            background:
                #111827;
        }}

        .model-row {{
            display: flex;

            justify-content:
                space-between;

            gap: 12px;

            padding: 5px 0;

            font-size: 13px;

            color: #cbd5e1;
        }}

        .model-name {{
            font-weight: 700;

            color: #f8fafc;
        }}

        .status-note {{
            margin-top: 10px;

            padding: 10px;

            border-left:
                3px solid #64748b;

            background:
                rgba(100,116,139,0.08);

            color: #cbd5e1;

            font-size: 12px;

            line-height: 1.5;
        }}

        .grounding-note {{
            margin-top: 10px;

            padding: 10px;

            border-radius: 6px;

            background:
                rgba(15,23,42,0.6);

            color: #cbd5e1;

            font-size: 12px;

            line-height: 1.5;
        }}

        .warning-note {{
            margin-top: 10px;

            padding: 12px;

            border:
                1px solid #eab308;

            border-radius: 7px;

            background:
                rgba(234,179,8,0.08);

            color: #fde68a;

            font-size: 12px;

            line-height: 1.55;
        }}

        @media (max-width: 700px) {{

            .grid,
            .images {{
                grid-template-columns:
                    1fr;

                display: block;
            }}

            .img-container {{
                margin-bottom: 16px;
            }}

            .model-row {{
                display: block;
            }}
        }}

        @media print {{

            body {{
                background: #fff;

                color: #000;

                margin: 15px;
            }}

            .certificate {{
                background: #fff;

                border:
                    1px solid #ccc;

                color: #000;

                box-shadow: none;
            }}

            .card,
            .evidence-item {{
                background: #fff;

                border:
                    1px solid #ccc;
            }}
        }}

    </style>

</head>

<body>

<div class="certificate">

    <!-- HEADER -->

    <div class="header">

        <div>

            <div class="title">
                ManufacturingRAG-QA
                Inspection Certificate
            </div>

            <div style="
                font-size: 13px;
                color: #94a3b8;
                margin-top: 4px;
            ">

                Standard:
                {standard_reference}

                |

                Operating Level:
                {operating_class}

            </div>

        </div>

        <div class="badge">

            {disposition}

        </div>

    </div>


    <!-- IDENTIFIERS + ASI -->

    <div class="grid">

        <div class="card">

            <div class="label">
                Report & Board Identifiers
            </div>

            <div class="val">
                ID:
                {report_id}
            </div>

            <div class="val"
                 style="margin-top: 4px;">

                Serial:
                {board_serial}

            </div>

            <div style="
                font-size: 12px;
                color: #94a3b8;
                margin-top: 4px;
            ">

                Lot:
                {lot_id}

                |

                Line:
                {line_id}

            </div>

        </div>


        <div class="card">

            <div class="label">
                Audit Severity Index (ASI)
            </div>

            <div class="val"
                 style="
                    color: {header_color};
                    font-size: 20px;
                 ">

                {severity_score}
                / 100.0

                <span style="
                    font-size: 13px;
                    font-weight: normal;
                    color: #94a3b8;
                ">

                    ({escape(str(severity_category))})

                </span>

            </div>

            <div class="severity-gauge">

                <div class="severity-bar"></div>

            </div>

            <div style="
                font-size: 12px;
                color: #94a3b8;
                margin-top: 6px;
            ">

                CBAM Confidence:
                {cbam_confidence}%

            </div>

            <div style="
                font-size: 12px;
                color: #94a3b8;
                margin-top: 4px;
            ">

                Risk Factor:
                {summary.get('risk_factor', 0)}

            </div>

        </div>

    </div>


    <!-- INSPECTION FINDING -->

    <div class="card"
         style="margin-bottom: 24px;">

        <div class="label">
            Inspection Finding
        </div>

        <div class="val"
             style="
                font-size: 17px;
                margin-bottom: 8px;
             ">

            {display_finding}

        </div>


        <div class="model-box">

            <div class="model-row">

                <span class="model-name">
                    CBAM Binary Decision
                </span>

                <span>
                    {escape(str(cbam_classification))}
                </span>

            </div>

            <div class="model-row">

                <span class="model-name">
                    CBAM Confidence
                </span>

                <span>
                    {cbam_confidence}%
                </span>

            </div>

            <div class="model-row">

                <span class="model-name">
                    ProtoNet Category
                </span>

                <span>
                    {proto_display}
                </span>

            </div>

            

        </div>


        <div class="status-note">

            <strong>Decision Explanation:</strong>

            {finding_status}

        </div>


        {(
            '''
            <div class="warning-note">
                <strong>Inspection Status:</strong>
                CBAM detected an anomaly, but the specific defect
                type has not been confirmed. This result requires
                containment and further defect identification.
                No defect-specific IPC clause or rework procedure
                has been assigned.
            </div>
            '''
            if is_unconfirmed_anomaly
            else ''
        )}


        <div class="detail"
             style="margin-top: 12px;">

            <strong>Reported Finding:</strong>
            {defect_type_display}

        </div>

        <div class="detail"
             style="margin-top: 6px;">

            <strong>Severity:</strong>
            {severity_level_display}

        </div>

        <div class="detail"
             style="margin-top: 6px;">

            <strong>Required Action:</strong>
            {action_required}

        </div>

    </div>


    <!-- IPC STANDARD -->

    <div class="card"
         style="margin-bottom: 24px;">

        <div class="label">
            IPC-A-610 Violation &
            Standard Grounding
        </div>

        <div class="val"
             style="
                font-size: 16px;
                margin-bottom: 8px;
             ">

            {ipc_display}

        </div>

        <div class="detail">

            <strong>Standard Source:</strong>
            {standard_source_display}

        </div>

        <div class="detail"
             style="margin-top: 8px;">

            <strong>Class Requirement:</strong>
            {class_requirement}

        </div>

        <div class="detail"
             style="margin-top: 8px;">

            <strong>Acceptance Level:</strong>
            {acceptance_level}

        </div>

        <div class="detail"
             style="
                margin-top: 8px;
                color: #fca5a5;
             ">

            <strong>Electrical Impact:</strong>
            {electrical_impact}

        </div>

        <div class="detail"
             style="margin-top: 8px;">

            <strong>Description:</strong>
            {description}

        </div>

    </div>


    <!-- IMAGES -->

    <div class="images">

        <div class="img-container">

            <div class="label">
                Original Optical Image
            </div>

            <img
                src="{original_image}"
                alt="Original Inspection Image"
            />

        </div>


        <div class="img-container">

            <div class="label">
                CBAM + Grad-CAM
                Saliency Overlay
            </div>

            <img
                src="{heatmap_image}"
                alt="Grad-CAM Overlay"
            />

        </div>

    </div>


    <!-- RAG EVIDENCE -->

    <div class="rag-box">

        <div class="label"
             style="color: #6ee7b7;">

            RAG Evidence & Grounding

        </div>


        <div class="detail"
             style="margin-top: 6px;">

            <strong>Knowledge Source:</strong>

            {knowledge_source}

        </div>


        <div class="detail"
             style="margin-top: 6px;">

            <strong>PDF Evidence Retrieved:</strong>

            {str(
                bool(
                    rag.get(
                        "pdf_evidence_used",
                        False
                    )
                )
            ).upper()}

            &nbsp; | &nbsp;

            <strong>Evidence Count:</strong>

            {rag.get(
                "evidence_count",
                0
            )}

        </div>


        <div class="detail"
             style="margin-top: 6px;">

            <strong>Structured Standards KB:</strong>

            {str(
                bool(
                    rag.get(
                        "structured_kb_used",
                        False
                    )
                )
            ).upper()}

        </div>


        <div class="grounding-note">

            {grounding_notice}

        </div>


        <div class="detail"
             style="
                margin-top: 10px;
                color: #cbd5e1;
             ">

            <strong>Retrieval Summary:</strong>
            <br>

            {retrieval_summary}

        </div>


        {evidence_html}

    </div>


    <!-- REWORK -->

    <div class="rework-box">

        <div class="label"
             style="color: #60a5fa;">

            IPC-7711 / 7721
            Standard Corrective
            Rework Procedure

        </div>

        <div style="
            font-size: 14px;
            color: #e2e8f0;
            margin-top: 4px;
            line-height: 1.55;
        ">

            {rework_procedure}

        </div>

        <div style="
            font-size: 12px;
            color: #94a3b8;
            margin-top: 8px;
            line-height: 1.5;
        ">

            <strong>
                Root Cause Hypotheses:
            </strong>

            {root_cause_text}

        </div>

    </div>


    <!-- FOOTER -->

    <div class="footer">

        <div>

            Audited by:
            {operator_id}

            |

            Automated
            CBAM-ProtoNet
            Inspection System

        </div>

        <div>

            Date:
            {report_timestamp}

        </div>

    </div>

</div>

</body>

</html>
"""

        return html


