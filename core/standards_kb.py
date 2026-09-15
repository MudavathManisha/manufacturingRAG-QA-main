"""
Manufacturing Standards Knowledge Base.
Structured repository of IPC-A-610G, IPC-7711/7721, J-STD-001H, and ISO-9001
clauses, Class 1/2/3 acceptance criteria, defect thresholds, and rework guidelines.
"""

from typing import List, Dict, Any, Optional

IPC_STANDARDS_DATABASE: List[Dict[str, Any]] = [
    {
        "id": "IPC-610-5.2.1",
        "standard": "IPC-A-610G",
        "section": "5.2.1",
        "title": "Solder Bridging / Solder Short",
        "category": "Soldering Anomalies",
        "defect_type": "Solder_Bridge",
        "description": "Solder bridging occurs when solder establishes an unintended conductive connection between adjacent non-common electrical traces, pads, or component leads.",
        "class_1_criteria": "Defect - Solder crosses spacing to adjacent non-common conductor.",
        "class_2_criteria": "Defect - Solder bridges across any non-common conductor or violates minimum electrical clearance.",
        "class_3_criteria": "Defect - Solder bridges across any adjacent conductor or reduces minimum electrical spacing below design specification.",
        "acceptance_level": "DEFECT Class 1, 2, 3",
        "severity_level": "Critical",
        "risk_factor": 0.98,
        "electrical_impact": "Direct short-circuit risk, power rail collapse, component latch-up or fire hazard.",
        "ipc_7721_rework_procedure": "Procedure 3.1.2 - Solder Wick De-bridging: Apply RMA/No-Clean liquid flux, place copper braid over bridge, heat with fine chisel tip at 315°C (lead-free: 350°C), wick excess solder, clean and inspect joint fillet.",
        "root_cause_factors": [
            "Excessive solder paste volume due to stencil aperture thickness/wear",
            "Improper stencil-to-board print alignment (< 25um accuracy)",
            "Incorrect reflow peak temperature or excessive solder paste slump",
            "Insufficient solder mask dam between fine-pitch IC leads"
        ]
    },
    {
        "id": "IPC-610-8.3.4",
        "standard": "IPC-A-610G",
        "section": "8.3.4",
        "title": "Missing Component",
        "category": "Component Placement",
        "defect_type": "Missing_Component",
        "description": "The designated electronic component is absent from its footprint on the assembled PCB.",
        "class_1_criteria": "Defect - Component missing from intended location.",
        "class_2_criteria": "Defect - Component missing from intended location.",
        "class_3_criteria": "Defect - Component missing from intended location.",
        "acceptance_level": "DEFECT Class 1, 2, 3",
        "severity_level": "Critical",
        "risk_factor": 0.95,
        "electrical_impact": "Open circuit, circuit inoperability, missing decoupling or pull-up/down termination.",
        "ipc_7721_rework_procedure": "Procedure 2.1.1 - Component Manual Replacement: Inspect pad integrity, re-apply controlled solder paste/flux, place replacement component using vacuum pick-up, reflow with hot air station or fine iron.",
        "root_cause_factors": [
            "Pick-and-place nozzle vacuum loss or clogged nozzle tip",
            "Component tape pocket jamming or peeling tape splice failure",
            "SMD feeder vibration or incorrect feeder pitch configuration",
            "Surface tension imbalance washing component into reflow air stream"
        ]
    },
    {
        "id": "IPC-610-8.3.3",
        "standard": "IPC-A-610G",
        "section": "8.3.3",
        "title": "Side Overhang / Component Misalignment",
        "category": "Component Placement",
        "defect_type": "Component_Misaligned",
        "description": "Component termination extends laterally off the side of the designated land pad.",
        "class_1_criteria": "Acceptable - Side overhang (A) is <= 50% of component termination width (W), provided minimum electrical clearance is maintained.",
        "class_2_criteria": "Acceptable - Side overhang (A) is <= 50% of component termination width (W). Defect if > 50% W or if clearance violated.",
        "class_3_criteria": "Acceptable - Side overhang (A) is <= 25% of component termination width (W). Defect if > 25% W.",
        "acceptance_level": "CONDITIONAL (Acceptable Class 1/2 <= 50% W, Defect Class 3 > 25% W)",
        "severity_level": "Major",
        "risk_factor": 0.72,
        "electrical_impact": "Reduced mechanical shear strength, parasitic capacitance shift, potential solder bridging to adjacent tracks under vibration.",
        "ipc_7721_rework_procedure": "Procedure 2.2.4 - Component Realignment: Apply mild RMA flux, heat both terminations simultaneously using dual-tip tweezer iron or hot air reflow nozzle, align component with precision tweezers, cool without mechanical disturbance.",
        "root_cause_factors": [
            "Pick-and-place optical centering camera calibration drift",
            "High conveyor acceleration/deceleration causing component shifting prior to reflow",
            "Uneven solder paste printing volume causing unequal surface tension (swimming)",
            "PCB fiducial mark oxidation or poor optical recognition"
        ]
    },
    {
        "id": "IPC-610-8.3.1",
        "standard": "IPC-A-610G",
        "section": "8.3.1",
        "title": "Tombstoning / Billboarding (Manhattan Effect)",
        "category": "Component Placement & Soldering",
        "defect_type": "Tombstoning",
        "description": "Chip component stands partially or completely upright on one end with the opposite termination lifted completely off the land pad.",
        "class_1_criteria": "Defect - Component lifted off pad; electrical connection broken or mechanically unstable.",
        "class_2_criteria": "Defect - Component lifted off pad; non-wetted open circuit.",
        "class_3_criteria": "Defect - Component lifted off pad; non-wetted open circuit.",
        "acceptance_level": "DEFECT Class 1, 2, 3",
        "severity_level": "Critical",
        "risk_factor": 0.92,
        "electrical_impact": "Complete electrical open circuit on the affected node, severe mechanical weakness.",
        "ipc_7721_rework_procedure": "Procedure 2.3.1 - Tombstone Desolder & Resolder: Desolder lifted component, wick pads clean, dispense micro-drop of fresh paste or flux, reposition flat, reflow both pads simultaneously.",
        "root_cause_factors": [
            "Thermal mass imbalance: one pad connected to a large ground plane without thermal relief",
            "Uneven solder paste volume printed between opposing pads",
            "Reflow oven heating ramp rate too steep (> 3°C/sec) causing rapid solvent boiling",
            "Uneven component termination metallization or solderability oxidation"
        ]
    },
    {
        "id": "IPC-610-5.2.4",
        "standard": "IPC-A-610G",
        "section": "5.2.4",
        "title": "Solder Balls / Splatter",
        "category": "Soldering Anomalies",
        "defect_type": "Solder_Ball",
        "description": "Small spherical solder particles adhering to solder mask, laminate, or adjacent leads.",
        "class_1_criteria": "Acceptable - Solder balls are entrapped in conformal coating or do not violate minimum electrical clearance (> 0.13mm).",
        "class_2_criteria": "Defect - Solder balls not entrapped/encapsulated, or diameter violates minimum electrical clearance, or > 5 balls per 100mm².",
        "class_3_criteria": "Defect - Any loose or un-entrapped solder ball, or diameter > 0.13mm, or within 0.13mm of adjacent conductors.",
        "acceptance_level": "CONDITIONAL (Class 1 Process Indicator, Defect Class 2/3 if loose/violates clearance)",
        "severity_level": "Moderate",
        "risk_factor": 0.65,
        "electrical_impact": "Vibration-induced dislodgement causing intermittent electrical shorts or ESD breakdown in field operation.",
        "ipc_7721_rework_procedure": "Procedure 3.2.1 - Solder Ball Removal & Cleaning: Scrape loose solder spheres using ESD-safe probe, brush area with IPA (Isopropyl Alcohol), rinse with deionized water, verify with 10x optical inspection.",
        "root_cause_factors": [
            "Moisture absorption in solder paste or PCB substrate prior to reflow",
            "Excessive solder paste slump or stencil squeegee pressure spilling paste onto solder mask",
            "Reflow preheat profile too rapid causing flux explosive outgassing",
            "Solder paste expired beyond shelf life or improperly thawed"
        ]
    },
    {
        "id": "IPC-610-5.2.3",
        "standard": "IPC-A-610G",
        "section": "5.2.3",
        "title": "Insufficient Solder / Poor Wetting / Non-Wetting",
        "category": "Soldering Anomalies",
        "defect_type": "Insufficient_Solder",
        "description": "Solder connection fails to form the required concave fillet or fails to cover the minimum required percentage of the pad/lead surface.",
        "class_1_criteria": "Defect - Fillet height (G) is < 25% of component termination thickness (T) or solder does not cover pad contact area.",
        "class_2_criteria": "Defect - Fillet height (G) is < 50% of termination thickness (T) or side fillet width < 50% of lead width.",
        "class_3_criteria": "Defect - Fillet height (G) is < 75% of termination thickness (T) or wetting contact angle theta > 90° (non-wetting).",
        "acceptance_level": "DEFECT Class 1, 2, 3",
        "severity_level": "Major",
        "risk_factor": 0.82,
        "electrical_impact": "High contact resistance, intermittent signal integrity degradation, thermal fatigue fracture under thermal cycling.",
        "ipc_7721_rework_procedure": "Procedure 3.3.2 - Solder Joint Touch-up: Clean oxidized flux, apply active tacky flux, feed 0.3mm wire solder to heel/toe fillet using temperature-controlled iron at 340°C, ensure 100% wetting with smooth concave meniscus.",
        "root_cause_factors": [
            "Insufficient solder paste volume deposited by clogged stencil aperture",
            "Pad or lead surface oxidation / nickel-gold plating passivation",
            "Insufficient reflow peak temperature or inadequate flux activation time",
            "Solder scavenging by adjacent via-in-pad without tenting"
        ]
    },
    {
        "id": "IPC-610-0.0.0",
        "standard": "IPC-A-610G",
        "section": "General",
        "title": "Good Assembly / Target Condition",
        "category": "Acceptance Standards",
        "defect_type": "Good_Assembly",
        "description": "Assembly meets or exceeds all Target conditions across IPC-A-610 Classes 1, 2, and 3.",
        "class_1_criteria": "Target - Complete compliance with design tolerances and electrical requirements.",
        "class_2_criteria": "Target - Centered components, full wetting, smooth concave fillets, zero bridging, zero contamination.",
        "class_3_criteria": "Target - Optimal mechanical joint strength, 100% wetting, minimum 75% fillet height, zero contamination.",
        "acceptance_level": "TARGET ACCEPTABLE All Classes",
        "severity_level": "None",
        "risk_factor": 0.0,
        "electrical_impact": "None - Optimal electrical continuity, mechanical resilience, and long-term reliability.",
        "ipc_7721_rework_procedure": "No rework required. Pass to next manufacturing stage.",
        "root_cause_factors": ["Operating within standard SMT statistical process control (Cpk > 1.67)."]
    },
    {
        "id": "IPC-610-CLASSES",
        "standard": "IPC-A-610G",
        "section": "1.4.1",
        "title": "Classification of Electronic Products (Class 1, 2, 3)",
        "category": "General Principles",
        "defect_type": "Product_Classification",
        "description": "IPC-A-610 establishes three product classes based on end-use reliability and operational criticality: Class 1 (General Electronic Products), Class 2 (Dedicated Service Electronic Products), and Class 3 (High Performance / Harsh Environment Electronic Products).",
        "class_1_criteria": "Class 1: Includes products suitable for applications where the major requirement is the function of the completed assembly (e.g. consumer electronics, toys, basic peripherals). Cosmetic imperfections are acceptable if function is uncompromised.",
        "class_2_criteria": "Class 2: Includes products where continued performance and extended life is required, and for which uninterrupted service is desired but not critical (e.g. business machines, communication equipment, industrial instruments).",
        "class_3_criteria": "Class 3: High Performance / Harsh Environment - Continued performance or performance-on-demand is critical; equipment downtime cannot be tolerated, end-use environment may be uncommonly harsh, and equipment must function when required (e.g. life support medical, aerospace, automotive safety systems, defense).",
        "acceptance_level": "MANDATORY CLASSIFICATION Baseline",
        "severity_level": "Informational",
        "risk_factor": 0.50,
        "electrical_impact": "Classification determines required inspection thresholds, solder fillet heights, clearance distances, and testing rigor.",
        "ipc_7721_rework_procedure": "Rework requirements must align with class criticality. Class 3 assemblies require strict logging, qualified operators, and thermal profile compliance.",
        "root_cause_factors": ["Improper class specification in contract manufacturing documents.", "Mismatched quality plan with customer end-use environment."]
    },
    {
        "id": "IPC-610-5.2.2",
        "standard": "IPC-A-610G",
        "section": "5.2.2",
        "title": "Cold Solder / Disturbed Solder Joint / Fractured Fillet",
        "category": "Soldering Anomalies",
        "defect_type": "Cold_Solder",
        "description": "Cold or disturbed solder connections exhibit poor wetting, grayish/chalky granular surface appearance, uneven meniscus, or visible micro-fissures caused by movement during solidification or insufficient reflow heat.",
        "class_1_criteria": "Defect - Solder connection displays incomplete wetting, cracked fillet, or lack of metallurgical bond.",
        "class_2_criteria": "Defect - Disturbed joint showing movement fissures, cold solder appearance with wetting contact angle > 90°, or fractured fillet.",
        "class_3_criteria": "Defect - Any evidence of cold solder, disturbed joint crystallization, micro-fissuring, or incomplete intermetallic layer formation.",
        "acceptance_level": "DEFECT Class 1, 2, 3",
        "severity_level": "Critical",
        "risk_factor": 0.89,
        "electrical_impact": "Elevated contact resistance, intermittent open circuits under thermal cycling or vibration, complete joint detachment in field.",
        "ipc_7721_rework_procedure": "Procedure 3.3.1 - Joint Reflow & Solder Replacement: Remove disturbed solder using desoldering braid, clean pad, apply liquid RMA/No-Clean flux, resolder joint with controlled thermal profile (peak 245°C for SAC305).",
        "root_cause_factors": [
            "Conveyor vibration or mechanical shock during reflow liquidus-to-solidus transition",
            "Peak reflow temperature below solder liquidus threshold (< 217°C for lead-free SAC305)",
            "Component termination or pad metallization oxidation inhibiting wetting",
            "Inadequate preheat resulting in thermal shock and incomplete flux activation"
        ]
    },
    {
        "id": "IPC-610-5.2.5",
        "standard": "IPC-A-610G",
        "section": "5.2.5",
        "title": "Dewetting and Non-Wetting",
        "category": "Soldering Anomalies",
        "defect_type": "Dewetting_Nonwetting",
        "description": "Non-wetting is the condition whereby liquid solder has contacted a surface but has not adhered, leaving exposed base metal. Dewetting is a condition where solder initially wet the surface but retracted into irregular mounds, leaving a thin solder coating over base metal without full fillet meniscus.",
        "class_1_criteria": "Defect - Non-wetting or dewetting resulting in less than required minimum contact area or fillet height.",
        "class_2_criteria": "Defect - Non-wetting on required solderable surface exceeding 25% of land/lead area, or dewetting causing non-conforming fillets.",
        "class_3_criteria": "Defect - Any non-wetting on functional joint area. Dewetting exceeding 5% of total solderable area on any termination or land.",
        "acceptance_level": "DEFECT Class 1, 2, 3",
        "severity_level": "Major",
        "risk_factor": 0.84,
        "electrical_impact": "Severe degradation of shear strength, high vulnerability to thermal fatigue, intermittent signal attenuation.",
        "ipc_7721_rework_procedure": "Procedure 3.3.3 - Surface Reactivation & Re-soldering: Clean joint, apply active acidic/RMA flux to dissolve oxide layer, tin area with fresh solder wire, wick excess, re-establish concave fillet.",
        "root_cause_factors": [
            "Heavy surface oxidation on copper land pads or component leads due to improper storage",
            "Exhausted or inactive flux in solder paste formulation",
            "Intermetallic compound (Cu6Sn5/Ni3Sn4) overgrowth from excessive thermal exposure",
            "Contamination by silicone, oils, or mold release agents"
        ]
    },
    {
        "id": "IPC-7711-REWORK",
        "standard": "IPC-7711/7721",
        "section": "General",
        "title": "Rework, Modification and Repair of Electronic Assemblies",
        "category": "Rework Guidelines",
        "defect_type": "Rework_General",
        "description": "IPC-7711/7721 provides standardized procedures for desoldering, component removal, land restoration, and re-soldering without degrading laminate integrity or adjacent components.",
        "class_1_criteria": "Acceptable - Rework achieves functional electrical connection and meets minimum mechanical requirements.",
        "class_2_criteria": "Acceptable - Rework restores assembly to full IPC-A-610 Class 2 acceptance criteria. No heat-induced discoloration of laminate exceeding 10% surrounding area.",
        "class_3_criteria": "Acceptable - Rework restores assembly to 100% IPC-A-610 Class 3 acceptance criteria. Thermal cycle count must be logged; maximum 2 rework cycles allowed on any single joint; zero laminate measling, blistering, or pad lifting.",
        "acceptance_level": "STANDARD COMPLIANT with Procedure Logging",
        "severity_level": "Process Standard",
        "risk_factor": 0.40,
        "electrical_impact": "Ensures reworked joints have equivalent electrical conductivity, intermetallic thickness (1-3 um), and fatigue life as original reflow joints.",
        "ipc_7721_rework_procedure": "General Rework Protocol: (1) Preheat board to 100-120°C to minimize thermal gradients; (2) Apply controlled liquid flux matching original chemistry; (3) Use temperature-regulated iron (315-350°C lead-free); (4) Maximum dwell time 3 seconds per joint; (5) Post-rework IPA/DI water cleaning and 10x-40x optical inspection.",
        "root_cause_factors": [
            "Operator thermal overdrive exceeding substrate glass transition temperature (Tg)",
            "Excessive tip mechanical pressure causing copper pad delamination",
            "Incompatible flux residue causing dendritic growth or electrochemical migration"
        ]
    },
    {
        "id": "J-STD-001H",
        "standard": "IPC/EIA J-STD-001H",
        "section": "4.1 & 8.0",
        "title": "Requirements for Soldered Electrical and Electronic Assemblies",
        "category": "Soldering Materials & Processes",
        "defect_type": "Soldering_Process_Requirements",
        "description": "J-STD-001H describes materials, methods, and verification criteria for producing high-quality soldered interconnections. Emphasizes process control, solder purity, flux compatibility, and ionic cleanliness.",
        "class_1_criteria": "Process Standard: Solder joints must exhibit good wetting and electrical continuity.",
        "class_2_criteria": "Process Standard: Solder alloys (SAC305, Sn63Pb37) must conform to purity limits. Flux residues must not cause corrosion or exceed ionic contamination limits (< 1.56 ug/cm² NaCl equivalent).",
        "class_3_criteria": "Process Standard: Strict material traceability. Cleanliness testing mandatory (ROSE testing < 1.30 ug/cm² NaCl equivalent). X-ray inspection required for area array packages (BGA voiding < 25%).",
        "acceptance_level": "MANDATORY Process Quality Baseline",
        "severity_level": "Process Standard",
        "risk_factor": 0.50,
        "electrical_impact": "Prevents electrochemical migration, dendritic shorting, and solder joint brittleness.",
        "ipc_7721_rework_procedure": "All materials utilized in rework must comply with J-STD-001 alloy and flux compatibility guidelines.",
        "root_cause_factors": [
            "Cross-contamination between leaded (SnPb) and lead-free (RoHS) alloys",
            "Inadequate cleaning allowing hygroscopic flux residues to remain on board",
            "Reflow profile exceeding component maximum thermal rating"
        ]
    },
    {
        "id": "ISO-9001-8.7",
        "standard": "ISO 9001:2015",
        "section": "8.7",
        "title": "Control of Nonconforming Outputs",
        "category": "Quality Management",
        "defect_type": "Nonconformance_Control",
        "description": "ISO 9001:2015 Clause 8.7 requires organizations to ensure that outputs that do not conform to requirements are identified, segregated, and controlled to prevent unintended delivery or use.",
        "class_1_criteria": "Mandatory: Non-conforming PCB assemblies must be identified and corrected before dispatch.",
        "class_2_criteria": "Mandatory: Non-conforming assemblies must be segregated, tagged with non-conformance ticket, and evaluated for rework, concession, or scrap.",
        "class_3_criteria": "Mandatory: Formal non-conformance logging (NCR), immediate quarantine containment, root-cause investigation (8D / 5-Why), corrective and preventive action (CAPA) implementation, and customer notification if escape occurred.",
        "acceptance_level": "MANDATORY QMS REQUIREMENT",
        "severity_level": "Management System",
        "risk_factor": 0.90,
        "electrical_impact": "Prevents defective boards from reaching downstream integration or end customers.",
        "ipc_7721_rework_procedure": "Reworked assemblies must be re-verified against original acceptance criteria and certified by authorized quality inspector before release from quarantine.",
        "root_cause_factors": [
            "Bypassing inspection stations under production schedule pressure",
            "Lack of clear boundary samples / golden boards for visual inspection staff",
            "Ineffective containment when automated optical inspection (AOI) detects repeated defects"
        ]
    },
    {
        "id": "ISO-9001-7.1.5",
        "standard": "ISO 9001:2015",
        "section": "7.1.5",
        "title": "Monitoring and Measuring Resources (Calibration & Verification)",
        "category": "Quality Management",
        "defect_type": "Calibration_Measurement_Traceability",
        "description": "Requires inspection, measuring, and test equipment (AOI, X-ray, solder iron stations, reflow profilers, optical microscopes) to be calibrated or verified at specified intervals against measurement standards traceable to international or national standards.",
        "class_1_criteria": "Inspection equipment must be functional and verified.",
        "class_2_criteria": "Equipment calibrated on schedule; out-of-calibration tools tagged and removed from line.",
        "class_3_criteria": "Strict NIST-traceable calibration certificates. If an instrument is found out of calibration, previous inspection lots must be quarantined and re-inspected.",
        "acceptance_level": "MANDATORY QMS REQUIREMENT",
        "severity_level": "Management System",
        "risk_factor": 0.75,
        "electrical_impact": "Ensures inspection measurement accuracy, prevents false acceptances and escapes.",
        "ipc_7721_rework_procedure": "Thermal profiling tools and rework soldering iron tips must have calibrated thermocouples verified every shift.",
        "root_cause_factors": ["Overdue calibration cycles", "Drift in AOI illumination LED intensity", "Degraded thermocouple leads in reflow profiler"]
    }
]


def get_all_standards() -> List[Dict[str, Any]]:
    """Returns the full list of IPC / ISO standard entries."""
    return IPC_STANDARDS_DATABASE


def get_standard_by_defect_type(defect_type: str) -> Optional[Dict[str, Any]]:
    """Fetches the corresponding IPC standard entry for a given defect type."""
    normalized_type = defect_type.strip().lower().replace("_", " ")
    for entry in IPC_STANDARDS_DATABASE:
        entry_defect = entry.get("defect_type", "").lower().replace("_", " ")
        entry_title = entry.get("title", "").lower()
        if entry_defect == normalized_type or entry_title == normalized_type:
            return entry

    # Partial match fallback
    for entry in IPC_STANDARDS_DATABASE:
        entry_defect = entry.get("defect_type", "").lower().replace("_", " ")
        entry_title = entry.get("title", "").lower()
        if normalized_type in entry_defect or normalized_type in entry_title:
            return entry
        if entry_defect in normalized_type or entry_title in normalized_type:
            return entry
    return None


def get_standard_by_id(clause_id: str) -> Optional[Dict[str, Any]]:
    """Fetches standard clause by unique ID."""
    for entry in IPC_STANDARDS_DATABASE:
        if entry["id"].lower() == clause_id.lower():
            return entry
    return None


def search_standards_kb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search structured standards knowledge base by relevance score.
    Returns matched entries with match scores.
    """
    query_clean = query.strip().lower()
    terms = [t for t in query_clean.replace("-", " ").replace("/", " ").split() if len(t) > 2]
    scored = []

    for entry in IPC_STANDARDS_DATABASE:
        score = 0
        haystack = " ".join([
            str(entry.get("id", "")),
            str(entry.get("standard", "")),
            str(entry.get("section", "")),
            str(entry.get("title", "")),
            str(entry.get("category", "")),
            str(entry.get("defect_type", "")),
            str(entry.get("description", "")),
            str(entry.get("class_1_criteria", "")),
            str(entry.get("class_2_criteria", "")),
            str(entry.get("class_3_criteria", "")),
            str(entry.get("ipc_7721_rework_procedure", "")),
            " ".join(entry.get("root_cause_factors", []))
        ]).lower()

        # Exact title / defect type bonus
        if entry.get("title", "").lower() in query_clean:
            score += 10
        if entry.get("defect_type", "").lower().replace("_", " ") in query_clean:
            score += 10

        # Term matches
        for term in terms:
            if term in haystack:
                score += 1.5

        # Class matching
        if "class 3" in query_clean and "class 3" in entry.get("class_3_criteria", "").lower():
            score += 3
        if "rework" in query_clean and entry.get("ipc_7721_rework_procedure"):
            score += 4
        if "iso 9001" in query_clean and "iso 9001" in entry.get("standard", "").lower():
            score += 8
        if "ipc" in query_clean and "ipc" in entry.get("standard", "").lower():
            score += 2

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

