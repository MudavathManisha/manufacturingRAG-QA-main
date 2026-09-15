from pathlib import Path
import re

path = Path(r".\core\audit_reporter.py")

text = path.read_text(encoding="utf-8")

# ------------------------------------------------------------
# FIX 1: Broken NORMAL / NON-DEFECT indentation
# ------------------------------------------------------------

text = text.replace(
'''        # --------------------------------------------------------------
        # NORMAL / NON-DEFECT CASE
        # --------------------------------------------------------------

                # NORMAL / NON-DEFECT CASE
        # Higher confidence that the assembly is normal should
''',
'''        # --------------------------------------------------------------
        # NORMAL / NON-DEFECT CASE
        # --------------------------------------------------------------

        # Higher confidence that the assembly is normal should
'''
)

# ------------------------------------------------------------
# FIX 2: Broken retrieval_summary block
# Handles spaces/tabs from the accidental edit.
# ------------------------------------------------------------

pattern = re.compile(
    r'''        retrieval_summary\s*=\s*len\(\s*
\s*rag\.get\("retrieved_evidence",\s*\[\]\)\s*
\s*\)\s*
\s*retrieval_summary\s*=\s*\(\s*
\s*f"\{evidence_count\} supporting PDF source\(s\) retrieved\. "\s*
\s*"Full passages are retained in the audit record\."\s*
\s*\)''',
    re.MULTILINE
)

replacement = '''        evidence_count = len(
            rag.get(
                "retrieved_evidence",
                []
            )
        )

        retrieval_summary = (
            f"{evidence_count} supporting PDF source(s) retrieved. "
            "Full passages are retained in the audit record."
        )'''

text, count = pattern.subn(replacement, text, count=1)

if count != 1:
    print("WARNING: retrieval_summary block was not matched.")
    print("Trying a broader replacement...")

    start_marker = '        retrieval_summary =len('
    end_marker = '        rework_procedure = escape('

    start = text.find(start_marker)

    if start != -1:
        end = text.find(end_marker, start)

        if end != -1:
            text = (
                text[:start]
                + replacement
                + "\n\n"
                + text[end:]
            )
            count = 1

if count != 1:
    raise SystemExit(
        "ERROR: Could not repair retrieval_summary automatically."
    )

# ------------------------------------------------------------
# FIX 3: Repair accidental mojibake in the anomaly display
# ------------------------------------------------------------

text = text.replace(
    "Anomaly â€” defect category",
    "Anomaly — defect category"
)

# ------------------------------------------------------------
# Write repaired file
# ------------------------------------------------------------

path.write_text(text, encoding="utf-8")

print("SUCCESS: audit_reporter.py repaired.")
