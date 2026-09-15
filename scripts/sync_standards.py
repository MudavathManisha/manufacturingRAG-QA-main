"""
Sync standards PDFs into local data/standards directory.
"""
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_STANDARDS_DIR = os.path.join(PROJECT_ROOT, "data", "standards")
SOURCE_STANDARDS_DIR = "D:/ManufacturingRAG-QA/data/standards"

os.makedirs(LOCAL_STANDARDS_DIR, exist_ok=True)

copied = 0
if os.path.exists(SOURCE_STANDARDS_DIR):
    for filename in os.listdir(SOURCE_STANDARDS_DIR):
        if filename.lower().endswith(".pdf"):
            src = os.path.join(SOURCE_STANDARDS_DIR, filename)
            dst = os.path.join(LOCAL_STANDARDS_DIR, filename)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copy2(src, dst)
                print(f"Copied: {filename} ({os.path.getsize(dst):,} bytes)")
                copied += 1
            else:
                print(f"Already present: {filename}")

extra_pdfs = [
    "PCBA-Checklist_0.pdf",
    "TIPCB assembly-guideness.pdf",
    "TI_PCB_Issue_guideliness.pdf",
    "infineon-integratedpowerstage-drmos-solderingguidelines-ap-en.pdf",
    "surface_mount_components.pdf"
]
for pdf in extra_pdfs:
    src = os.path.join("D:/", pdf)
    dst = os.path.join(LOCAL_STANDARDS_DIR, pdf)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Copied from root: {pdf}")
        copied += 1

print(f"Total PDFs in {LOCAL_STANDARDS_DIR}: {len(os.listdir(LOCAL_STANDARDS_DIR))}")
