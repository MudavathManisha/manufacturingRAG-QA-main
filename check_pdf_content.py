import fitz
import glob
import os

pdfs = glob.glob(r"D:\ManufacturingRAG-QA\data\standards\*.pdf")

print("PDFs:", len(pdfs))

for pdf in pdfs:
    print("\n" + "=" * 70)
    print(os.path.basename(pdf))
    print("=" * 70)

    doc = fitz.open(pdf)
    found = False

    for page_num, page in enumerate(doc, start=1):
        text = " ".join(page.get_text("text").split())
        lower = text.lower()

        if any(keyword in lower for keyword in [
            "solder bridge",
            "solder bridging",
            "5.2.1"
        ]):
            found = True

            print(f"\nPAGE {page_num}:")
            print(text[:2000])

    if not found:
        print("No matching text found.")

    doc.close()