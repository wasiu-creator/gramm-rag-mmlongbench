"""
Download all 135 MMLongBench-Doc PDFs from HuggingFace.

Usage:
    python download_pdfs.py

PDFs are saved to data/mmlongbench/pdfs/.
Skips files already present (safe to re-run).
Requires: pip install huggingface_hub
"""

import shutil
from pathlib import Path

HF_REPO  = "yubo2333/MMLongBench-Doc"
PDF_DIR  = Path(__file__).parent / "data" / "mmlongbench" / "pdfs"
RAW_DIR  = PDF_DIR / "hf_raw"

PDF_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading PDFs from {HF_REPO} to {PDF_DIR} ...")
print("(~2 GB — may take 5–30 min depending on connection)\n")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("Run: pip install huggingface_hub")

snapshot_download(
    repo_id=HF_REPO,
    repo_type="dataset",
    local_dir=str(RAW_DIR),
    allow_patterns=["*.pdf"],
    ignore_patterns=["*.parquet", "*.json", "*.arrow", ".gitattributes"],
)

# Flatten nested subfolders into PDF_DIR
copied = skipped = failed = 0
for src in RAW_DIR.rglob("*.pdf"):
    dest = PDF_DIR / src.name
    if dest.exists():
        skipped += 1
        continue
    try:
        shutil.copy2(str(src), str(dest))
        copied += 1
    except Exception as e:
        print(f"  WARNING: could not copy {src.name}: {e}")
        failed += 1

total = len(list(PDF_DIR.glob("*.pdf")))
print(f"\nDone. Copied: {copied}  Skipped (cached): {skipped}  Failed: {failed}")
print(f"Total PDFs in {PDF_DIR}: {total}")
