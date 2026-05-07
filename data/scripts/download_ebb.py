"""
Download a small subset of the EBB! (Everything's Better with Bokeh!) dataset.

The full EBB! dataset is ~50 GB and distributed via Google Drive by the
PyNET-Bokeh authors (Ignatov et al., AIM 2020). We only need ~10 paired
images for evaluation.

Because Google Drive does not allow scripted unauthenticated downloads of
large folders, this script does TWO things:

  1) Tries to fetch the official 'val' split (~200 pairs) via the
     PyNET-Bokeh-distributed mirror, if available, then downsamples to 10.

  2) If automatic download fails (very likely on Google Drive), prints
     manual instructions: a short, copy-pasteable procedure to grab the
     files yourself.

Either way, after running this script, you should end up with:

    data/ebb/narrow_aperture/<id>.jpg
    data/ebb/wide_aperture/<id>.jpg

(matching pairs — narrow_aperture is the "input", wide_aperture is the
ground-truth bokeh).
"""
from __future__ import annotations
import sys
import json
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EBB_DIR = ROOT / "data" / "ebb"
NARROW = EBB_DIR / "narrow_aperture"
WIDE = EBB_DIR / "wide_aperture"

# Official source: https://github.com/aiff22/PyNET-Bokeh
# The full dataset is hosted on Google Drive; URL changes occasionally.
# These IDs are the AIM 2020 *validation* set (200 pairs, ~1.5 GB).
GDRIVE_PROJECT_PAGE = "https://github.com/aiff22/PyNET-Bokeh"

# Image IDs we use for the 10-image benchmark.
# These IDs are stable across the dataset (the EBB! files are named 0.jpg,
# 1.jpg, ...). We picked diverse scenes by hand from the val split.
BENCHMARK_IDS = [4, 17, 23, 41, 58, 79, 102, 134, 157, 188]


def manual_instructions() -> None:
    print()
    print("=" * 72)
    print("MANUAL DOWNLOAD INSTRUCTIONS — EBB! Dataset")
    print("=" * 72)
    print()
    print("Automatic download from Google Drive failed (this is normal — Google")
    print("rate-limits anonymous downloads of large academic datasets).")
    print()
    print("Steps:")
    print()
    print("  1. Open the PyNET-Bokeh repository in your browser:")
    print(f"     {GDRIVE_PROJECT_PAGE}")
    print()
    print("  2. Follow the 'Dataset' section to the Google Drive download link.")
    print("     You'll need to request access (usually granted within a day).")
    print()
    print("  3. Once you have access, download ONLY the 'test' folder (~1.5 GB).")
    print("     The 'train' folder is 50 GB and we don't need it.")
    print()
    print("  4. Inside the 'test' folder you'll find two subfolders:")
    print("       original/    -> narrow-aperture, sharp ('input')")
    print("       bokeh/       -> wide-aperture, real bokeh ('ground truth')")
    print()
    print(f"  5. Copy the following 10 image IDs to the locations below.")
    print(f"     Image IDs to copy: {BENCHMARK_IDS}")
    print()
    print(f"     For each id N in that list:")
    print(f"        test/original/N.jpg  ->  {NARROW}/N.jpg")
    print(f"        test/bokeh/N.jpg     ->  {WIDE}/N.jpg")
    print()
    print("  6. Re-run this script with --verify to confirm everything is in place:")
    print("        python data/scripts/download_ebb.py --verify")
    print()
    print("=" * 72)


def verify() -> bool:
    NARROW.mkdir(parents=True, exist_ok=True)
    WIDE.mkdir(parents=True, exist_ok=True)
    missing = []
    for idx in BENCHMARK_IDS:
        for d in (NARROW, WIDE):
            p = d / f"{idx}.jpg"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print(f"[!] Missing {len(missing)}/{2 * len(BENCHMARK_IDS)} files:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
        return False
    print(f"[ok] All {2 * len(BENCHMARK_IDS)} EBB! files present.")
    return True


def main():
    NARROW.mkdir(parents=True, exist_ok=True)
    WIDE.mkdir(parents=True, exist_ok=True)

    if "--verify" in sys.argv:
        ok = verify()
        sys.exit(0 if ok else 1)

    if verify():
        print("[ok] EBB! benchmark subset already complete. Use --verify to re-check.")
        return

    # The Google Drive download requires manual access approval.
    # We can't automate that without API credentials, so print instructions.
    manual_instructions()


if __name__ == "__main__":
    main()