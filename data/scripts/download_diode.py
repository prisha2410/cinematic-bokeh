"""
Download a 10-image subset of the DIODE validation split.

DIODE (Vasiljevic et al., 2019) provides RGB + dense depth from a FARO laser
scanner. The full dataset is ~80 GB; we only need ~10 representative images
from the validation split for evaluation.

The dataset is hosted at https://diode-dataset.org/ as a single tarball per
split. We download the validation tarball (~1.5 GB) and extract only the
images we need, then optionally clean up the tarball.

Output:
    data/diode/rgb/<scene_id>.png
    data/diode/depth/<scene_id>.npy        (float32 depth in meters)
    data/diode/depth_mask/<scene_id>.npy   (uint8 validity mask, 1 = valid)
"""
from __future__ import annotations
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIODE_DIR = ROOT / "data" / "diode"
RGB_DIR = DIODE_DIR / "rgb"
DEPTH_DIR = DIODE_DIR / "depth"
MASK_DIR = DIODE_DIR / "depth_mask"

# Validation split is small enough to download. Indoor + outdoor sub-splits.
VAL_INDOOR_URL = "http://diode-dataset.org/data/val_indoor.tar.gz"
VAL_OUTDOOR_URL = "http://diode-dataset.org/data/val_outdoor.tar.gz"

# 10 scene IDs we use for the benchmark, chosen for variety
# (5 indoor + 5 outdoor, spanning close/medium/far focus distances).
# DIODE filenames look like:  scene_<XXXXX>/scan_<XXXXX>/<view_id>.png
# We'll list specific PNG paths we want from the val tarballs.
BENCHMARK_PATHS = [
    # Indoor
    "indoors/scene_00019/scan_00183/00019_00183_indoors_010_010.png",
    "indoors/scene_00021/scan_00187/00021_00187_indoors_280_010.png",
    "indoors/scene_00022/scan_00191/00022_00191_indoors_080_010.png",
    "indoors/scene_00023/scan_00193/00023_00193_indoors_010_010.png",
    "indoors/scene_00024/scan_00195/00024_00195_indoors_180_010.png",
    # Outdoor
    "outdoor/scene_00020/scan_00185/00020_00185_outdoor_180_010.png",
    "outdoor/scene_00022/scan_00189/00022_00189_outdoor_080_010.png",
    "outdoor/scene_00023/scan_00192/00023_00192_outdoor_010_010.png",
    "outdoor/scene_00024/scan_00194/00024_00194_outdoor_280_010.png",
    "outdoor/scene_00025/scan_00197/00025_00197_outdoor_080_010.png",
]


def download(url: str, dest: Path):
    if dest.exists():
        print(f"[ok] {dest.name} already downloaded ({dest.stat().st_size / 1e9:.1f} GB)")
        return
    print(f"[..] downloading {url}")
    print(f"     -> {dest}")
    print(f"     This is large (~750 MB per split); grab a coffee.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[ok] downloaded {dest.stat().st_size / 1e9:.2f} GB")


def extract_subset(tarball: Path, wanted_paths: list[str]):
    """
    Extract only the RGB + depth + mask files matching our benchmark IDs.
    DIODE conventions:
      <stem>.png         RGB
      <stem>_depth.npy   depth in meters
      <stem>_depth_mask.npy  validity mask
    """
    RGB_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    wanted_stems = {Path(p).stem for p in wanted_paths}
    extracted = 0
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            stem_no_ext = name.replace("_depth.npy", "").replace("_depth_mask.npy", "").replace(".png", "")
            if stem_no_ext not in wanted_stems:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()

            if name.endswith("_depth.npy"):
                out = DEPTH_DIR / name
            elif name.endswith("_depth_mask.npy"):
                out = MASK_DIR / name
            elif name.endswith(".png"):
                out = RGB_DIR / name
            else:
                continue

            out.write_bytes(data)
            extracted += 1
    print(f"[ok] extracted {extracted} files from {tarball.name}")


def verify() -> bool:
    DIODE_DIR.mkdir(parents=True, exist_ok=True)
    expected_stems = [Path(p).stem for p in BENCHMARK_PATHS]
    missing = []
    for stem in expected_stems:
        if not (RGB_DIR / f"{stem}.png").exists():
            missing.append(f"{stem}.png")
        if not (DEPTH_DIR / f"{stem}_depth.npy").exists():
            missing.append(f"{stem}_depth.npy")
        if not (MASK_DIR / f"{stem}_depth_mask.npy").exists():
            missing.append(f"{stem}_depth_mask.npy")
    if missing:
        print(f"[!] Missing {len(missing)} files:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
        return False
    print(f"[ok] All DIODE benchmark files present ({len(expected_stems)} scenes).")
    return True


def main():
    DIODE_DIR.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "_cache"
    cache.mkdir(exist_ok=True)

    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)

    if verify():
        print("[ok] DIODE benchmark already complete.")
        return

    indoor_tar = cache / "val_indoor.tar.gz"
    outdoor_tar = cache / "val_outdoor.tar.gz"

    try:
        download(VAL_INDOOR_URL, indoor_tar)
        download(VAL_OUTDOOR_URL, outdoor_tar)
    except Exception as e:
        print(f"[!] Automatic download failed: {e}")
        print(f"    You can manually download from https://diode-dataset.org/")
        print(f"    and place the tarballs at {cache}/")
        sys.exit(1)

    extract_subset(indoor_tar, BENCHMARK_PATHS)
    extract_subset(outdoor_tar, BENCHMARK_PATHS)

    print()
    print("Tip: the cached tarballs are large and only needed once.")
    print(f"     You can delete them now to save ~1.5 GB:")
    print(f"        rm -rf {cache}")

    verify()


if __name__ == "__main__":
    main()