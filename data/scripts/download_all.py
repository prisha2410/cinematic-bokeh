"""Run all three dataset downloads in sequence."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(script: str):
    print()
    print("=" * 72)
    print(f"  {script}")
    print("=" * 72)
    rc = subprocess.call([sys.executable, str(ROOT / "data" / "scripts" / script)])
    return rc


def main():
    print("Downloading benchmark datasets...")
    rcs = {
        "Unsplash": run("download_unsplash.py"),
        "DIODE":    run("download_diode.py"),
        "EBB!":     run("download_ebb.py"),
    }
    print()
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    for name, rc in rcs.items():
        status = "OK" if rc == 0 else "needs manual steps"
        print(f"  {name:10s} {status}")
    print()
    if rcs["EBB!"] != 0:
        print("EBB! requires Google Drive access — see the printed instructions above.")


if __name__ == "__main__":
    main()