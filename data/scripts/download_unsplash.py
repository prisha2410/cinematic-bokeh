"""
Download a curated set of 9 Unsplash photos for in-the-wild evaluation.
(Flower photo removed — CDN URLs unstable; manually add object_02_flower.jpg if desired.)
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNSPLASH_DIR = ROOT / "data" / "unsplash"

BENCHMARK_PHOTOS = [
    ("portrait_01_woman_lights.jpg",
     "https://images.unsplash.com/photo-1488161628813-04466f872be2?w=1600",
     "Brooke Cagle", "JBwcenOuRCg"),
    ("portrait_02_man_window.jpg",
     "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=1600",
     "Christopher Campbell", "rDEOVtE7vOs"),
    ("portrait_03_curly_hair.jpg",
     "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=1600",
     "Mike Von", "Z08DwAEjEzg"),
    ("portrait_04_bokeh_lights.jpg",
     "https://images.unsplash.com/photo-1502323777036-f29e3972d82f?w=1600",
     "Allef Vinicius", "OkRCFEMv9oU"),
    ("portrait_05_outdoor.jpg",
     "https://images.unsplash.com/photo-1463453091185-61582044d556?w=1600",
     "Ben Parker", "I-LsRdQuZmM"),
    ("animal_01_dog.jpg",
     "https://images.unsplash.com/photo-1450778869180-41d0601e046e?w=1600",
     "Jamie Street", "fyaarp4dfFk"),
    ("animal_02_cat.jpg",
     "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1600",
     "Manja Vitolic", "gKXKBY-C-Dk"),
    ("object_01_coffee.jpg",
     "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=1600",
     "Nathan Dumlao", "ZuIDLSz3XLg"),
    ("portrait_06_glasses.jpg",
     "https://images.unsplash.com/photo-1521119989659-a83eee488004?w=1600",
     "Eye for Ebony", "TtBeP5tk2eA"),
]


def download_one(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        print(f"[..] {dest.name}")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; cinematic-bokeh-research/1.0)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"[!] failed: {e}")
        return False


def verify() -> bool:
    UNSPLASH_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name, _, _, _ in BENCHMARK_PHOTOS
               if not (UNSPLASH_DIR / name).exists()]
    if missing:
        print(f"[!] Missing {len(missing)}/{len(BENCHMARK_PHOTOS)} Unsplash photos:")
        for m in missing[:5]:
            print(f"    {m}")
        return False
    print(f"[ok] All {len(BENCHMARK_PHOTOS)} Unsplash photos present.")
    return True


def write_attribution():
    out = UNSPLASH_DIR / "CREDITS.md"
    lines = [
        "# Unsplash photo credits",
        "",
        "All photos in this directory are from Unsplash and are used under",
        "the [Unsplash License](https://unsplash.com/license).",
        "",
        "While not required, we provide attribution as a courtesy:",
        "",
        "| File | Photographer | Photo ID |",
        "|---|---|---|",
    ]
    for name, _url, photographer, pid in BENCHMARK_PHOTOS:
        lines.append(
            f"| `{name}` | {photographer} | [{pid}](https://unsplash.com/photos/{pid}) |"
        )
    out.write_text("\n".join(lines))
    print(f"[ok] wrote attribution credits -> {out}")


def main():
    UNSPLASH_DIR.mkdir(parents=True, exist_ok=True)

    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)

    n_ok = 0
    for name, url, _photographer, _pid in BENCHMARK_PHOTOS:
        dest = UNSPLASH_DIR / name
        if download_one(url, dest):
            n_ok += 1
    write_attribution()
    print(f"[ok] {n_ok}/{len(BENCHMARK_PHOTOS)} Unsplash photos ready.")
    if n_ok < len(BENCHMARK_PHOTOS):
        print("[!] Some downloads failed. Re-run the script to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()