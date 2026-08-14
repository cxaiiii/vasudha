"""Assemble the shippable folder and zip it.

PyInstaller wipes dist/Vasudha on every build, so the files a user needs but
the bundle does not produce have to be copied in afterwards. Doing that by hand
is exactly the kind of step that gets forgotten once — and forgetting
model_source.json ships a binary whose download button cannot work.

    python scripts/package_release.py            # stage files, report
    python scripts/package_release.py --zip      # also produce the archive
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def default_dist() -> Path:
    """Prefer dist/, but fall back to dist_build/ — PyInstaller cannot clean
    dist/Vasudha while any shell has it as a working directory, so builds get
    redirected there and the release step must follow."""
    primary = ROOT / "dist" / "Vasudha"
    alternate = ROOT / "dist_build" / "Vasudha"
    if (primary / "Vasudha.exe").is_file() and not (alternate / "Vasudha.exe").is_file():
        return primary
    if (alternate / "Vasudha.exe").is_file():
        newer = max((primary, alternate),
                    key=lambda p: (p / "Vasudha.exe").stat().st_mtime
                    if (p / "Vasudha.exe").is_file() else 0)
        return newer
    return primary

DIST = ROOT / "dist" / "Vasudha"   # replaced in main()

# (source, name in the shipped folder, why it must be there)
EXTRAS = [
    (ROOT / "packaging" / "dist_README.txt", "README.txt",
     "what to do on first launch"),
    (ROOT / "model_source.json", "model_source.json",
     "download URL + checksum; without it first-run download cannot work"),
    (ROOT / "LICENSE", "LICENSE", "Apache 2.0"),
]


def human(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="also build the archive")
    ap.add_argument("--name", default="Vasudha-windows-x64")
    ap.add_argument("--dist", type=Path, default=None,
                    help="folder holding Vasudha.exe (default: newest build)")
    args = ap.parse_args()

    global DIST
    DIST = args.dist or default_dist()
    exe = DIST / "Vasudha.exe"
    if not exe.is_file():
        print(f"No build at {exe}\nRun PyInstaller first.", file=sys.stderr)
        return 1

    print(f"staging into {DIST}\n")
    missing = False
    for source, name, why in EXTRAS:
        target = DIST / name
        if not source.is_file():
            print(f"  MISSING  {name:<22} {why}")
            missing = True
            continue
        shutil.copyfile(source, target)
        print(f"  ok       {name:<22} {why}")

    # The one that silently breaks a release if it is empty rather than absent.
    import json
    try:
        cfg = json.loads((DIST / "model_source.json").read_text(encoding="utf-8"))
        if not cfg.get("url"):
            print("\n  WARNING: model_source.json has no url — the Download button "
                  "will fail for every user.")
            missing = True
        else:
            print(f"\n  download url set: {cfg['url'][:64]}…")
    except Exception:  # noqa: BLE001
        print("\n  WARNING: model_source.json is unreadable")
        missing = True

    files = [p for p in DIST.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"\n  {len(files)} files, {human(total)}")

    if args.zip:
        archive = ROOT / "dist" / f"{args.name}.zip"
        print(f"\ncompressing -> {archive.name}  (a few minutes)")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in files:
                zf.write(path, Path("Vasudha") / path.relative_to(DIST))
        size = archive.stat().st_size
        print(f"  {archive.name}  {human(size)}  ({100 * size / total:.0f}% of unpacked)")

    if missing:
        print("\nSomething above needs fixing before this is shippable.")
        return 1
    print("\nready to ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
