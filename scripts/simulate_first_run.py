"""Wipe every trace of Vasudha from this machine and launch it cold.

The first-run experience is the hardest thing to test on a development box,
because the machine that built the app is the one machine guaranteed to have
everything already: a model in ollama, cached settings, a seeded persona
folder. This removes all of it so the next launch is genuinely someone's first.

    python scripts/simulate_first_run.py             # show what would change
    python scripts/simulate_first_run.py --go        # do it, then launch
    python scripts/simulate_first_run.py --go --keep-model

--keep-model leaves the downloaded weights in place, for testing the persona
and chat flow without re-downloading 2.1 GB.

What it CANNOT simulate: a machine with no Ollama installed. If ollama is
running, the app will find it and skip the download path entirely — so stop the
ollama service first if that is the path you want to see. The script says so
when it detects one.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _newest_exe() -> Path:
    """dist/ or dist_build/, whichever holds the newer binary. Builds get
    redirected to dist_build when a shell is sitting inside dist/Vasudha and
    Windows will not let PyInstaller clean it."""
    candidates = [ROOT / "dist" / "Vasudha" / "Vasudha.exe",
                  ROOT / "dist_build" / "Vasudha" / "Vasudha.exe"]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return candidates[0]
    return max(existing, key=lambda p: p.stat().st_mtime)


EXE = _newest_exe()


def app_data() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / "Vasudha"


def describe(path: Path) -> str:
    if not path.exists():
        return "absent"
    if path.is_file():
        return f"file, {path.stat().st_size / 1e6:.1f} MB"
    files = list(path.rglob("*"))
    size = sum(f.stat().st_size for f in files if f.is_file())
    return f"{len([f for f in files if f.is_file()])} files, {size / 1e6:.1f} MB"


def ollama_running() -> bool:
    try:
        import requests
        requests.get("http://127.0.0.1:11434/api/tags", timeout=1.5)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually delete, then launch")
    ap.add_argument("--keep-model", action="store_true",
                    help="leave downloaded weights in place")
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()

    root = app_data()
    targets = [
        ("settings.json", root / "settings.json", "persona choice, engine, generation"),
        ("models.json", root / "models.json", "which .gguf is adopted"),
        ("personas/", root / "personas", "seeded persona files + .seeded marker"),
        ("chats/", root / "chats", "saved conversations"),
    ]
    if not args.keep_model:
        targets.append(("models/", root / "models", "downloaded weights"))

    print(f"app data: {root}\n")
    for label, path, why in targets:
        print(f"  {label:<16} {describe(path):<24} {why}")

    if not args.go:
        print("\nDry run. Add --go to wipe these and launch cold.")
        return 0

    print()
    for label, path, _ in targets:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  removed {label}")
        except OSError as exc:
            print(f"  could NOT remove {label}: {exc}", file=sys.stderr)

    # The download link lives beside the exe; without it a cold start has
    # nothing to fetch and the first-run screen is a dead end.
    source = ROOT / "model_source.json"
    beside = EXE.parent / "model_source.json"
    if source.is_file() and EXE.parent.is_dir():
        shutil.copyfile(source, beside)
        print(f"  copied model_source.json next to the exe")

    if ollama_running():
        print("\n  NOTE: ollama is running, so Vasudha will use it and skip the")
        print("        download screen entirely. Stop it first to see what a user")
        print("        without ollama gets.")

    if args.no_launch:
        return 0
    if not EXE.is_file():
        print(f"\nNo binary at {EXE} — build it first.", file=sys.stderr)
        return 1

    print(f"\nlaunching {EXE.name} cold…")
    subprocess.Popen([str(EXE)], cwd=str(EXE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
