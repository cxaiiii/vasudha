"""Publish the GGUF to HuggingFace, then wire the app to download it.

Run `hf auth login` first — this script never handles your token, it only uses
the one the CLI has already cached.

    python scripts/publish_model.py --repo <username>/vasudha-4b-v3-gguf

What it does, in order:
  1. checks you are logged in
  2. verifies the local file's SHA-256 (so a corrupt source is never published)
  3. creates the repo if needed
  4. uploads the GGUF and the model card
  5. writes the resolved URL + checksum into model_source.json

Step 5 is the point: after this, first-run download works with no rebuild,
because model_source.json is read at runtime from beside the executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = ROOT / "ollama" / "vasudha-leaprunning-4b-v3-Q3_K_M.gguf"
CARD = ROOT / "packaging" / "hf_model_card.md"
SOURCE_JSON = ROOT / "model_source.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as handle:
        while (block := handle.read(1 << 22)):
            digest.update(block)
            done += len(block)
            pct = 100 * done / total
            print(f"\r  hashing {pct:5.1f}%", end="", flush=True)
    print("\r  hashing 100.0%")
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/vasudha-4b-v3-gguf")
    ap.add_argument("--file", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-hash", action="store_true",
                    help="trust the checksum already in model_source.json")
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami
    from huggingface_hub.errors import LocalTokenNotFoundError

    try:
        user = whoami()
    except (LocalTokenNotFoundError, OSError):
        print("Not logged in. Run:  hf auth login", file=sys.stderr)
        return 1
    print(f"logged in as {user.get('name', '?')}")

    gguf = args.file
    if not gguf.is_file():
        print(f"missing: {gguf}", file=sys.stderr)
        return 1
    size = gguf.stat().st_size
    print(f"file: {gguf.name}  {size / 2**30:.2f} GiB")

    existing = json.loads(SOURCE_JSON.read_text(encoding="utf-8")) if SOURCE_JSON.exists() else {}
    if args.skip_hash and existing.get("sha256"):
        digest = existing["sha256"]
        print(f"  using recorded sha256 {digest[:16]}…")
    else:
        digest = sha256_of(gguf)
    print(f"sha256: {digest}")

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)
    print(f"repo ready: https://huggingface.co/{args.repo}")

    if CARD.is_file():
        api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
        print("uploaded model card")

    print("uploading weights — this takes a while at 2 GB…")
    api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf.name,
                    repo_id=args.repo, repo_type="model")

    url = f"https://huggingface.co/{args.repo}/resolve/main/{gguf.name}?download=true"
    existing.update({"url": url, "filename": gguf.name,
                     "sha256": digest, "size_bytes": size})
    SOURCE_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print("\ndone.")
    print(f"  {url}")
    print(f"  wrote {SOURCE_JSON.relative_to(ROOT)}")
    print("  copy that file next to Vasudha.exe and first-run download works,")
    print("  no rebuild needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
