"""
Fast downloader for a public Google Drive link (file or folder).

Unlike Google Drive's browser "Download" button — which zips folders into
multiple .zip parts once the folder is large — this pulls every file down
individually, in parallel, straight into a local directory.

Usage:
    python scripts/download_gdrive.py <google_drive_url> [output_dir] [--workers N]

Examples:
    python scripts/download_gdrive.py "https://drive.google.com/drive/folders/XXXX" ./data
    python scripts/download_gdrive.py "https://drive.google.com/file/d/XXXX/view" ./data/video.mp4

Requires: pip install gdown
"""

import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown
from tqdm import tqdm

MAX_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 15


def is_folder_url(url: str) -> bool:
    return "/folders/" in url or "folderview" in url


def download_folder(url: str, output_dir: str, workers: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("Listing files in the folder (no download yet)...")
    files = gdown.download_folder(
        url=url,
        output=output_dir,
        quiet=True,
        skip_download=True,  # just build the file list + directory structure first
    )

    if not files:
        print("No files found (link may be private or empty).")
        return

    total = len(files)
    print(f"Found {total} files. Downloading with {workers} parallel workers...\n")

    file_bar = tqdm(total=total, desc="Files", unit="file", position=0)
    byte_bar = tqdm(total=None, desc="Data", unit="B", unit_scale=True, position=1)
    lock = threading.Lock()
    failed = []

    def fetch(entry):
        local_path = entry.local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Stagger thread starts so we don't fire a burst of simultaneous
        # requests at Drive, which is what trips its abuse detection.
        time.sleep(random.uniform(0, 2))

        seen = 0

        def on_progress(bytes_so_far, _bytes_total):
            nonlocal seen
            with lock:
                byte_bar.update(bytes_so_far - seen)
            seen = bytes_so_far

        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                gdown.download(
                    id=entry.id,
                    output=local_path,
                    quiet=True,
                    resume=True,
                    progress=on_progress,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                seen = 0  # reset byte counter; resume=True will re-report from scratch on retry
                if attempt < MAX_ATTEMPTS:
                    wait = BASE_BACKOFF_SECONDS * attempt + random.uniform(0, 5)
                    file_bar.write(
                        f"Retry {attempt}/{MAX_ATTEMPTS - 1} in {wait:.0f}s: {local_path} ({e})"
                    )
                    time.sleep(wait)

        if last_error is not None:
            file_bar.write(f"FAILED (gave up after {MAX_ATTEMPTS} attempts): {local_path} ({last_error})")
            with lock:
                failed.append(entry)
        file_bar.update(1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, entry) for entry in files]
        for f in as_completed(futures):
            f.result()

    file_bar.close()
    byte_bar.close()

    if failed:
        print(f"\n{len(failed)}/{total} files failed after retries. Just rerun the same command —")
        print("already-downloaded files are skipped automatically, so it'll only pick up the rest.")
        print("If most/all files fail again immediately, Drive is rate-limiting this IP: rerun with --workers 1.")
    else:
        print("\nAll downloads finished.")


def download_single_file(url: str, output_path: str | None) -> None:
    print("Downloading single file...")
    gdown.download(url=url, output=output_path, quiet=False, resume=True)


def main():
    parser = argparse.ArgumentParser(description="Fast public Google Drive downloader")
    parser.add_argument("url", help="Public Google Drive file or folder URL")
    parser.add_argument("output", nargs="?", default=None, help="Output directory (folder link) or file path (file link)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel downloads for folders (default: 3; lower if Drive rate-limits you)")
    args = parser.parse_args()

    if is_folder_url(args.url):
        output_dir = args.output or "gdrive_download"
        download_folder(args.url, output_dir, args.workers)
    else:
        download_single_file(args.url, args.output)


if __name__ == "__main__":
    main()
