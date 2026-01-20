"""Validate dataset/result files against data_manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate(manifest_path: Path, root: Path) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest entries must be a list.")

    failures = 0
    for entry in entries:
        rel_path = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")

        if not rel_path or expected_hash is None or expected_size is None:
            print(f"[error] Invalid entry: {entry}")
            failures += 1
            continue

        path = root / rel_path
        if not path.exists():
            print(f"[missing] {rel_path}")
            failures += 1
            continue

        size = path.stat().st_size
        if size != expected_size:
            print(f"[size] {rel_path} expected={expected_size} actual={size}")
            failures += 1
            continue

        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            print(f"[hash] {rel_path} expected={expected_hash} actual={actual_hash}")
            failures += 1
            continue

    if failures == 0:
        print(f"[ok] {len(entries)} files validated")
    else:
        print(f"[fail] {failures} issue(s) found")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data files against data_manifest.json.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data_manifest.json",
        help="Path to data manifest JSON.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repository root containing the listed paths.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(args.root)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    failures = _validate(manifest_path, root)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
