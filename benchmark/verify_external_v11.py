"""Verify locally downloaded external v1.1 evaluation sources against the manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .artifacts import sha256_file


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, got {actual}")


def verify_external_sources(manifest_path: str | Path, workspace: str | Path) -> dict:
    """Verify downloaded LongMemEval, LongBench and RULER source pins."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(workspace)
    reports = []
    for source in manifest["tracks"]:
        status = source["status"]
        if not status.startswith(("admitted", "downloaded", "generator_downloaded")):
            continue
        name = source["name"]
        if name == "LongMemEval-S-Cleaned":
            file_path = root / ".datasets" / "longmemeval-cleaned" / source["file"]
            _verify_hash(file_path, source["sha256"])
            reports.append({"name": name, "verified": True, "file": str(file_path)})
        elif name == "LongBench-v1-Chinese":
            repository = root / ".datasets" / "longbench"
            if _git_revision(repository) != source["repository_revision"]:
                raise ValueError(f"LongBench revision does not match {source['repository_revision']}")
            _verify_hash(repository / "data.zip", source["data_zip_sha256"])
            data_dir = root / source["local_path"]
            total = 0
            for item in source["selected_files"]:
                file_path = data_dir / item["name"]
                _verify_hash(file_path, item["sha256"])
                row_count = sum(1 for line in file_path.open(encoding="utf-8") if line.strip())
                if row_count != item["case_count"]:
                    raise ValueError(f"{file_path}: expected {item['case_count']} rows, got {row_count}")
                total += row_count
            if total != source["case_count"]:
                raise ValueError(f"LongBench: expected {source['case_count']} rows, got {total}")
            reports.append({"name": name, "verified": True, "case_count": total})
        elif name == "RULER":
            repository = root / source["local_path"]
            if _git_revision(repository) != source["repository_revision"]:
                raise ValueError(f"RULER revision does not match {source['repository_revision']}")
            _verify_hash(repository / source["config_file"], source["config_sha256"])
            reports.append({"name": name, "verified": True, "config": source["config_file"]})
    return {"valid": True, "sources": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify downloaded v1.1 external sources.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.1" / "source_manifest.json",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify_external_sources(args.manifest, args.workspace), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
