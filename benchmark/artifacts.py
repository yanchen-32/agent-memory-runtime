from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_key_value_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def _linux_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _memory_total_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def collect_environment(root: Path, benchmark_path: Path) -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pytest", "sentence-transformers", "torch", "transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    os_release = _read_key_value_file(Path("/etc/os-release"))
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": sha256_file(benchmark_path),
        "platform": platform.platform(),
        "os_name": os_release.get("PRETTY_NAME") or os_release.get("NAME"),
        "machine": platform.machine(),
        "processor": _linux_cpu_model(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "memory_total_kib": _memory_total_kib(),
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "packages": packages,
    }


def write_formal_artifacts(
    output_dir: Path,
    rows: list[dict],
    summary_payload: dict,
    manifest: dict,
) -> None:
    """Write the protocol-required, machine-readable evidence bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "raw_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flat_rows = []
    for experiment, values in summary_payload.get("experiments", {}).items():
        for item in values:
            flat_rows.append({"experiment": experiment, **item})
    fieldnames = sorted({key for row in flat_rows for key in row})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    environment = manifest.get("environment", {})
    (output_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run.log").write_text(
        f"generated_at={manifest.get('generated_at')}\n"
        f"status=completed\n"
        f"rows={len(rows)}\n",
        encoding="utf-8",
    )
    (output_dir / "figures").mkdir(exist_ok=True)
