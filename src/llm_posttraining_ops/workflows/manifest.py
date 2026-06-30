"""Environment and input manifest for reproducible workflow runs."""

from __future__ import annotations

import json
import platform as platform_module
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from llm_posttraining_ops import __version__

MANIFEST_SCHEMA_VERSION = "1.0"
KEY_DEPENDENCIES = (
    "accelerate",
    "fastapi",
    "peft",
    "pydantic",
    "torch",
    "transformers",
    "trl",
    "typer",
)


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    """Version, platform, model, and data provenance for one run."""

    schema_version: str
    package_version: str
    python_version: str
    platform: str
    seed: int
    run_id: str
    models: dict[str, str | None]
    data_paths: dict[str, str]
    dependencies: dict[str, str]
    git_commit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dependency_versions(
    package_names: tuple[str, ...] = KEY_DEPENDENCIES,
) -> dict[str, str]:
    """Return installed dependency versions, omitting unavailable packages."""

    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_commit_hash(repository: str | Path = ".") -> str | None:
    """Return the current commit hash when the source is a Git checkout."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def build_reproducibility_manifest(
    *,
    run_id: str,
    seed: int,
    models: Mapping[str, str | None],
    data_paths: Mapping[str, str],
    repository: str | Path = ".",
) -> ReproducibilityManifest:
    """Capture the environment and workflow inputs."""

    return ReproducibilityManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        package_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform_module.platform(),
        seed=seed,
        run_id=run_id,
        models=dict(sorted(models.items())),
        data_paths=dict(sorted(data_paths.items())),
        dependencies=dependency_versions(),
        git_commit=git_commit_hash(repository),
    )


def write_reproducibility_manifest(
    manifest: ReproducibilityManifest,
    path: str | Path,
) -> Path:
    """Write a stable manifest JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path
