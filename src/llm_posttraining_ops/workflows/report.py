"""Final Markdown reporting for end-to-end workflow runs."""

from __future__ import annotations

from pathlib import Path

from llm_posttraining_ops.workflows.registry import ExperimentRegistry

DEFAULT_WORKFLOW_REPORT_PATH = Path("reports/workflow_report.md")


def render_workflow_report(
    registry: ExperimentRegistry,
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> str:
    """Render stage outcomes and artifact locations."""

    lines = [
        "# End-to-End Workflow Report",
        "",
        f"- Run ID: `{registry.run_id}`",
        f"- Status: **{registry.status.upper()}**",
        f"- Run directory: `{run_dir}`",
        f"- Started: {registry.started_at}",
        f"- Ended: {registry.ended_at}",
        f"- Reproducibility manifest: `{manifest_path}`",
        f"- Workflow summary: `{summary_path}`",
        "",
        "## Stage results",
        "",
        "| Stage | Status | Artifacts | Error or reason |",
        "| --- | --- | ---: | --- |",
    ]
    for stage in registry.stages:
        detail = stage.error or stage.reason or ""
        lines.append(
            f"| {stage.name} | {stage.status} | {len(stage.artifacts)} | {detail} |"
        )

    lines.extend(
        [
            "",
            "## Artifact registry",
            "",
            "| Name | Path |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| {name} | `{path}` |" for name, path in sorted(registry.artifacts.items())
    )
    lines.extend(
        [
            "",
            "Skipped stages are explicit and do not fail the workflow. Any failed",
            "stage marks the final workflow result as failed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_workflow_report(
    registry: ExperimentRegistry,
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    path: str | Path = DEFAULT_WORKFLOW_REPORT_PATH,
) -> Path:
    """Write the final workflow Markdown report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_workflow_report(
            registry,
            run_dir=run_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return output_path
