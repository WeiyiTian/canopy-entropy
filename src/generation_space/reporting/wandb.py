from dataclasses import fields
from pathlib import Path

import torch
import wandb

from src.constants import GENERATION_SPACE_METRIC_KEYS, LENGTH_BUCKET_NAMES
from src.metrics import Correlation
from src.visualization import plot_sequence_length_histogram

from ..core.structures import GenerationMetadata, PromptRolloutStats


def resolve_run_name(
    wandb_run_name: str | None,
    model_name: str,
    file_name: str,
    model_variant: str,
) -> str:
    """
    Resolves the shared W&B run name prefix for generation and compute jobs.

    Args:
        wandb_run_name: Explicit run name prefix, if provided by the caller.
        model_name: Generation model family.
        file_name: Prompt file path used to derive a default run name.
        model_variant: Model variant identifier.

    Returns:
        Explicit `wandb_run_name` when provided; otherwise defaults to
        of `{model_name}_{model_variant}_{prompt_file_stem}`.
    """
    if wandb_run_name:
        return wandb_run_name
    return f"{model_name}_{model_variant}_{Path(file_name).stem}"


def build_scalar_metrics_table(metrics: dict[str, float | int]) -> wandb.Table:
    """
    Builds a two-column W&B table from flattened scalar metrics.

    Args:
        metrics: Mapping from metric names to numeric values.

    Returns:
        W&B table with `metric` and `value` columns.
    """
    table = wandb.Table(columns=["metric", "value"])
    for metric_name, metric_value in metrics.items():
        table.add_data(metric_name, metric_value)
    return table


def build_flat_scalar_summary(
    metadata: GenerationMetadata,
    pooled_metrics: dict[str, object],
) -> dict[str, float | int]:
    """
    Flattens pooled generation-space metrics into W&B summary scalars.

    Args:
        metadata: Serializable run metadata.
        pooled_metrics: Mapping with scalar values, nested dictionaries, and objects.

    Returns:
        Flat scalar mapping.
    """
    summary: dict[str, float | int] = {"num_prompts": metadata.num_prompts}
    for variant in ("raw", "kept"):
        bundle = pooled_metrics[variant]
        for key in GENERATION_SPACE_METRIC_KEYS:
            value = bundle[key]
            if isinstance(value, Correlation):
                for field in fields(value):
                    summary[f"{variant}.{key}.{field.name}"] = float(getattr(value, field.name).item())
            elif isinstance(value, dict):
                for bucket_name in LENGTH_BUCKET_NAMES:
                    summary[f"{variant}.{key}.{bucket_name}"] = float(value[bucket_name].item())
            else:
                summary[f"{variant}.{key}"] = float(value.item())
    return summary


def log_metric_artifacts(
    metadata: GenerationMetadata,
    pooled_metrics: dict[str, object],
    prompt_results: list[PromptRolloutStats],
    prompt_stats_path: Path,
    pooled_metrics_path: Path,
    sequence_length_hist_path: Path,
) -> dict[str, float | int]:
    """
    Logs metric summaries and plots to the W&B run.

    Args:
        metadata: Serializable run metadata.
        pooled_metrics: Pooled metrics for raw and reward-filtered rollouts.
        prompt_results: Per-prompt rollout stats used to build the sequence length histogram.
        prompt_stats_path: Path to the serialized per-prompt statistics.
        pooled_metrics_path: Path to the serialized pooled metrics.
        sequence_length_hist_path: Destination for the generated histogram image.

    Returns:
        Flat scalar metric summary logged to W&B.
    """
    plot_sequence_length_histogram(
        sequence_lengths=torch.cat([pr.raw_sequence_lengths for pr in prompt_results]),
        output_path=sequence_length_hist_path,
        title=(
            f"{metadata.model_name} ({metadata.model_variant}) - "
            f"{Path(metadata.prompt_file).stem}: Sequence Length Distribution"
        ),
    )
    scalar_summary = build_flat_scalar_summary(metadata, pooled_metrics)

    wandb.log(
        {
            "scalar_metrics_table": build_scalar_metrics_table(scalar_summary),
            "sequence_length_histogram": wandb.Image(str(sequence_length_hist_path)),
        }
    )
    wandb.run.summary.update(scalar_summary)
    wandb.run.summary["prompt_file"] = metadata.prompt_file
    wandb.run.summary["model_name"] = metadata.model_name
    wandb.run.summary["model_variant"] = metadata.model_variant
    wandb.run.summary["prompt_stats_path"] = str(prompt_stats_path)
    wandb.run.summary["pooled_metrics_path"] = str(pooled_metrics_path)
    wandb.run.summary["sequence_length_histogram_path"] = str(sequence_length_hist_path)

    return scalar_summary
