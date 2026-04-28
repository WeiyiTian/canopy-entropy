from pathlib import Path

import pandas as pd
import torch

from src.generation_space.core import GenerationMetadata, PromptRolloutStats
from src.generation_space.io import build_rollout_metadata_path, load_prompt_stats


def build_panel(prompt_stats_paths: list[Path]) -> pd.DataFrame:
    """
    Stacks per-run prompt-stats blocks into one panel across all runs.

    Args:
        prompt_stats_paths: Paths to `prompt_rollout_stats.pt` files, one per
            (model, variant, dataset) run.

    Returns:
        DataFrame with one row per (model, variant, dataset, prompt). Columns:'
        - task: dataset stem.
        - prompt_id: zero-indexed position of the prompt within the run.
        - model_name: model family.
        - model_variant: model variant.
        - D: semantic diversity `D_{tmp}` for this prompt.
        - R_bar: mean per-rollout entropy rate `R_bar_{tmp}` for this prompt.
        - N_bar: mean rollout length `(1/M) * sum_i N^(i)` for this prompt.
        - M: number of raw rollouts for this prompt.
        - prompt_uid: `"{task}/{prompt_id}"`, a unique cross-task prompt identifier.

    Notes:
        Rows with NaN `D` (semantic diversity) are dropped.
    """
    blocks = [_load_panel_block(Path(p)) for p in prompt_stats_paths]
    panel = pd.concat(blocks, ignore_index=True) # stacking row-wise
    panel["prompt_uid"] = panel["task"].astype(str) + "/" + panel["prompt_id"].astype(str)

    # removes rows where column "D" is NaN
    panel = panel.dropna(subset=["D"]).reset_index(drop=True)
    return panel


def _load_panel_block(prompt_stats_path: Path) -> pd.DataFrame:
    """
    Loads one prompt-stats file and returns its prompt rows as a dataframe.

    Args:
        prompt_stats_path: Path to a list of `PromptRolloutStats` for one
            (model, variant, dataset) run.

    Returns:
        DataFrame with one row per prompt in this run. Columns for each prompt:
        - task: dataset stem.
        - prompt_id: zero-indexed position of the prompt within the run.
        - model_name: model family.
        - model_variant: model variant.
        - D: semantic diversity `D_{tmp}` for this prompt.
        - R_bar: mean per-rollout entropy rate `R_bar_{tmp}` for this prompt.
        - N_bar: mean rollout length `(1/M) * sum_i N^(i)` for this prompt.
        - M: number of raw rollouts for this prompt.
    """
    metadata = GenerationMetadata.load(build_rollout_metadata_path(prompt_stats_path.parent))
    prompt_results: list[PromptRolloutStats] = load_prompt_stats(prompt_stats_path, device="cpu")
    
    rows = [
        {
            "task": Path(metadata.prompt_file).stem,
            "prompt_id": prompt_id,
            "model_name": metadata.model_name,
            "model_variant": metadata.model_variant,
            "D": _to_float(stats.raw_metrics.semantic_diversity),
            "R_bar": _to_float(stats.raw_metrics.entropy_rate),
            "N_bar": _to_float(stats.raw_sequence_lengths.to(torch.float32).mean()),
            "M": int(stats.raw_sequence_lengths.numel()),
        }
        for prompt_id, stats in enumerate(prompt_results)
    ]
    return pd.DataFrame.from_records(rows)


def _to_float(value: torch.Tensor) -> float:
    """Converts a single-element tensor to a Python float."""
    return float(value.item())
