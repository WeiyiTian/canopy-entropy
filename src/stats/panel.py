from pathlib import Path

import pandas as pd
import torch

from src.generation_space.core import GenerationMetadata, PromptRolloutStats
from src.generation_space.io import (
    build_metadata_path,
    count_prompt_shards,
    load_prompt_shard_tensor,
    load_prompt_stats,
)
from src.metrics.semantic_metrics import calculate_relaxed_semantic_entropy


def build_panel(prompt_stats_paths: list[Path]) -> pd.DataFrame:
    """
    Stacks per-run prompt-stats blocks into one panel across all runs.

    Args:
        prompt_stats_paths: Paths to `prompt_rollout_stats.pt` files, one per
            (model, variant, dataset) run.

    Returns:
        DataFrame with one row per (model, variant, dataset, prompt). Columns:
        - task: dataset stem.
        - prompt_id: zero-indexed position of the prompt within the run.
        - model_name: model family.
        - model_variant: model variant.
        - D: semantic diversity `D_{tmp}` for this prompt.
        - log_VS: Vendi entropy `H_{tmp}` for this prompt.
        - VS: Vendi Score `exp(H_{tmp})`, the effective number of distinct
          responses among the M rollouts, bounded in `[1, M]`.
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


def build_rse_panel(prompt_stats_paths: list[Path], judgment_artifact: str) -> pd.DataFrame:
    """
    Stacks per-run RSE blocks into one panel across all runs.

    Args:
        prompt_stats_paths: Paths to `prompt_rollout_stats.pt` files, one per
            (model, variant, dataset) run.
        judgment_artifact: Judgment artifact name, including the judge label.

    Returns:
        DataFrame with one row per (model, variant, dataset, prompt) that was judged.
        Columns:
        - task: dataset stem.
        - prompt_id: zero-indexed position of the prompt within the run.
        - model_name: model family.
        - model_variant: model variant.
        - RSE: Relaxed Semantic Entropy `RSE_{tmp}` over the m judged rollouts.
        - RSE_norm: `RSE_{tmp} / log(m)`, bounded in `[0, 1]`.
        - n_clusters: number of semantic equivalence classes among the m judged
          rollouts, bounded in `[1, m]`.
        - m: number of judged rollouts, which bounds `RSE` at `log(m)`.
        - R_bar: mean per-rollout entropy rate `R_bar_{tmp}` for this prompt.
        - N_bar: mean rollout length `(1/M) * sum_i N^(i)` for this prompt.
        - M: number of raw rollouts for this prompt.
        - prompt_uid: `"{task}/{prompt_id}"`, a unique cross-task prompt identifier.

    Notes:
        `RSE` is computed over the m judged rollouts while `R_bar` and `N_bar` are 
        computed over all M.
    """
    blocks = [_load_rse_block(Path(path), judgment_artifact) for path in prompt_stats_paths]
    panel = pd.concat(blocks, ignore_index=True) # stacking row-wise
    panel["prompt_uid"] = panel["task"].astype(str) + "/" + panel["prompt_id"].astype(str)
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
        - log_VS: Vendi entropy `H_{tmp}` for this prompt.
        - VS: Vendi Score `exp(H_{tmp})` for this prompt.
        - R_bar: mean per-rollout entropy rate `R_bar_{tmp}` for this prompt.
        - N_bar: mean rollout length `(1/M) * sum_i N^(i)` for this prompt.
        - M: number of raw rollouts for this prompt.
    """
    metadata = GenerationMetadata.load(build_metadata_path(prompt_stats_path.parent))
    prompt_results: list[PromptRolloutStats] = load_prompt_stats(prompt_stats_path, device="cpu")
    
    rows = [
        {
            "task": Path(metadata.prompt_file).stem,
            "prompt_id": prompt_id,
            "model_name": metadata.model_name,
            "model_variant": metadata.model_variant,
            "D": _to_float(stats.raw_metrics.semantic_diversity),
            "log_VS": _to_float(stats.raw_metrics.vendi_entropy),
            "VS": _to_float(stats.raw_metrics.vendi_entropy.exp()),
            "R_bar": _to_float(stats.raw_metrics.entropy_rate),
            "N_bar": _to_float(stats.raw_sequence_lengths.to(torch.float32).mean()),
            "M": int(stats.raw_sequence_lengths.numel()),
        }
        for prompt_id, stats in enumerate(prompt_results)
    ]
    return pd.DataFrame.from_records(rows)


def _load_rse_block(prompt_stats_path: Path, judgment_artifact: str) -> pd.DataFrame:
    """
    Loads one run's judgment shards and returns its judged prompt rows as a dataframe.

    Args:
        prompt_stats_path: Path to a list of `PromptRolloutStats` for one
            (model, variant, dataset) run.
        judgment_artifact: Judgment artifact name including the judge label.

    Returns:
        DataFrame with one row per judged prompt in this run. Columns for each prompt:
        - task: dataset stem.
        - prompt_id: zero-indexed position of the prompt within the run.
        - model_name: model family.
        - model_variant: model variant.
        - RSE: Relaxed Semantic Entropy `RSE_{tmp}` over the m judged rollouts.
        - RSE_norm: `RSE_{tmp} / log(m)`, bounded in `[0, 1]`.
        - n_clusters: number of semantic equivalence classes among the m judged
          rollouts, bounded in `[1, m]`.
        - m: number of judged rollouts, which bounds `RSE` at `log(m)`.
        - R_bar: mean per-rollout entropy rate `R_bar_{tmp}` for this prompt.
        - N_bar: mean rollout length `(1/M) * sum_i N^(i)` for this prompt.
        - M: number of raw rollouts for this prompt.
    """
    run_dir = prompt_stats_path.parent
    metadata = GenerationMetadata.load(build_metadata_path(run_dir))
    prompt_results: list[PromptRolloutStats] = load_prompt_stats(prompt_stats_path, device="cpu")
    num_judged = count_prompt_shards(run_dir, judgment_artifact)

    rows = []
    for prompt_id, stats in enumerate(prompt_results[:num_judged]):
        rse = calculate_relaxed_semantic_entropy(
            load_prompt_shard_tensor(run_dir, judgment_artifact, prompt_id, "judgments")
        )
        rows.append({
            "task": Path(metadata.prompt_file).stem,
            "prompt_id": prompt_id,
            "model_name": metadata.model_name,
            "model_variant": metadata.model_variant,
            "RSE": _to_float(rse.entropy),
            "RSE_norm": _to_float(rse.normalized_entropy),
            "n_clusters": int(rse.num_clusters),
            "m": int(rse.num_responses),
            "R_bar": _to_float(stats.raw_metrics.entropy_rate),
            "N_bar": _to_float(stats.raw_sequence_lengths.to(torch.float32).mean()),
            "M": int(stats.raw_sequence_lengths.numel()),
        })
    return pd.DataFrame.from_records(rows)


def _to_float(value: torch.Tensor) -> float:
    """Converts a single-element tensor to a Python float."""
    return float(value.item())
