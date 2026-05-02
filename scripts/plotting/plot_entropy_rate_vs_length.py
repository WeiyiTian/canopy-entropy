from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import ROLLOUT_SHARDS_ARTIFACT
from src.generation_space.core import GenerationMetadata, PromptRollouts
from src.generation_space.io import (
    build_prompt_shard_path,
    build_rollout_metadata_path,
    build_run_dir,
    verify_prompt_shards_complete,
)
from src.metrics.generation_tree import aggregate_running_entropy_rate
from src.visualization import plot_entropy_rate_trajectory

load_dotenv()


@torch.no_grad()
def compute_run_entropy_rate_trajectory(
    outputs_root: Path,
    family: str,
    dataset: str,
    variant: str,
    run_name: str,
    bin_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Aggregates the running entropy rate trajectory across all prompt shards in a run.

    Returns (positions, mean_rate, active_count) of length B = max_new_tokens // bin_step,
    with positions[b] = bin_step * (b + 1). Returns None if rollout metadata is absent.
    """
    run_dir = build_run_dir(outputs_root, f"{dataset}.jsonl", family, variant, run_name)
    metadata_path = build_rollout_metadata_path(run_dir)
    if not metadata_path.exists():
        return None
    metadata = GenerationMetadata.load(metadata_path)
    verify_prompt_shards_complete(run_dir, ROLLOUT_SHARDS_ARTIFACT, metadata.num_prompts)

    n_bins = metadata.max_new_tokens // bin_step
    positions = torch.arange(1, n_bins + 1) * bin_step # [B]
    entropy_rate_sum = torch.zeros(n_bins, dtype=torch.float64)
    active_count = torch.zeros(n_bins, dtype=torch.int64)

    for prompt_index in range(metadata.num_prompts):
        shard_path = build_prompt_shard_path(run_dir, ROLLOUT_SHARDS_ARTIFACT, prompt_index)
        rollouts = PromptRollouts.load(shard_path, device="cpu")

        shard_entropy_rate_sum, shard_active_count = aggregate_running_entropy_rate(
            rollouts.step_conditional_entropy, rollouts.sequence_lengths, positions
        )
        entropy_rate_sum += shard_entropy_rate_sum # [B]
        active_count += shard_active_count # [B]

    mean_entropy_rate = entropy_rate_sum / active_count.clamp_min(1)
    return positions, mean_entropy_rate, active_count


@hydra.main(version_base=None, config_path="../../configs", config_name="plot_entropy_rate_vs_length")
def main(cfg: DictConfig) -> None:
    families = list(cfg.matrix.models)
    variants = list(cfg.matrix.variants)
    datasets = list(cfg.matrix.datasets)

    outputs_root = Path(cfg.paths.outputs_root)
    cells = [
        (family, dataset, variant)
        for family in families
        for dataset in datasets
        for variant in variants
    ]

    curves_by_cell = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                compute_run_entropy_rate_trajectory,
                outputs_root, family, dataset, variant, cfg.matrix.run_name, cfg.bin_step,
            ): (family, dataset, variant)
            for family, dataset, variant in cells
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Computing trajectories", dynamic_ncols=True
        ):
            family, dataset, variant = futures[future]
            result = future.result()
            if result is None:
                print(f"Skipping {family}/{dataset}/{variant}/{cfg.matrix.run_name}: no rollout metadata")
                continue
            curves_by_cell[(family, dataset)][variant] = result

    # trim each dataset panel to the largest bin count where every curve still has min_active_rollouts survivors
    for dataset in datasets:
        panel = [
            variant_curves for (_, panel_dataset), variant_curves in curves_by_cell.items()
            if panel_dataset == dataset
        ]
        panel_length = min(
            int((active_count >= cfg.min_active_rollouts).sum())
            for variant_curves in panel
            for _, _, active_count in variant_curves.values()
        )
        for variant_curves in panel:
            for variant, curve in variant_curves.items():
                variant_curves[variant] = tuple(curve_tensor[:panel_length] for curve_tensor in curve)

    output_path = Path(cfg.output.path)
    plot_entropy_rate_trajectory(
        curves_by_cell=curves_by_cell,
        families=families,
        datasets=datasets,
        output_path=output_path,
        label_fontsize=cfg.label_fontsize,
        line_width=cfg.line_width,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
