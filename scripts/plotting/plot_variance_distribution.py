from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from src.generation_space.io import build_run_dir, load_prompt_stats
from src.visualization import plot_variance_distribution

load_dotenv()


def compute_run_per_prompt_variances(
    outputs_root: Path,
    family: str,
    dataset: str,
    variant: str,
    run_name: str,
    prompt_stats_file: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Loads the saved per-prompt rollout stats for one run and returns the within-prompt
    sample variance of generation length and entropy rate, one value per prompt.

    Returns (var_lengths, var_entropy_rates), each of shape [P]. Returns None if the
    prompt stats artifact is missing.
    """
    file_name = f"{dataset}.jsonl"
    run_dir = build_run_dir(outputs_root, file_name, family, variant, run_name)
    stats_path = run_dir / prompt_stats_file
    if not stats_path.exists():
        return None

    prompt_results = load_prompt_stats(stats_path, device="cpu")

    var_lengths = []
    var_entropy_rates = []
    for prompt_result in prompt_results:
        lengths = prompt_result.raw_sequence_lengths.to(dtype=torch.float64)
        entropies = prompt_result.raw_sequence_conditional_entropy.to(dtype=torch.float64)
        if lengths.numel() < 2:
            continue
        rates = entropies / lengths.clamp(min=1.0)
        var_lengths.append(lengths.var(unbiased=True))
        var_entropy_rates.append(rates.var(unbiased=True))

    return torch.stack(var_lengths), torch.stack(var_entropy_rates)


@hydra.main(version_base=None, config_path="../../configs", config_name="plot_variance_distribution")
def main(cfg: DictConfig) -> None:
    families = list(cfg.matrix.models)
    variants = list(cfg.matrix.variants)
    datasets = list(cfg.matrix.datasets)
    metric = cfg.metric

    if metric not in {"length", "entropy_rate"}:
        raise ValueError(f"Unknown metric: {metric} (expected 'length' or 'entropy_rate')")

    outputs_root = Path(cfg.paths.outputs_root)
    cells = [
        (family, dataset, variant)
        for family in families
        for dataset in datasets
        for variant in variants
    ]

    variances_by_cell = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                compute_run_per_prompt_variances,
                outputs_root, family, dataset, variant, cfg.matrix.run_name, cfg.prompt_stats_file,
            ): (family, dataset, variant)
            for family, dataset, variant in cells
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Computing variances", dynamic_ncols=True
        ):
            family, dataset, variant = futures[future]
            result = future.result()
            if result is None:
                print(f"Skipping {family}/{dataset}/{variant}/{cfg.matrix.run_name}: no prompt stats")
                continue
            var_lengths, var_entropy_rates = result
            variances_by_cell[(family, dataset)][variant] = (
                var_lengths if metric == "length" else var_entropy_rates
            )

    output_path = Path(cfg.output.path)
    plot_variance_distribution(
        variances_by_cell=variances_by_cell,
        families=families,
        datasets=datasets,
        output_path=output_path,
        metric=metric,
        n_eval_points=cfg.n_eval_points,
        label_fontsize=cfg.label_fontsize,
        line_width=cfg.line_width,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
