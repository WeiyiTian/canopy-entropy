import json
from dataclasses import asdict
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.generation_space.io import build_run_dir, load_prompt_stats
from src.stats import compute_paired_bootstrap_comparison

load_dotenv()


METRIC_SCALE_TYPES = {
    "gen_ppl": "positive",
    "branching_factor": "positive",
    "ce_star_max": "positive",
    "entropy_rate_vs_length.covariance": "signed",
    "entropy_rate_vs_length.pearson": "signed",
    "entropy_rate_vs_length.spearman": "signed",
    "entropy_rate_vs_length.kendall": "signed",
}


@hydra.main(version_base=None, config_path="../../configs", config_name="bootstrap_base_vs_instruct")
def main(cfg: DictConfig) -> None:
    all_results = {}
    for model in cfg.matrix.models:
        for dataset in cfg.matrix.datasets:
            base_path = (
                build_run_dir(cfg.paths.outputs_root, dataset, model, "base", cfg.matrix.run_name)
                / cfg.prompt_stats_file
            )
            instruct_path = (
                build_run_dir(cfg.paths.outputs_root, dataset, model, "instruct", cfg.matrix.run_name)
                / cfg.prompt_stats_file
            )
            if not base_path.exists() or not instruct_path.exists():
                print(f"skipping {model}/{dataset}: missing prompt stats")
                continue

            pair_results = compute_paired_bootstrap_comparison(
                base_stats=load_prompt_stats(base_path, device="cpu"),
                instruct_stats=load_prompt_stats(instruct_path, device="cpu"),
                metric_scale_types=METRIC_SCALE_TYPES,
                n_boot=cfg.bootstrap.n_boot,
                seed=cfg.bootstrap.seed,
                ci_level=cfg.bootstrap.ci_level,
            )
            all_results[f"{model}/{dataset}"] = {
                metric_name: {
                    estimate_name: asdict(bootstrap_estimate)
                    for estimate_name, bootstrap_estimate in metric_estimates.items()
                }
                for metric_name, metric_estimates in pair_results.items()
            }

            print(f"computed {model}/{dataset}")

    results_path = Path(cfg.output.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=4))
    print(f"wrote {len(all_results)} (model, dataset) comparisons to {results_path}")


if __name__ == "__main__":
    main()
