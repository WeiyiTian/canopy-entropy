import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig

from src.generation_space.io import build_run_dir
from src.stats import build_panel, fit_quadratic_robustness

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="fit_quadratic_robustness")
def main(cfg: DictConfig) -> None:
    prompt_stats_paths = [
        build_run_dir(cfg.paths.outputs_root, dataset, model, variant, cfg.matrix.run_name)
        / cfg.prompt_stats_file
        for model in cfg.matrix.models
        for dataset in cfg.matrix.datasets
        for variant in cfg.matrix.variants
    ]

    panel = build_panel(prompt_stats_paths)
    print(f"panel: {len(panel)} rows across {panel['prompt_id'].nunique()} prompts, "
          f"{panel[['model_name', 'model_variant']].drop_duplicates().shape[0]} (model, variant) pairs.")

    result = fit_quadratic_robustness(
        panel=panel,
        groups_col=cfg.regression.groups,
        ft_variant=cfg.regression.ft_variant,
    )

    print("=== Linear (ML) ===")
    print(result.linear_summary)
    print("\n=== Quadratic (ML) ===")
    print(result.quadratic_summary)

    print(f"\nR_bar mean (centering constant) = {result.r_bar_mean:.6f}")
    print(f"\nLinear: beta = {result.linear_beta.estimate:.4f} (SE={result.linear_beta.stderr:.4f}, "
          f"p={result.linear_beta.pvalue:.4g})  "
          f"eta = {result.linear_eta.estimate:.4f} (SE={result.linear_eta.stderr:.4f}, "
          f"p={result.linear_eta.pvalue:.4g})")
    print(f"Quadratic: beta = {result.quadratic_beta.estimate:.4f} (SE={result.quadratic_beta.stderr:.4f}, "
          f"p={result.quadratic_beta.pvalue:.4g})  "
          f"eta = {result.quadratic_eta.estimate:.4f} (SE={result.quadratic_eta.stderr:.4f}, "
          f"p={result.quadratic_eta.pvalue:.4g})")
    print(f"Quadratic: q (R_c^2) = {result.quadratic_q.estimate:.4g} (SE={result.quadratic_q.stderr:.4g}, "
          f"p={result.quadratic_q.pvalue:.4g})  "
          f"k (R_c^2:FT) = {result.quadratic_k.estimate:.4g} (SE={result.quadratic_k.stderr:.4g}, "
          f"p={result.quadratic_k.pvalue:.4g})")
    print(f"\nJoint LRT (df={result.lrt_df}): chi2 = {result.lrt_statistic:.4f}, "
          f"p = {result.lrt_pvalue:.4g}")
    print(f"delta_beta = {result.delta_beta:+.4f}  (within 1 SE: {result.delta_beta_in_one_se})")
    print(f"delta_eta  = {result.delta_eta:+.4f}  (within 1 SE: {result.delta_eta_in_one_se})")

    results_path = Path(cfg.output.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: v for k, v in asdict(result).items()
        if k not in {"linear_summary", "quadratic_summary"}
    }
    results_path.write_text(json.dumps(payload, indent=4))
    print(f"\nwrote results to {results_path}")


if __name__ == "__main__":
    main()
