import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig

from src.generation_space.io import build_run_dir
from src.stats import build_panel, fit_interaction_regression

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="fit_interaction_regression")
def main(cfg: DictConfig) -> None:
    prompt_stats_paths = [
        build_run_dir(cfg.paths.outputs_root, dataset, model, variant, cfg.panel.run_name)
        / cfg.panel.prompt_stats_file
        for model in cfg.panel.models
        for dataset in cfg.panel.datasets
        for variant in cfg.panel.variants
    ]

    panel = build_panel(prompt_stats_paths)
    print(f"panel: {len(panel)} rows across {panel['prompt_id'].nunique()} prompts, "
          f"{panel[['model_name', 'model_variant']].drop_duplicates().shape[0]} (model, variant) pairs.")

    result = fit_interaction_regression(
        panel=panel,
        formula=cfg.regression.formula,
        groups_col=cfg.regression.groups,
        ft_variant=cfg.regression.ft_variant,
    )

    print(result.summary)
    print(f"\nbeta (base slope) = {result.beta.estimate:.4f}  SE={result.beta.stderr:.4f}  "
          f"p={result.beta.pvalue:.4g}  95% CI=[{result.beta.ci_low:.4f}, {result.beta.ci_high:.4f}]")
    print(f"tau  (FT shift)   = {result.tau.estimate:.4f}  SE={result.tau.stderr:.4f}  "
          f"p={result.tau.pvalue:.4g}  95% CI=[{result.tau.ci_low:.4f}, {result.tau.ci_high:.4f}]")
    print(f"eta  (FT extra)   = {result.eta.estimate:.4f}  SE={result.eta.stderr:.4f}  "
          f"p={result.eta.pvalue:.4g}  95% CI=[{result.eta.ci_low:.4f}, {result.eta.ci_high:.4f}]")

    results_path = Path(cfg.output.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in asdict(result).items() if k != "summary"}
    results_path.write_text(json.dumps(payload, indent=4))
    print(f"\nwrote results to {results_path}")


if __name__ == "__main__":
    main()
