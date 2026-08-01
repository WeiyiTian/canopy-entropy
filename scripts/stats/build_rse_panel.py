from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.constants import JUDGMENT_SHARDS_ARTIFACT
from src.generation_space.io import build_run_dir
from src.stats import build_rse_panel

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="build_rse_panel")
def main(cfg: DictConfig) -> None:
    prompt_stats_paths = [
        build_run_dir(cfg.paths.outputs_root, dataset, model, variant, cfg.run_name)
        / cfg.prompt_stats_file
        for model in cfg.matrix.models
        for dataset in cfg.matrix.datasets
        for variant in cfg.matrix.variants
    ]

    panel = build_rse_panel(prompt_stats_paths, f"{JUDGMENT_SHARDS_ARTIFACT}/{cfg.judge.name}")
    print(f"panel: {len(panel)} rows across {panel['prompt_id'].nunique()} prompts, "
          f"{panel[['model_name', 'model_variant']].drop_duplicates().shape[0]} (model, variant) pairs.")

    # a large saturated share means m is too small to separate the most diverse prompts from each other
    print(f"saturated at log(m): {(panel['n_clusters'] == panel['m']).mean():.1%}")
    print(panel.groupby("model_variant")[["RSE", "RSE_norm", "n_clusters"]].mean().round(3))

    panel_csv = Path(cfg.output.panel_csv)
    panel_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_csv, index=False)
    print(f"wrote rse panel to {panel_csv}")


if __name__ == "__main__":
    main()
