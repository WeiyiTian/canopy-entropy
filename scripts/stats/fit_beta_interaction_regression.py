import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig

from src.generation_space.io import build_run_dir
from src.stats import build_panel

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="fit_beta_interaction_regression")
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

    panel_csv = Path(cfg.output.panel_csv)
    panel_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_csv, index=False)
    print(f"wrote panel to {panel_csv}")

    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not found on PATH; install R to run the beta regression.")

    r_script_src = Path(cfg.r_script).resolve()
    r_script_local = panel_csv.parent / r_script_src.name
    shutil.copy(r_script_src, r_script_local)

    log_path = Path(cfg.output.r_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        ["Rscript", r_script_local.name],
        cwd=panel_csv.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(proc.returncode)
    print(f"wrote R log to {log_path}")


if __name__ == "__main__":
    main()
