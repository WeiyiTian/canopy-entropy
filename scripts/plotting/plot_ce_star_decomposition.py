import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from src.generation_space.io import build_run_dir
from src.metrics.generation_tree import CEStarMaxDecomposition, ce_star_decomposition_from_pooled
from src.visualization import plot_ce_star_decomposition

load_dotenv()


def load_run_decomposition(
    results_root: Path,
    file_name: str,
    family: str,
    variant: str,
    run_name: str,
    pooled_metrics_file: str,
) -> CEStarMaxDecomposition | None:
    """Reads the saved raw pooled metrics for one run and returns its CE*_max decomposition,
    or None if missing."""
    run_dir = build_run_dir(results_root, file_name, family, variant, run_name)
    pooled_path = run_dir / pooled_metrics_file
    if not pooled_path.exists():
        return None
    pooled = json.loads(pooled_path.read_text())["raw"]
    return ce_star_decomposition_from_pooled(
        ce_star_max=float(pooled["ce_star_max"]),
        gen_ppl=float(pooled["gen_ppl"]),
        branching_factor=float(pooled["branching_factor"]),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="plot_ce_star_decomposition")
def main(cfg: DictConfig) -> None:
    families = list(cfg.matrix.models)
    variants = list(cfg.matrix.variants)
    datasets = list(cfg.matrix.datasets)

    results_root = Path(cfg.paths.results_root)
    cells = [
        (family, dataset, variant)
        for family in families
        for dataset in datasets
        for variant in variants
    ]

    decompositions_by_cell = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                load_run_decomposition,
                results_root,
                f"{dataset}.jsonl",
                family,
                variant,
                cfg.matrix.run_name,
                cfg.pooled_metrics_file,
            ): (family, dataset, variant)
            for family, dataset, variant in cells
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading decompositions", dynamic_ncols=True
        ):
            family, dataset, variant = futures[future]
            decomp = future.result()
            if decomp is None:
                print(f"Skipping {family}/{dataset}/{variant}/{cfg.matrix.run_name}: no pooled metrics")
                continue
            decompositions_by_cell[(family, dataset)][variant] = decomp

    output_path = Path(cfg.output.path)
    plot_ce_star_decomposition(
        decompositions_by_cell=decompositions_by_cell,
        families=families,
        datasets=datasets,
        variants=variants,
        output_path=output_path,
        label_fontsize=cfg.label_fontsize,
        bar_height=cfg.bar_height,
        line_width=cfg.line_width,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
