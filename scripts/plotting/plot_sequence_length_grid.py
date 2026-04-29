from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import ROLLOUT_SHARDS_ARTIFACT
from src.generation_space.core import GenerationMetadata
from src.generation_space.io import (
    build_rollout_metadata_path,
    build_run_dir,
    load_prompt_shard_tensor,
    verify_prompt_shards_complete,
)
from src.visualization.sequence_length_hist import plot_sequence_length_grid

load_dotenv()


def load_run_sequence_lengths(
    outputs_root: Path,
    family: str,
    dataset: str,
    variant: str,
    run_name: str,
) -> torch.Tensor | None:
    """
    Loads and concatenates sequence_lengths across all prompt shards of one run.
    Returns None if the run directory or metadata is missing.
    """
    file_name = f"{dataset}.jsonl"
    run_dir = build_run_dir(outputs_root, file_name, family, variant, run_name)
    metadata_path = build_rollout_metadata_path(run_dir)
    if not metadata_path.exists():
        return None
    metadata = GenerationMetadata.load(metadata_path)
    verify_prompt_shards_complete(run_dir, ROLLOUT_SHARDS_ARTIFACT, metadata.num_prompts)

    chunks = [
        load_prompt_shard_tensor(
            run_dir, ROLLOUT_SHARDS_ARTIFACT, prompt_index, "sequence_lengths", device="cpu"
        )
        for prompt_index in range(metadata.num_prompts)
    ]
    return torch.cat(chunks)


@hydra.main(version_base=None, config_path="../../configs", config_name="plot_sequence_length_grid")
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

    lengths_by_cell = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                load_run_sequence_lengths,
                outputs_root, family, dataset, variant, cfg.matrix.run_name,
            ): (family, dataset, variant)
            for family, dataset, variant in cells
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading lengths", dynamic_ncols=True
        ):
            family, dataset, variant = futures[future]
            lengths = future.result()
            if lengths is None:
                print(f"Skipping {family}/{dataset}/{variant}/{cfg.matrix.run_name}: no rollout metadata")
                continue
            lengths_by_cell[(family, dataset)][variant] = lengths

    output_path = Path(cfg.output.path)
    plot_sequence_length_grid(
        data=lengths_by_cell,
        families=families,
        datasets=datasets,
        output_path=output_path,
        bins=cfg.bins,
        label_fontsize=cfg.label_fontsize,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
