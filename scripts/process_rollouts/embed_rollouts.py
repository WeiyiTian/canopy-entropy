from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.constants import EMBEDDING_SHARDS_ARTIFACT, ROLLOUT_SHARDS_ARTIFACT
from src.generation_space.core import GenerationMetadata, PromptRollouts
from src.generation_space.io import (
    build_prompt_shard_dir,
    build_prompt_shard_path,
    build_rollout_metadata_path,
    build_run_dir,
    count_prompt_shards,
    reset_prompt_shards,
    verify_prompt_shards_complete,
)

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="embed_rollouts")
def main(cfg: DictConfig) -> None:
    run_dir = build_run_dir(
        cfg.paths.outputs_root,
        cfg.dataset.file_name,
        cfg.model.name,
        cfg.model.variant,
        cfg.run_name,
    )
    metadata_path = build_rollout_metadata_path(run_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Rollout metadata not found at {metadata_path}. Run generate_rollouts.py first."
        )
    metadata = GenerationMetadata.load(metadata_path)
    verify_prompt_shards_complete(run_dir, ROLLOUT_SHARDS_ARTIFACT, metadata.num_prompts)

    embedding_artifact = f"{EMBEDDING_SHARDS_ARTIFACT}/{cfg.embedding.name}"
    embedding_dir = build_prompt_shard_dir(run_dir, embedding_artifact)
    if not cfg.resume:
        reset_prompt_shards(run_dir, embedding_artifact)
    existing_shards = count_prompt_shards(run_dir, embedding_artifact)
    if existing_shards >= metadata.num_prompts:
        print(f"Embeddings already complete: {existing_shards}/{metadata.num_prompts} shards at {embedding_dir}")
        return
    embedding_dir.mkdir(parents=True, exist_ok=True)

    print("Loading semantic model...")
    semantic_model = SentenceTransformer(
        str(Path(cfg.paths.model_root) / cfg.embedding.name),
        device=cfg.inference.device,
        model_kwargs={"dtype": torch.bfloat16},
    )
    print("Finished loading semantic model.")

    with tqdm(
        total=metadata.num_prompts,
        initial=existing_shards,
        desc="Embedding rollouts",
        dynamic_ncols=True,
    ) as progress:
        for prompt_index in range(existing_shards, metadata.num_prompts):
            prompt_output_path = build_prompt_shard_path(run_dir, embedding_artifact, prompt_index)
            _, generated_texts = PromptRollouts.load_texts(
                build_prompt_shard_path(run_dir, ROLLOUT_SHARDS_ARTIFACT, prompt_index)
            )
            embeddings = semantic_model.encode(
                [f"search_document: {text}" for text in generated_texts],
                batch_size=cfg.embedding.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).to("cpu", dtype=torch.float32) # [M, D]
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            save_file({"embeddings": embeddings}, str(prompt_output_path))
            progress.update(1)

    print(f"Wrote {metadata.num_prompts} embedding shards to {embedding_dir}")


if __name__ == "__main__":
    main()
