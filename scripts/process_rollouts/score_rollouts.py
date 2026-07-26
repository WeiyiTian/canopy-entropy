from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from safetensors.torch import save_file
from tqdm import tqdm

from src.constants import REWARD_SHARDS_ARTIFACT, ROLLOUT_SHARDS_ARTIFACT
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
from src.models import SkyworkRewardPipeline

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="score_rollouts")
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

    reward_artifact = f"{REWARD_SHARDS_ARTIFACT}/{cfg.reward.name}"
    reward_dir = build_prompt_shard_dir(run_dir, reward_artifact)
    if not cfg.resume:
        reset_prompt_shards(run_dir, reward_artifact)
    existing_shards = count_prompt_shards(run_dir, reward_artifact)
    if existing_shards >= metadata.num_prompts:
        print(f"Rewards already complete: {existing_shards}/{metadata.num_prompts} shards at {reward_dir}")
        return
    reward_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reward model...")
    reward_model = SkyworkRewardPipeline(
        model_path=Path(cfg.paths.model_root) / cfg.reward.name,
        device=cfg.inference.device,
        batch_size=cfg.reward.batch_size,
    )
    print("Finished loading reward model.")

    with tqdm(
        total=metadata.num_prompts,
        initial=existing_shards,
        desc="Scoring rewards",
        dynamic_ncols=True,
    ) as progress:
        for prompt_index in range(existing_shards, metadata.num_prompts):
            prompt, generated_texts = PromptRollouts.load_texts(
                build_prompt_shard_path(run_dir, ROLLOUT_SHARDS_ARTIFACT, prompt_index)
            )
            reward_scores = reward_model.score_batch(prompt, generated_texts).to("cpu", dtype=torch.float32)
            prompt_output_path = build_prompt_shard_path(run_dir, reward_artifact, prompt_index)
            save_file({"rewards": reward_scores}, str(prompt_output_path))
            progress.update(1)

    print(f"Wrote {metadata.num_prompts} reward shards to {reward_dir}")


if __name__ == "__main__":
    main()
