import json

import hydra
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.constants import (
    EMBEDDING_SHARDS_ARTIFACT,
    REWARD_SHARDS_ARTIFACT,
    ROLLOUT_SHARDS_ARTIFACT,
)
from src.generation_space.core import (
    GenerationMetadata,
    PromptRolloutStats,
    PromptRollouts,
    build_reward_filter_mask,
    compute_pooled_metrics,
    compute_prompt_rollout_stats,
)
from src.generation_space.io import (
    build_prompt_shard_path,
    build_rollout_metadata_path,
    build_run_dir,
    count_prompt_shards,
    load_prompt_shard_tensor,
    save_pooled_metrics,
    save_prompt_stats,
    verify_prompt_shards_complete,
)
from src.generation_space.reporting import log_metric_artifacts, resolve_run_name

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="compute_gen_space")
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

    num_rewards = count_prompt_shards(run_dir, REWARD_SHARDS_ARTIFACT)
    num_embeddings = count_prompt_shards(run_dir, EMBEDDING_SHARDS_ARTIFACT)
    has_rewards = cfg.use_reward_filter and num_rewards >= metadata.num_prompts
    has_embeddings = num_embeddings >= metadata.num_prompts
    if not cfg.use_reward_filter:
        print("Reward filtering disabled.")
    elif not has_rewards:
        print(
            f"Reward shards missing/incomplete ({num_rewards}/{metadata.num_prompts}); "
            f"proceeding without reward filtering."
        )
    if not has_embeddings:
        print(
            f"Embedding shards missing/incomplete ({num_embeddings}/{metadata.num_prompts}); "
            f"proceeding without semantic metrics."
        )

    prompt_stats_path = run_dir / cfg.prompt_stats_file
    results_run_dir = build_run_dir(
        cfg.paths.results_root,
        cfg.dataset.file_name,
        cfg.model.name,
        cfg.model.variant,
        cfg.run_name,
    )
    pooled_metrics_path = results_run_dir / cfg.pooled_metrics_file
    sequence_length_hist_path = results_run_dir / "sequence_length_histogram.png"

    wandb_run_name = resolve_run_name(
        cfg.wandb.run_name, cfg.model.name, cfg.dataset.file_name, cfg.model.variant
    )
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{wandb_run_name}-compute",
        group=wandb_run_name,
        job_type="compute",
        mode=cfg.wandb.mode,
        config={
            **OmegaConf.to_container(cfg, resolve=True),
            **metadata.to_dict(),
        },
    )

    device = torch.device(cfg.inference.device)
    prompt_results: list[PromptRolloutStats] = []
    for prompt_index in tqdm(
        range(metadata.num_prompts),
        desc="Computing metrics",
        dynamic_ncols=True,
    ):
        shard = PromptRollouts.load(
            build_prompt_shard_path(run_dir, ROLLOUT_SHARDS_ARTIFACT, prompt_index),
            device=device,
        )

        if has_rewards:
            reward_scores = load_prompt_shard_tensor(
                run_dir, REWARD_SHARDS_ARTIFACT, prompt_index, "rewards", device
            )
            keep_mask = build_reward_filter_mask(reward_scores, cfg.keep_fraction)
        else:
            reward_scores = torch.zeros(metadata.n_samples, dtype=torch.float32, device=device)
            keep_mask = torch.ones(metadata.n_samples, dtype=torch.bool, device=device)

        normalized_embeddings = (
            load_prompt_shard_tensor(
                run_dir, EMBEDDING_SHARDS_ARTIFACT, prompt_index, "embeddings", device
            )
            if has_embeddings
            else None
        )

        prompt_result = compute_prompt_rollout_stats(
            prompt=shard.prompt,
            step_conditional_entropy=shard.step_conditional_entropy,
            sequence_lengths=shard.sequence_lengths,
            normalized_embeddings=normalized_embeddings,
            reward_scores=reward_scores,
            keep_mask=keep_mask,
            max_new_tokens=metadata.max_new_tokens,
        )
        prompt_results.append(prompt_result.to_cpu())
        del shard, prompt_result

    pooled_metrics = compute_pooled_metrics(prompt_results)
    save_prompt_stats(prompt_results, prompt_stats_path)
    save_pooled_metrics(pooled_metrics, pooled_metrics_path)

    scalar_summary = log_metric_artifacts(
        metadata=metadata,
        pooled_metrics=pooled_metrics,
        prompt_results=prompt_results,
        prompt_stats_path=prompt_stats_path,
        pooled_metrics_path=pooled_metrics_path,
        sequence_length_hist_path=sequence_length_hist_path,
    )
    run.finish()
    print(json.dumps(scalar_summary, indent=4))


if __name__ == "__main__":
    main()
