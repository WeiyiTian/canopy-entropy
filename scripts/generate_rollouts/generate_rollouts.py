import json
from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.constants import ROLLOUT_SHARDS_ARTIFACT
from src.generation_space.core import GenerationMetadata, PromptRollouts
from src.generation_space.io import (
    build_prompt_shard_path,
    build_metadata_path,
    build_run_dir,
    count_prompt_shards,
    load_prompts,
    reset_prompt_shards,
    resume_rollouts,
)
from src.generation_space.reporting import resolve_run_name
from src.metrics.generation_tree import step_conditional_entropy_from_logprobs
from src.models import build_model_path, generate_step_scores, load_generation_backend

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="generate_rollouts")
def main(cfg: DictConfig) -> None:
    run_dir = build_run_dir(
        cfg.paths.outputs_root,
        cfg.dataset.file_name,
        cfg.model.name,
        cfg.model.variant,
        cfg.run_name,
    )
    is_resume = cfg.resume and build_metadata_path(run_dir).exists()

    if cfg.sampling.top_k is not None and cfg.sampling.top_k > 0:
        assert cfg.sampling.logprobs >= cfg.sampling.top_k, (
            f"retained logprobs ({cfg.sampling.logprobs}) must be >= top_k ({cfg.sampling.top_k})."
        )

    prompt_file = Path(cfg.paths.data_dir) / cfg.dataset.file_name
    prompts = load_prompts(prompt_file, cfg.dataset.num_prompts)
    requested_metadata = GenerationMetadata(
        prompt_file=str(prompt_file),
        num_prompts=len(prompts),
        model_name=cfg.model.name,
        model_variant=cfg.model.variant,
        n_samples=cfg.sampling.n_samples,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temperature,
        top_k=cfg.sampling.top_k,
        top_p=cfg.sampling.top_p,
        seed=cfg.sampling.seed,
        logprobs=cfg.sampling.logprobs,
        enable_thinking=cfg.model.enable_thinking,
    )

    if is_resume:
        start_index = resume_rollouts(run_dir, requested_metadata)
        print(f"Resuming rollouts from index {start_index} at {run_dir}")
        if start_index >= len(prompts):
            print(f"Rollouts already complete: {start_index}/{len(prompts)} shards at {run_dir}")
            return
    else:
        discarded_shards = count_prompt_shards(run_dir, ROLLOUT_SHARDS_ARTIFACT)
        if discarded_shards > 0 and not cfg.force:
            raise ValueError(
                f"Refusing to delete {discarded_shards} rollout shards at {run_dir}. "
                f"Pass force=true."
            )
        reset_prompt_shards(run_dir, ROLLOUT_SHARDS_ARTIFACT)
        requested_metadata.save(build_metadata_path(run_dir))
        start_index = 0

    run_name = resolve_run_name(
        cfg.wandb.run_name, cfg.model.name, cfg.dataset.file_name, cfg.model.variant
    )
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{run_name}-generate",
        group=run_name,
        job_type="generate",
        mode=cfg.wandb.mode,
        config={
            **OmegaConf.to_container(cfg, resolve=True),
            "resumed_from_shard": start_index,
        },
    )

    model_path = build_model_path(cfg.paths.model_root, cfg.model.name, cfg.model.variant)
    model, tokenizer = load_generation_backend(
        model_path=model_path,
        backend=cfg.inference.backend,
        device=cfg.inference.device,
        tensor_parallel_size=cfg.inference.tensor_parallel_size,
        max_model_len=cfg.inference.max_model_len,
        max_logprobs=cfg.inference.max_logprobs,
        gpu_memory_utilization=cfg.inference.gpu_memory_utilization,
        seed=cfg.sampling.seed,
    )

    remaining_prompts = prompts[start_index:]
    with tqdm(
        total=len(prompts),
        initial=start_index,
        desc="Generating rollouts",
        dynamic_ncols=True,
        position=2,
        leave=True,
    ) as progress:
        for batch_start in range(0, len(remaining_prompts), cfg.inference.prompt_batch_size):
            prompt_batch = remaining_prompts[batch_start: batch_start + cfg.inference.prompt_batch_size]
            batch_results = generate_step_scores(
                prompts=prompt_batch,
                model=model,
                n_samples=cfg.sampling.n_samples,
                max_new_tokens=cfg.sampling.max_new_tokens,
                backend=cfg.inference.backend,
                tokenizer=tokenizer,
                temperature=cfg.sampling.temperature,
                top_k=cfg.sampling.top_k,
                top_p=cfg.sampling.top_p,
                seed=cfg.sampling.seed,
                logprobs=cfg.sampling.logprobs,
                sample_batch_size=cfg.inference.sample_batch_size,
                device=cfg.inference.device,
                enable_thinking=cfg.model.enable_thinking,
                use_chat_template=cfg.model.variant != "base",
            )

            absolute_offset = start_index + batch_start
            for prompt_offset, prompt_result in enumerate(batch_results):
                (
                    generated_texts,
                    sequence_step_logprobs,
                    sequence_lengths,
                    generated_token_ids,
                    prompt_token_ids,
                ) = prompt_result

                step_conditional_entropy = [
                    step_conditional_entropy_from_logprobs(logprobs=log_probs)
                    for log_probs in sequence_step_logprobs
                ]

                PromptRollouts(
                    prompt=prompt_batch[prompt_offset],
                    prompt_token_ids=prompt_token_ids,
                    generated_texts=generated_texts,
                    generated_token_ids=generated_token_ids,
                    sequence_lengths=sequence_lengths,
                    step_conditional_entropy=step_conditional_entropy,
                    sequence_step_logprobs=sequence_step_logprobs if cfg.sampling.save_logprobs else None,
                ).save(
                    build_prompt_shard_path(
                        run_dir, ROLLOUT_SHARDS_ARTIFACT, absolute_offset + prompt_offset
                    )
                )

            progress.update(len(prompt_batch))

    final_shard_count = count_prompt_shards(run_dir, ROLLOUT_SHARDS_ARTIFACT)
    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "num_prompts": len(prompts),
        "shards_on_disk": final_shard_count,
        "shards_generated_this_run": final_shard_count - start_index,
    }
    wandb.run.summary.update(summary)
    run.finish()
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
