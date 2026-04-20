import argparse
import json
from pathlib import Path

import wandb
from tqdm import tqdm

from src.generation_space.core import GenerationMetadata, PromptRollouts
from src.generation_space.io import (
    build_rollout_metadata_path,
    build_rollout_shard_path,
    count_existing_shards,
    load_prompts,
    reset_rollout_dir,
    resume_rollouts,
)
from src.generation_space.reporting import resolve_run_name, to_wandb_config
from src.metrics import step_conditional_entropy_from_logprobs
from src.models import generate_step_scores, load_generation_backend
from src.settings import settings
from src.utils import build_artifact_path, build_model_path, clear_runtime_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-prompt rollout shards with vLLM/local backend.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(settings.data_dir))
    parser.add_argument("--model-root", type=Path, default=Path(settings.model_dir))
    parser.add_argument("--outputs-root", type=Path, default=Path(settings.outputs_dir))
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-variant", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--backend", choices=["local", "vllm"], default="vllm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logprobs", type=int, default=100)
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-logprobs", type=int, default=100)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--save-logprobs", action="store_true")
    parser.add_argument("--rollout-file", type=str, default="generation_rollouts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="GenPPL")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rollout_dir = build_artifact_path(
        args.outputs_root,
        args.file_name,
        args.model_name,
        args.model_variant,
        args.rollout_file,
    )
    is_resume = args.resume and build_rollout_metadata_path(rollout_dir).exists()

    prompt_file = args.data_dir / args.file_name
    prompts = load_prompts(prompt_file, args.num_prompts)
    requested_metadata = GenerationMetadata(
        prompt_file=str(prompt_file),
        num_prompts_processed=len(prompts),
        model_name=args.model_name,
        model_variant=args.model_variant,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        logprobs=args.logprobs,
        enable_thinking=args.enable_thinking,
    )

    if is_resume:
        start_index = resume_rollouts(rollout_dir, requested_metadata)
        if start_index >= len(prompts):
            print(f"Rollouts already complete: {start_index}/{len(prompts)} shards at {rollout_dir}")
            return
    else:
        reset_rollout_dir(rollout_dir, requested_metadata)
        start_index = 0

    run_name = resolve_run_name(
        args.wandb_run_name, args.model_name, args.file_name, args.model_variant
    )
    run = wandb.init(
        project=args.wandb_project,
        name=f"{run_name}-generate",
        group=run_name,
        job_type="generate",
        mode=args.wandb_mode,
        config={**to_wandb_config(args), "resumed_from_shard": start_index},
    )

    model_path = build_model_path(args.model_root, args.model_name, args.model_variant)
    model, tokenizer = load_generation_backend(
        model_path=model_path,
        backend=args.backend,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_logprobs=args.max_logprobs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    remaining_prompts = prompts[start_index:]
    with tqdm(
        total=len(prompts),
        initial=start_index,
        desc="Generating rollouts",
        dynamic_ncols=True,
    ) as progress:
        
        for batch_start in range(0, len(remaining_prompts), args.prompt_batch_size):
            prompt_batch = remaining_prompts[batch_start: batch_start + args.prompt_batch_size]
            batch_results = generate_step_scores(
                prompts=prompt_batch,
                model=model,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                backend=args.backend,
                tokenizer=tokenizer,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                logprobs=args.logprobs,
                sample_batch_size=args.sample_batch_size,
                device=args.device,
                enable_thinking=args.enable_thinking,
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
                    sequence_step_logprobs=sequence_step_logprobs if args.save_logprobs else None,
                ).save(build_rollout_shard_path(rollout_dir, absolute_offset + prompt_offset))
            
            progress.update(len(prompt_batch))

    del model, tokenizer
    clear_runtime_memory()

    final_shard_count = count_existing_shards(rollout_dir)
    summary = {
        "status": "ok",
        "rollout_dir": str(rollout_dir),
        "num_prompts": len(prompts),
        "shards_on_disk": final_shard_count,
        "shards_generated_this_run": final_shard_count - start_index,
    }
    wandb.run.summary.update(summary)
    run.finish()
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
