import asyncio
import hashlib
import os
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import JUDGMENT_SHARDS_ARTIFACT, ROLLOUT_SHARDS_ARTIFACT
from src.generation_space.core import GenerationMetadata, JudgeMetadata, PromptRollouts
from src.generation_space.io import (
    build_metadata_path,
    build_prompt_shard_dir,
    build_prompt_shard_path,
    build_run_dir,
    load_judge_prompt,
    reset_prompt_shards,
    resume_judgments,
    save_judgment_shard,
    verify_prompt_shards_complete,
)
from src.models import JudgeStatus, SimilarityJudge, build_judgment_matrix

load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="judge_pairs")
def main(cfg: DictConfig) -> None:
    asyncio.run(judge_rollouts(cfg))


async def judge_rollouts(cfg: DictConfig) -> None:
    """Collects pairwise similarity verdicts for the prompts of one rollout run."""
    run_dir = build_run_dir(
        cfg.paths.outputs_root,
        cfg.dataset.file_name,
        cfg.model.name,
        cfg.model.variant,
        cfg.run_name,
    )
    metadata_path = build_metadata_path(run_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Rollout metadata not found at {metadata_path}. Run generate_rollouts.py first."
        )
    metadata = GenerationMetadata.load(metadata_path)
    verify_prompt_shards_complete(run_dir, ROLLOUT_SHARDS_ARTIFACT, metadata.num_prompts)

    prompt_path, prompt_template = load_judge_prompt(
        Path(cfg.judge.prompts_dir), Path(metadata.prompt_file).stem
    )
    judge_metadata = JudgeMetadata(
        model=cfg.judge.model,
        base_url=cfg.judge.base_url,
        prompt_file=str(prompt_path),
        # bytes => hash => hex
        prompt_hash=hashlib.sha256(prompt_template.encode()).hexdigest()[:16],
        num_rollouts=cfg.judge.num_rollouts,
        seed=cfg.judge.seed,
        temperature=cfg.judge.temperature,
        top_p=cfg.judge.top_p,
        max_tokens=cfg.judge.max_tokens,
    )

    num_prompts = metadata.num_prompts
    if cfg.judge.max_prompts is not None:
        num_prompts = min(num_prompts, cfg.judge.max_prompts)

    judgment_artifact = f"{JUDGMENT_SHARDS_ARTIFACT}/{cfg.judge.name}"
    judgment_dir = build_prompt_shard_dir(run_dir, judgment_artifact)
    if not cfg.resume:
        reset_prompt_shards(run_dir, judgment_artifact)
    existing_shards = resume_judgments(run_dir, judgment_artifact, judge_metadata)
    if existing_shards >= num_prompts:
        print(f"Judgments already complete: {existing_shards}/{num_prompts} shards at {judgment_dir}")
        return

    judge = SimilarityJudge(
        model=cfg.judge.model,
        base_url=cfg.judge.base_url,
        api_key=os.environ[cfg.judge.api_key_env],
        prompt_template=prompt_template,
        max_concurrency=cfg.judge.max_concurrency,
        temperature=cfg.judge.temperature,
        top_p=cfg.judge.top_p,
        max_tokens=cfg.judge.max_tokens,
        timeout=cfg.judge.timeout,
        max_retries=cfg.judge.max_retries,
    )
    await verify_judge_ready(judge)

    num_calls = 0
    num_similar = 0
    num_usable = 0
    with tqdm(
        total=num_prompts,
        initial=existing_shards,
        desc="Judging pairs",
        dynamic_ncols=True,
    ) as progress:
        for prompt_index in range(existing_shards, num_prompts):
            prompt, generated_texts = PromptRollouts.load_texts(
                build_prompt_shard_path(run_dir, ROLLOUT_SHARDS_ARTIFACT, prompt_index)
            )
            subsample_indices = draw_subsample(
                num_rollouts=len(generated_texts),
                num_judged=cfg.judge.num_rollouts,
                seed=cfg.judge.seed,
                prompt_index=prompt_index,
            )
            responses = [generated_texts[index] for index in subsample_indices.tolist()]

            matrix = await build_judgment_matrix(judge, prompt, responses)
            if matrix.failure_rate > cfg.judge.max_failure_rate:
                breakdown = ", ".join(
                    f"{matrix.count_status(status)} {status.value}"
                    for status in JudgeStatus
                    if status is not JudgeStatus.OK
                )
                raise RuntimeError(
                    f"Prompt {prompt_index}: {matrix.failure_rate:.1%} of "
                    f"{len(matrix.pair_judgments)} judge calls yielded no verdict "
                    f"({breakdown}), above the {cfg.judge.max_failure_rate:.1%} threshold. "
                    f"Such calls resolve to not similar and would inflate diversity, so no "
                    f"shard was written."
                )

            save_judgment_shard(
                output_path=build_prompt_shard_path(run_dir, judgment_artifact, prompt_index),
                judgments=matrix.judgments,
                subsample_indices=subsample_indices,
                pair_judgments=[asdict(judgment) for judgment in matrix.pair_judgments],
            )

            num_calls += len(matrix.pair_judgments)
            num_similar += sum(judgment.is_similar for judgment in matrix.pair_judgments)
            num_usable += matrix.count_status(JudgeStatus.OK)
            progress.set_postfix(
                ok=f"{num_usable / num_calls:.0%}",
                similar=f"{num_similar / num_calls:.0%}",
            )
            progress.update(1)

    print(f"Wrote {num_prompts - existing_shards} judgment shards to {judgment_dir}")


async def verify_judge_ready(judge: SimilarityJudge) -> None:
    """
    Judges one hand-made pair of identical responses before the main loop.

    A wrong model identifier, endpoint or key otherwise surfaces only as a run whose every
    call fails, after a prompt's worth of requests have already been spent. Identical
    responses are the least ambiguous similar pair there is, so a verdict of different
    means the prompt is not eliciting the expected format even though the call succeeded.

    Raises:
        RuntimeError: If the probe returns no verdict, or calls identical responses different.
    """
    response = "The answer is 42."
    judgment, = await judge.judge_pairs("Probe.", [response, response], [(0, 1)])
    if judgment.status is not JudgeStatus.OK:
        raise RuntimeError(f"Judge probe returned no verdict ({judgment.status}): {judgment.text}")
    if not judgment.is_similar:
        raise RuntimeError(
            f"Judge probe called two identical responses different, so the judge prompt is "
            f"not working as intended. Reply was: {judgment.text!r}"
        )
    print("Judge probe OK")


def draw_subsample(num_rollouts: int, num_judged: int, seed: int, prompt_index: int) -> torch.Tensor:
    """
    Selects the indices of rollouts to judge, deterministically per prompt.

    Args:
        num_rollouts: Rollouts available for this prompt.
        num_judged (m): Number of rollouts to judge.
        seed: Base seed shared by the run.
        prompt_index: Zero-based prompt index, offsetting the seed so prompts draw
            independently while staying reproducible.

    Returns:
        Sorted tensor of shape [m] holding the chosen rollout indices.
    """
    if num_judged >= num_rollouts:
        return torch.arange(num_rollouts)
    generator = torch.Generator().manual_seed(seed + prompt_index) # local generator
    return torch.randperm(num_rollouts, generator=generator)[:num_judged].sort().values


if __name__ == "__main__":
    main()
