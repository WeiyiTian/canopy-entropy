from dataclasses import replace
from pathlib import Path

from src.constants import ROLLOUT_SHARDS_ARTIFACT

from .artifacts import build_metadata_path, build_prompt_shard_dir, count_prompt_shards
from ..core.structures import GenerationMetadata, JudgeMetadata


def resume_rollouts(run_dir: Path, requested_metadata: GenerationMetadata) -> int:
    """
    Checks that saved metadata matches the request, overwrites metadata for the current run,
    and returns shard count.

    Args:
        run_dir: Directory holding an existing rollout run's metadata and shard subdirectory.
        requested_metadata: New metadata requested for the resumed run.

    Returns:
        Number of already saved prompt shards. Raises ValueError if existing metadata differs from
        `requested_metadata` on any field other than `num_prompts` (excluded so the prompt set can
        be extended across resumes).
    """
    existing = GenerationMetadata.load(build_metadata_path(run_dir))
    if replace(existing, num_prompts=0) != replace(requested_metadata, num_prompts=0):
        raise ValueError(
            "Resume aborted: requested args do not match existing rollout metadata.\n"
            f"existing:  {existing}\nrequested: {requested_metadata}"
        )
    if requested_metadata.num_prompts != existing.num_prompts:
        print(
            f"num_prompts changed: {existing.num_prompts} -> {requested_metadata.num_prompts}. "
            f"Overwriting metadata."
        )
    requested_metadata.save(build_metadata_path(run_dir))
    return count_prompt_shards(run_dir, ROLLOUT_SHARDS_ARTIFACT)


def resume_judgments(run_dir: Path, artifact: str, requested_metadata: JudgeMetadata) -> int:
    """
    Checks that saved judging metadata matches the request, writes the metadata for the
    current run, and returns shard count.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Judgment artifact name including the judge label.
        requested_metadata: Judge configuration requested for the resumed run.

    Returns:
        Number of already saved prompt shards. Raises ValueError if any shard exists that
        was judged under different settings.
    """
    existing_shards = count_prompt_shards(run_dir, artifact)
    metadata_path = build_metadata_path(build_prompt_shard_dir(run_dir, artifact))
    if existing_shards > 0:
        existing = JudgeMetadata.load(metadata_path)
        if existing != requested_metadata:
            raise ValueError(
                f"Resume aborted: {existing_shards} shards under '{artifact}' were judged "
                f"with different settings.\n"
                f"existing:  {existing}\nrequested: {requested_metadata}\n"
                f"Use a different judge name, or set resume=false to discard them."
            )
    requested_metadata.save(metadata_path)
    return existing_shards
