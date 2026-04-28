from dataclasses import replace
from pathlib import Path

from src.constants import ROLLOUT_SHARDS_ARTIFACT

from .artifacts import count_prompt_shards
from ..core.structures import GenerationMetadata


def build_rollout_metadata_path(rollout_dir: Path) -> Path:
    """
    Returns the metadata file path for a rollout directory.

    Args:
        rollout_dir: Directory holding the rollout run's metadata and shard subdirectory.

    Returns:
        Path to `metadata.json` inside `rollout_dir`.
    """
    return rollout_dir / "metadata.json"


def resume_rollouts(rollout_dir: Path, requested_metadata: GenerationMetadata) -> int:
    """
    Checks that saved metadata matches the request, overwrites metadata for the current run,
    and returns shard count.

    Args:
        rollout_dir: Directory holding an existing rollout run's metadata and shard subdirectory.
        requested_metadata: New metadata requested for the resumed run.

    Returns:
        Number of already saved prompt shards. Raises ValueError if existing metadata differs from
        `requested_metadata` on any field other than `num_prompts` (excluded so the prompt set can
        be extended across resumes).
    """
    existing = GenerationMetadata.load(build_rollout_metadata_path(rollout_dir))
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
    requested_metadata.save(build_rollout_metadata_path(rollout_dir))
    return count_prompt_shards(rollout_dir, ROLLOUT_SHARDS_ARTIFACT)
