from dataclasses import replace
from pathlib import Path

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


def build_prompt_shards_dir(rollout_dir: Path) -> Path:
    """
    Returns the directory that stores per-prompt rollout shards.

    Args:
        rollout_dir: Directory holding the rollout run's metadata and shard subdirectory.

    Returns:
        Path to the `prompt_shards/` subdirectory of `rollout_dir`.
    """
    return rollout_dir / "prompt_shards"


def build_rollout_shard_path(rollout_dir: Path, prompt_index: int) -> Path:
    """
    Returns the shard path for one prompt index.

    Args:
        rollout_dir: Directory holding the rollout run's metadata and shard subdirectory.
        prompt_index: Zero-based index of the prompt.

    Returns:
        Path to the prompt's `.safetensors` shard, named with a zero-padded 6-digit index.
    """
    return build_prompt_shards_dir(rollout_dir) / f"{prompt_index:06d}.safetensors"


def count_existing_shards(rollout_dir: Path) -> int:
    """
    Counts saved rollout shard files in the rollout directory.

    Args:
        rollout_dir: Directory holding the rollout run's metadata and shard subdirectory.

    Returns:
        Number of `.safetensors` files in the prompt shards directory, or 0 if dir doesn't exist.
    """
    shards_dir = build_prompt_shards_dir(rollout_dir)
    if not shards_dir.exists():
        return 0
    return sum(1 for _ in shards_dir.glob("*.safetensors"))


def verify_rollouts_complete(rollout_dir: Path, metadata: GenerationMetadata) -> None:
    """
    Verifies that at least `metadata.num_prompts` shards exist on disk for the rollout run.

    Args:
        rollout_dir: Directory holding the rollout run's metadata and shard subdirectory.
        metadata: Loaded run metadata.

    Raises:
        ValueError: If fewer shards exist than `metadata.num_prompts`, indicating the
            generation run did not complete. Orphan shards past `metadata.num_prompts`
            from a prior larger run are allowed and ignored.
    """
    existing_shards = count_existing_shards(rollout_dir)
    if existing_shards < metadata.num_prompts:
        raise ValueError(
            f"Incomplete rollouts at {rollout_dir}: {existing_shards} shards on disk, "
            f"metadata.num_prompts={metadata.num_prompts}. "
            f"Run generate_rollouts.py with resume=True to finish."
        )


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
    return count_existing_shards(rollout_dir)


def reset_rollout_dir(rollout_dir: Path, requested_metadata: GenerationMetadata) -> None:
    """
    Creates the directory if it doesn't exist, removes old shards, and overwrites metadata
    for a new rollout run.

    Args:
        rollout_dir: Directory holding the rollout run's metadata and shard subdirectory.
        requested_metadata: Metadata overwritten to `metadata.json` in `rollout_dir`.
    """
    prompt_shards_dir = build_prompt_shards_dir(rollout_dir)
    prompt_shards_dir.mkdir(parents=True, exist_ok=True)
    for stale_shard in prompt_shards_dir.glob("*.safetensors"):
        stale_shard.unlink()
    requested_metadata.save(build_rollout_metadata_path(rollout_dir))
