from pathlib import Path

import torch
from safetensors import safe_open


def build_run_dir(
    root: Path | str,
    prompt_file: Path | str,
    model_name: str,
    model_variant: str,
    run_name: str,
) -> Path:
    """
    Constructs the run directory for one (model, dataset, variant, run_name) slot,
    structured as `{root}/{model_name}/{prompt_stem}/{model_variant}/{run_name}`.

    Args:
        root: Output or results root directory.
        prompt_file: Prompt file path or name, whose stem is used.
        model_name: Model family directory name.
        model_variant: Variant directory name.
        run_name: Run slot name within the (model, dataset, variant) scope.

    Returns:
        Path to the run directory, where per-run artifacts are placed.
    """
    return Path(root) / model_name / Path(prompt_file).stem / model_variant / run_name


def build_prompt_shard_dir(run_dir: Path, artifact: str) -> Path:
    """
    Returns the subdirectory holding per-prompt shards of a given artifact under a run directory.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Artifact name.

    Returns:
        Path to dir `{run_dir}/{artifact}`.
    """
    return Path(run_dir) / artifact


def build_prompt_shard_path(run_dir: Path, artifact: str, prompt_index: int) -> Path:
    """
    Returns the safetensors shard path for one prompt's entry in a per-prompt artifact.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Artifact name.
        prompt_index: Zero-based prompt index.

    Returns:
        Path to the prompt's `.safetensors` shard, named with a zero-padded 6-digit index.
    """
    return build_prompt_shard_dir(run_dir, artifact) / f"{prompt_index:06d}.safetensors"


def count_prompt_shards(run_dir: Path, artifact: str) -> int:
    """
    Counts existing per-prompt shard files for one artifact under a run directory.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Artifact name.

    Returns:
        Number of `.safetensors` shard files in the artifact subdirectory, or 0 if it doesn't exist.
    """
    shards_dir = build_prompt_shard_dir(run_dir, artifact)
    if not shards_dir.exists():
        return 0
    return sum(1 for _ in shards_dir.glob("*.safetensors"))


def reset_prompt_shards(run_dir: Path, artifact: str) -> None:
    """
    Creates the artifact subdirectory if missing and removes any existing shard files in it.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Artifact name.
    """
    shards_dir = build_prompt_shard_dir(run_dir, artifact)
    shards_dir.mkdir(parents=True, exist_ok=True)
    for stale_shard in shards_dir.glob("*.safetensors"):
        stale_shard.unlink()


def verify_prompt_shards_complete(run_dir: Path, artifact: str, expected_count: int) -> None:
    """
    Verifies that at least `expected_count` shards exist for the artifact on disk.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Artifact name.
        expected_count: Minimum number of shards expected on disk.

    Raises:
        ValueError: If fewer shards exist than `expected_count`. Orphan shards past
            `expected_count` from a prior larger run are allowed and ignored.
    """
    existing_shards = count_prompt_shards(run_dir, artifact)
    if existing_shards < expected_count:
        raise ValueError(
            f"Incomplete shards for artifact '{artifact}' at {run_dir}: "
            f"{existing_shards} shards on disk, expected {expected_count}."
        )


def load_prompt_shard_tensor(
    run_dir: Path,
    artifact: str,
    prompt_index: int,
    key: str,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Loads a tensor from one prompt's per-prompt safetensors shard.

    Args:
        run_dir: Directory holding one pipeline run's artifacts.
        artifact: Artifact name.
        prompt_index: Zero-based prompt index.
        key: Tensor key inside the safetensors shard.
        device: Device to load the tensor onto.

    Returns:
        Tensor stored under `key` in the shard.
    """
    path = build_prompt_shard_path(run_dir, artifact, prompt_index)
    with safe_open(str(path), framework="pt", device=str(device)) as f:
        return f.get_tensor(key)
