import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from ..core.structures import PromptRolloutStats


def save_prompt_stats(
    prompt_results: list[PromptRolloutStats],
    output_path: Path,
) -> None:
    """
    Saves the per-prompt rollout stats list as a `.pt` file.

    Args:
        prompt_results: List of per-prompt rollout stats `PromptRolloutStats`.
        output_path: Destination `.pt` file, whose parent dir will be created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_results, output_path)


def load_prompt_stats(
    output_path: Path,
    device: str | torch.device = "cpu",
) -> list[PromptRolloutStats]:
    """
    Loads the saved per-prompt rollout stats.

    Args:
        output_path: Path to the saved `.pt` file.
        device: Device used to load any tensors in the saved stats.

    Returns:
        List of per-prompt rollout stats `PromptRolloutStats`.
    """
    return torch.load(output_path, map_location=str(device), weights_only=False)


def flatten_pooled_scalars(pooled_metrics: dict) -> dict[str, float]:
    """
    Flattens the raw metrics in a pooled metrics json dictionary.
    Nested dicts are flattened one level with `parent.child` keys.

    Args:
        pooled_metrics: Dictionary of json parsed pooled metric contents.

    Returns:
        Mapping from metric names to scalar float values.
    """
    raw = pooled_metrics["raw"]
    out: dict[str, float] = {}
    for key, val in raw.items():
        if isinstance(val, (int, float)):
            out[key] = float(val)
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                out[f"{key}.{sub_key}"] = float(sub_val)
    return out


def save_pooled_metrics(
    pooled_metrics: dict[str, object],
    output_path: Path,
) -> None:
    """
    Saves the cross-prompt pooled metric summary as JSON.

    Args:
        pooled_metrics: Dictionary of pooled metric results, which will be
            converted to Python primitives before serialization.
        output_path: Destination `.json` file, whose parent dir will be created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_tensors_to_python(pooled_metrics), indent=4))


def _tensors_to_python(obj: object) -> object:
    """
    Recursively converts tensors and dataclasses into JSON-serializable Python values.

    Args:
        obj: Value to convert.

    Returns:
        JSON-serializable copy of `obj`, with tensors converted to scalars or lists.
    """
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif is_dataclass(obj):
        return _tensors_to_python(asdict(obj))
    elif isinstance(obj, dict):
        return {key: _tensors_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_tensors_to_python(value) for value in obj]
    return obj
