from pathlib import Path

from src.constants import MODEL_VARIANTS


def build_model_path(model_root: Path | str, model_name: str, model_variant: str) -> Path:
    """
    Constructs the path for a locally downloaded model variant.

    Args:
        model_root: Root directory containing locally available model weights.
        model_name: Model family key.
        model_variant: Model variant key, e.g., base or instruct.

    Returns:
        Path to the selected model variant.
    """
    return Path(model_root) / MODEL_VARIANTS[model_name][model_variant]


def build_artifact_path(
    root: Path | str,
    prompt_file: Path | str,
    model_name: str,
    model_variant: str,
    artifact_name: str,
) -> Path:
    """
    Constructs the path to an artifact based on the model and prompt category information,
    structured as `{root}/{model_name}/{prompt_stem}/{model_variant}/{artifact_name}`.

    Args:
        root: Output or results root directory.
        prompt_file: Prompt file path or name, whose stem is used.
        model_name: Model family directory name.
        model_variant: Variant directory name.
        artifact_name: Artifact file or directory name.

    Returns:
        Path to the requested artifact.
    """
    return Path(root) / model_name / Path(prompt_file).stem / model_variant / artifact_name
