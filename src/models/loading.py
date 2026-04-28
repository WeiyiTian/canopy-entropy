from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM

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


def load_generation_backend(
    model_path: Path | str,
    backend: Literal["local", "vllm"],
    device: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_logprobs: int,
    gpu_memory_utilization: float,
    seed: int,
) -> tuple[PreTrainedModel | LLM, PreTrainedTokenizerBase | None]:
    """
    Loads the configured generation backend and any tokenizer it requires.

    Args:
        model_path: Local model path used to initialize the generation backend.
        backend: Generation backend, either local or vllm.
        device: Target device where the model should run.
        tensor_parallel_size: Number of GPUs to shard model weights across for vLLM.
        max_model_len: Maximum supported context length (prompt + output) for vLLM.
        max_logprobs: Maximum number of token logprobs to retain per token for vLLM.
        gpu_memory_utilization: Fraction of each GPU memory budget available to vLLM.
        seed: Random seed used by vLLM sampling.

    Returns:
        (model, tokenizer): the loaded model and tokenizer.
    """
    if backend == "local":
        tokenizer = load_tokenizer(model_path)
        model = load_local_model(model_path, device)
        return model, tokenizer
    
    elif backend == "vllm":
        tokenizer = load_tokenizer(model_path)
        model = load_vllm_model(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_logprobs=max_logprobs,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        return model, tokenizer

    raise ValueError("backend must be one of {'local', 'vllm'}")


def load_tokenizer(model_path: Path | str) -> PreTrainedTokenizerBase:
    """
    Loads a tokenizer and ensure a valid padding token is configured.

    Args:
        model_path: Local model path for tokenizer loading.

    Returns:
        Initialized tokenizer with `pad_token_id` set. If padding is missing,
        `EOS` is reused as the pad token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_hf_model(
    model_cls: type[PreTrainedModel],
    model_path: Path | str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> PreTrainedModel:
    """
    Loads a Hugging Face model, moves it to device, and switches it to eval mode.

    Args:
        model_cls: Hugging Face model class with `from_pretrained`.
        model_path: Local model path for model loading.
        device: Target device where the model should run.
        dtype: Weight dtype for inference. Defaults to bf16.
        **kwargs: Additional kwargs forwarded to `from_pretrained`.

    Returns:
        Initialized Hugging Face model in evaluation mode.
    """
    model = model_cls.from_pretrained(model_path, dtype=dtype, **kwargs)
    model.to(device)
    model.eval()

    return model


def load_local_model(model_path: Path | str, device: torch.device | str) -> PreTrainedModel:
    """
    Loads a local causal language model, moves it to device, and switches to eval mode.

    Args:
        model_path: Local model path for model loading.
        device: Target device where the model should run.

    Returns:
        Initialized causal language model in evaluation mode.
    """
    return load_hf_model(AutoModelForCausalLM, model_path, device=device)


def load_vllm_model(
    model_path: Path | str,
    tensor_parallel_size: int,
    max_model_len: int = 8192,
    max_logprobs: int = 100,
    logprobs_mode: str = "processed_logprobs",
    gpu_memory_utilization: float = 0.9,
    seed: int = 42,
) -> LLM:
    """
    Initializes a vLLM engine for autoregressive generation.

    Args:
        model_path: Local model path for vLLM model loading.
        tensor_parallel_size: Number of GPUs to shard model weights across.
        max_model_len: Maximum supported context length (prompt + output).
        max_logprobs: Maximum number of token logprobs to retain per token.
        logprobs_mode: Logprob output mode (for example, "processed_logprobs").
        gpu_memory_utilization: Fraction of each GPU memory budget available to vLLM.
        seed: Random seed used by vLLM sampling.

    Returns:
        Initialized `LLM` instance.
    """
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_logprobs=max_logprobs,
        logprobs_mode=logprobs_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=seed,
    )
    return llm
