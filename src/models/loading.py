import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from vllm import LLM


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
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
    model_path: str,
    device: torch.device | str,
    **kwargs,
) -> PreTrainedModel:
    """
    Loads a Hugging Face model, ensures padding is configured, moves it to device,
    and switches it to eval mode.

    Args:
        model_cls: Hugging Face model class with `from_pretrained`.
        model_path: Local model path for model loading.
        device: Target device where the model should run.
        **kwargs: Additional kwargs forwarded to `from_pretrained`.

    Returns:
        Initialized Hugging Face model in evaluation mode.
    """
    model = model_cls.from_pretrained(model_path, **kwargs)
    if getattr(model.config, "pad_token_id", None) is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    model.eval()

    return model


def load_local_model(model_path: str, device: torch.device | str) -> PreTrainedModel:
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
    model_path: str,
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
        model_path: Local model path for vLLM odel loading.
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
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_logprobs=max_logprobs,
        logprobs_mode=logprobs_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed
    )
    return llm
