import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from vllm import LLM, SamplingParams


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_local_model(model_path: str, device: torch.device | str) -> PreTrainedModel:
    """Loads a causal, moves it to device, and sets it to eval mode."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if getattr(model.config, "pad_token_id", None) is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    model.eval()

    return model


def load_vllm_model(
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int = 8192,
    max_logprobs: int = 100,
    logprobs_mode: str = "processed_logprobs",
    gpu_memory_utilization: float = 0.9,
    seed: int = 42,
) -> LLM:
    """Initialize and return a vLLM LLM instance.

    Args:
        tensor_parallel_size: Number of GPUs to shard model weights across.
        max_model_len: Maximum supported context length (prompt + output).
        max_logprobs: Maximum number of token logprobs to retain per token.
        logprobs_mode: "processed_logprobs" would return logprobs after processing.
        gpu_memory_utilization: Fraction of each GPU memory budget available to vLLM.
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
