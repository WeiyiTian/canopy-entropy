import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_local_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if getattr(model.config, "pad_token_id", None) is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    model.eval()

    return model


def load_vllm_model(
    model_path,
    tensor_parallel_size,
    max_model_len=3000,
    max_logprobs=100,
    gpu_memory_utilization=0.9,
    seed=42,
):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_logprobs=max_logprobs,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed
    )
    return llm
