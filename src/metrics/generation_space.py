from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from ..models.inference import generate_step_scores
from .core import (
    calculate_branching_factor, 
    calculate_diversity_correlation,
    expected_total_uncertainty_sequence_step_scores,
    calculate_gen_ppl
)


@torch.inference_mode()
def estimate_generation_space(
    prompt: str,
    model: AutoModelForCausalLM | LLM,
    n_samples: int,
    max_new_tokens: int,
    backend: Literal["local", "vllm"] = "vllm",
    tokenizer: AutoTokenizer | None = None,
    temperature: float = 1.0,
    top_k: int | None = 0,
    top_p: float | None = 1.0,
    seed: int | None = 42,
    logprobs: int = -1,
    sample_batch_size: int = 8,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """
    Estimate generation-space metrics from sampled rollouts for one prompt.

    Args:
        prompt: Single input prompt string used for sampled generations.
        model: Backend model instance. Use `AutoModelForCausalLM` with local backend
            and `LLM` with vllm backend.
        n_samples: Number of rollout samples (M) for the given prompt.
        max_new_tokens: Maximum number of generated tokens per rollout.
        backend: Generation backend, either local or vllm.
        tokenizer: Tokenizer required only for the local backend.
        temperature: Sampling temperature.
        top_k: Top-k truncation parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for sampling. For vLLM, seed is incremented by batch index.
        logprobs: Number of candidate-token scores retained per step.
        sample_batch_size: Batch size used only for vLLM sampling loops.
        device: Device override. Backend-specific default is used if not set.

    Returns:
        dict: Dictionary with keys
        - generated_texts: List of M decodede strings from rollout outputs.
        - step_conditional_entropy: List of length M. The i-th tensor has shape [T_i]
            with per-step entropies `H(Y^(i)_t | X, y^(i)_<t)`.
        - sequence_conditional_entropy: Tensor of shape [M], where i-th entry is
            `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        - sequence_lengths: Tensor of shape [M], where i-th entry is generated token length `N^(i)`.
        - tm_star_max: Scalar tensor that represents the expected total uncertainty, i.e.,
            the mean of the sequence entropies, i.e., `(1/M) * sum_i H^(i)_sum`.
        - gen_ppl: Scalar tensor `exp(tm_star_max / E[N])`.
        - branching_factor: Scalar tensor `exp((1/M) * sum_i r^(i)_N)`,
            where `r^(i)_N = H^(i)_sum / N^(i)`.
        - diversity_correlation: Scalar tensor of Pearson correlation over rollout pairs
            `(N^(i), r^(i)_N)`.
        - diversity_covariance: Scalar tensor of covariance over rollout pairs
            `(N^(i), r^(i)_N)`.
    """
    generated_texts, sequence_step_scores, sequence_lengths = generate_step_scores(
        prompt=prompt,
        model=model,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        backend=backend,
        tokenizer=tokenizer,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        logprobs=logprobs,
        sample_batch_size=sample_batch_size,
        device=device,
    )

    step_entropy, sequence_entropy, tm_star_max = expected_total_uncertainty_sequence_step_scores(
        # Step scores are already normalized post-policy scores from generation.
        sequence_step_scores=sequence_step_scores,
    )

    gen_ppl = calculate_gen_ppl(tm_star_max, sequence_lengths)
    branching_factor = calculate_branching_factor(sequence_entropy, sequence_lengths)
    diversity_correlation, diversity_covariance = calculate_diversity_correlation(
        sequence_entropy,
        sequence_lengths,
    )

    return {
        "generated_texts": generated_texts,
        "step_conditional_entropy": step_entropy,
        "sequence_conditional_entropy": sequence_entropy,
        "sequence_lengths": sequence_lengths,
        "tm_star_max": tm_star_max,
        "gen_ppl": gen_ppl,
        "branching_factor": branching_factor,
        "diversity_correlation": diversity_correlation,
        "diversity_covariance": diversity_covariance,
    }
