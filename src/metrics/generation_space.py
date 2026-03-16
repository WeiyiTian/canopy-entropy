from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from ..models.inference import generate_step_scores
from .generation_tree import (
    calculate_prompt_controlled_diversity,
    calculate_rollout_summary,
    step_entropy_and_sequence_entropy,
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
        - tm_star_max: Scalar tensor that represents the expected cumulative branching uncertainty, i.e.,
            the mean of the sequence entropies `(1/M) * sum_i H^(i)_sum`.
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

    step_entropy, sequence_entropy = step_entropy_and_sequence_entropy(
        # Step scores are already normalized post-policy scores from generation.
        sequence_step_scores=sequence_step_scores,
    )

    summary = calculate_rollout_summary(sequence_entropy, sequence_lengths)

    return {
        "generated_texts": generated_texts,
        "step_conditional_entropy": step_entropy,
        "sequence_conditional_entropy": sequence_entropy,
        "sequence_lengths": sequence_lengths,
        **summary,
    }


@torch.inference_mode()
def aggregate_generation_space_results(
    prompt_results: list[dict[str, Any]],
) -> dict[str, object]:
    """
    Aggregates multiple `estimate_generation_space` outputs across prompts.

    Args:
        prompt_results: List of length P where each element is a result dict from
        `estimate_generation_space` for one prompt. 
        For prompt p with M_p rollouts:
            `generated_texts`: list of length M_p.
            `step_conditional_entropy`: list of length M_p; i-th tensor shape [T_(p,i)].
            `sequence_conditional_entropy`: tensor shape [M_p].
            `sequence_lengths`: tensor shape [M_p].
            `diversity_correlation`: scalar tensor for prompt p.
            `diversity_covariance`: scalar tensor for prompt p.

    Returns:
        dict: Dictionary with keys
        - generated_texts: List of all generated texts with total length `M_total = sum_p M_p`.
        - step_conditional_entropy: List of all per-rollout step-entropy tensors with
            total length `M_total`.
        - sequence_conditional_entropy: Tensor of shape `[M_total]` from prompt-wise concatenation.
        - sequence_lengths: Tensor of shape `[M_total]` from prompt-wise concatenation.
        - tm_star_max: Scalar tensor computed on pooled rollouts that represents the expected 
            cumulative branching uncertainty, i.e., `(1/M_total) * sum_i H^(i)_sum`.
        - gen_ppl: Scalar tensor computed on pooled rollouts `exp(tm_star_max / E[N])`.
        - branching_factor: Scalar tensor computed on pooled rollouts 
            `exp((1/M_total) * sum_i r^(i)_N)`, where `r^(i)_N = H^(i)_sum / N^(i)`.
        - diversity_correlation (within-prompt effects): Scalar tensor from prompt-controlled
            aggregation over {corr_p} Fisher-z weighted by `M_p - 3`.
        - diversity_covariance (within-prompt effects): Scalar tensor from prompt-controlled
            aggregation over {cov_p} weighted by `M_p - 1`.
        - diversity_correlation_pooled (between-prompt effects): Scalar tensor computed 
            on concatenated rollout pairs `(N^(i), r^(i)_N)` over p prompts.
        - diversity_covariance_pooled (between-prompt effects): Scalar tensor computed 
            on concatenated rollout pairs `(N^(i), r^(i)_N)` over p prompts.

    Notes:
        - The first four outputs (`generated_texts`, `step_conditional_entropy`,
            `sequence_conditional_entropy`, `sequence_lengths`) are direct
            concatenations across prompts with no additional transformation.
        - Entropy/length-driven scalars (`tm_star_max`, `gen_ppl`, `branching_factor`) are
            recomputed from pooled rollout tensors. 
        - Diversity is reported in both prompt-controlled and pooled forms to 
            separate within-prompt and between-prompt effects.
        - Law of total variance: Cov(N, r) = E_p[Cov(N,r | p)] + Cov_p(E[N|p], E[r|p])
            = within-prompt effects + between-prompt effects
    """

    generated_texts = [text for result in prompt_results for text in result["generated_texts"]]
    step_entropy = [entropy for result in prompt_results for entropy in result["step_conditional_entropy"]]
    
    sequence_entropy_chunks = [result["sequence_conditional_entropy"] for result in prompt_results]
    sequence_length_chunks = [result["sequence_lengths"] for result in prompt_results]
    prompt_correlations = [result["diversity_correlation"] for result in prompt_results]
    prompt_covariances = [result["diversity_covariance"] for result in prompt_results]
    prompt_sample_sizes = [lengths.numel() for lengths in sequence_length_chunks]

    sequence_entropy = torch.cat(sequence_entropy_chunks, dim=0)
    sequence_lengths = torch.cat(sequence_length_chunks, dim=0)
    pooled_summary = calculate_rollout_summary(sequence_entropy, sequence_lengths)
    prompt_controlled_diversity = calculate_prompt_controlled_diversity(
        correlations=prompt_correlations,
        covariances=prompt_covariances,
        sample_sizes=prompt_sample_sizes,
    )

    return {
        "generated_texts": generated_texts,
        "step_conditional_entropy": step_entropy,
        "sequence_conditional_entropy": sequence_entropy,
        "sequence_lengths": sequence_lengths,
        "tm_star_max": pooled_summary["tm_star_max"],
        "gen_ppl": pooled_summary["gen_ppl"],
        "branching_factor": pooled_summary["branching_factor"],
        "diversity_correlation": prompt_controlled_diversity["diversity_correlation"],
        "diversity_covariance": prompt_controlled_diversity["diversity_covariance"],
        "diversity_correlation_pooled": pooled_summary["diversity_correlation"],
        "diversity_covariance_pooled": pooled_summary["diversity_covariance"],
    }
