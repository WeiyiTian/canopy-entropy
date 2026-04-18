from __future__ import annotations
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.metrics import PromptMetrics


@dataclass(slots=True)
class PromptRollouts:
    """
    Container for a prompt and its M sampled rollouts, where all rollout-level
    fields (texts, tokens, lengths, entropies, and optional log-probabilities)
    are aligned by index.

    Attributes:
        prompt: Input prompt string shared by all rollouts.
        prompt_token_ids: Tensor for the tokenized prompt.
        generated_texts: List of M decoded rollout strings.
        generated_token_ids: List of M; tensor i has shape [T_i],
            containing the token IDs of the i-th generated sequence.
        sequence_lengths: Tensor of shape [M] with generated token lengths T_i.
        step_conditional_entropy: List of M; tensor i has shape [T_i] with
            per-step entropies `H(Y_t | X, y_<t)`.
        sequence_step_logprobs: Optional list of M; tensor i has shape
            [T_i, K_i_max], containing the normalized top-candidate log
            probabilities for each generated step.

    Notes:
        T_i: generated token length of rollout i.
        K_i_max: retained candidate log-probabilities per step for rollout i
            (padded with `-inf` to this width).
    """

    prompt: str
    prompt_token_ids: torch.Tensor
    generated_texts: list[str]
    generated_token_ids: list[torch.Tensor]
    sequence_lengths: torch.Tensor
    step_conditional_entropy: list[torch.Tensor]
    sequence_step_logprobs: list[torch.Tensor] | None = None

    def to_cpu(self) -> PromptRollouts:
        """Returns a new object with all tensors moved to CPU."""
        return PromptRollouts(
            prompt=self.prompt,
            prompt_token_ids=self.prompt_token_ids.cpu(),
            generated_texts=self.generated_texts,
            generated_token_ids=[t.cpu() for t in self.generated_token_ids],
            sequence_lengths=self.sequence_lengths.cpu(),
            step_conditional_entropy=[t.cpu() for t in self.step_conditional_entropy],
            sequence_step_logprobs=[t.cpu() for t in self.sequence_step_logprobs]
                if self.sequence_step_logprobs is not None
                else None,
        )

    def save(self, path: Path) -> None:
        """Saves rollout data to a safetensors file with metadata for prompt and texts."""
        self_cpu = self.to_cpu()
        meta = {
            "prompt": self_cpu.prompt,
            "generated_texts": json.dumps(self_cpu.generated_texts),
        }

        tensors: dict[str, torch.Tensor] = {
            "prompt_token_ids": self_cpu.prompt_token_ids,
            "sequence_lengths": self_cpu.sequence_lengths,
            "generated_token_ids_flat": torch.cat(self_cpu.generated_token_ids, dim=0),
            "step_conditional_entropy_flat": torch.cat(self_cpu.step_conditional_entropy, dim=0),
        }
        if self_cpu.sequence_step_logprobs is not None:
            for rollout_idx, step_logprobs in enumerate(self_cpu.sequence_step_logprobs):
                tensors[f"sequence_step_logprobs.{rollout_idx:05d}"] = step_logprobs
        
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata=meta)

    @classmethod
    def load(cls, path: Path, device: str | torch.device) -> PromptRollouts:
        """Loads rollout data from a safetensors file to the specified device."""
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            meta = f.metadata() or {}
            tensors = {key: f.get_tensor(key) for key in f.keys()}
        sequence_lengths = tensors["sequence_lengths"]
        num_rollouts = int(sequence_lengths.numel())
        has_logprobs = "sequence_step_logprobs.00000" in tensors

        return cls(
            prompt=meta["prompt"],
            prompt_token_ids=tensors["prompt_token_ids"],
            generated_texts=json.loads(meta["generated_texts"]),
            generated_token_ids=list(torch.split(
                tensors["generated_token_ids_flat"], sequence_lengths.tolist(), dim=0
            )),
            sequence_lengths=sequence_lengths,
            step_conditional_entropy=list(torch.split(
                tensors["step_conditional_entropy_flat"], sequence_lengths.tolist(), dim=0
            )),
            sequence_step_logprobs=(
                [tensors[f"sequence_step_logprobs.{i:05d}"] for i in range(num_rollouts)]
                if has_logprobs else None
            ),
        )

    @classmethod
    def load_texts(cls, path: Path) -> tuple[str, list[str]]:
        """Loads just the prompt and generated texts from the safetensors metadata."""
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
        return meta["prompt"], json.loads(meta["generated_texts"])


@dataclass(slots=True)
class PromptRolloutStats:
    """
    Generation space stats computed for a single prompt's rollouts.

    Attributes:
        prompt: Input prompt string shared by all rollouts.
        raw_sequence_lengths: Tensor of shape [M] with pre-filter lengths.
        reward_scores: Tensor of shape [M] aligned with raw rollout order.
        keep_mask: Boolean tensor of shape [M] marking retained rollouts after filter.
        raw_metrics: `PromptMetrics` containing metrics computed from all M raw rollouts.
        kept_sequence_lengths: Tensor of shape [M_kept] with post-filter lengths.
        kept_step_conditional_entropy: List of M_kept tensors; tensor i
            has shape [T_i] with per-step entropies `H(Y_t | X, y_<t)`.
        kept_sequence_conditional_entropy: Tensor of shape [M_kept] with the
            sum of step entropies per kept rollout, tensor i is
            `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        kept_metrics: `PromptMetrics` containing metrics computed from the M_kept 
            retained rollouts.

    Notes:
        M: number of raw sampled rollouts for the prompt.
        M_kept: number of rollouts retained after reward filtering.
        T_i: generated token length of the i-th kept rollout.
    """

    prompt: str
    raw_sequence_lengths: torch.Tensor
    reward_scores: torch.Tensor
    keep_mask: torch.Tensor
    raw_metrics: PromptMetrics
    kept_sequence_lengths: torch.Tensor
    kept_step_conditional_entropy: list[torch.Tensor]
    kept_sequence_conditional_entropy: torch.Tensor
    kept_metrics: PromptMetrics

    def to_cpu(self) -> PromptRolloutStats:
        """Returns a new object with all tensors moved to CPU."""
        return PromptRolloutStats(
            prompt=self.prompt,
            raw_sequence_lengths=self.raw_sequence_lengths.cpu(),
            reward_scores=self.reward_scores.cpu(),
            keep_mask=self.keep_mask.cpu(),
            raw_metrics=self.raw_metrics.to_cpu(),
            kept_sequence_lengths=self.kept_sequence_lengths.cpu(),
            kept_step_conditional_entropy=[
                entropy.cpu() for entropy in self.kept_step_conditional_entropy
            ],
            kept_sequence_conditional_entropy=self.kept_sequence_conditional_entropy.cpu(),
            kept_metrics=self.kept_metrics.to_cpu(),
        )


@dataclass(slots=True)
class GenerationMetadata:
    """
    Serializable run metadata describing the prompts, model, and sampling settings.

    Attributes:
        prompt_file: Path to the prompt file used for generation.
        num_prompts_processed: Number of input prompts for which rollouts were generated.
        model_name: Generation model family.
        model_variant: Model variant identifier.
        n_samples: Number of rollouts sampled per prompt.
        max_new_tokens: Maximum generated tokens per rollout.
        temperature: Sampling temperature.
        top_k: Top-k truncation value.
        top_p: Nucleus sampling probability.
        seed: Seed passed to the generation backend.
        logprobs: Number of logprobs retained per decoding step.
        enable_thinking: Whether rollouts were generated with reasoning enabled.
    """

    prompt_file: str
    num_prompts_processed: int
    model_name: str
    model_variant: str
    n_samples: int
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    logprobs: int
    enable_thinking: bool

    def to_dict(self) -> dict[str, object]:
        """Serializes metadata fields to a flat dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GenerationMetadata:
        """Reconstructs metadata from a dictionary."""
        return cls(**{field.name: payload[field.name] for field in fields(cls)})

    def save(self, path: Path) -> None:
        """Saves metadata as a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: Path) -> GenerationMetadata:
        """Loads metadata from a JSON file."""
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
