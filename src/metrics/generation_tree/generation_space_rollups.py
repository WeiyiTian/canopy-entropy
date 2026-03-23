from __future__ import annotations

from dataclasses import dataclass
from itertools import compress
from typing import Any, cast

import torch

from ...constants import LENGTH_BUCKET_NAMES

from .rollout_statistics import (
    calculate_prompt_controlled_diversity,
    calculate_rollout_summary,
)


@dataclass(slots=True)
class RawRollouts:
    """
    Unfiltered rollout data for one prompt.

    Attributes:
        generated_texts: All decoded rollout texts.
        sequence_step_scores: Token-level score tensors aligned with `generated_texts`.
        sequence_lengths: Generated token lengths aligned with `generated_texts`.
    """

    generated_texts: list[str]
    sequence_step_scores: list[torch.Tensor]
    sequence_lengths: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_texts": self.generated_texts,
            "sequence_step_scores": self.sequence_step_scores,
            "sequence_lengths": self.sequence_lengths,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawRollouts:
        return cls(
            generated_texts=data["generated_texts"],
            sequence_step_scores=data["sequence_step_scores"],
            sequence_lengths=data["sequence_lengths"],
        )


@dataclass(slots=True)
class FilteredRollouts:
    """
    Reward-filtered view over one prompt's raw rollouts.

    Attributes:
        raw_rollouts: Unfiltered rollout data.
        reward_scores: Reward scores aligned with the raw rollouts.
        kept_mask: Boolean mask over raw rollouts indicating retained samples.

    Views:
        raw_generated_texts: All decoded rollout texts before filtering.
        raw_sequence_step_scores: Token-level score tensors aligned with the raw rollouts.
        raw_sequence_lengths: Generated token lengths before filtering.
        kept_generated_texts: Retained rollout texts in original rollout order.
        kept_sequence_step_scores: Retained token-level score tensors in original rollout order.
        kept_sequence_lengths: Retained rollout lengths in original rollout order.
    """

    raw_rollouts: RawRollouts
    reward_scores: torch.Tensor
    kept_mask: torch.Tensor

    @property
    def raw_generated_texts(self) -> list[str]:
        return self.raw_rollouts.generated_texts

    @property
    def raw_sequence_step_scores(self) -> list[torch.Tensor]:
        return self.raw_rollouts.sequence_step_scores

    @property
    def raw_sequence_lengths(self) -> torch.Tensor:
        return self.raw_rollouts.sequence_lengths

    @property
    def kept_generated_texts(self) -> list[str]:
        return list(compress(self.raw_generated_texts, self.kept_mask.tolist()))

    @property
    def kept_sequence_step_scores(self) -> list[torch.Tensor]:
        return list(compress(self.raw_sequence_step_scores, self.kept_mask.tolist()))

    @property
    def kept_sequence_lengths(self) -> torch.Tensor:
        return self.raw_sequence_lengths[self.kept_mask]

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_rollouts": self.raw_rollouts.to_dict(),
            "reward_scores": self.reward_scores,
            "kept_mask": self.kept_mask,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilteredRollouts:
        return cls(
            raw_rollouts=RawRollouts.from_dict(data["raw_rollouts"]),
            reward_scores=data["reward_scores"],
            kept_mask=data["kept_mask"],
        )


@dataclass(slots=True)
class ResponseRecord:
    """
    One raw rollout response annotated with reward-filtering and entropy metadata.

    Attributes:
        generated_text: Decoded rollout text for this response.
        sequence_length: Generated token length for this response.
        reward_score: Reward score assigned during filtering.
        is_kept: Whether the response survived reward filtering.
        step_conditional_entropy: Per-step entropy tensor for kept responses.
        sequence_conditional_entropy: Total sequence entropy tensor for kept responses.
    """

    generated_text: str
    sequence_length: torch.Tensor
    reward_score: torch.Tensor
    is_kept: bool
    step_conditional_entropy: torch.Tensor | None
    sequence_conditional_entropy: torch.Tensor | None

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_text": self.generated_text,
            "sequence_length": self.sequence_length,
            "reward_score": self.reward_score,
            "is_kept": self.is_kept,
            "step_conditional_entropy": self.step_conditional_entropy,
            "sequence_conditional_entropy": self.sequence_conditional_entropy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseRecord:
        return cls(
            generated_text=data["generated_text"],
            sequence_length=data["sequence_length"],
            reward_score=data["reward_score"],
            is_kept=data["is_kept"],
            step_conditional_entropy=data["step_conditional_entropy"],
            sequence_conditional_entropy=data["sequence_conditional_entropy"],
        )

