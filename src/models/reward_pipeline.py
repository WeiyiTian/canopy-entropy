from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from .loading import load_hf_model, load_tokenizer


class SkyworkRewardPipeline:
    """Score chat messages with Skywork Reward V2."""

    def __init__(
        self,
        model_path: Path | str,
        device: torch.device | str,
        batch_size: int = 8,
    ) -> None:
        """
        Loads the reward model and tokenizer.

        Args:
            model_path: Local model path for model and tokenizer loading.
            device: Device where the model should run.
            batch_size: Batch size for batched scoring.
        """
        self.device = torch.device(device)
        self.model = load_hf_model(
            AutoModelForSequenceClassification,
            model_path,
            device=self.device,
            num_labels=1,
        )
        self.tokenizer = load_tokenizer(model_path)
        self.batch_size = batch_size

    @torch.inference_mode()
    def score_batch(
        self,
        prompt: str,
        responses: Sequence[str],
    ) -> torch.Tensor:
        """
        Scores multiple prompt-response pairs for one prompt while preserving input order.
        Processes responses in batches sorted by input length for efficiency.

        Args:
            prompt: User prompt.
            responses: Sequence of M assistant responses.

        Returns:
            Tensor [M] of reward scores aligned with responses.
        """
        conversations = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for response in responses
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            padding=False,
            truncation=False,
        )
        sorted_indices = sorted(range(len(input_ids)), key=lambda idx: len(input_ids[idx]))
        sorted_indices_tensor = torch.tensor(sorted_indices, dtype=torch.long, device=self.device)

        scores = torch.empty(len(responses), dtype=torch.float32, device=self.device)
        for start in range(0, len(sorted_indices), self.batch_size):
            batch_indices = sorted_indices[start : start + self.batch_size]
            batch_inputs = self.tokenizer.pad(
                [{"input_ids": input_ids[idx]} for idx in batch_indices],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**batch_inputs).logits[:, 0]
            scores[sorted_indices_tensor[start : start + self.batch_size]] = logits

        return scores
