from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification

from .loading import load_hf_model, load_tokenizer


class SkyworkRewardPipeline:
    """Score chat messages with Skywork Reward V2."""

    def __init__(
        self,
        model_path: str,
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

    def _build_conversation(self, prompt: str, response: str) -> str:
        """Format one prompt-response pair with the reward model chat template."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        # Remove a potential duplicate BOS token inserted by the chat template.
        if self.tokenizer.bos_token and conversation.startswith(self.tokenizer.bos_token):
            conversation = conversation[len(self.tokenizer.bos_token):]
        return conversation

    @torch.inference_mode()
    def score_batch(
        self,
        prompt: str,
        responses: Sequence[str],
    ) -> list[float]:
        """
        Scores multiple prompt-response pairs while preserving input order.
        Processes responses in batches sorted by input length for efficiency.

        Args:
            prompt: User prompt.
            responses: Assistant responses.

        Returns:
            Reward scores aligned with `responses`.
        """
        conversations = [self._build_conversation(prompt, response) for response in responses]
        tokenized = self.tokenizer(
            conversations,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = tokenized["input_ids"]
        sorted_indices = sorted(range(len(input_ids)), key=lambda idx: len(input_ids[idx]))

        scores = [0.0] * len(responses)
        for start in range(0, len(sorted_indices), self.batch_size):
            batch_indices = sorted_indices[start : start + self.batch_size]
            batch_inputs = self.tokenizer.pad(
                [{"input_ids": input_ids[idx]} for idx in batch_indices],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**batch_inputs).logits[:, 0].tolist()
            for original_idx, score in zip(batch_indices, logits):
                scores[original_idx] = score

        return scores

    def __call__(
        self,
        prompt: str,
        response: str | Sequence[str],
    ) -> float | list[float]:
        """
        Scores one or more prompt-response pairs.

        Args:
            prompt: User prompt.
            response: Assistant response or responses.

        Returns:
            Scalar reward score for a single response, otherwise a list of scores.
        """
        if isinstance(response, str):
            return self.score_batch(prompt, [response])[0]
        return self.score_batch(prompt, response)
