import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase


@torch.inference_mode()
def rescore_eos_logprob_trajectories(
    prompt: str,
    generated_token_ids: list[torch.Tensor],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """
    Computes per-step EOS logprob trajectories for realized generations from one prompt.
    Each rollout is scored with one forward pass over the full realized prompt+generation 
    sequence, and the per-step EOS logprobs are then sliced from the model's next-token logits.

    Args:
        prompt: Prompt text shared by all generated rollouts.
        generated_token_ids: List of rollout token-id tensors. The i-th tensor has
            shape `[T_i]` and stores the generated token ids for rollout i.
        model: Causal language model used for rescoring.
        tokenizer: Tokenizer that provides the EOS and padding token ids.
        batch_size: Number of trajectories to score in one forward pass.

    Returns:
        List of length `M`, where the i-th tensor has shape `[T_i]`. Entry `t`
        (zero-based) equals `log P(EOS | prompt, y_<t)`, where `y_<t` denotes the
        first `t` generated tokens, and `y_<0` is the empty prefix.
    """
    device = model.device
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device=device)
    prompt_len = prompt_ids.numel()

    eos_token_id = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
    eos_token_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
    eos_token_ids_tensor = torch.tensor(eos_token_ids, dtype=torch.long, device=device)

    generated_token_ids = [
        token_ids.to(device=device, dtype=torch.long)
        for token_ids in generated_token_ids
    ]
    sorted_indices = sorted(
        range(len(generated_token_ids)),
        key=lambda idx: generated_token_ids[idx].numel(),
    )
    sequence_eos_logprobs = [None] * len(generated_token_ids)

    for start in range(0, len(sorted_indices), batch_size):
        batch_indices = sorted_indices[start : start + batch_size]
        batch_generated_token_ids = [generated_token_ids[idx] for idx in batch_indices]

        batch_full_sequences = [
            torch.cat((prompt_ids, token_ids), dim=0)
            for token_ids in batch_generated_token_ids
        ]
        batch_lengths = torch.tensor(
            [sequence.numel() for sequence in batch_full_sequences],
            dtype=torch.long,
            device=device,
        ) # [B]
        batch_generated_lengths = batch_lengths - prompt_len # [B]

        batch_input_ids = pad_sequence(
            batch_full_sequences,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ) # [B, S], where S = prompt_len + max(T_i) in batch
        attention_mask = (
            torch.arange(batch_input_ids.shape[1], device=device).unsqueeze(0)
            < batch_lengths.unsqueeze(1)
        ) #[1, S] < [B, 1] -> [B, S]

        # [B, S, V]
        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits
        batch_generation_logits = batch_logits[:, prompt_len - 1: -1, :] # [B, max(T_i), V]

        batch_generation_logprobs = torch.log_softmax(batch_generation_logits, dim=-1)
        batch_eos_logprobs = torch.logsumexp(
            # [B, max(T_i), num_eos_tokens]
            batch_generation_logprobs.index_select(dim=-1, index=eos_token_ids_tensor),
            dim=-1,
        ).cpu() # [B, max(T_i)]

        for batch_idx, (original_idx, generated_length) in enumerate(
            zip(batch_indices, batch_generated_lengths.tolist(), strict=True)
        ):
            sequence_eos_logprobs[original_idx] = batch_eos_logprobs[batch_idx, :generated_length]

    return sequence_eos_logprobs
