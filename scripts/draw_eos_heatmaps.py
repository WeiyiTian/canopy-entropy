import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.generation_space import (
    GenerationMetadata,
    iter_rollout_records,
    rollout_metadata_path,
)
from src.utils.paths import build_artifact_path, build_model_path
from src.utils.torch_ops import clear_runtime_memory
from src.models import load_local_model, load_tokenizer, score_eos_trajectories
from src.settings import settings
from src.visualization import (
    plot_eos_rollout_heatmap,
    plot_eos_topk_membership_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute, save, load, and plot EOS rollout heatmaps.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="compute_plot",
        choices=("compute_plot", "plot_only", "compute_only"),
    )
    parser.add_argument("--model-root", type=Path, default=Path(settings.model_dir))
    parser.add_argument("--outputs-root", type=Path, default=Path(settings.outputs_dir))
    parser.add_argument("--results-root", type=Path, default=Path(settings.results_dir))
    parser.add_argument("--file-name", type=str, default="coding.jsonl")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-7b")
    parser.add_argument("--model-variant", type=str, default="instruct")
    parser.add_argument("--rollout-file", type=str, default="generation_rollouts")
    parser.add_argument("--save-file", type=str, default="eos_heatmap_trajectories.pt")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--model-device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--eos-batch-size", type=int, default=4)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument(
        "--group-mode",
        type=str,
        default="combined",
        choices=("combined", "prompt_blocks"),
    )
    parser.add_argument(
        "--sort-mode",
        type=str,
        default="length",
        choices=("length", "none"),
    )
    parser.add_argument("--label-fontsize", type=float, default=12.0)
    parser.add_argument("--logprob-output-file", type=str, default="eos_logprob_heatmap.png")
    parser.add_argument(
        "--topk-output-file",
        type=str,
        default="eos_topk_membership_heatmap.png",
    )
    return parser.parse_args()


def _compute_eos_payload(args: argparse.Namespace) -> dict[str, object]:
    """
    Loads saved rollouts, rescoring them with the HF model, and builds the EOS payload.
    """
    rollout_dir = build_artifact_path(
        args.outputs_root,
        args.file_name,
        args.model_name,
        args.model_variant,
        args.rollout_file,
    )
    metadata = GenerationMetadata.load(rollout_metadata_path(rollout_dir))
    prompt_count = metadata.num_prompts_processed

    model_path = build_model_path(
        args.model_root,
        metadata.model_name,
        metadata.model_variant,
    )
    tokenizer = load_tokenizer(model_path)
    model = load_local_model(model_path, args.model_device)

    prompt_eos_logprob_trajectories = []
    prompt_eos_topk_membership_trajectories = []
    records = iter_rollout_records(rollout_dir)
    for _, rollout in tqdm(
        records,
        total=prompt_count,
        desc="Scoring EOS trajectories",
        dynamic_ncols=True,
    ):
        scored = score_eos_trajectories(
            prompt_token_ids=rollout.prompt_token_ids,
            generated_token_ids=rollout.generated_token_ids,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.eos_batch_size,
            top_k=args.top_k,
        )
        prompt_eos_logprob_trajectories.append(scored.eos_logprobs)
        prompt_eos_topk_membership_trajectories.append(scored.eos_in_topk)

    del model, tokenizer
    clear_runtime_memory()

    return {
        "metadata": metadata.to_dict(),
        "rollout_path": str(rollout_dir),
        "top_k": args.top_k,
        "prompt_eos_logprob_trajectories": prompt_eos_logprob_trajectories,
        "prompt_eos_topk_membership_trajectories": prompt_eos_topk_membership_trajectories,
    }


def _plot_eos_payload(
    args: argparse.Namespace,
    eos_payload: dict[str, object],
) -> tuple[Path, Path]:
    """
    Plots both EOS heatmaps from a precomputed payload and returns their paths.
    """
    metadata = eos_payload["metadata"]
    prompt_file = str(metadata["prompt_file"])
    prompt_eos_logprob_trajectories = eos_payload["prompt_eos_logprob_trajectories"]
    prompt_eos_topk_membership_trajectories = eos_payload["prompt_eos_topk_membership_trajectories"]
    top_k = int(eos_payload["top_k"])

    logprob_output_path = build_artifact_path(
        args.results_root,
        prompt_file,
        args.model_name,
        args.model_variant,
        f"{args.group_mode}_{args.logprob_output_file}",
    )
    topk_output_path = build_artifact_path(
        args.results_root,
        prompt_file,
        args.model_name,
        args.model_variant,
        f"{args.group_mode}_{args.topk_output_file}",
    )

    plot_eos_rollout_heatmap(
        prompt_eos_logprob_trajectories=prompt_eos_logprob_trajectories,
        output_path=logprob_output_path,
        title=f"EOS Logprob Heatmap ({args.model_variant}; {Path(args.file_name).stem})",
        max_positions=args.max_positions,
        group_mode=args.group_mode,
        sort_mode=args.sort_mode,
        label_fontsize=args.label_fontsize,
    )
    plot_eos_topk_membership_heatmap(
        prompt_eos_topk_membership_trajectories=prompt_eos_topk_membership_trajectories,
        output_path=topk_output_path,
        title=f"EOS Top-{top_k} Membership Heatmap ({args.model_variant}; {Path(args.file_name).stem})",
        max_positions=args.max_positions,
        group_mode=args.group_mode,
        sort_mode=args.sort_mode,
        label_fontsize=args.label_fontsize,
    )

    return logprob_output_path, topk_output_path


def main() -> None:
    args = parse_args()
    save_path = build_artifact_path(
        args.outputs_root,
        args.file_name,
        args.model_name,
        args.model_variant,
        args.save_file,
    )

    if args.mode == "plot_only":
        eos_payload: dict[str, object] = torch.load(
            save_path,
            map_location="cpu",
            weights_only=False,
        )
    else:
        eos_payload = _compute_eos_payload(args)
        if not args.no_save:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(eos_payload, save_path)

    logprob_output_path = None
    topk_output_path = None
    if args.mode != "compute_only":
        logprob_output_path, topk_output_path = _plot_eos_payload(args, eos_payload)

    prompt_eos_logprob_trajectories = eos_payload["prompt_eos_logprob_trajectories"]


    print(
        json.dumps(
            {
                "mode": args.mode,
                "file_name": args.file_name,
                "model_variant": args.model_variant,
                "save_path": str(save_path),
                "logprob_output_path": None if not logprob_output_path else str(logprob_output_path),
                "topk_output_path": None if not topk_output_path else str(topk_output_path),
                "num_prompts": len(prompt_eos_logprob_trajectories),
                "num_rollouts": sum(
                    len(prompt_rollouts) for prompt_rollouts in prompt_eos_logprob_trajectories
                ),
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()
