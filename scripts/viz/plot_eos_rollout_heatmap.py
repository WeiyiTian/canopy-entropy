import argparse
import json
from pathlib import Path

import torch

from src.settings import settings
from src.utils import build_output_path
from src.visualization import plot_eos_rollout_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EOS rollout trajectories as a heatmap with active-count strip.",
    )
    parser.add_argument("--outputs-root", type=Path, default=Path(settings.outputs_dir))
    parser.add_argument("--results-root", type=Path, default=Path(settings.results_dir))
    parser.add_argument("--file-name", type=str, default="coding.jsonl")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-7b")
    parser.add_argument("--model-variant", type=str, default="instruct")
    parser.add_argument("--input-file", type=str, default="eos_logprob_trajectories.pt")
    parser.add_argument("--output-file", type=str, default="eos_logprob_heatmap.png")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = build_output_path(
        args.outputs_root,
        args.file_name,
        args.model_name,
        args.model_variant,
        args.input_file,
    )
    eos_data: dict[str, object] = torch.load(input_path, map_location="cpu", weights_only=False)
    metadata = eos_data["metadata"]
    prompt_eos_logprob_trajectories = eos_data["prompt_eos_logprob_trajectories"]

    output_path = build_output_path(
        args.results_root,
        str(metadata["prompt_file"]),
        args.model_name,
        args.model_variant,
        f"{args.group_mode}_{args.output_file}",
    )
    plot_eos_rollout_heatmap(
        prompt_eos_logprob_trajectories=prompt_eos_logprob_trajectories,
        output_path=output_path,
        title=f"EOS Logprob Heatmap ({args.model_variant}; {Path(args.file_name).stem})",
        max_positions=args.max_positions,
        group_mode=args.group_mode,
        sort_mode=args.sort_mode,
        label_fontsize=args.label_fontsize,
    )

    print(
        json.dumps(
            {
                "file_name": args.file_name,
                "model_variant": args.model_variant,
                "input_path": str(input_path),
                "output_path": str(output_path),
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
