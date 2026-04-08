from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors


def plot_eos_rollout_heatmap(
    prompt_eos_logprob_trajectories: Sequence[Sequence[torch.Tensor]],
    output_path: Path,
    title: str = "EOS Logprob Heatmap",
    max_positions: int | None = None,
    group_mode: str = "combined",
    sort_mode: str = "length",
    label_fontsize: float = 12.0,
    color_quantiles: tuple[float, float] = (0.02, 0.98),
    dpi: int = 150,
) -> None:
    """
    Plots all EOS rollout trajectories as a heatmap with an aligned active-count strip.

    Args:
        prompt_eos_logprob_trajectories: Nested prompt -> rollout -> trajectory structure.
        output_path: Image output path.
        title: Figure title.
        max_positions: Optional cap on plotted token positions.
        group_mode: `combined` for one global block, or `prompt_blocks` to
            keep prompts separated along the y-axis.
        sort_mode: `length` to sort trajectories by increasing length within
            each block, or `none` to preserve the input order.
        label_fontsize: Font size used for axis labels and tick labels.
        color_quantiles:Clips the colormap range to the specified lower and upper quantiles 
            of the data to reduce outlier influence.
        dpi: Output image DPI.
    """
    heatmap_data = build_eos_heatmap_data(
        prompt_eos_logprob_trajectories=prompt_eos_logprob_trajectories,
        max_positions=max_positions,
        group_mode=group_mode,
        sort_mode=sort_mode,
    )
    matrix = heatmap_data["matrix"] # [num_total_rollouts, K]

    finite_values = matrix[np.isfinite(matrix)]
    lower_quantile, upper_quantile = color_quantiles
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#F2F2F2")

    _plot_eos_heatmap(
        heatmap_data=heatmap_data,
        output_path=output_path,
        title=title,
        group_mode=group_mode,
        label_fontsize=label_fontsize,
        dpi=dpi,
        cmap=cmap,
        norm=colors.Normalize(
            vmin=float(np.quantile(finite_values, lower_quantile)),
            vmax=float(np.quantile(finite_values, upper_quantile)),
        ),
        colorbar_label="EOS Logprob",
    )


def plot_eos_topk_membership_heatmap(
    prompt_eos_topk_membership_trajectories: Sequence[Sequence[torch.Tensor]],
    output_path: Path,
    title: str = "EOS Top-k Membership Heatmap",
    max_positions: int | None = None,
    group_mode: str = "combined",
    sort_mode: str = "length",
    label_fontsize: float = 12.0,
    dpi: int = 150,
    colors_present: tuple[str, str] = ("#D9D9D9", "#1F77B4"),
) -> None:
    """
    Plots per-step EOS top-k membership as a binary heatmap with an active-count strip.

    Args:
        prompt_eos_topk_membership_trajectories: Nested prompt -> rollout -> trajectory
            structure. Each innermost tensor is bool-like and has shape [T_i].
        output_path: Image output path.
        title: Figure title.
        max_positions: Optional cap on plotted token positions.
        group_mode: `combined` for one global block, or `prompt_blocks` to
            keep prompts separated along the y-axis.
        sort_mode: `length` to sort trajectories by increasing length within
            each block, or `none` to preserve the input order.
        label_fontsize: Font size used for axis labels and tick labels.
        dpi: Output image DPI.
        colors_present: Tuple `(absent_color, present_color)` for the binary map.
    """
    heatmap_data = build_eos_heatmap_data(
        prompt_eos_logprob_trajectories=prompt_eos_topk_membership_trajectories,
        max_positions=max_positions,
        group_mode=group_mode,
        sort_mode=sort_mode,
    )
    cmap = colors.ListedColormap([colors_present[0], colors_present[1]])
    cmap.set_bad(color="#F2F2F2")

    _plot_eos_heatmap(
        heatmap_data=heatmap_data,
        output_path=output_path,
        title=title,
        group_mode=group_mode,
        label_fontsize=label_fontsize,
        dpi=dpi,
        cmap=cmap,
        norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N),
        colorbar_label="EOS in Top-k",
        colorbar_ticks=[0, 1],
        colorbar_ticklabels=["Absent", "Present"],
    )


def _plot_eos_heatmap(
    heatmap_data: dict[str, np.ndarray | list[int] | list[float] | list[str]],
    output_path: Path,
    title: str,
    group_mode: str,
    label_fontsize: float,
    dpi: int,
    *,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    colorbar_label: str,
    colorbar_ticks: Sequence[float] | None = None,
    colorbar_ticklabels: Sequence[str] | None = None,
) -> None:
    """
    Renders the shared EOS heatmap layout and writes the figure to disk.

    Args:
        heatmap_data: Output from `build_eos_heatmap_data`, including the dense
            heatmap matrix, per-position active counts, and optional prompt-block
            metadata for y-axis grouping.
        output_path: Image output path.
        title: Figure title.
        group_mode: `combined` for one global rollout block, or `prompt_blocks`
            to draw prompt separators and prompt-level y-axis labels.
        label_fontsize: Font size used for axis labels and tick labels.
        dpi: Output image DPI.
        cmap: Colormap used for the heatmap body.
        norm: Mapping from data to color indices.
        colorbar_label: Colorbar label.
        colorbar_ticks: Optional explicit tick locations for the colorbar.
        colorbar_ticklabels: Optional colorbar tick labels.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = heatmap_data["matrix"] # [num_total_rollouts, K]
    active_counts = heatmap_data["active_counts"] # [K]
    prompt_boundaries = heatmap_data["prompt_boundaries"]
    prompt_centers = heatmap_data["prompt_centers"]
    prompt_labels = heatmap_data["prompt_labels"]

    rc_params = {
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": label_fontsize,
        "ytick.labelsize": label_fontsize,
        "figure.titlesize": label_fontsize + 2,
    }

    with plt.rc_context(rc_params):
        fig = plt.figure(figsize=(16, 12))
        grid = fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=(1.0, 7.0),
            width_ratios=(40.0, 2.0),
            hspace=0.05,
            wspace=0.08,
        )
        counts_ax = fig.add_subplot(grid[0, 0])
        heatmap_ax = fig.add_subplot(grid[1, 0], sharex=counts_ax)
        colorbar_ax = fig.add_subplot(grid[1, 1])

        image = heatmap_ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper", # y goes downwards
            extent=(0.5, matrix.shape[1] + 0.5, matrix.shape[0] - 0.5, -0.5), # (xmin, xmax, ymin, ymax)
            cmap=cmap,
            norm=norm,
        )

        positions = np.arange(1, active_counts.shape[0] + 1)
        counts_ax.fill_between(positions, active_counts, step="mid", alpha=0.25, color="#4C78A8")
        counts_ax.plot(positions, active_counts, linewidth=1.75, color="#4C78A8")

        if group_mode == "prompt_blocks":
            for boundary in prompt_boundaries[:-1]:
                heatmap_ax.axhline(boundary - 0.5, color="white", linewidth=0.8, alpha=0.85)
            heatmap_ax.set_yticks(prompt_centers)
            heatmap_ax.set_yticklabels(prompt_labels)
            heatmap_ax.set_ylabel("Prompt Block")
        elif group_mode == "combined":
            heatmap_ax.set_ylabel("Rollout")

        colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=colorbar_ticks)
        if colorbar_ticklabels is not None:
            colorbar.ax.set_yticklabels(colorbar_ticklabels)
        colorbar.set_label(colorbar_label)

        counts_ax.set_ylabel("Active")
        counts_ax.grid(axis="y", alpha=0.2) # only horizontal grid lines
        counts_ax.tick_params(axis="x", labelbottom=False)
        counts_ax.spines["top"].set_visible(False)
        counts_ax.spines["right"].set_visible(False)

        heatmap_ax.set_xlabel("Generated Token Position")
        fig.suptitle(title, y=0.9)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_eos_heatmap_data(
    prompt_eos_logprob_trajectories: Sequence[Sequence[torch.Tensor]],
    max_positions: int | None = None,
    group_mode: str = "combined",
    sort_mode: str = "length",
) -> dict[str, np.ndarray | list[int] | list[float] | list[str]]:
    """
    Converts nested variable-length EOS trajectories into a dense heatmap matrix.

    Args:
        prompt_eos_logprob_trajectories: Nested prompt-major structure with shape
            `[num_prompts][num_rollouts_for_prompt]`. Each innermost tensor has
            shape [T_i], where T_i is that rollout's generated length.
        max_positions: Optional maximum number of token positions to keep on the
            x-axis. If provided, trajectories are truncated to at most this length.
        group_mode: Controls how rows are grouped before plotting.
            - combined: flattens all prompts into one block.
            - prompt_blocks: keeps one block per prompt in input order.
        sort_mode: Row-ordering policy applied within each block defined by group_mode. 
            - length: sorts trajectories by increasing length.
            - none: preserves the original order within each block.

    Returns:
        Dictionary:
        - matrix: Array of shape `[num_total_rollouts, K]`, where K is 
          the longest retained trajectory length. Trailing positions after
          EOS are filled with `.
        - active_counts: Array of shape [K] with the number of active
          rollouts at each token position.
        - prompt_boundaries: Exclusive row-end indices for prompt blocks.
          Empty when `group_mode="combined"`.
        - prompt_centers: Y-axis center positions for prompt blocks.
        - prompt_labels: Labels aligned with `prompt_centers`.
    """
    row_trajectories: list[np.ndarray] = []
    prompt_boundaries: list[int] = []
    prompt_centers: list[float] = []
    prompt_labels: list[str] = []

    if group_mode == "combined":
        row_trajectories = [
            _trajectory_to_numpy(trajectory)
            for prompt_trajectories in prompt_eos_logprob_trajectories
            for trajectory in prompt_trajectories
        ]
        if sort_mode == "length":
            row_trajectories.sort(key=len)
    else:
        row_start = 0
        for prompt_index, prompt_trajectories in enumerate(prompt_eos_logprob_trajectories):
            trajectories = [_trajectory_to_numpy(trajectory) for trajectory in prompt_trajectories]
            if sort_mode == "length":
                trajectories.sort(key=len)

            row_trajectories.extend(trajectories)
            row_end_exclusive = row_start + len(trajectories) # next row start index
            prompt_boundaries.append(row_end_exclusive)
            prompt_centers.append((row_start + row_end_exclusive - 1) / 2.0)
            prompt_labels.append(f"P{prompt_index + 1}")
            row_start = row_end_exclusive

    longest_trajectory = max(len(trajectory) for trajectory in row_trajectories)
    num_positions = longest_trajectory if max_positions is None else min(
        longest_trajectory, max_positions)
    
    # [num_total_rollouts, max_trajectory_length]
    matrix = np.full((len(row_trajectories), num_positions), np.nan, dtype=np.float32)
    for row_index, trajectory in enumerate(row_trajectories):
        clipped_trajectory = trajectory[:num_positions] # could be shorter than num_positions
        matrix[row_index, : len(clipped_trajectory)] = clipped_trajectory

    active_counts = np.sum(np.isfinite(matrix), axis=0, dtype=np.int64) # [max_trajectory_length]

    return {
        "matrix": matrix,
        "active_counts": active_counts,
        "prompt_boundaries": prompt_boundaries,
        "prompt_centers": prompt_centers,
        "prompt_labels": prompt_labels,
    }


def _trajectory_to_numpy(trajectory: torch.Tensor) -> np.ndarray:
    """Detach one trajectory and convert it to a CPU numpy array."""
    return trajectory.detach().to(device="cpu", dtype=torch.float32).numpy()
