from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

import torch

VARIANT_COLORS = {
    "base": "#6299CA",
    "instruct": "#C87271",
}


def plot_sequence_length_histogram(
    sequence_lengths: torch.Tensor,
    output_path: str | Path,
    bins: int = 50,
    title: str = "Sequence Length Distribution",
    label_fontsize: float = 12.0,
) -> None:
    """
    Saves a sequence length density plot.

    Args:
        sequence_lengths: 1-D tensor of integer lengths.
        output_path: Destination image path.
        bins: Number of histogram bins.
        title: Plot title.
        label_fontsize: Font size used for the title, axis labels, and ticks.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rc_params = {
        "axes.titlesize": label_fontsize + 2,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": label_fontsize,
        "ytick.labelsize": label_fontsize,
    }

    lengths = [int(x) for x in sequence_lengths.detach().cpu().tolist()]
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(lengths, bins=bins, density=True, color="#92B1D9", edgecolor="#FFFFFF")
        ax.set_title(title)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_sequence_length_grid(
    lengths_by_cell: dict[tuple[str, str], dict[str, torch.Tensor]],
    families: list[str],
    datasets: list[str],
    output_path: str | Path,
    bins: int = 50,
    label_fontsize: float = 11.0,
) -> None:
    """
    Saves a (model families x datasets) grid of overlaid sequence length densities.

    Args:
        lengths_by_cell: Maps (family, dataset) -> {variant: 1-D tensor of lengths}.
        families: Model families to render as rows, in top to bottom order.
        datasets: Dataset names to render as columns, in left to right order.
        output_path: Destination image path.
        bins: Number of histogram bins (shared across cells).
        label_fontsize: Font size for titles, labels, and ticks.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(families), len(datasets)
    rc_params = {
        "axes.titlesize": label_fontsize + 1,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": label_fontsize - 1,
        "ytick.labelsize": label_fontsize - 1,
        "legend.fontsize": label_fontsize + 1,
        "figure.labelsize": label_fontsize + 1,
    }

    legend_handles: dict[str, object] = {}
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 2.8 * n_rows),
            squeeze=False,
            sharex="col",
        )

        for row, family in enumerate(families):
            for col, dataset in enumerate(datasets):
                ax = axes[row][col]
                variants = lengths_by_cell[(family, dataset)]
                if variants:
                    union_max_len = max(int(variant_lengths.max()) for variant_lengths in variants.values())
                    edges = np.linspace(0, union_max_len + 1, bins + 1)
                    for variant, len_tensor in variants.items():
                        lengths = [int(x) for x in len_tensor.detach().cpu().tolist()]
                        _, _, patches = ax.hist(
                            lengths,
                            bins=edges,
                            density=True,
                            color=VARIANT_COLORS[variant],
                            edgecolor="#FFFFFF",
                            alpha=.55,
                            label=variant,
                        )
                        legend_handles.setdefault(variant, patches[0])
                
                if row == 0:
                    ax.set_title(dataset.capitalize())
                if col == 0:
                    ax.set_ylabel(family)
                    
                ax.grid(axis="y", alpha=0.25)
                ax.spines[["top", "right"]].set_visible(False)
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-3, -3))
                ax.yaxis.set_major_formatter(formatter)
                ax.yaxis.get_offset_text().set_visible(False)

        fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.12, wspace=0.22, hspace=0.22)
        fig.supylabel(r"Density ($\times 10^{-3}$)", x=0.01)
        fig.supxlabel("Sequence Length", y=0.05)
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.00),
            ncol=len(legend_handles),
            frameon=False,
        )

    fig.savefig(output, dpi=150)
    plt.close(fig)