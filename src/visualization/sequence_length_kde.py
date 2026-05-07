from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, SymmetricalLogLocator
import numpy as np

import torch
from scipy.stats import gaussian_kde

from src.visualization.plotting_constants import FAMILY_VARIANT_COLORS, VARIANT_LINESTYLES


def plot_sequence_length_kde(
    lengths_by_cell: dict[tuple[str, str], dict[str, torch.Tensor]],
    families: list[str],
    datasets: list[str],
    output_path: str | Path,
    symlog_linthresh: float = 1.0,
    symlog_linscale: float = 0.1,
    n_eval_points: int = 1000,
    label_fontsize: float = 11.0,
    line_width: float = 1.6,
) -> None:
    """
    Saves a 1-row grid of sequence length KDEs with all (family, variant) pairs overlaid per panel.

    Args:
        lengths_by_cell: Maps (family, dataset) -> {variant: 1-D tensor of lengths}.
        families: Model families overlaid within each panel.
        datasets: Dataset names rendered as columns, in left to right order.
        output_path: Destination image path.
        symlog_linthresh: Threshold below which the x-axis is linear (length 0 sits at the left edge).
        symlog_linscale: Visual width of the linear region in units of one log decade.
        n_eval_points: Number of points on which the KDE is evaluated per cell.
        label_fontsize: Font size for titles, labels, and ticks.
        line_width: KDE curve linewidth.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_cols = len(datasets)
    rc_params = {
        "axes.titlesize": label_fontsize + 1,
        "axes.labelsize": label_fontsize + 1,
        "xtick.labelsize": label_fontsize - 1,
        "ytick.labelsize": label_fontsize - 1,
        "legend.fontsize": label_fontsize,
        "figure.labelsize": label_fontsize + 1,
    }

    line_handles = {}
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4 * n_cols, 4),
            squeeze=False,
        )

        for col, dataset in enumerate(datasets):
            ax = axes[0][col]
            union_max_len = 0
            for family in families:
                for variant_lengths in lengths_by_cell[(family, dataset)].values():
                    union_max_len = max(union_max_len, int(variant_lengths.max()))

            xs = np.linspace(0, union_max_len + 1, n_eval_points)
            for family in families:
                variants = lengths_by_cell[(family, dataset)]
                for variant, len_tensor in variants.items():
                    lengths = len_tensor.detach().cpu().numpy()
                    kde = gaussian_kde(lengths)
                    line, = ax.plot(
                        xs,
                        kde(xs),
                        color=FAMILY_VARIANT_COLORS[family][variant],
                        linestyle=VARIANT_LINESTYLES[variant],
                        linewidth=line_width,
                    )
                    line_handles.setdefault((family, variant), line)

            ax.set_title(dataset.capitalize(), fontweight="bold")
            ax.set_xlabel("Sequence Length (log)")
            ax.set_xscale("symlog", linthresh=symlog_linthresh, linscale=symlog_linscale)
            ax.set_xticks([1, 10, 100, 1000, 10000])
            ax.xaxis.set_minor_locator(
                SymmetricalLogLocator(linthresh=symlog_linthresh, base=10.0, subs=range(2, 10))
            )
            ax.set_xlim(0, union_max_len + 1)

            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -3))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_visible(False)

        legend_keys = [
            (family, variant)
            for family in families
            for variant in VARIANT_LINESTYLES
            if (family, variant) in line_handles
        ]

        fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.30, wspace=0.22)
        fig.supylabel(r"Density ($\times 10^{-3}$)", x=0.01, y=0.60)
        fig.legend(
            handles=[line_handles[key] for key in legend_keys],
            labels=[f"{family.removesuffix('b')}B {variant}" for family, variant in legend_keys],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=len(families),
            frameon=False,
        )

    fig.savefig(output, dpi=150)
    plt.close(fig)
