from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from .plotting_constants import FAMILY_VARIANT_COLORS, VARIANT_LINESTYLES


METRIC_AXIS_LABELS = {
    "length": r"$\widehat{\mathrm{Var}}(N \mid X)$",
    "entropy_rate": r"$\widehat{\mathrm{Var}}(r \mid X)$",
}


def plot_variance_distribution(
    variances_by_cell: dict[tuple[str, str], dict[str, torch.Tensor]],
    families: list[str],
    datasets: list[str],
    output_path: str | Path,
    metric: Literal["length", "entropy_rate"],
    n_eval_points: int = 400,
    label_fontsize: float = 11.0,
    line_width: float = 1.6,
) -> None:
    """
    Saves a (model families x datasets) grid of within-prompt variance KDEs, with
    base and instruct variants overlaid in each panel and a global epsilon guide
    line at the smallest observed variance.

    Args:
        variances_by_cell: Maps (family, dataset) -> {variant: 1-D tensor of shape [P]},
            holding the per-prompt within-prompt sample variance of either
            sequence length N or entropy rate r.
        families: Model families rendered as rows, top to bottom.
        datasets: Dataset names rendered as columns, left to right.
        output_path: Destination image path.
        metric: Either "length" or "entropy_rate" for x-axis label.
        n_eval_points: Number of points on which the KDE is evaluated per cell.
        label_fontsize: Font size for titles, labels, and ticks.
        line_width: KDE curve linewidth.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    global_min = min(
        float(variant_vars.min())
        for variant_map in variances_by_cell.values()
        for variant_vars in variant_map.values()
        if variant_vars.numel() > 0
    )
    global_max = max(
        float(variant_vars.max())
        for variant_map in variances_by_cell.values()
        for variant_vars in variant_map.values()
        if variant_vars.numel() > 0
    )

    n_rows, n_cols = len(families), len(datasets)
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
            n_rows, n_cols,
            figsize=(4 * n_cols, 2.6 * n_rows),
            sharex=True,
            squeeze=False,
        )

        log_grid = np.linspace(np.log(global_min), np.log(global_max), n_eval_points)
        xs = np.exp(log_grid)

        for row, family in enumerate(families):
            for col, dataset in enumerate(datasets):
                ax = axes[row][col]
                variants = variances_by_cell[(family, dataset)]

                for variant, variant_vars in variants.items():
                    log_values = np.log(variant_vars.detach().cpu().numpy())
                    kde = gaussian_kde(log_values)
                    line, = ax.plot(
                        xs,
                        kde(log_grid),
                        color=FAMILY_VARIANT_COLORS[family][variant],
                        linestyle=VARIANT_LINESTYLES[variant],
                        linewidth=line_width,
                    )
                    line_handles.setdefault((family, variant), line)

                ax.axvline(global_min, color="grey", linestyle=":", linewidth=line_width, alpha=0.7)
                ax.set_xscale("log")
                ax.grid(axis="y", alpha=0.25)
                ax.spines[["top", "right"]].set_visible(False)

                if row == 0:
                    ax.set_title(dataset.capitalize(), fontweight="bold")
                if col == 0:
                    ax.set_ylabel(family.removesuffix("b") + "B")

        for col in range(n_cols):
            axes[-1][col].set_xlabel(METRIC_AXIS_LABELS[metric] + " (log)")

        legend_keys = [
            (family, variant)
            for family in families
            for variant in VARIANT_LINESTYLES
            if (family, variant) in line_handles
        ]
        eps_handle = plt.Line2D([], [], color="grey", linestyle=":", linewidth=1.0)

        fig.align_ylabels(axes[:, 0])
        fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.16, hspace=0.30, wspace=0.22)
        fig.legend(
            handles=[line_handles[key] for key in legend_keys] + [eps_handle],
            labels=[f"{family.removesuffix('b')}B {variant}" for family, variant in legend_keys]
                + [f"min observed = {global_min:.2g}"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(families) + 1,
            frameon=False,
        )

    fig.savefig(output, dpi=150)
    plt.close(fig)
