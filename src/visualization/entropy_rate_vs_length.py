from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import EngFormatter, MaxNLocator

from .plotting_constants import FAMILY_VARIANT_COLORS, VARIANT_LINESTYLES


def plot_entropy_rate_trajectory(
    curves_by_cell: dict[tuple[str, str], dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    families: list[str],
    datasets: list[str],
    output_path: str | Path,
    label_fontsize: float = 11.0,
    line_width: float = 1.6,
) -> None:
    """
    Plots running entropy-rate trajectories with a supplementary active-rollout
    count row beneath, one column per dataset.

    Args:
        curves_by_cell: Mapping (family, dataset) to variant curves. Each curve
            has (positions, mean_rate, active_count) of which all have shape [B].
            `mean_rate[b]` averages rollouts active at token position `positions[b]`.
        families: Model families to overlay within each dataset panel.
        datasets: Dataset names rendered as columns, left to right.
        output_path: Destination image path.
        label_fontsize: Base font size.
        line_width: Curve linewidth.
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
            2, n_cols,
            figsize=(4 * n_cols, 5.0),
            sharex="col",
            squeeze=False,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        for col, dataset in enumerate(datasets):
            rate_ax, count_ax = axes[0][col], axes[1][col]
            for family in families:
                for variant, (positions, mean_rate, active_count) in curves_by_cell[(family, dataset)].items():
                    xs = positions.detach().cpu().numpy()
                    style = dict(
                        color=FAMILY_VARIANT_COLORS[family][variant],
                        linestyle=VARIANT_LINESTYLES[variant],
                        linewidth=line_width,
                    )
                    line, = rate_ax.plot(xs, mean_rate.detach().cpu().numpy(), **style)
                    count_ax.plot(xs, active_count.detach().cpu().numpy(), **style)
                    line_handles.setdefault((family, variant), line)

            rate_ax.set_title(dataset.capitalize(), fontweight="bold")
            count_ax.set_xlabel(r"Token Position $t$")
            for ax in (rate_ax, count_ax):
                ax.grid(axis="both", alpha=0.25)
                ax.spines[["top", "right"]].set_visible(False)

        axes[0][0].set_ylabel("Entropy Rate")
        axes[1][0].set_ylabel("# Active")
        for col in range(n_cols):
            axes[1][col].yaxis.set_major_formatter(EngFormatter(places=0, sep=""))
            axes[1][col].xaxis.set_major_formatter(EngFormatter(places=0, sep=""))
            axes[1][col].xaxis.set_major_locator(MaxNLocator(nbins=5))

        legend_keys = [
            (family, variant)
            for family in families
            for variant in VARIANT_LINESTYLES
            if (family, variant) in line_handles
        ]

        fig.align_ylabels(axes[:, 0])
        fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.24, hspace=0.18, wspace=0.22)
        fig.legend(
            handles=[line_handles[key] for key in legend_keys],
            labels=[f"{family.removesuffix('b')}B {variant}" for family, variant in legend_keys],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=len(families),
            frameon=False,
        )

    fig.savefig(output, dpi=150)
    plt.close(fig)
