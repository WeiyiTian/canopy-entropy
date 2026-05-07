import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.metrics.generation_tree import CEStarMaxDecomposition


LENGTH_DRIVEN_COLOR = "#dfa6a7"
COVARIANCE_COLOR = "#5a966c"
BAND_COLOR = "#f5f5f5"
REFERENCE_COLOR = "#666666"

FAMILY_GAP = 0.6


def plot_ce_star_decomposition(
    decompositions_by_cell: dict[tuple[str, str], dict[str, CEStarMaxDecomposition]],
    families: list[str],
    datasets: list[str],
    variants: list[str],
    output_path: str | Path,
    label_fontsize: float = 9.0,
    bar_height: float = 0.6,
    line_width: float = 0.8,
) -> None:
    """
    Saves horizontal stacked-bar panels for the normalized CE*_max decomposition.

    Each dataset is rendered as a panel. Rows are grouped by model family, with
    one row per variant. Bars show `length_share + cov_share = 1`: the
    length-driven term extends right from zero, while positive covariance fills
    the gap to the `x=1` reference and negative covariance extends left.

    Args:
        decompositions_by_cell: Maps `(family, dataset)` -> `{variant: CEStarMaxDecomposition}`.
        families: Model families ordered bottom to top within each panel.
        datasets: Dataset panels ordered left to right.
        variants: Model variants drawn as adjacent rows within each family group.
        output_path: Destination image path.
        label_fontsize: Base font size.
        bar_height: Bar thickness.
        line_width: Reference line width.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rc_params = {
        "font.size": label_fontsize - 2,
        "axes.titlesize": label_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": label_fontsize - 1,
        "ytick.labelsize": label_fontsize - 1,
        "legend.fontsize": label_fontsize,
    }

    family_spacing = len(variants) + FAMILY_GAP
    n_total_rows = len(families) * len(variants)

    all_decomps = [decomp for cell in decompositions_by_cell.values() for decomp in cell.values()]
    x_min = min(0.0, *(d.length_entropy_rate_cov_share for d in all_decomps))
    x_max = max(1.0, *(d.length_driven_share for d in all_decomps))
    x_lo = x_min - 0.75
    x_hi = x_max + 0.45

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            1, len(datasets),
            figsize=(1.5 * len(datasets) + 0.8, 0.32 * n_total_rows + 1.4),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for ax, dataset in zip(axes[0], datasets, strict=True):
            yticks: list[float] = []
            ytick_labels: list[str] = []
            for fam_idx, family in enumerate(families):
                family_display = re.match(r"[A-Za-z]+", family).group()
                for var_idx, variant in enumerate(variants):
                    y_pos = fam_idx * family_spacing + var_idx
                    if var_idx > 0:
                        ax.axhspan(y_pos - 0.5, y_pos + 0.5, facecolor=BAND_COLOR, zorder=0)
                    yticks.append(y_pos)
                    ytick_labels.append(f"{family_display}-{variant}")

                    decomp = decompositions_by_cell[(family, dataset)][variant]
                    if decomp is None:
                        continue
                    length_share = decomp.length_driven_share
                    cov_share = decomp.length_entropy_rate_cov_share

                    ax.barh(y_pos, length_share, left=0, height=bar_height, color=LENGTH_DRIVEN_COLOR, alpha=.5)
                    ax.barh(
                        y_pos, cov_share,
                        left=length_share if cov_share >= 0 else 0.0,
                        height=bar_height, color=COVARIANCE_COLOR,
                    )
                    if cov_share >= 0:
                        ax.text(
                            length_share + cov_share + 0.02, y_pos, f"{cov_share:+.2f}",
                            ha="left", va="center", color=COVARIANCE_COLOR,
                        )
                    else:
                        ax.text(
                            cov_share - 0.02, y_pos, f"{cov_share:+.2f}",
                            ha="right", va="center", color=COVARIANCE_COLOR,
                        )

            ax.axvline(1.0, linestyle="--", color=REFERENCE_COLOR, linewidth=line_width)
            ax.axvline(0.0, linestyle="--", color=REFERENCE_COLOR, linewidth=line_width)

            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
            ax.set_ylim(min(yticks) - 0.7, max(yticks) + 0.7)
            ax.set_xlim(x_lo, x_hi)
            ax.set_title(dataset.capitalize(), fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", length=0)

        legend_handles = [
            Patch(facecolor=LENGTH_DRIVEN_COLOR, alpha=0.5,
                  label=r"Length-driven $\;\dfrac{\mathbb{E}[N]\mathbb{E}[r_N]}{\mathrm{CE}^\star_{\max}}$"),
            Patch(facecolor=COVARIANCE_COLOR,
                  label=r"(Length, entropy rate) coupling $\;\dfrac{\mathrm{Cov}(N, r_N)}{\mathrm{CE}^\star_{\max}}$"),
        ]
        fig.subplots_adjust(left=0.16, right=0.99, top=0.92, bottom=0.18, wspace=0.10)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            frameon=False,
        )

    fig.savefig(output, dpi=150)
    plt.close(fig)
