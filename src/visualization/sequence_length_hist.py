from pathlib import Path
import matplotlib.pyplot as plt

import torch


def plot_sequence_length_histogram(
    sequence_lengths: torch.Tensor,
    output_path: str | Path,
    bins: int = 50,
    title: str = "Sequence Length Distribution",
    label_fontsize: float = 12.0,
) -> None:
    """
    Saves a static histogram for generated sequence lengths.

    Args:
        sequence_lengths: 1-D tensor of integer lengths.
        output_path: Destination image path.
        bins: Number of histogram bins.
        title: Plot title.
        label_fontsize: Font size used for the title, axis labels, and ticks.
    """

    lengths = [int(x) for x in sequence_lengths.detach().cpu().tolist()]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rc_params = {
        "axes.titlesize": label_fontsize + 2,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": label_fontsize,
        "ytick.labelsize": label_fontsize,
    }

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(lengths, bins=bins, color="#92B1D9", edgecolor="#FFFFFF")
        ax.set_title(title)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

    fig.savefig(output, dpi=150)
    plt.close(fig)
