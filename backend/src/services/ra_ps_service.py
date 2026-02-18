from typing import Sequence, Tuple, Union
import matplotlib.pyplot as plt

def plot_ra_ps_series(
    ax: plt.Axes,
    data: Sequence[Sequence[float]],
    label: str,
    color: str,
    marker: str = "o",
    markersize: float = 8.0,
    capsize: float = 5.0,
    linestyle: str = "none",
) -> None:
    if not data:
        return

    ra_list = [row[0] for row in data]
    ps_list = [row[1] for row in data]
    rms_list = [row[2] for row in data]

    ax.errorbar(
        ra_list, ps_list, yerr=rms_list, label=label, color=color,
        fmt=marker, markersize=markersize, capsize=capsize,
        linestyle=linestyle, ecolor=color, alpha=0.8
    )

def plot_ra_ps_summary(
    series_list: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str = "RA vs Ps",
    xlabel: str = "RA (ohm um^2)",
    ylabel: str = "Ps (%)",
    figsize: Tuple[float, float] = (8, 6),
    grid: bool = True,
    xscale: str = "log",
    yscale: str = "linear",
    xlim: Union[Tuple[float, float], None] = None,
    ylim: Union[Tuple[float, float], None] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    
    if len(series_list) != len(labels) or len(series_list) != len(colors):
        raise ValueError("Length mismatch between series, labels, and colors")

    fig, ax = plt.subplots(figsize=figsize)

    for i, data in enumerate(series_list):
        plot_ra_ps_series(ax, data, label=labels[i], color=colors[i])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if grid: ax.grid(True, linestyle=":", alpha=0.6)
    if labels: ax.legend(fontsize=12, frameon=False)

    fig.tight_layout()
    return fig, ax
