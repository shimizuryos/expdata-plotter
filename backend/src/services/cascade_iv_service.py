from typing import List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from ..models.analysis_types import ParsedIVSeries

def plot_iv_series(iv_series_list: List[ParsedIVSeries], labels: Optional[List[str]] = None) -> Tuple[Any, Any, Any]:
    # Simplified plotting logic adapted from original
    fig, (ax_iv, ax_rv) = plt.subplots(1, 2, figsize=(20, 8), squeeze=True)

    color_options: list[str] = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black",
    ]

    for j, series in enumerate(iv_series_list):
        ids = series.id_mA
        vds = series.vd_mV
        rs = series.r_ohm
        color_j = color_options[j % len(color_options)]
        label = labels[j] if labels and j < len(labels) else f"label {j+1}"

        # IV
        ax_iv.plot(
            vds, ids, marker="o", linestyle="none", markersize=6.0,
            color=color_j, linewidth=1.2, alpha=0.9, label=label
        )

        # RV
        filtered_pairs = [(v, r) for v, r in zip(vds, rs) if not (-20 <= v <= 20)]
        if filtered_pairs:
            vds_rv, rs_rv = zip(*filtered_pairs)
            ax_rv.plot(
                vds_rv, rs_rv, marker="o", linestyle="none", markersize=6.0,
                color=color_j, linewidth=1.2, alpha=0.9, label=label
            )

    ax_iv.set_title("IV")
    ax_iv.set_xlabel("Vd (mV)", fontsize=17)
    ax_iv.set_ylabel("Id (mA)", fontsize=17)
    ax_iv.grid(True, linestyle=":", alpha=0.4)
    if any(ax_iv.get_legend_handles_labels()[0]):
        ax_iv.legend(frameon=False, fontsize=13)

    ax_rv.set_title("RV")
    ax_rv.set_xlabel("Vd (mV)", fontsize=17)
    ax_rv.set_ylabel("R (ohm)", fontsize=17)
    ax_rv.set_yscale("log")
    ax_rv.grid(True, linestyle=":", alpha=0.4)
    if any(ax_rv.get_legend_handles_labels()[0]):
        ax_rv.legend(frameon=False, fontsize=13)

    fig.tight_layout()
    return fig, ax_iv, ax_rv

def plot_grouped_iv_series(
    grouped_iv_series: List[List[ParsedIVSeries]],
    grouped_labels: Optional[List[List[str]]] = None,
    rv_exclude_window_mV: Optional[float] = 20.0,
) -> Tuple[Any, Any, Any]:
    fig, (ax_iv, ax_rv) = plt.subplots(1, 2, figsize=(20, 8), squeeze=True)

    cmap_names: list[str] = [
        "Blues", "Oranges", "Greens", "Reds", "Purples",
        "Greys", "YlGn", "PuBu", "BuPu", "GnBu",
    ]

    for g_idx, series_list in enumerate(grouped_iv_series):
        if not series_list:
            continue
        cmap = plt.get_cmap(cmap_names[g_idx % len(cmap_names)])
        n = len(series_list)
        shades = [0.35 + 0.55 * (k / max(n - 1, 1)) for k in range(n)]

        for k, series in enumerate(series_list):
            color_gk = cmap(shades[k])
            label = (
                grouped_labels[g_idx][k]
                if (grouped_labels and g_idx < len(grouped_labels) and k < len(grouped_labels[g_idx]))
                else f"group{g_idx+1}-{k+1}"
            )

            ax_iv.plot(
                series.vd_mV, series.id_mA, marker="o", linestyle="none", markersize=4.0,
                color=color_gk, linewidth=1.2, alpha=0.9, label=label,
            )

            if rv_exclude_window_mV is None:
                vds_rv, rs_rv = series.vd_mV, series.r_ohm
            else:
                w = float(rv_exclude_window_mV)
                filtered_pairs = [(v, r) for v, r in zip(series.vd_mV, series.r_ohm) if not (-w <= v <= w)]
                if not filtered_pairs:
                    vds_rv, rs_rv = [], []
                else:
                    vds_rv, rs_rv = zip(*filtered_pairs)

            if vds_rv:
                ax_rv.plot(
                    vds_rv, rs_rv, marker="o", linestyle="none", markersize=4.0,
                    color=color_gk, linewidth=1.2, alpha=0.9, label=label,
                )

    ax_iv.set_title("IV")
    ax_iv.grid(True, linestyle=":", alpha=0.4)
    ax_rv.set_title("RV")
    ax_rv.set_yscale("log")
    ax_rv.grid(True, linestyle=":", alpha=0.4)
    
    if any(ax_iv.get_legend_handles_labels()[0]):
        ax_iv.legend(frameon=False, fontsize=13)

    fig.tight_layout()
    return fig, ax_iv, ax_rv
