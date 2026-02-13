from typing import List, Sequence, Tuple, Union
import matplotlib.pyplot as plt

def plot_ra_ps_series(
    ax: plt.Axes,
    data: Sequence[Sequence[float]],
    label: str,
    color: str,
    marker: str = "o",
    markersize: float = 8.0,
    capsize: float = 5.0,
    linestyle: str = "none",  # デフォルトは点プロット（線なし）
) -> None:
    """
    1つのデータ系列（RA, Ps, RMS のリストのリスト）を指定された ax にプロットする。
    
    Parameters:
    - ax: プロット対象の Matplotlib Axes
    - data: [[RA, Ps, RMS], [RA, Ps, RMS], ...] の形式のリスト
    - label: 凡例用ラベル
    - color: プロットの色
    - marker: マーカー形状
    - markersize: マーカーサイズ
    - capsize: エラーバーのキャップサイズ
    - linestyle: 線種（デフォルトは 'none' でマーカーのみ）
    """
    if not data:
        return

    # データの分解
    # 行ごとに [RA, Ps, RMS] があると想定
    ra_list = [row[0] for row in data]
    ps_list = [row[1] for row in data]
    rms_list = [row[2] for row in data]

    # エラーバー付きプロット
    # RMS を Y軸の誤差として使用
    ax.errorbar(
        ra_list,
        ps_list,
        yerr=rms_list,
        label=label,
        color=color,
        fmt=marker,        # マーカー形状
        markersize=markersize,
        capsize=capsize,   # エラーバーの横棒の長さ
        linestyle=linestyle,
        ecolor=color,      # エラーバーの色（プロット色と同じにする）
        alpha=0.8
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
    """
    複数のデータ系列を同一グラフにプロットする。
    
    Parameters:
    - series_list: データ系列のリスト。各要素は [[RA, Ps, RMS], ...]
    - labels: 各系列のラベルリスト
    - colors: 各系列の色リスト
    - title: グラフタイトル
    - xlabel: X軸ラベル
    - ylabel: Y軸ラベル
    - figsize: 図のサイズ
    - grid: グリッド表示有無
    - xscale: X軸スケール ("linear", "log" 等)
    - yscale: Y軸スケール
    - xlim: X軸の範囲 (min, max)
    - ylim: Y軸の範囲 (min, max)

    Returns:
    - (fig, ax)
    """
    if len(series_list) != len(labels) or len(series_list) != len(colors):
        raise ValueError("series_list, labels, colors の長さは一致する必要があります。")

    fig, ax = plt.subplots(figsize=figsize)

    for i, data in enumerate(series_list):
        plot_ra_ps_series(
            ax,
            data,
            label=labels[i],
            color=colors[i]
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12, direction="in")
    
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
        
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(True, which="major", linestyle=":", alpha=0.6)
        if xscale == "log" or yscale == "log":
             ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    if labels:
        ax.legend(fontsize=12, frameon=False)

    fig.tight_layout()
    return fig, ax

