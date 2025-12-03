import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

@dataclass
class ParsedIVSeries:
    id_mA: List[float]
    vd_mV: List[float]
    r_ohm: List[float]
    warnings: List[str]

def parse_iv_csv(file_path: str) -> ParsedIVSeries:
    warnings: List[str] = []
    id_list: List[float] = []
    vd_list: List[float] = []
    r_list: List[float] = []

    # 読み込み（すべて文字列として読み、後で必要なところだけ数値化）
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            dtype=str,
            engine="python",
            names=list(range(256)),       # 列数を固定して可変列行に対応
            skip_blank_lines=True,        # 空行をスキップ
            on_bad_lines="skip",          # 破損行はスキップ
        )
    except Exception as exc:
        raise ValueError(f"CSV の読み込みに失敗しました: {exc}")

    # DataName 行の検出
    name_rows = df.index[df[0].astype(str).str.strip() == "DataName"].tolist()
    if not name_rows:
        raise ValueError("DataName 行が見つかりません")
    name_idx = name_rows[0]
    names = df.loc[name_idx, 1:].tolist()
    # NaN (pandasの欠損) を空文字に
    names = [("" if pd.isna(x) else str(x).strip()) for x in names]

    # 必須カラムの位置
    try:
        name_to_idx = {name: i for i, name in enumerate(names)}
        id_idx = name_to_idx["Id"]
        vd_idx = name_to_idx["Vd"]
        r_idx = name_to_idx["R"]
    except KeyError as exc:
        raise ValueError(f"必須カラムが不足しています: {exc}")

    header_len = len(names)

    def to_float(token: str) -> Optional[float]:
        if token is None:
            return None
        t = str(token).strip()
        if t == "" or t.lower() == "nan":
            return None
        try:
            return float(t)
        except ValueError:
            return None

    # DataValue 行の抽出
    value_df = df[df[0].astype(str).str.strip() == "DataValue"]
    for i, row in value_df.iterrows():
        values = row[1:].tolist()
        # 不足分を埋める
        if len(values) < header_len:
            values += [""] * (header_len - len(values))

        id_val = to_float(values[id_idx])
        vd_val = to_float(values[vd_idx])
        r_val = to_float(values[r_idx])

        if id_val is None or vd_val is None or r_val is None:
            warnings.append(f"line {i+1}: 数値変換不可 (Id='{values[id_idx]}', Vd='{values[vd_idx]}', R='{values[r_idx]}')")
            continue

        # 単位換算: A -> mA, V -> mV
        id_list.append(id_val * 1e3)
        vd_list.append(vd_val * 1e3)
        r_list.append(r_val)

    if not id_list:
        raise ValueError("有効な DataValue 行が見つかりませんでした")

    return ParsedIVSeries(id_mA=id_list, vd_mV=vd_list, r_ohm=r_list, warnings=warnings)

def plot_iv_series(iv_series_list: List[ParsedIVSeries], labels: Optional[List[str]] = None):
    # 図準備（ノートブックのロジックに倣う）
    fig, (ax_iv, ax_rv) = plt.subplots(1, 2, figsize=(20, 8), squeeze=True)

    # スタイルの簡易サイクル
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

        # IV: Id(mA) vs Vd(mV)
        ax_iv.plot(
            vds, ids,
            marker="o",
            linestyle="none",
            markersize=6.0,
            color=color_j,
            linewidth=1.2,
            alpha=0.9,
            label=label
        )

        # RV 用に Vd≈0 近傍（±50 mV）を除外
        filtered_pairs = [(v, r) for v, r in zip(vds, rs) if not (-20 <= v <= 20)]
        if filtered_pairs:
            vds_rv, rs_rv = zip(*filtered_pairs)
            ax_rv.plot(
                vds_rv, rs_rv,
                marker="o",
                linestyle="none",
                markersize=6.0,
                color=color_j,
                linewidth=1.2,
                alpha=0.9,
                label=label
            )

    # 体裁（ノートブック準拠）
    ax_iv.set_title("IV")
    ax_iv.set_xlabel("Vd (mV)", fontsize=17)
    ax_iv.set_ylabel("Id (mA)", fontsize=17)
    ax_iv.tick_params(labelsize=17, direction="in")
    ax_iv.set_box_aspect(1)

    ax_rv.set_title("RV")
    ax_rv.set_xlabel("Vd (mV)", fontsize=17)
    ax_rv.set_ylabel("R (ohm)", fontsize=17)
    ax_rv.set_yscale("log")
    ax_rv.tick_params(labelsize=17, direction="in")
    ax_rv.set_box_aspect(1)

    # グリッド
    ax_iv.grid(True, linestyle=":", alpha=0.4)
    ax_rv.grid(True, linestyle=":", alpha=0.4)

    # 凡例
    if any(ax_iv.get_legend_handles_labels()[0]):
        ax_iv.legend(frameon=False, fontsize=13)
    if any(ax_rv.get_legend_handles_labels()[0]):
        ax_rv.legend(frameon=False, fontsize=13)

    fig.tight_layout()
    return fig, ax_iv, ax_rv

def plot_grouped_iv_series(
    grouped_iv_series: List[List[ParsedIVSeries]],
    grouped_labels: Optional[List[List[str]]] = None,
    rv_exclude_window_mV: Optional[float] = 20.0,
):
    # 図準備
    fig, (ax_iv, ax_rv) = plt.subplots(1, 2, figsize=(20, 8), squeeze=True)

    # グループごとに同系統色（カラーマップの濃淡）を使用
    cmap_names: list[str] = [
        "Blues", "Oranges", "Greens", "Reds", "Purples",
        "Greys", "YlGn", "PuBu", "BuPu", "GnBu",
    ]

    for g_idx, series_list in enumerate(grouped_iv_series):
        if not series_list:
            continue
        cmap = plt.get_cmap(cmap_names[g_idx % len(cmap_names)])
        n = len(series_list)
        # 薄い→濃いの範囲を適度に確保
        shades = [0.35 + 0.55 * (k / max(n - 1, 1)) for k in range(n)]

        for k, series in enumerate(series_list):
            color_gk = cmap(shades[k])
            label = (
                grouped_labels[g_idx][k]
                if (grouped_labels and g_idx < len(grouped_labels) and k < len(grouped_labels[g_idx]))
                else f"group{g_idx+1}-{k+1}"
            )

            # IV: Id(mA) vs Vd(mV)
            ax_iv.plot(
                series.vd_mV, series.id_mA,
                marker="o",
                linestyle="none",
                markersize=4.0,
                color=color_gk,
                linewidth=1.2,
                alpha=0.9,
                label=label,
            )

            # RV: R(ohm) vs Vd(mV)（ゼロ近傍を除外可）
            if rv_exclude_window_mV is None:
                vds_rv = series.vd_mV
                rs_rv = series.r_ohm
            else:
                w = float(rv_exclude_window_mV)
                filtered_pairs = [(v, r) for v, r in zip(series.vd_mV, series.r_ohm) if not (-w <= v <= w)]
                if not filtered_pairs:
                    vds_rv, rs_rv = [], []
                else:
                    vds_rv, rs_rv = zip(*filtered_pairs)

            if vds_rv:
                ax_rv.plot(
                    vds_rv, rs_rv,
                    marker="o",
                    linestyle="none",
                    markersize=4.0,
                    color=color_gk,
                    linewidth=1.2,
                    alpha=0.9,
                    label=label,
                )

    # 体裁（既存関数と同様）
    ax_iv.set_title("IV")
    ax_iv.set_xlabel("Vd (mV)", fontsize=17)
    ax_iv.set_ylabel("Id (mA)", fontsize=17)
    ax_iv.tick_params(labelsize=17, direction="in")
    ax_iv.set_box_aspect(1)

    ax_rv.set_title("RV")
    ax_rv.set_xlabel("Vd (mV)", fontsize=17)
    ax_rv.set_ylabel("R (ohm)", fontsize=17)
    ax_rv.set_yscale("log")
    ax_rv.tick_params(labelsize=17, direction="in")
    ax_rv.set_box_aspect(1)

    ax_iv.grid(True, linestyle=":", alpha=0.4)
    ax_rv.grid(True, linestyle=":", alpha=0.4)

    if any(ax_iv.get_legend_handles_labels()[0]):
        ax_iv.legend(frameon=False, fontsize=13)
    if any(ax_rv.get_legend_handles_labels()[0]):
        ax_rv.legend(frameon=False, fontsize=13)

    fig.tight_layout()
    return fig, ax_iv, ax_rv