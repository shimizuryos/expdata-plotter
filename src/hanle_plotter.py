from typing import List, Tuple, Union, Sequence
import math
import re
import matplotlib.pyplot as plt


def read_hanle_raw_data(file_path: str) -> Tuple[List[float], List[float]]:
    """
    指定ファイルを読み込み、1行目を無視し、2行目以降の各行から
    先頭2列の数値を抽出して (magnetic_field_Oe, voltage_uV) を返す。

    - 空行、先頭が '#' の行、2列未満の行、数値化できない行はスキップする
    - 区切りは任意の空白（スペース/タブ）として扱う
    - 単位: magnetic_field_Oe は Oe, voltage_uV は uV
    """
    magnetic_field_Oe: List[float] = []
    voltage_uV: List[float] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                # 1行目は無視
                continue
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            cols = text.split()  # 任意の空白で分割（スペース/タブ対応）
            if len(cols) < 2:
                continue
            try:
                magnetic_field_Oe.append(float(cols[0]))
                voltage_uV.append(float(cols[1])*1_000_000)
            except ValueError:
                # 数値化できない行はスキップ
                continue

    return magnetic_field_Oe, voltage_uV



def read_hanle_data(file_path: str) -> List[List[List[float]]]:
    """
    区切り行(= =)で複数セクションに分かれたHanleデータを読み込み、
    セクションごとに [magnetic_field_list, voltage_list] の形でまとめた
    リストを返す。

    - 区切り行の判定は、空白を取り除いた結果が "==" に一致する行（"= =" も許容）
    - 空行、先頭が '#' の行、2列未満の行、数値化できない行はスキップ
    - 各データ行は先頭2列を使用（単位変換なし）
    - 区切り行の間のデータが1セクション。ファイル末尾時点で未確定セクションがあれば追加
    """

    def is_delim(line_text: str) -> bool:
        s = line_text.strip()
        if not s:
            return False
        return s.replace(" ", "") == "=="

    series_list: List[List[List[float]]] = []
    current_b: List[float] = []
    current_v: List[float] = []

    def flush_current_if_needed():
        if current_b and current_v and len(current_b) == len(current_v):
            series_list.append([current_b.copy(), current_v.copy()])

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()

            # 区切り判定
            if is_delim(text):
                flush_current_if_needed()
                current_b.clear()
                current_v.clear()
                continue

            if not text or text.startswith("#"):
                continue

            cols = text.split()
            if len(cols) < 2:
                continue
            try:
                b_val = float(cols[0])
                v_val = float(cols[1])
            except ValueError:
                continue

            current_b.append(b_val)
            current_v.append(v_val * 1_000_000)

    # ファイル末尾で残っていれば追加
    flush_current_if_needed()

    return series_list


def read_hanle_broad(
    file_path: str,
) -> Tuple[Tuple[float, float, float, float, float, float, float, float, float], List[List[float]], List[List[float]], List[List[float]]]:
    """
    `read_hanle_data`の結果から、
    - 1番目: 実験データ(exp_data)
    - 2番目: フィッティングデータ(fitting_data)
    - 4番目: ブロードフィッティングデータ(broad_fitting_data)
    に加えて、1行目のパラメータをタプルで返す。

    戻り値: (
        params_tuple,
        exp_data, fitting_data, broad_fitting_data
    )
    - params_tuple は以下の順で9要素のタプル:
      (A_b3t, W_b3t, A_n3t, W_n3t, Ts, Voff_b, Voff_n, yokozure, alpha)
    - データは [magnetic_field_list, voltage_list]
    """
    # 1行目のパラメータを解析
    def _parse_header_params(path: str) -> Tuple[float, float, float, float, float, float, float, float, float]:
        def _find_value(pattern: str, text: str) -> float:
            m = re.search(pattern, text)
            if not m:
                return float('nan')
            try:
                return float(m.group(1))
            except ValueError:
                return float('nan')

        float_pattern = r"([+\-]?\d*(?:\.\d+)?(?:[Ee][+\-]?\d+)?)"
        with open(path, "r", encoding="utf-8") as rf:
            first_line = rf.readline().strip()
        # 先頭の # を除去
        if first_line.startswith("#"):
            first_line = first_line[1:]

        A_b3t   = _find_value(rf"A_b3t\s*=\s*{float_pattern}", first_line)
        W_b3t   = _find_value(rf"W_b3t\s*=\s*{float_pattern}", first_line)
        A_n3t   = _find_value(rf"A_n3t\s*=\s*{float_pattern}", first_line)
        W_n3t   = _find_value(rf"W_n3t\s*=\s*{float_pattern}", first_line)
        Ts      = _find_value(rf"Ts\s*=\s*{float_pattern}", first_line)
        Voff_b  = _find_value(rf"Voff_b\s*=\s*{float_pattern}", first_line)
        Voff_n  = _find_value(rf"Voff_n\s*=\s*{float_pattern}", first_line)
        # yokozure と alpha は '=' が無いケースに対応
        yokozure = _find_value(rf"yokozure\s+{float_pattern}", first_line)
        alpha    = _find_value(rf"alpha\s+{float_pattern}", first_line)

        return (A_b3t, W_b3t, A_n3t, W_n3t, Ts, Voff_b, Voff_n, yokozure, alpha)

    params = _parse_header_params(file_path)
    series = read_hanle_data(file_path)

    # 注: exp と fitting の順序が逆（0: fitting, 1: exp）
    exp_data = series[1]
    fitting_data = series[0]
    if len(series) == 4:
        broad_fitting_data = series[3]
        return params, exp_data, fitting_data, broad_fitting_data
    elif len(series) == 3:
        broad_fitting_data = series[2]
        return params, exp_data, fitting_data, broad_fitting_data
    else:
        raise ValueError("broadデータ抽出に必要な4セクションが見つかりませんでした")


def read_hanle_n_only(
    file_path: str,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    `read_hanle_data`の結果から、
    - 1番目: 実験データ(exp_data)
    - 2番目: フィッティングデータ(fitting_data)
    を抽出して返す。

    それぞれ [magnetic_field_list, voltage_list] の2要素リスト。
    """
    series = read_hanle_data(file_path)
    if len(series) < 2:
        raise ValueError("n_onlyデータ抽出に必要な2セクションが見つかりませんでした")
    # 注: exp と fitting の順序が逆（0: fitting, 1: exp）
    exp_data = series[1]
    fitting_data = series[0]
    return exp_data, fitting_data


def plot_hanle_broad_and_narrow(
    broad_exp_data_series: Sequence[List[List[float]]],
    broad_fitting_data_series: Sequence[List[List[float]]],
    broad_broad_fitting_series: Sequence[List[List[float]]],
    narrow_exp_data_series: Sequence[List[List[float]]],
    narrow_fitting_data_series: Sequence[List[List[float]]],
    *,
    labels: Sequence[str] | None = None,
    grid: bool = True,
    figsize: tuple[float, float] = (20, 8),
    xlim: tuple[float, float] | None = None,
    broad_xlim: tuple[float, float] | None = (-3000.0, 3000.0),
    narrow_xlim: tuple[float, float] | None = (-300.0, 300.0),
    ylim: tuple[float, float] | None = None,
):
    """
    5つのシリーズ群を受け取り、
    - broad_* の3系列（exp, fitting, broad_fitting）を同一グラフ
    - narrow_* の2系列（exp, fitting）を同一グラフ
    に描画する。

    仕様:
    - 各シリーズ要素は [magnetic_field_list, voltage_list]
    - fitting系はマーカーなし（線のみ）
    - 同じインデックスのシリーズは同色（exp/fitting/broad_fittingで統一）
    - ラベルは label + "_exp_data" / "_fitting_data" / "_broad_fitting" を付加
      （labelは labels を使用。未指定時は "series{idx}"）

    戻り値:
    (fig, broad_ax, narrow_ax)
    """

    # ベース色（シリーズ間の色）
    base_colors: list[str] = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black",
    ]
    def base_color(j: int) -> str:
        return base_colors[j % len(base_colors)]

    # exp と fitting で色を分けるための色補助（同シリーズで色系統を変える）
    # 例: exp はベース色、fitting は少し暗めの同系色にするため alpha/line style だけでなく色も調整
    def variant_color(color_name: str) -> str:
        # matplotlib の既定色名に対してダークトーンを選ぶのは難しいため、
        # シンプルに同色で区別は線種・太さ・alphaで行い、
        # 要件の「expとfittingで色を分ける」に対応するため、
        # 既定の 'black' を fitting 時に用いるケースを避けるための簡易マップを用意
        mapping = {
            "tab:blue": "royalblue",
            "tab:orange": "darkorange",
            "tab:green": "seagreen",
            "tab:red": "firebrick",
            "tab:purple": "indigo",
            "tab:brown": "saddlebrown",
            "tab:pink": "deeppink",
            "tab:gray": "dimgray",
            "tab:olive": "darkolivegreen",
            "tab:cyan": "teal",
            "black": "dimgray",
        }
        return mapping.get(color_name, color_name)

    # 1つの図に broad / narrow を横並びで描画
    fig, (broad_ax, narrow_ax) = plt.subplots(1, 2, figsize=figsize, squeeze=True)
    num_broad = max(
        len(broad_exp_data_series),
        len(broad_fitting_data_series),
        len(broad_broad_fitting_series),
    )
    # シリーズ数が1本のみのときは exp を黒で描画する
    num_narrow_for_flag = max(
        len(narrow_exp_data_series),
        len(narrow_fitting_data_series),
    )
    single_series = (max(num_broad, num_narrow_for_flag) == 1)
    for j in range(num_broad):
        # 同じシリーズで broad/narrow の exp/fitting が同じ色系になるよう、
        # exp と fitting の可視的区別はスタイル（マーカー/線種/alpha/太さ）で行う
        base_col = base_color(j)
        label_base = ((labels[j % len(labels)]) if labels else f"series{j+1}")

        # exp
        if j < len(broad_exp_data_series):
            b_list, v_list = broad_exp_data_series[j]
            broad_ax.plot(
                b_list,
                v_list,
                linestyle="-",
                marker="o",
                markersize=6.0,
                color=("black" if single_series else base_col),
                linewidth=1.2,
                alpha=0.9,
                label=f"{label_base}_exp",
            )

        # fitting（線のみ）
        if j < len(broad_fitting_data_series):
            b_list, v_list = broad_fitting_data_series[j]
            broad_ax.plot(
                b_list,
                v_list,
                linestyle="--",
                marker="none",
                color=variant_color(base_col),
                linewidth=1.8,
                alpha=0.9,
                label=f"{label_base}_fitting",
            )

        # broad_fitting（線のみ）
        if j < len(broad_broad_fitting_series):
            b_list, v_list = broad_broad_fitting_series[j]
            broad_ax.plot(
                b_list,
                v_list,
                linestyle=":",
                marker="none",
                color=variant_color(base_col),
                linewidth=1.6,
                alpha=0.9,
                label=f"{label_base}_broad_fitting",
            )

    broad_ax.set_xlabel("B (Oe)", fontsize=17)
    broad_ax.set_ylabel("V (uV)", fontsize=17)
    broad_ax.tick_params(labelsize=17, direction="in")
    broad_ax.set_box_aspect(1)
    # xlim の適用: broad は broad_xlim を優先、未指定時に xlim を適用
    if broad_xlim is not None:
        broad_ax.set_xlim(broad_xlim)
    elif xlim is not None:
        broad_ax.set_xlim(xlim)
    if ylim is not None:
        broad_ax.set_ylim(ylim)
    if grid:
        broad_ax.grid(True, linestyle=":", alpha=0.4)
    if any(broad_ax.get_legend_handles_labels()[0]):
        # 1列6件まで
        legend_cols = math.ceil(num_broad / 6) if num_broad > 0 else 1
        broad_ax.legend(frameon=False, fontsize=13, ncol=legend_cols)
    # narrow 側
    num_narrow = max(
        len(narrow_exp_data_series),
        len(narrow_fitting_data_series),
    )
    for j in range(num_narrow):
        base_col = base_color(j)
        label_base = ((labels[j % len(labels)]) if labels else f"series{j+1}")

        # exp
        if j < len(narrow_exp_data_series):
            b_list, v_list = narrow_exp_data_series[j]
            narrow_ax.plot(
                b_list,
                v_list,
                linestyle="-",
                marker="o",
                markersize=6.0,
                color=("black" if single_series else base_col),
                linewidth=1.2,
                alpha=0.9,
                label=f"{label_base}_exp",
            )

        # fitting（線のみ）
        if j < len(narrow_fitting_data_series):
            b_list, v_list = narrow_fitting_data_series[j]
            narrow_ax.plot(
                b_list,
                v_list,
                linestyle="--",
                marker="none",
                color=variant_color(base_col),
                linewidth=1.8,
                alpha=0.9,
                label=f"{label_base}_fitting",
            )

    narrow_ax.set_xlabel("B (Oe)", fontsize=17)
    narrow_ax.set_ylabel("V (uV)", fontsize=17)
    narrow_ax.tick_params(labelsize=17, direction="in")
    narrow_ax.set_box_aspect(1)
    # xlim の適用: narrow は narrow_xlim を優先、未指定時に xlim を適用
    if narrow_xlim is not None:
        narrow_ax.set_xlim(narrow_xlim)
    elif xlim is not None:
        narrow_ax.set_xlim(xlim)
    if ylim is not None:
        narrow_ax.set_ylim(ylim)
    if grid:
        narrow_ax.grid(True, linestyle=":", alpha=0.4)
    if any(narrow_ax.get_legend_handles_labels()[0]):
        legend_cols = math.ceil(num_narrow / 6) if num_narrow > 0 else 1
        narrow_ax.legend(frameon=False, fontsize=13, ncol=legend_cols)
    fig.tight_layout()
    return fig, broad_ax, narrow_ax

def plot_hanle_by_index(
    label_indices: Union[int, Sequence[int]],
    magnetic_field_series: Sequence[Sequence[float]] | None = None,
    voltage_series: Sequence[Sequence[float]] | None = None,
    marker_choice: Union[int, str, Sequence[Union[int, str]]] = 0,
    linestyle_choice: Union[int, str, Sequence[Union[int, str]]] = 0,
    size_choice: Union[int, float, Sequence[Union[int, float]]] = 0.8,
    color_choice: Union[int, str, Sequence[Union[int, str]]] = "auto",
    title: str = "Hanle (uV vs Oe)",
    grid: bool = True,
    figsize: tuple[float, float] = (10, 8),
    xlim: tuple[float, float]=(-3000, 3000),
    ylim: tuple[float, float] | None = None,
    legend_labels: Sequence[str] | None = None,
):
    """
    Hanleの生データ系列から、指定したラベル番号(1始まり)のものを選んで
    同一座標上に Voltage(uV) vs Magnetic Field (Oe) を重ね描きする。

    - label_indices: 1始まりの番号(例: 1 または [1,3])
    - magnetic_field_series, voltage_series: 各シリーズ(配列の配列)
      例) magnetic_field_series = [b_1, b_2, ...], voltage_series = [v_1, v_2, ...]
    - marker_choice / linestyle_choice / size_choice / color_choice:
      単一指定(全系列共通) または シーケンス指定(系列ごと設定)。
      color_choice に "auto" を含む場合は自動配色
    - figsize: 図サイズ
    - xlim/ylim: 軸範囲 (Noneなら自動)
    - legend_labels: 凡例ラベル(系列ごとに循環適用)。未指定時は自動ラベル

    戻り値: (fig, ax)
    """

    # スタイル候補
    marker_options: list[str] = ["o", "s", "^", "D", "x", "+", "*", ".", "v", "P", "h"]
    linestyle_options: list[str] = ["-", "--", ":", "-.", "none"]
    size_options: list[float] = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    color_options: list[str] = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black",
    ]

    def _resolve_from_options(choice, options, *, allow_auto: bool = False):
        if isinstance(choice, int):
            return options[choice % len(options)]
        if isinstance(choice, str):
            if allow_auto and choice == "auto":
                return "auto"
            if choice in options:
                return choice
            return choice
        return options[0]

    def _resolve_size(choice) -> float:
        if isinstance(choice, int):
            return float(size_options[choice % len(size_options)])
        if isinstance(choice, (float,)):
            return float(choice)
        return float(size_options[0])

    def _choice_for_series(choice_or_seq, j: int):
        if isinstance(choice_or_seq, (list, tuple)):
            return choice_or_seq[j % len(choice_or_seq)]
        return choice_or_seq

    # データ整合チェック
    if magnetic_field_series is None or voltage_series is None:
        raise ValueError("magnetic_field_series, voltage_series を指定してください")
    if not (len(magnetic_field_series) == len(voltage_series)):
        raise ValueError("magnetic_field_series と voltage_series の長さが一致していません")

    # ラベル番号を 1始まり → 0始まり に変換
    if isinstance(label_indices, int):
        idx_list = [label_indices]
    else:
        idx_list = list(label_indices)
    zero_based_indices = [i - 1 for i in idx_list]

    # 図準備
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=True)

    # 描画
    for j, idx in enumerate(zero_based_indices):
        if idx < 0 or idx >= len(magnetic_field_series):
            raise IndexError(f"label index {idx+1} は範囲外です (1..{len(magnetic_field_series)})")

        b_vals = magnetic_field_series[idx]
        v_vals = voltage_series[idx]

        # 系列ごとのスタイル解決
        marker_j = _resolve_from_options(_choice_for_series(marker_choice, j), marker_options)
        linestyle_j = _resolve_from_options(_choice_for_series(linestyle_choice, j), linestyle_options)
        size_j = _resolve_size(_choice_for_series(size_choice, j))
        color_sel = _choice_for_series(color_choice, j)
        color_j = _resolve_from_options(color_sel, color_options, allow_auto=True)
        if color_j == "auto":
            color_j = color_options[j % len(color_options)]

        # ラベル: 指定があれば循環適用、なければ自動
        if legend_labels and len(legend_labels) > 0:
            label = legend_labels[j % len(legend_labels)]
        else:
            label = f"label {idx+1}"

        # Voltage (uV) vs Magnetic Field (Oe)
        ax.plot(
            b_vals, v_vals,
            linestyle=linestyle_j,
            marker=marker_j,
            markersize=size_j,
            color=color_j,
            linewidth=1.2,
            alpha=0.9,
            label=label,
        )

    ax.set_xlabel("B (Oe)", fontsize=17)
    ax.set_ylabel("V (uV)", fontsize=17)
    ax.tick_params(labelsize=17, direction="in")
    ax.set_box_aspect(1)

    # 軸範囲の適用（指定があれば）
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(True, linestyle=":", alpha=0.4)

    # 凡例列数 (1列あたり6個まで)
    num_plots = len(zero_based_indices)
    legend_cols = math.ceil(num_plots / 6) if num_plots > 0 else 1
    if any(ax.get_legend_handles_labels()[0]):
        ax.legend(frameon=False, fontsize=13, ncol=legend_cols)

    fig.tight_layout()
    return fig, ax

