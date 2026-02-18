from typing import List, Tuple, Union, Sequence, Optional
import math
import matplotlib.pyplot as plt

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
    base_colors: list[str] = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black",
    ]
    def base_color(j: int) -> str:
        return base_colors[j % len(base_colors)]

    fig, (broad_ax, narrow_ax) = plt.subplots(1, 2, figsize=figsize, squeeze=True)
    
    # Plotting logic for broad
    num_broad = max(len(broad_exp_data_series), len(broad_fitting_data_series), len(broad_broad_fitting_series))
    for j in range(num_broad):
        base_col = base_color(j)
        label_base = ((labels[j % len(labels)]) if labels else f"series{j+1}")

        if j < len(broad_exp_data_series):
            b_list, v_list = broad_exp_data_series[j]
            broad_ax.plot(b_list, v_list, "-", marker="o", markersize=6.0, color=base_col, alpha=0.9, label=f"{label_base}_exp")
            
        if j < len(broad_fitting_data_series):
            b_list, v_list = broad_fitting_data_series[j]
            broad_ax.plot(b_list, v_list, "--", color=base_col, alpha=0.9, label=f"{label_base}_fit")

        if j < len(broad_broad_fitting_series):
            b_list, v_list = broad_broad_fitting_series[j]
            broad_ax.plot(b_list, v_list, ":", color=base_col, alpha=0.9, label=f"{label_base}_broad_fit")

    broad_ax.set_title("Hanle Broad")
    if broad_xlim: broad_ax.set_xlim(broad_xlim)
    if grid: broad_ax.grid(True, linestyle=":", alpha=0.4)
    broad_ax.legend()

    # Plotting logic for narrow
    num_narrow = max(len(narrow_exp_data_series), len(narrow_fitting_data_series))
    for j in range(num_narrow):
        base_col = base_color(j)
        label_base = ((labels[j % len(labels)]) if labels else f"series{j+1}")

        if j < len(narrow_exp_data_series):
            b_list, v_list = narrow_exp_data_series[j]
            narrow_ax.plot(b_list, v_list, "-", marker="o", markersize=6.0, color=base_col, alpha=0.9, label=f"{label_base}_exp")

        if j < len(narrow_fitting_data_series):
            b_list, v_list = narrow_fitting_data_series[j]
            narrow_ax.plot(b_list, v_list, "--", color=base_col, alpha=0.9, label=f"{label_base}_fit")

    narrow_ax.set_title("Hanle Narrow")
    if narrow_xlim: narrow_ax.set_xlim(narrow_xlim)
    if grid: narrow_ax.grid(True, linestyle=":", alpha=0.4)
    narrow_ax.legend()

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
    if isinstance(label_indices, int):
        idx_list = [label_indices]
    else:
        idx_list = list(label_indices)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for j, idx in enumerate(idx_list):
        real_idx = idx - 1
        if real_idx < 0 or real_idx >= len(magnetic_field_series):
            continue
            
        b_vals = magnetic_field_series[real_idx]
        v_vals = voltage_series[real_idx]
        
        label = legend_labels[j] if legend_labels and j < len(legend_labels) else f"label {idx}"
        ax.plot(b_vals, v_vals, label=label)

    ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if grid: ax.grid(True)
    ax.legend()
    
    return fig, ax

def calc_hanle_n_only_rms_from_files(file_paths: Sequence[str]) -> List[float]:
    # This logic involves loading, but since it's a calculation utility specific to hanle analysis
    # and uses a specific loader, we can import loader here or keep it.
    # To separate concerns properly, data should be passed in, but strictly following legacy signature:
    # We must import read_hanle_n_only from data_loader
    from .data_loader import read_hanle_n_only
    
    rms_list: List[float] = []
    for path in file_paths:
        try:
            exp_data, fitting_data = read_hanle_n_only(path)
            if not exp_data or not fitting_data:
                rms_list.append(float("nan"))
                continue
            
            exp_v = exp_data[1]
            fit_v = fitting_data[1]
            
            n = min(len(exp_v), len(fit_v))
            if n == 0:
                rms_list.append(float("nan"))
                continue

            diff_sq_sum = sum((exp_v[i] - fit_v[i])**2 for i in range(n))
            rms = math.sqrt(diff_sq_sum / n)
            rms_list.append(rms)
        except Exception:
            rms_list.append(float("nan"))
            
    return rms_list
