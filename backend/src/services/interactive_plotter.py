from typing import Sequence, Tuple, Union, Any, Dict, List
import plotly.graph_objects as go
import json
import math
from ..models.analysis_types import RAPsSeries

def create_ps_ra_plot(
    series_list: List[RAPsSeries],
    title: str = "RA vs Ps",
    xlabel: str = "RA (ohm um^2)",
    ylabel: str = "Ps (%)",
    xscale: str = "log",
    yscale: str = "linear",
    xlim: Union[Tuple[float, float], None] = None,
    ylim: Union[Tuple[float, float], None] = None,
) -> Dict[str, Any]:
    """
    Generate Plotly figure for RA vs Ps plot.
    Returns the figure as a dict (compatible with plotly.js).
    """
    fig = go.Figure()

    for series in series_list:
        if not series.points:
            continue
            
        ra_list = [p.ra for p in series.points]
        ps_list = [p.ps for p in series.points]
        rms_list = [p.rms for p in series.points]
        # Use point label if available, otherwise empty string
        point_labels = [p.label if p.label else "" for p in series.points]
        
        fig.add_trace(go.Scatter(
            x=ra_list,
            y=ps_list,
            mode='markers',
            name=series.label,
            customdata=point_labels,
            error_y=dict(
                type='data',
                array=rms_list,
                visible=True,
                color=series.color,
                thickness=1.5,
                width=3
            ),
            marker=dict(
                color=series.color,
                size=10,
                symbol='circle'
            ),
            hovertemplate=(
                f"<b>{series.label}</b><br>" +
                "Label: %{customdata}<br>" +
                "RA: %{x:.2f}<br>" +
                "Ps: %{y:.2f}%<br>" +
                "RMS: %{error_y.array:.2f}<extra></extra>"
            )
        ))

    # Layout configuration
    axis_type_x = "log" if xscale == "log" else "linear"
    axis_type_y = "log" if yscale == "log" else "linear"

    layout_update = dict(
        title=title,
        xaxis=dict(
            title=xlabel,
            type=axis_type_x,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title=ylabel,
            type=axis_type_y,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        plot_bgcolor='white',
        hovermode='closest',
        width=800,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )
    
    if xlim:
        layout_update['xaxis']['range'] = [math.log10(x) for x in xlim] if axis_type_x == "log" else xlim
        # Note: Plotly log axis range is in log10 units if specific range is set? 
        # Actually usually it's just value if not manually setting range array.
        # Constructing range for log axis in plotly can be tricky (log10 vs value).
        # We will omit manual range for now to let autoscaling work, unless strict requirement.
        pass

    if ylim:
        layout_update['yaxis']['range'] = ylim

    fig.update_layout(**layout_update)

    return json.loads(fig.to_json())

def create_iv_plot(
    iv_data: Any, # Placeholder type
    title: str = "IV Characteristics"
) -> Dict[str, Any]:
    # Placeholder
    fig = go.Figure()
    fig.update_layout(title=title)
    return json.loads(fig.to_json())

def create_hanle_plot(
    hanle_data: Any,
    title: str = "Hanle Effect"
) -> Dict[str, Any]:
    fig.update_layout(title=title)
    return json.loads(fig.to_json())

from ..models.analysis_types import LogRAVSeries

def create_log_ra_v_plot(
    series_list: List[LogRAVSeries],
    title: str = "Log RA vs V",
    xlabel: str = "Voltage (mV)",
    ylabel: str = "RA (ohm um^2)",
) -> Dict[str, Any]:
    """
    Generate Plotly figure for Log RA vs V plot.
    """
    fig = go.Figure()

    # Group series by group_label to handle legend
    seen_groups = set()
    
    for series in series_list:
        if not series.vd_mV or not series.ra_ohm_um2:
            continue
            
        # Filter out points in range -5mV to 5mV
        x_data = []
        y_data = []
        for x, y in zip(series.vd_mV, series.ra_ohm_um2):
            if not (-5 <= x <= 5):
                x_data.append(x)
                y_data.append(y)

        if not x_data:
            continue
            
        show_legend = False
        if series.group_label not in seen_groups:
            show_legend = True
            seen_groups.add(series.group_label)
            
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers', 
            name=series.group_label if show_legend else series.label, # Use group label for legend item
            legendgroup=series.group_label, # Group all traces together
            showlegend=show_legend,
            line=dict(color=series.color, width=1),
            marker=dict(size=4),
            hovertemplate=(
                f"<b>{series.label}</b><br>" +
                f"Group: {series.group_label}<br>" +
                "V: %{x:.1f} mV<br>" +
                "RA: %{y:.2g} ohm um^2<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=xlabel,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title=ylabel,
            type="log", # Requested Log RA
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        plot_bgcolor='white',
        hovermode='closest',
        width=1000,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )

    return json.loads(fig.to_json())
