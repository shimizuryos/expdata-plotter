from fastapi import APIRouter, HTTPException
from ..services.data_loader import load_ps_ra_data
from ..services.interactive_plotter import create_ps_ra_plot, create_iv_plot, create_hanle_plot
import os
import shutil
from fastapi import UploadFile, File
from ..utils import units

router = APIRouter()

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from typing import Optional
from fastapi import UploadFile, File, Form
from ..services.data_loader import (
    read_hanle_raw_data, 
    read_hanle_broad, 
    read_hanle_n_only
)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    hanle_type: Optional[str] = Form(None)
):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    response_data = {"info": f"file '{file.filename}' saved at '{file_location}'"}
    
    if hanle_type:
        try:
            if hanle_type == "raw":
                parsed = read_hanle_raw_data(file_location)
                response_data["hanle_parsed"] = "success"
                # You could extract basic info like len(parsed.voltage_V)
            elif hanle_type == "broad":
                parsed = read_hanle_broad(file_location)
                response_data["hanle_parsed"] = "success" if parsed else "failed"
            elif hanle_type == "narrow":
                parsed = read_hanle_n_only(file_location)
                response_data["hanle_parsed"] = "success" if parsed else "failed"
            else:
                response_data["hanle_parsed"] = "unsupported format"
        except Exception as e:
            response_data["hanle_parsed_error"] = str(e)

    return response_data

@router.get("/plots/ps-ra")
def get_ps_ra_plot():
    # Resolve path relative to this file to avoid CWD dependency
    # File: backend/src/api/endpoints.py
    # Goal: expdata-plotter/data/ps_ra_data.yaml
    # Path: ../../../data/ps_ra_data.yaml (relative to file)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    data_path = os.path.join(project_root, "data/ps_ra_data.yaml")
    
    if not os.path.exists(data_path):
         # Try fallback for local dev if structure differs
         # e.g. if running from root
         fallback = os.path.join(os.getcwd(), "data/ps_ra_data.yaml")
         if os.path.exists(fallback):
             data_path = fallback
             
    if not os.path.exists(data_path):
         raise HTTPException(status_code=404, detail=f"Data file not found at {data_path}")

    # load_ps_ra_data now returns List[RAPsSeries]
    # create_ps_ra_plot accepts list[RAPsSeries]
    series_list = load_ps_ra_data(data_path)
    if not series_list:
        raise HTTPException(status_code=404, detail="No data found in yaml")
        
    fig_json = create_ps_ra_plot(series_list)
    
    # Also return series metadata for the fit UI
    series_meta = [{
        "label": "all",
        "color": "black",
        "r_p": series_list[0].r_p_ohm if series_list else 0,
        "n_points": sum(len(s.points) for s in series_list),
    }]
    for s in series_list:
        series_meta.append({
            "label": s.label,
            "color": s.color,
            "r_p": s.r_p_ohm,
            "n_points": len(s.points),
        })
    return {"plot": fig_json, "series": series_meta}


# ─── Ps-RA Fitting ───────────────────────────
from pydantic import BaseModel
from typing import Optional
import uuid
import threading
from ..services.ps_ra_fitting import (
    run_fitting, generate_fit_curves, calc_tox_from_Tp,
    get_fit_progress, clear_fit_progress, model_Ps_RA,
)
import numpy as np
import plotly.graph_objects as go
import json


class PsRaFitRequest(BaseModel):
    series_label: str
    
    fix_ps_a: bool = True
    fix_ps_b: bool = True
    fix_lam_a: bool = False
    fix_lam_b: bool = False
    fix_c: bool = False
    fix_d_b: bool = False
    
    init_ps_a: float = 0.5
    init_ps_b: float = 0.3
    init_lam_a: float = 1e-9
    init_lam_b: float = 5e-9
    init_c: float = 1.0
    init_d_b: float = 1.0
    
    V_B: float = 0.1
    weight_ratio: float = 1.0  # σ_lnR / σ_P


# In-memory store for completed fit results (job_id -> result)
_fit_results: dict = {}
_fit_results_lock = threading.Lock()


def _resolve_data_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    data_path = os.path.join(project_root, "data/ps_ra_data.yaml")
    if not os.path.exists(data_path):
        fallback = os.path.join(os.getcwd(), "data/ps_ra_data.yaml")
        if os.path.exists(fallback):
            data_path = fallback
    return data_path


def _build_fit_plots(
    series_list, target_series, params, fit_curves, body: PsRaFitRequest,
):
    """Build 3 Plotly plots: Ps-tox, RA-tox, Ps-RA."""
    # Data points
    pts = target_series.points
    tox_data = calc_tox_from_Tp(np.array([units.s_to_min(p.tp_s) for p in pts]))
    ra_data = np.array([units.convert_RA_ohm_m2_to_ohm_um2(p.ra_ohm_m2) for p in pts])
    ps_data = np.array([p.ps for p in pts])
    area_data = np.array([units.convert_area_m2_to_um2(p.area_m2) for p in pts])
    labels = [p.label or "" for p in pts]
    
    # Correct RA for parasitic resistance
    ra_corr = ra_data - target_series.r_p_ohm * area_data

    tox_fit = np.array(fit_curves["tox"])
    ps_fit = np.array(fit_curves["Ps"])
    ra_fit = np.array(fit_curves["RA"])

    # ── Plot 1: Ps vs t_ox ──
    fig_ps_tox = go.Figure()
    fig_ps_tox.add_trace(go.Scatter(
        x=(tox_data * 1e9).tolist(), y=ps_data.tolist(),
        mode="markers", name=target_series.label,
        marker=dict(color=target_series.color, size=8),
        text=labels, hovertemplate="<b>%{text}</b><br>t_ox: %{x:.2f} nm<br>Ps: %{y:.3f}<extra></extra>",
    ))
    fig_ps_tox.add_trace(go.Scatter(
        x=(tox_fit * 1e9).tolist(), y=ps_fit.tolist(),
        mode="lines", name="Fit",
        line=dict(color="black", dash="dash"),
    ))
    fig_ps_tox.update_layout(
        title="Ps vs t_ox", xaxis_title="t_ox (nm)", yaxis_title="Ps",
        plot_bgcolor="white", hovermode="closest",
    )

    # ── Plot 2: RA vs t_ox (log scale) ──
    fig_ra_tox = go.Figure()
    fig_ra_tox.add_trace(go.Scatter(
        x=(tox_data * 1e9).tolist(), y=ra_corr.tolist(),
        mode="markers", name=target_series.label,
        marker=dict(color=target_series.color, size=8),
        text=labels, hovertemplate="<b>%{text}</b><br>t_ox: %{x:.2f} nm<br>RA: %{y:.1f}<extra></extra>",
    ))
    fig_ra_tox.add_trace(go.Scatter(
        x=(tox_fit * 1e9).tolist(), y=ra_fit,
        mode="lines", name="Fit",
        line=dict(color="black", dash="dash"),
    ))
    fig_ra_tox.update_layout(
        title="RA vs t_ox", xaxis_title="t_ox (nm)",
        yaxis_title="RA (Ω·μm²)", yaxis_type="log",
        plot_bgcolor="white", hovermode="closest",
    )

    # ── Plot 3: Ps vs RA (original scatter + fit curve) ──
    fig_ps_ra = go.Figure()
    # Add all series as scatter
    for s in series_list:
        ra_list = [units.convert_RA_ohm_m2_to_ohm_um2(p.ra_ohm_m2) for p in s.points]
        ps_list = [p.ps for p in s.points]
        rms_list = [p.rms for p in s.points]
        s_labels = [p.label or "" for p in s.points]
        fig_ps_ra.add_trace(go.Scatter(
            x=ra_list, y=ps_list, mode="markers", name=s.label,
            customdata=s_labels,
            error_y=dict(type="data", array=rms_list, visible=True, color=s.color),
            marker=dict(color=s.color, size=10),
            hovertemplate=(
                f"<b>{s.label}</b><br>Label: %{{customdata}}<br>"
                "RA: %{x:.2f}<br>Ps: %{y:.3f}<extra></extra>"
            ),
        ))
    # Add fit curve (Ps vs RA parametric)
    fig_ps_ra.add_trace(go.Scatter(
        x=ra_fit, y=ps_fit.tolist(),
        mode="lines", name="Fit Curve",
        line=dict(color="black", width=2, dash="dash"),
    ))
    fig_ps_ra.update_layout(
        title="Ps vs RA (with Fit)", xaxis_title="RA (Ω·μm²)",
        yaxis_title="Ps", xaxis_type="log",
        plot_bgcolor="white", hovermode="closest",
    )

    return {
        "ps_tox": json.loads(fig_ps_tox.to_json()),
        "ra_tox": json.loads(fig_ra_tox.to_json()),
        "ps_ra": json.loads(fig_ps_ra.to_json()),
    }


class PsRaPreviewRequest(BaseModel):
    series_label: str
    init_ps_a: float = 0.5
    init_ps_b: float = 0.3
    init_lam_a: float = 1e-9
    init_lam_b: float = 5e-9
    init_c: float = 1.0
    init_d_b: float = 1.0
    V_B: float = 0.1
    weight_ratio: float = 1.0


@router.post("/plots/ps-ra/preview")
def preview_ps_ra_model(body: PsRaPreviewRequest):
    """Evaluate model with given params and overlay on data (no fitting)."""
    data_path = _resolve_data_path()
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data file not found")

    series_list = load_ps_ra_data(data_path)

    # Resolve target series
    if body.series_label == "all":
        from ..models.analysis_types import RAPsSeries
        all_points = []
        r_p_val = series_list[0].r_p_ohm if series_list else 0.0
        for s in series_list:
            all_points.extend(s.points)
        target = RAPsSeries(points=all_points, label="all", color="black", r_p_ohm=r_p_val)
    else:
        target = None
        for s in series_list:
            if s.label == body.series_label:
                target = s
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Series '{body.series_label}' not found")

    params = {
        "D_A": 1.0, "D_B": body.init_d_b,
        "lambda_A": body.init_lam_a, "lambda_B": body.init_lam_b,
        "C": body.init_c,
        "P_S_A": body.init_ps_a, "P_S_B": body.init_ps_b
    }

    # Calculate current cost and residual for the UI
    from ..services.ps_ra_fitting import LikelihoodCalculator
    pts = target.points
    Tp_arr = np.array([units.s_to_min(p.tp_s) for p in pts])
    tox_arr_all = calc_tox_from_Tp(Tp_arr)
    Ps_arr = np.array([p.ps for p in pts])
    RA_arr = np.array([units.convert_RA_ohm_m2_to_ohm_um2(p.ra_ohm_m2) for p in pts])
    A_arr = np.array([units.convert_area_m2_to_um2(p.area_m2) for p in pts])
    RA_corr_arr = RA_arr - target.r_p_ohm * A_arr

    valid_mask = np.isfinite(RA_corr_arr) & (RA_corr_arr > 0) & np.isfinite(Ps_arr)
    tox_v = tox_arr_all[valid_mask]
    Ps_v = Ps_arr[valid_mask]
    lnRA_v = np.log(RA_corr_arr[valid_mask])

    calc = LikelihoodCalculator(
        tox_v=tox_v, Ps_v=Ps_v, lnRA_v=lnRA_v,
        V_B=body.V_B, sigma_P=1.0, sigma_lnR=body.weight_ratio,
        fix_flags={"P_S_A": True, "P_S_B": True, "lambda_A": True, "lambda_B": True, "C": True, "D_B": True},
        init_vals={"P_S_A": body.init_ps_a, "P_S_B": body.init_ps_b, "lambda_A": body.init_lam_a, "lambda_B": body.init_lam_b, "C": body.init_c, "D_B": body.init_d_b, "D_A": 1.0}
    )
    res_fun = calc.residuals(np.array([]))
    cost = float(0.5 * np.sum(res_fun**2))
    residual_norm = float(np.linalg.norm(res_fun))

    # Generate curves over full t_ox range from all series
    all_tp = []
    for s in series_list:
        all_tp.extend([units.s_to_min(p.tp_s) for p in s.points])
    tox_all = calc_tox_from_Tp(np.array(all_tp))
    tox_min = float(tox_all.min()) * 0.2
    tox_max = float(tox_all.max()) * 2.0

    fit_curves = generate_fit_curves(
        params, (tox_min, tox_max), n_points=500, V_B=body.V_B,
    )

    # Reuse the same plot builder with a dummy PsRaFitRequest
    dummy_body = PsRaFitRequest(
        series_label=body.series_label, V_B=body.V_B,
    )
    plots = _build_fit_plots(series_list, target, params, fit_curves, dummy_body)

    return {
        "status": "preview",
        "params": params,
        "info": {
            "cost": cost,
            "residual_norm": residual_norm,
            "nfev": 1
        },
        "plots": plots,
    }


@router.post("/plots/ps-ra/fit")
def start_ps_ra_fit(body: PsRaFitRequest):
    """Start async Ps-RA fitting job. Returns job_id for polling."""
    data_path = _resolve_data_path()
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data file not found")

    series_list = load_ps_ra_data(data_path)
    
    # Handle "all" - merge all series into one virtual target
    if body.series_label == "all":
        from ..models.analysis_types import RAPsSeries, RAPsPoint
        all_points = []
        r_p_val = series_list[0].r_p_ohm if series_list else 0.0
        for s in series_list:
            all_points.extend(s.points)
        if len(all_points) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 data points")
        target = RAPsSeries(points=all_points, label="all", color="black", r_p_ohm=r_p_val)
    else:
        target = None
        for s in series_list:
            if s.label == body.series_label:
                target = s
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Series '{body.series_label}' not found")
        if len(target.points) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 data points")

    job_id = str(uuid.uuid4())[:8]

    def _run():
        pts = target.points
        Tp = np.array([units.s_to_min(p.tp_s) for p in pts])
        Ps = np.array([p.ps for p in pts])
        RA = np.array([units.convert_RA_ohm_m2_to_ohm_um2(p.ra_ohm_m2) for p in pts])
        A  = np.array([units.convert_area_m2_to_um2(p.area_m2) for p in pts])

        fix_flags = {
            "P_S_A": body.fix_ps_a, "P_S_B": body.fix_ps_b,
            "lambda_A": body.fix_lam_a, "lambda_B": body.fix_lam_b,
            "C": body.fix_c, "D_B": body.fix_d_b,
        }
        init_vals = {
            "P_S_A": body.init_ps_a, "P_S_B": body.init_ps_b,
            "lambda_A": body.init_lam_a, "lambda_B": body.init_lam_b,
            "C": body.init_c, "D_B": body.init_d_b,
            "D_A": 1.0,
        }

        try:
            params, info = run_fitting(
                Tp, Ps, RA, A,
                R_para_ohm=target.r_p_ohm,
                V_B=body.V_B,
                weight_ratio=body.weight_ratio,
                fix_flags=fix_flags,
                init_vals=init_vals,
                job_id=job_id,
            )

            # Generate fit curves over extended t_ox range
            # Use ALL series data to determine range for wider coverage
            all_tp = []
            for s in series_list:
                all_tp.extend([units.s_to_min(p.tp_s) for p in s.points])
            tox_all = calc_tox_from_Tp(np.array(all_tp))
            tox_min = float(tox_all.min()) * 0.2
            tox_max = float(tox_all.max()) * 2.0
            fit_curves = generate_fit_curves(
                params, (tox_min, tox_max), n_points=500, V_B=body.V_B,
            )

            plots = _build_fit_plots(series_list, target, params, fit_curves, body)

            with _fit_results_lock:
                _fit_results[job_id] = {
                    "status": "done",
                    "params": params,
                    "info": info,
                    "plots": plots,
                }
        except Exception as e:
            with _fit_results_lock:
                _fit_results[job_id] = {"status": "error", "error": str(e)}

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"job_id": job_id}


@router.get("/plots/ps-ra/fit/{job_id}")
def get_ps_ra_fit_result(job_id: str):
    """Poll for fit progress / result."""
    # Check completed results first
    with _fit_results_lock:
        result = _fit_results.get(job_id)
    if result:
        return result

    # Check progress
    progress = get_fit_progress(job_id)
    if progress:
        return progress

    raise HTTPException(status_code=404, detail="Job not found")

@router.get("/plots/iv")
def get_iv_plot():
    # Placeholder: In future, load data/exp_data.csv or similar
    fig_json = create_iv_plot(None)
    return fig_json

@router.get("/plots/hanle")
def get_hanle_plot():
    # Placeholder
    fig_json = create_hanle_plot(None)
    return fig_json

from ..services.data_loader import load_log_ra_v_data
from ..services.interactive_plotter import create_log_ra_v_plot
import yaml

@router.get("/plot/{plot_id}")
def get_plot(plot_id: str):
    """
    Generic endpoint to load a plot by ID (key in YAML).
    Searches known YAML files for the key.
    """
    # Define known data files to search
    # In a real app, maybe scan the directory.
    # For now, we know the files.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    data_dir = os.path.join(project_root, "data")
    
    # Files to check
    yaml_files = ["ps_ra_data.yaml", "iv_plot_data.yaml"]
    
    found_file = None
    plot_type = None
    
    for yf in yaml_files:
        path = os.path.join(data_dir, yf)
        if not os.path.exists(path):
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
                if payload and plot_id in payload:
                    found_file = path
                    plot_data = payload[plot_id]
                    plot_type = plot_data.get("plot_type")
                    break
        except Exception:
            continue
            
    if not found_file:
         raise HTTPException(status_code=404, detail=f"Plot ID '{plot_id}' not found in data files")

    if plot_type == "ps-ra":
        # Current logic for ps-ra is slightly specific to a structure, 
        # but let's assume we can reuse the loader if adapted, or keep the old endpoint for that.
        # The user's request for "log_ra_v" is the main focus.
        # If plot_type is ps-ra, we might need to adapt `load_ps_ra_data` to take a key?
        # The existing `load_ps_ra_data` loads the WHOLE file and returns a list of series.
        # It doesn't filter by key. 
        # But `ps_ra_data.yaml` has `cofe-ps-ra` as the key.
        # Let's support `log_ra_v` first.
        pass
        
    if plot_type == "log_ra_v":
        series_list = load_log_ra_v_data(found_file, plot_id)
        if not series_list:
             raise HTTPException(
                 status_code=404,
                 detail=(
                     f"No data found for plot '{plot_id}'. "
                     "IV data files may be on an external drive that is not mounted. "
                     "Check that the file_path entries in iv_plot_data.yaml are accessible."
                 ),
             )
        
        # Determine title from ID or config?
        title = plot_id
        fig_json = create_log_ra_v_plot(series_list, title=title)
        
        # User requested PR text? "PR用の文章も一緒に出力"
        # We can return a wrapper object: { plot: fig_json, description: ... }
        # But existing frontend expects pure plotly json? 
        # If I change the return type, I might break frontend if it uses this endpoint.
        # But this is a NEW endpoint `/plot/{plot_id}`.
        
        response = {
            "plot": fig_json,
            "description": f"Log RA vs V plot for {plot_id}",
            "plot_type": plot_type
        }
        return response

    raise HTTPException(status_code=400, detail=f"Unknown or unsupported plot type: {plot_type}")
