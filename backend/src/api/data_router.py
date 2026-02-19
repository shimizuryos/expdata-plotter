from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import hashlib
from datetime import datetime

from ..models.db_models import Sample, DeviceGroup, Device, LayerStructure, Measurement
from ..repositories.sample_repository import SampleRepository
from ..repositories.measurement_repository import MeasurementRepository

router = APIRouter(prefix="/data", tags=["data-management"])

sample_repo = SampleRepository()
measurement_repo = MeasurementRepository()

# --- Pydantic Models for Request Body ---

class LayerStructureModel(BaseModel):
    material: str
    thick_nm_variation: Optional[List[float]] = None

class CreateSampleRequest(BaseModel):
    id: str
    name: str
    device_type: str = "three_terminal_hanle"
    structures: List[LayerStructureModel]
    note: str = ""

class VariationModel(BaseModel):
    suffix: str
    area_um2: float

class CreateDeviceGroupRequest(BaseModel):
    coord: List[int] # [x, y]
    thick_nm: Dict[str, float]
    shared_properties: Dict[str, Any]
    group_class: str = "single" # "single" or "area_variation"
    variations: List[VariationModel]

class CreateMeasurementRequest(BaseModel):
    sample_id: str
    device_id: str
    measurement_type: str # "IV", "Hanle"
    metadata: Dict[str, Any]
    file_ref: str
    derived: Optional[Dict[str, Any]] = None
    set_as_default: bool = False

# --- Endpoints ---

@router.get("/samples")
def list_samples():
    """
    List all samples (lightweight view).
    """
    samples = sample_repo.list_all()
    # Return lightweight list to avoid transferring huge JSON
    return [
        {
            "id": s.id,
            "name": s.name,
            "device_type": s.device_type,
            "note": s.note,
            "created_at": None # To implement if storing in DB row creation time
        }
        for s in samples
    ]

@router.post("/samples", status_code=201)
def create_sample(req: CreateSampleRequest):
    """
    Create a new sample.
    """
    if sample_repo.get_by_id(req.id):
        raise HTTPException(status_code=400, detail="Sample ID already exists")

    structures = [
        LayerStructure(material=s.material, thick_nm_variation=s.thick_nm_variation)
        for s in req.structures
    ]

    sample = Sample(
        id=req.id,
        name=req.name,
        device_type=req.device_type,
        structures=structures,
        device_groups=[],
        note=req.note
    )
    
    sample_repo.insert(sample)
    return {"id": sample.id}

@router.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    """
    Get full sample details including device groups.
    """
    sample = sample_repo.get_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample

@router.get("/samples/{sample_id}/device-groups")
def list_device_groups(sample_id: str):
    """
    List device groups for a sample.
    """
    sample = sample_repo.get_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample.device_groups

@router.post("/samples/{sample_id}/device-groups", status_code=201)
def create_device_group(sample_id: str, req: CreateDeviceGroupRequest):
    """
    Create a new device group. 
    Supports 'area_variation' to bulk create devices.
    """
    sample = sample_repo.get_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    # Validate coord
    coord_tuple = tuple(req.coord)
    for dg in sample.device_groups:
        if dg.coord == coord_tuple:
             raise HTTPException(status_code=400, detail=f"DeviceGroup at {coord_tuple} already exists")

    # Generate Devices
    devices = []
    
    # Base device ID prefix: "{x}-{y}"
    prefix = f"{req.coord[0]}-{req.coord[1]}"
    
    for v in req.variations:
        # device_id format: "{prefix}-{suffix}"
        if v.suffix:
             d_id = f"{prefix}-{v.suffix}"
        else:
             d_id = prefix # No suffix case
             
        # Check global uniqueness within sample
        existing_devs = sample_repo.get_devices(sample_id)
        if any(d.device_id == d_id for d in existing_devs):
             raise HTTPException(status_code=400, detail=f"Device ID {d_id} already exists in sample")
             
        devices.append(Device(device_id=d_id, area_um2=v.area_um2))

    new_group = DeviceGroup(
        coord=coord_tuple,
        thick_nm=req.thick_nm,
        shared_properties=req.shared_properties,
        devices=devices
    )
    
    # Use repo method (which we added in Phase 2)
    # sample_repo.add_device_group handles the append + save
    # Note: add_device_group in repo implementation appended to list.
    # We can use that or do it manually here since we fetched sample.
    # Let's use repo method to be clean, but we need to re-fetch or use logic.
    # The repo method `add_device_group` fetches sample by ID internally.
    try:
        sample_repo.add_device_group(sample_id, new_group)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "coord": new_group.coord,
        "devices": [{"device_id": d.device_id, "area_um2": d.area_um2} for d in devices]
    }

@router.post("/measurements", status_code=201)
def create_measurement(req: CreateMeasurementRequest):
    """
    Register a measurement.
    """
    # Verify sample/device existence
    sample = sample_repo.get_by_id(req.sample_id)
    if not sample:
         raise HTTPException(status_code=404, detail="Sample not found")
         
    devices = sample_repo.get_devices(req.sample_id)
    if not any(d.device_id == req.device_id for d in devices):
         raise HTTPException(status_code=404, detail="Device not found in sample")

    # Generate ID
    # Hash of key fields + timestamp to ensure uniqueness but determinism for same file?
    # Actually timestamp makes it non-deterministic.
    ts = datetime.now().isoformat()
    raw_str = f"{req.sample_id}_{req.device_id}_{req.measurement_type}_{req.file_ref}_{ts}"
    meas_id = hashlib.sha256(raw_str.encode()).hexdigest()[:16]
    
    measurement = Measurement(
        id=meas_id,
        sample_id=req.sample_id,
        device_id=req.device_id,
        measurement_type=req.measurement_type,
        metadata=req.metadata,
        data=None, # File reference only, data loaded on demand usually
        derived=req.derived,
        file_ref=req.file_ref,
        measured_at=ts
    )
    
    measurement_repo.insert(measurement)
    
    if req.set_as_default:
        try:
            sample_repo.set_device_default_measurement(
                req.sample_id, req.device_id, 
                req.measurement_type, meas_id
            )
        except ValueError as e:
             # Logic error, device should exist
             print(f"Warning: Failed to set default: {e}")

    return {"id": meas_id, "set_as_default": req.set_as_default}

@router.put("/measurements/{measurement_id}/set-default")
def set_default_measurement(measurement_id: str):
    """
    Set a measurement as default for its device/type.
    """
    meas = measurement_repo.get_by_id(measurement_id)
    if not meas:
        raise HTTPException(status_code=404, detail="Measurement not found")
        
    try:
        sample_repo.set_device_default_measurement(
            meas.sample_id, meas.device_id,
            meas.measurement_type, meas.id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return {"ok": True}
class AppendDevicesRequest(BaseModel):
    coord: List[int]
    variations: List[VariationModel]

@router.post("/samples/{sample_id}/device-groups/append", status_code=201)
def append_devices_to_group(sample_id: str, req: AppendDevicesRequest):
    """
    Append new devices (variations) to an existing device group.
    """
    sample = sample_repo.get_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    coord_tuple = tuple(req.coord)
    target_group = next((g for g in sample.device_groups if g.coord == coord_tuple), None)
    
    if not target_group:
         raise HTTPException(status_code=404, detail=f"DeviceGroup at {coord_tuple} not found")

    # Prefix
    prefix = f"{req.coord[0]}-{req.coord[1]}"
    new_devices = []

    existing_devs = sample_repo.get_devices(sample_id) # Flattened list of all devices in sample

    for v in req.variations:
        if v.suffix:
             d_id = f"{prefix}-{v.suffix}"
        else:
             d_id = prefix
        
        # Check uniqueness
        if any(d.device_id == d_id for d in existing_devs):
             raise HTTPException(status_code=400, detail=f"Device ID {d_id} already exists")
        
        # Also check in current batch to avoid duplicates
        if any(d.device_id == d_id for d in new_devices):
             raise HTTPException(status_code=400, detail=f"Duplicate Device ID {d_id} in request")

        new_devices.append(Device(device_id=d_id, area_um2=v.area_um2))

    # Add to group
    # We need to save the sample. 
    # Since we modified the object in memory (if we append to target_group.devices), we can save.
    # But `sample_repo` might need to be explicit.
    # The `sample` object contains `device_groups` list of `DeviceGroup` objects.
    # `target_group` is a reference to one of them.
    # So `target_group.devices.extend(new_devices)` should work if we then save `sample`.
    
    for d in new_devices:
        try:
            # We can reuse add_device_to_group for each, but that saves every time. Inefficient.
            # Better to append all and save once.
            # But we don't have a public "save_sample" method exposed in repo (it is _save_sample).
            # We should probably expose `update(sample)` or similar.
            # Or just use `add_device_to_group` loop for safety/consistency with repo logic.
            sample_repo.add_device_to_group(sample_id, coord_tuple, d)
        except ValueError as e:
             raise HTTPException(status_code=400, detail=str(e))
             
    return {
        "coord": list(coord_tuple),
        "added_devices": [d.device_id for d in new_devices]
    }

@router.get("/samples/{sample_id}/devices/{device_id}/measurements")
def list_device_measurements(sample_id: str, device_id: str):
    """
    List measurements for a specific device.
    """
    return measurement_repo.find_by_device(sample_id, device_id)

@router.delete("/samples/{sample_id}/devices/{device_id}")
def delete_device(sample_id: str, device_id: str):
    """
    Delete a device from a sample.
    """
    try:
        sample_repo.remove_device_from_group(sample_id, device_id)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/measurements/{measurement_id}")
def delete_measurement(measurement_id: str):
    """
    Delete a measurement.
    """
    # Check existence
    if not measurement_repo.get_by_id(measurement_id):
        raise HTTPException(status_code=404, detail="Measurement not found")
    
    measurement_repo.delete(measurement_id)
    return {"ok": True}

# ---------------------------------------------------------------------------
# IV Data Loading & Plotting
# ---------------------------------------------------------------------------
from ..services.data_loader import load_iv_data
import plotly.graph_objects as go
import json as _json
import numpy as np

class IVLoadRequest(BaseModel):
    file_ref: str
    area_um2: float
    r_p: float = 0.0  # Resistance in Ohms to subtract (V_corr = V - I * R)
    label: str = ""

class IVLoadMultiRequest(BaseModel):
    entries: List[IVLoadRequest]

def _build_iv_plots(entries: List[dict]):
    """
    Build IV and logR-V Plotly JSON from parsed IV entries.
    entries: list of {label, file_ref, area_um2, r_p, parsed: ParsedIVSeries}
    Returns {iv_plot, log_r_v_plot}
    """
    iv_fig = go.Figure()
    rv_fig = go.Figure()

    for ent in entries:
        parsed = ent["parsed"]
        label = ent.get("label", "")
        # area = ent["area_um2"] # Used for RA calculation if needed
        r_p = ent.get("r_p", 0.0)

        if not parsed.id_mA:
            continue

        id_arr = np.array(parsed.id_mA)
        vd_arr = np.array(parsed.vd_mV)
        r_arr = np.array(parsed.r_ohm)

        # Apply Series R correction: V_corrected = V - I * R_series
        # I is in mA, R_series in Ohm, V in mV => I(A) * R(Ohm) * 1000 = mV
        if r_p != 0:
            vd_arr = vd_arr - (id_arr / 1000.0) * r_p * 1000.0
            # Recalculate R from corrected V
            with np.errstate(divide='ignore', invalid='ignore'):
                r_arr = np.where(id_arr != 0, vd_arr / id_arr, 0)

        # IV plot
        iv_fig.add_trace(go.Scatter(
            x=vd_arr.tolist(), y=id_arr.tolist(),
            mode='lines+markers', name=label,
            marker=dict(size=3),
            hovertemplate=f"<b>{label}</b><br>V: %{{x:.2f}} mV<br>I: %{{y:.4f}} mA<extra></extra>"
        ))

        # Filter near-zero voltage for log plot
        mask = np.abs(vd_arr) > 5
        r_filtered = np.abs(r_arr[mask])
        v_filtered = vd_arr[mask]

        rv_fig.add_trace(go.Scatter(
            x=v_filtered.tolist(), y=r_filtered.tolist(),
            mode='lines+markers', name=label,
            marker=dict(size=3),
            hovertemplate=f"<b>{label}</b><br>V: %{{x:.2f}} mV<br>R: %{{y:.4g}} Ω<extra></extra>"
        ))

    iv_fig.update_layout(
        title="IV Characteristics", xaxis_title="Voltage (mV)", yaxis_title="Current (mA)",
        plot_bgcolor='white', hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
    )

    rv_fig.update_layout(
        title="Log R vs V", xaxis_title="Voltage (mV)", yaxis_title="R (Ω)",
        yaxis_type="log",
        plot_bgcolor='white', hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
    )

    return {
        "iv_plot": _json.loads(iv_fig.to_json()),
        "log_r_v_plot": _json.loads(rv_fig.to_json()),
    }


@router.post("/iv/load")
def iv_load(req: IVLoadRequest):
    """Load a single IV file and return Plotly JSON for IV + log R-V."""
    parsed = load_iv_data(req.file_ref)
    if not parsed.id_mA:
        return {"iv_plot": None, "log_r_v_plot": None, "warnings": ["No data found"], "raw": []}
    
    entry = {
        "parsed": parsed,
        "label": req.label or req.file_ref.split('/')[-1],
        "area_um2": req.area_um2,
        "r_p": req.r_p,
    }
    result = _build_iv_plots([entry])
    result["warnings"] = parsed.warnings
    result["raw"] = parsed.raw_data_lines
    return result


@router.post("/iv/load-multi")
def iv_load_multi(req: IVLoadMultiRequest):
    """Load multiple IV files and return overlaid Plotly JSON."""
    entries = []
    all_warnings = []
    for e in req.entries:
        parsed = load_iv_data(e.file_ref)
        all_warnings.extend(parsed.warnings)
        entries.append({
            "parsed": parsed,
            "label": e.label or e.file_ref.split('/')[-1],
            "area_um2": e.area_um2,
            "r_p": e.r_p,
        })

    result = _build_iv_plots(entries)
    result["warnings"] = all_warnings
    return result


# ---------------------------------------------------------------------------
# Parasitic Resistance Fitting
# ---------------------------------------------------------------------------
class FitParasiticRequest(BaseModel):
    entries: List[IVLoadRequest]
    initial_r_para: float = 0.0 # initial_r_para not strictly needed for analytic solution but kept for compatibility/extensions
    mode: str = "full"  # "full" or "step"

@router.post("/iv/fit-parasitic")
def fit_parasitic(req: FitParasiticRequest):
    """
    Calculate parasitic resistance R_para such that the peaks of the RA curves
    (derived from corrected R) of the two smallest area devices match.
    Formula: R_para = (R_max1 * A1 - R_max2 * A2) / (A1 - A2)
    """
    # Sort entries by area and pick smallest two
    sorted_entries = sorted(req.entries, key=lambda e: e.area_um2)
    if len(sorted_entries) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 devices for fitting")

    small_two = sorted_entries[:2]
    e1, e2 = small_two[0], small_two[1]
    A1, A2 = e1.area_um2, e2.area_um2

    # Parse data
    p1 = load_iv_data(e1.file_ref)
    p2 = load_iv_data(e2.file_ref)
    if not p1.id_mA or not p2.id_mA:
         raise HTTPException(status_code=400, detail="Data missing in one of the smallest devices")

    # Get Max Resistance (R_max) from raw data (approximate peak of R-V curve)
    # R_ohm is already in parsed data
    # Filter for reasonable voltage range to avoid noise at V=0?
    # Usually peak is at V=0.
    def get_max_r(p):
        r_arr = np.array(p.r_ohm)
        # Filter NaNs or Infs
        r_arr = r_arr[np.isfinite(r_arr)]
        if len(r_arr) == 0: return 0.0
        return np.max(r_arr)

    R_max1 = get_max_r(p1)
    R_max2 = get_max_r(p2)

    # Analytic solution
    # r_para = (R_max1 * A1 - R_max2 * A2) / (A1 - A2)
    if abs(A1 - A2) < 1e-6:
         # Same area, cannot determine R_para by this method
         optimal_r_para = 0.0
    else:
        optimal_r_para = (R_max1 * A1 - R_max2 * A2) / (A1 - A2)

    # Sanity check: R_para should not exceed R_max1 (otherwise R_corrected < 0)
    # But strictly speaking it could if the model is perfect.
    # However, usually R_para is positive.
    if optimal_r_para < 0:
        optimal_r_para = 0.0 # Clamp to 0 if calculation yields negative

    # Build corrected plots for ALL entries
    all_entries = []
    for e in req.entries:
        p = load_iv_data(e.file_ref)
        all_entries.append({
            "parsed": p,
            "label": e.label or e.file_ref.split('/')[-1],
            "area_um2": e.area_um2,
            "r_p": optimal_r_para,
        })

    plots = _build_iv_plots(all_entries)

    return {
        "r_para": optimal_r_para,
        "plots": plots,
    }


@router.put("/samples/{sample_id}/r-parasitic")
def update_r_parasitic(sample_id: str, body: dict = Body(...)):
    """Save r_parasitic to the sample."""
    r_para = body.get("r_parasitic")
    if r_para is None:
        raise HTTPException(status_code=400, detail="r_parasitic is required")
    try:
        sample_repo.update_r_parasitic(sample_id, float(r_para))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Heatmap Data
# ---------------------------------------------------------------------------
@router.get("/samples/{sample_id}/heatmap-data")
def get_heatmap_data(sample_id: str):
    """
    Collect Hanle-derived values (Ps, RA, RMS) for each device,
    keyed by (x, y) coordinate.
    """
    sample = sample_repo.get_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    devices_data = []
    # Compute x/y ranges
    coords = [d["coord"] for group in sample.device_groups for d in group.devices] # This line was incorrect, it should be group.coord
    
    x_range = [None, None]
    y_range = [None, None]

    for group in sample.device_groups:
        cx, cy = group.coord
        if x_range[0] is None or cx < x_range[0]: x_range[0] = cx
        if x_range[1] is None or cx > x_range[1]: x_range[1] = cx
        if y_range[0] is None or cy < y_range[0]: y_range[0] = cy
        if y_range[1] is None or cy > y_range[1]: y_range[1] = cy

        for device in group.devices:
            entry = {
                "coord": list(group.coord),
                "device_id": device.device_id,
                "area_um2": device.area_um2,
                "hanle": None,
            }

            # Try to get default Hanle measurement
            hanle_id = device.default_measurements.get("Hanle")
            if hanle_id:
                meas = measurement_repo.get_by_id(hanle_id)
                if meas and meas.derived:
                    entry["hanle"] = meas.derived

            devices_data.append(entry)

    # The original code had a bug in calculating x_range/y_range if devices_data was empty.
    # The new logic for x_range/y_range calculation is more robust.
    # If no device groups, x_range/y_range will remain [None, None].
    # The instruction provided a different heatmap-data logic, which I will incorporate.
    # The instruction's heatmap-data logic is more detailed and includes json.loads for derived data.

    # Re-implementing get_heatmap_data based on the provided instruction's snippet
    devices_data = []
    x_range = [None, None]
    y_range = [None, None]

    for group in sample.device_groups:
        cx, cy = group.coord
        if x_range[0] is None or cx < x_range[0]: x_range[0] = cx
        if x_range[1] is None or cx > x_range[1]: x_range[1] = cx
        if y_range[0] is None or cy < y_range[0]: y_range[0] = cy
        if y_range[1] is None or cy > y_range[1]: y_range[1] = cy

        for device in group.devices:
            # We want metrics from Analysis (Summary) if available, or just defaults
            # Actually heatmap usually visualizes "Ps", "RA", "RMS" derived from Hanle
            # Logic: Fetch 'default' Hanle measurement for the device, parse its results.
            # For now, let's assume we look up the Measurement object.
            
            hanle_id = device.default_measurements.get("Hanle")
            metrics = None
            if hanle_id:
                m = measurement_repo.get_by_id(hanle_id) # Changed from measurement_repo.get to get_by_id for consistency
                if m and m.derived:
                    try:
                        # Ensure json is imported if not already
                        import json
                        derived = json.loads(m.derived) if isinstance(m.derived, str) else m.derived
                        # derived structure: {"spin_signal": ..., "ra": ..., "rms": ...}
                        metrics = {
                            "ps_percent": derived.get("spin_signal"),
                            "ra_ohm_um2": derived.get("ra"),
                            "rms": derived.get("rms"),
                        }
                    except:
                        pass
            
            devices_data.append({
                "device_id": device.device_id,
                "coord": group.coord,
                "area_um2": device.area_um2,
                "hanle": metrics
            })

    return {
        "devices": devices_data,
        "x_range": x_range,
        "y_range": y_range,
        "r_parasitic": sample.r_parasitic,
    }

