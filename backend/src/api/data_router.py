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
