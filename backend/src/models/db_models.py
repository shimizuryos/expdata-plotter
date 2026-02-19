from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class LayerStructure:
    material: str
    thick_nm_variation: Optional[List[float]] = None

@dataclass
class Device:
    device_id: str
    area_um2: float
    note: str = ""
    # Map measurement_type -> measurement_id (e.g. {"Hanle": "meas_123"})
    default_measurements: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeviceGroup:
    coord: tuple[int, int]
    thick_nm: Dict[str, float]       # e.g. {"MgO": 1.0}
    shared_properties: Dict[str, Any] # e.g. {"parasitic_resistance_ohm": 120.5}
    devices: List[Device]

@dataclass
class Sample:
    id: str
    name: str
    device_type: str
    structures: List[LayerStructure]
    device_groups: List[DeviceGroup]
    note: str = ""
    r_parasitic: Optional[float] = None  # R_para (ohm), shared across all devices

@dataclass
class Measurement:
    id: str
    sample_id: str
    device_id: str
    measurement_type: str             # "IV", "Hanle"
    metadata: Dict[str, Any]          # {temp_K, ...}
    data: Optional[Dict[str, Any]] = None       # measurement-specific raw data
    derived: Optional[Dict[str, Any]] = None    # {ra_ohm_um2, ps_percent, rms}
    file_ref: Optional[str] = None
    measured_at: Optional[str] = None
