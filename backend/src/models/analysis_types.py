from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict

class ParsedIVSeries(BaseModel):
    id_A: List[float] = Field(description="Current in Amperes")
    vd_V: List[float] = Field(description="Voltage in Volts")
    r_ohm: List[float] = Field(description="Resistance in Ohms")
    warnings: List[str] = Field(default_factory=list)

class RAPsPoint(BaseModel):
    ra_ohm_m2: float = Field(description="Resistance-Area product in Ohm * m^2")
    ps: float = Field(description="Spin polarization as a fraction (0.0 to 1.0)")
    rms: float = Field(description="Root mean square error")
    tp_s: float = Field(default=0.0, description="Processing time in seconds")
    area_m2: float = Field(default=1e-12, description="Area in m^2")
    label: Optional[str] = None

    @field_validator('ps')
    @classmethod
    def validate_ps(cls, v: float) -> float:
        if v > 1.0 or v < 0.0:
            raise ValueError(f"Ps must be a fraction between 0.0 and 1.0, got {v} (if this is a percentage, divide by 100)")
        return v

    @field_validator('area_m2')
    @classmethod
    def validate_area(cls, v: float) -> float:
        if v > 1.0:
            raise ValueError(f"Area {v} m^2 is too large for a nanodevice. Ensure you are converting from um^2 or cm^2 to m^2.")
        return v

class RAPsSeries(BaseModel):
    points: List[RAPsPoint] = Field(default_factory=list)
    label: str
    color: str
    r_p_ohm: float = Field(default=0.0, description="Parasitic resistance in Ohms")

class LogRAVSeries(BaseModel):
    vd_V: List[float]
    ra_ohm_m2: List[float]
    label: str
    color: str
    group_label: str

class HanleRawSeries(BaseModel):
    magnetic_field_T: List[float] = Field(description="Magnetic field in Tesla")
    voltage_V: List[float] = Field(description="Voltage in Volts")

class HanleBroadSeries(BaseModel):
    params: Dict[str, float] = Field(description="Parameters from the file header")
    exp_data: List[List[float]] = Field(description="Experimental data, typically [B (T), V (V)]")
    fitting_data: List[List[float]] = Field(description="Fitting data from the file")
    broad_fitting_data: List[List[float]] = Field(description="Broad fitting data from the file")

class HanleNarrowSeries(BaseModel):
    exp_data: List[List[float]] = Field(description="Experimental data")
    fitting_data: List[List[float]] = Field(description="Fitting data")

# Using discriminated union or basic Union for measurements
Measurement = Union[ParsedIVSeries, RAPsSeries, LogRAVSeries, HanleRawSeries, HanleBroadSeries, HanleNarrowSeries]

class Device(BaseModel):
    name: str
    x_coord: int
    y_coord: int
    measurements: List[Measurement] = Field(default_factory=list)

class Sample(BaseModel):
    name: str
    max_x: int = Field(default=22, description="Grid width")
    max_y: int = Field(default=22, description="Grid height")
    devices: List[Device] = Field(default_factory=list)
