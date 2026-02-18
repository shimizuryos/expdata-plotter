from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ParsedIVSeries:
    id_mA: List[float]
    vd_mV: List[float]
    r_ohm: List[float]
    warnings: List[str]

@dataclass
class RAPsPoint:
    ra: float
    ps: float
    rms: float
    label: Optional[str] = None

@dataclass
class RAPsSeries:
    points: List[RAPsPoint]
    label: str
    color: str
