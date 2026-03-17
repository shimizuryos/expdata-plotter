# backend/src/utils/units.py

# Length
def um_to_m(val: float) -> float:
    return val * 1e-6

def m_to_um(val: float) -> float:
    return val * 1e6

def cm_to_m(val: float) -> float:
    return val * 1e-2

def m_to_cm(val: float) -> float:
    return val * 1e2

# Voltage
def mV_to_V(val: float) -> float:
    return val * 1e-3

def V_to_mV(val: float) -> float:
    return val * 1e3

def uV_to_V(val: float) -> float:
    return val * 1e-6

def V_to_uV(val: float) -> float:
    return val * 1e6

# Current
def mA_to_A(val: float) -> float:
    return val * 1e-3

def A_to_mA(val: float) -> float:
    return val * 1e3

# Time
def min_to_s(val: float) -> float:
    return val * 60.0

def s_to_min(val: float) -> float:
    return val / 60.0

# Magnetic Field
def Oe_to_T(val: float) -> float:
    return val * 1e-4

def T_to_Oe(val: float) -> float:
    return val * 1e4

# Dimensionless
def percent_to_fraction(val: float) -> float:
    return val / 100.0

def fraction_to_percent(val: float) -> float:
    return val * 100.0

def identity(val: float) -> float:
    return val

# Physical quantities
def convert_area_um2_to_m2(area_um2: float) -> float:
    return um_to_m(um_to_m(area_um2))

def convert_area_m2_to_um2(area_m2: float) -> float:
    return m_to_um(m_to_um(area_m2))

def convert_RA_ohm_um2_to_ohm_m2(ra_ohm_um2: float) -> float:
    return ra_ohm_um2 * 1e-12

def convert_RA_ohm_m2_to_ohm_um2(ra_ohm_m2: float) -> float:
    return ra_ohm_m2 * 1e12
