"""
Ps-RA curve fitting service.

Two-channel tunnel model with likelihood-based chi-squared optimisation.
Uses scipy.optimize.least_squares with log-parametrisation for
positivity and numerical stability.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Any, Optional, Tuple, List
import threading

# ─────────────────────────────────────────────
# Physical constants for t_ox calculation
# ─────────────────────────────────────────────
ALPHA = 5.4e-10   # m
BETA  = 2.9e-20   # m^2 / min


def calc_tox_from_Tp(Tp_min: np.ndarray) -> np.ndarray:
    """Oxide thickness from plasma oxidation time."""
    Tp_min = np.asarray(Tp_min, dtype=float)
    return (-ALPHA + np.sqrt(ALPHA**2 + 4 * BETA * Tp_min)) / 2.0


# ─────────────────────────────────────────────
# Two-channel tunnel model
# ─────────────────────────────────────────────
def model_Ps_RA(
    tox: np.ndarray,
    D_A: float, D_B: float,
    lam_A: float, lam_B: float,
    C: float = 1.0,
    P_S_A: float = 0.5, P_S_B: float = 0.3,
    V_B: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (Ps, RA) from the two-channel tunnel model.
    Returns arrays of same length as *tox*.
    """
    tox = np.asarray(tox, dtype=float)

    xA = np.clip(-tox / lam_A, -700, 50)
    xB = np.clip(-tox / lam_B, -700, 50)

    eA = np.exp(xA)
    eB = np.exp(xB)

    J_A_plus  = C * (1 + P_S_A) * D_A * eA / 2.0
    J_A_minus = C * (1 - P_S_A) * D_A * eA / 2.0
    J_B_plus  = C * (1 + P_S_B) * D_B * eB / 2.0
    J_B_minus = C * (1 - P_S_B) * D_B * eB / 2.0

    J_plus  = J_A_plus  + J_B_plus
    J_minus = J_A_minus + J_B_minus

    J_s = J_plus - J_minus
    J   = J_plus + J_minus

    epsJ = 1e-30
    J_safe = np.where(np.abs(J) < epsJ, np.sign(J) * epsJ + epsJ, J)

    Ps = J_s / J_safe
    RA = V_B / J_safe

    return Ps, RA


# ─────────────────────────────────────────────
# Fitting engine
# ─────────────────────────────────────────────

# Global store for progressive results (job_id -> snapshots)
_fit_progress: Dict[str, Dict[str, Any]] = {}
_fit_progress_lock = threading.Lock()

PARAM_SPECS = {
    "P_S_A": {"is_log": False, "bounds": (-1.0, 1.0)},
    "P_S_B": {"is_log": False, "bounds": (-1.0, 1.0)},
    "lambda_A": {"is_log": True, "bounds": (np.log(0.1e-9), np.log(100e-9))},
    "lambda_B": {"is_log": True, "bounds": (np.log(0.1e-9), np.log(100e-9))},
    "C": {"is_log": True, "bounds": (np.log(1e-10), np.log(1e10))},
    "D_B": {"is_log": True, "bounds": (np.log(1e-20), np.log(1e+20))},
}

class LikelihoodCalculator:
    def __init__(self, tox_v: np.ndarray, Ps_v: np.ndarray, lnRA_v: np.ndarray, 
                 V_B: float, sigma_P: float, sigma_lnR: float, 
                 fix_flags: Dict[str, bool], init_vals: Dict[str, float],
                 job_id: Optional[str] = None):
        self.tox_v = tox_v
        self.Ps_v = Ps_v
        self.lnRA_v = lnRA_v
        self.V_B = V_B
        self.sigma_P = sigma_P
        self.sigma_lnR = sigma_lnR
        self.fix_flags = fix_flags
        self.init_vals = init_vals
        self.job_id = job_id
        
        self.D_A_val = 1.0  # strictly fixed
        self.fit_vars = [k for k in ["P_S_A", "P_S_B", "lambda_A", "lambda_B", "C", "D_B"] 
                         if not self.fix_flags.get(k, False)]
        self.iteration = 0

    def get_initial_x0_and_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0_list = []
        lb_list = []
        ub_list = []
        for k in self.fit_vars:
            val = self.init_vals[k]
            spec = PARAM_SPECS[k]
            if spec["is_log"]:
                val = max(1e-30, val)
                x0_list.append(np.log(val))
            else:
                val = max(spec["bounds"][0], min(spec["bounds"][1], val))
                x0_list.append(val)
            lb_list.append(spec["bounds"][0])
            ub_list.append(spec["bounds"][1])
        return np.array(x0_list, dtype=float), np.array(lb_list, dtype=float), np.array(ub_list, dtype=float)

    def decode_params(self, x: np.ndarray) -> Dict[str, float]:
        params = self.init_vals.copy()
        for i, k in enumerate(self.fit_vars):
            spec = PARAM_SPECS[k]
            if spec["is_log"]:
                params[k] = float(np.exp(x[i]))
            else:
                params[k] = float(x[i])
        return params

    def get_full_params_dict(self, x: Optional[np.ndarray] = None) -> Dict[str, float]:
        if x is not None:
            params = self.decode_params(x)
        else:
            params = self.init_vals.copy()
        params["D_A"] = self.D_A_val
        return params

    def residuals(self, x: np.ndarray) -> np.ndarray:
        params = self.get_full_params_dict(x)
        Ps_calc, RA_calc = model_Ps_RA(
            self.tox_v, params["D_A"], params["D_B"], 
            params["lambda_A"], params["lambda_B"],
            C=params["C"], P_S_A=params["P_S_A"], P_S_B=params["P_S_B"], 
            V_B=self.V_B
        )

        self.Ps_v_safe = np.maximum(np.abs(self.Ps_v), 1e-3)

        # Using relative difference error against the measured value: (calc - exp) / max(exp, 1e-3)
        RA_calc_safe = np.maximum(RA_calc, 1e-30)
        
        rP = (Ps_calc - self.Ps_v) / (self.Ps_v_safe * self.sigma_P)
        # Using logarithmic difference for RA as RA spans many orders of magnitude
        rR = (np.log(RA_calc_safe) - self.lnRA_v) / self.sigma_lnR
        
        self.iteration += 1

        if self.job_id and self.iteration % 10 == 0:
            cost = float(0.5 * np.sum(np.concatenate([rP, rR])**2))
            with _fit_progress_lock:
                _fit_progress[self.job_id] = {
                    "status": "running",
                    "iteration": self.iteration,
                    "params": params,
                    "cost": cost,
                }

        return np.concatenate([rP, rR])

def run_fitting(
    Tp_measured: np.ndarray,
    Ps_measured: np.ndarray,
    RA_measured: np.ndarray,
    area_measured: np.ndarray,  # μm²
    R_para_ohm: float,
    *,
    V_B: float = 0.1,
    weight_ratio: float = 1.0,  # σ_lnR / σ_P  (>1 → RA matters less)
    fix_flags: Optional[Dict[str, bool]] = None,
    init_vals: Optional[Dict[str, float]] = None,
    job_id: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run least_squares fitting with dynamic parameter toggling.
    D_A is uniformly fixed to 1.0. 
    Other 6 parameters (P_S_A, P_S_B, lambda_A, lambda_B, C, D_B) can be fixed or optimized.
    """
    Tp  = np.asarray(Tp_measured, dtype=float)
    tox = calc_tox_from_Tp(Tp)

    A        = np.asarray(area_measured, dtype=float)
    RA_para  = R_para_ohm * A
    RA_corr  = np.asarray(RA_measured, dtype=float) - RA_para

    # Validity mask
    valid = np.isfinite(RA_corr) & (RA_corr > 0) & np.isfinite(Ps_measured)
    tox_v  = tox[valid]
    Ps_v   = np.asarray(Ps_measured, dtype=float)[valid]
    RA_v   = RA_corr[valid]
    lnRA_v = np.log(RA_v)

    sigma_P   = 1.0
    sigma_lnR = weight_ratio  

    fix_flags = fix_flags or {}
    # Default valid fallback for unspecified params
    defaults = {
        "P_S_A": 0.5, "P_S_B": 0.3,
        "lambda_A": 1e-9, "lambda_B": 5e-9,
        "C": 1.0, "D_B": 1.0
    }
    if init_vals:
        defaults.update(init_vals)
    init_vals = defaults

    calc = LikelihoodCalculator(
        tox_v=tox_v, Ps_v=Ps_v, lnRA_v=lnRA_v, 
        V_B=V_B, sigma_P=sigma_P, sigma_lnR=sigma_lnR, 
        fix_flags=fix_flags, init_vals=init_vals, job_id=job_id
    )

    if not calc.fit_vars:
        # Everything is fixed, just return cost
        x0 = np.array([])
        res_fun = calc.residuals(x0)
        cost = float(0.5 * np.sum(res_fun**2))
        params = calc.get_full_params_dict()
        info = {
            "success": True,
            "cost": cost,
            "nfev": 1,
            "message": "All parameters fixed",
            "residual_norm": float(np.linalg.norm(res_fun)),
            "iterations": 1,
        }
    else:
        x0, lb, ub = calc.get_initial_x0_and_bounds()

        res = least_squares(
            calc.residuals, x0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
        )

        params = calc.get_full_params_dict(res.x)
        info = {
            "success": bool(res.success),
            "cost": float(res.cost),
            "nfev": int(res.nfev),
            "message": res.message,
            "residual_norm": float(np.linalg.norm(res.fun)),
            "iterations": calc.iteration,
        }

    # Print final residual
    print(f"--- Fitting Completed ---")
    print(f"Final Residual Norm: {info['residual_norm']}")
    print(f"Final Cost: {info['cost']}")

    # Mark job done
    if job_id:
        with _fit_progress_lock:
            _fit_progress[job_id] = {
                "status": "done",
                "iteration": calc.iteration,
                "params": params,
                "cost": info["cost"],
            }

    return params, info


def generate_fit_curves(
    params: Dict[str, float],
    tox_range: Tuple[float, float],
    n_points: int = 200,
    *,
    V_B: float = 0.1,
) -> Dict[str, List[float]]:
    """
    Generate smooth fit curves over a continuous t_ox range.
    Returns dict with tox, Ps, RA arrays.
    """
    tox_arr = np.linspace(tox_range[0], tox_range[1], n_points)
    
    # D_A and C might not be explicitly populated if generated elsewhere without it, but 1.0 is default
    D_A = params.get("D_A", 1.0)
    C = params.get("C", 1.0)
    
    Ps_fit, RA_fit = model_Ps_RA(
        tox_arr,
        D_A, params["D_B"],
        params["lambda_A"], params["lambda_B"],
        C=C, 
        P_S_A=params.get("P_S_A", 0.5), 
        P_S_B=params.get("P_S_B", 0.3), 
        V_B=V_B,
    )
    return {
        "tox": tox_arr.tolist(),
        "Ps": Ps_fit.tolist(),
        "RA": RA_fit.tolist(),
    }


def get_fit_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """Get progress snapshot for a running/completed fit job."""
    with _fit_progress_lock:
        return _fit_progress.get(job_id)


def clear_fit_progress(job_id: str):
    """Clean up progress data for a completed job."""
    with _fit_progress_lock:
        _fit_progress.pop(job_id, None)
