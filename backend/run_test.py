import sys
import numpy as np

from src.services.ps_ra_fitting import model_Ps_RA, calc_tox_from_Tp
from src.services.data_loader import load_ps_ra_data

def run_test():
    # Load data
    data_path = '../data/ps_ra_data.yaml'
    series_list = load_ps_ra_data(data_path)
    
    all_points = []
    for s in series_list:
        all_points.extend(s.points)
    
    Tp_arr = np.array([p.tp_min for p in all_points])
    tox_arr = calc_tox_from_Tp(Tp_arr)
    Ps_exp = np.array([p.ps for p in all_points])
    RA_exp = np.array([p.ra for p in all_points])
    A_arr = np.array([p.area_um2 for p in all_points])
    
    # Needs to subtract R_p
    # Let's just do it directly for the first series 
    # Actually wait, we should do it per series, but let's assume one series R_p for simplicity to test
    R_p = series_list[0].r_p
    RA_exp_corr = RA_exp - R_p * A_arr
    
    valid = (RA_exp_corr > 0)
    tox_arr = tox_arr[valid]
    Ps_exp = Ps_exp[valid]
    RA_exp = RA_exp_corr[valid]
    
    # Target params
    D_A = 1.0
    D_B = 1e3
    C = 1e-3
    lam_A = 2e-10
    lam_B = 6e-11
    P_S_A = 0.22
    P_S_B = 0.0
    V_B = 0.1
    
    Ps_calc, RA_calc = model_Ps_RA(
        tox_arr, D_A, D_B, lam_A, lam_B,
        C=C, P_S_A=P_S_A, P_S_B=P_S_B, V_B=V_B
    )
    
    sigma_P = 1.0
    sigma_lnR = 1.0
    
    print("--- Intermediate Calculations ---")
    for i, tox in enumerate(tox_arr):
        print(f"tox: {tox*1e9:.3f} nm")
        print(f"  Ps_exp: {Ps_exp[i]:.4f} | Ps_calc: {Ps_calc[i]:.4f}")
        print(f"  RA_exp: {RA_exp[i]:.4e} | RA_calc: {RA_calc[i]:.4e}")
        
        # Current logic
        Ps_calc_safe = abs(Ps_calc[i]) if abs(Ps_calc[i]) > 1e-30 else 1e-30
        RA_calc_safe = RA_calc[i] if RA_calc[i] > 1e-30 else 1e-30
        
        rP_current = (Ps_calc[i] - Ps_exp[i]) / (Ps_calc_safe * sigma_P)
        rR_current = (RA_calc[i] - RA_exp[i]) / (RA_calc_safe * sigma_lnR)
        
        # Originally you requested a strictly log-scale difference for RA if RA_exp handles log
        rR_log = np.log(RA_calc_safe) - np.log(max(RA_exp[i], 1e-30))
        
        print(f"  -> rP: {rP_current:.4f}")
        print(f"  -> rR (current relative): {rR_current:.4f}")
        print(f"  -> rR (log difference): {rR_log:.4f}")
        print()
    
    # Aggregation
    rP_all = (Ps_calc - Ps_exp) / np.maximum(np.abs(Ps_calc), 1e-30)
    rR_current_all = (RA_calc - RA_exp) / np.maximum(RA_calc, 1e-30)
    rR_log_all = np.log(np.maximum(RA_calc, 1e-30)) - np.log(np.maximum(RA_exp, 1e-30))
    
    print("--- Cost (Sum of Squares) ---")
    cost_P = 0.5 * np.sum(rP_all**2)
    cost_R_current = 0.5 * np.sum(rR_current_all**2)
    cost_R_log = 0.5 * np.sum(rR_log_all**2)
    
    print(f"Cost P  : {cost_P:.4e}")
    print(f"Cost R (relative): {cost_R_current:.4e}")
    print(f"Cost R (log): {cost_R_log:.4e}")

if __name__ == '__main__':
    run_test()
