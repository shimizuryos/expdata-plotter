import os
import sys
import numpy as np

# Add src to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from services.ps_ra_fitting import model_Ps_RA, calc_tox_from_Tp
from services.data_loader import load_ps_ra_data

def run_test():
    # Load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/ps_ra_data.yaml'))
    series_list = load_ps_ra_data(data_path)
    
    label_to_test = "CoFe" # Let's just grab the first or all
    target = None
    all_points = []
    for s in series_list:
        all_points.extend(s.points)
    
    Tp_arr = np.array([p.tp_min for p in all_points])
    tox_arr = calc_tox_from_Tp(Tp_arr)
    Ps_exp = np.array([p.ps for p in all_points])
    RA_exp = np.array([p.ra for p in all_points])
    
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
        rP_current = (Ps_calc[i] - Ps_exp[i]) / (abs(Ps_calc[i]) * sigma_P if abs(Ps_calc[i]) > 1e-30 else 1e-30)
        rR_current = (RA_calc[i] - RA_exp[i]) / (RA_calc[i] * sigma_lnR if RA_calc[i] > 1e-30 else 1e-30)
        
        # Originally you requested a strictly log-scale difference for RA if RA_exp handles log
        # Note: If RA spans many orders of magnitude, (RA_calc - RA_exp)/RA_calc == 1 - RA_exp/RA_calc
        # But log(RA_calc) - log(RA_exp) = log(RA_calc / RA_exp) represents ratio nicely and symmetrically.
        rR_log = np.log(max(RA_calc[i], 1e-30)) - np.log(RA_exp[i])
        
        print(f"  -> rP: {rP_current:.4f}")
        print(f"  -> rR (current relative): {rR_current:.4f}")
        print(f"  -> rR (log difference): {rR_log:.4f}")
        print()
    
    # Aggregation
    rP_all = (Ps_calc - Ps_exp) / np.maximum(np.abs(Ps_calc), 1e-30)
    rR_current_all = (RA_calc - RA_exp) / np.maximum(RA_calc, 1e-30)
    rR_log_all = np.log(np.maximum(RA_calc, 1e-30)) - np.log(RA_exp)
    
    print("--- Cost (Sum of Squares) ---")
    cost_P = 0.5 * np.sum(rP_all**2)
    cost_R_current = 0.5 * np.sum(rR_current_all**2)
    cost_R_log = 0.5 * np.sum(rR_log_all**2)
    
    print(f"Cost P  : {cost_P:.4e}")
    print(f"Cost R (relative): {cost_R_current:.4e}")
    print(f"Cost R (log): {cost_R_log:.4e}")

if __name__ == '__main__':
    run_test()
