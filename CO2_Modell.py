Filename: co2_modell.py
Author: Eike Roth.
Date: 2026-02-27

Description: 
This script performs the numerical simulation for the 'Lake' and 'Fountain' 
CO2 models with different embedment of the atmosphere into the global carbon cycle. 
For the period from 1850 to the present day, it is calculated, 
which external inflows were required to increase the concentration as observed, 
and for the next 100 years, it is calculated in reverse,
how the concentration will develop under different release scenarios.

Disclaimer: 
Parts of this code were generated with the assistance of Google Gemini 
(accessed Jan/Feb 2026) based on the provided specifications. 
The code was manually verified and tested by the author.
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import csv
import traceback

# =============================================================================
# INPUT SECTION
# =============================================================================

# Visualization Range
PLOT_START_YEAR = 1850
PLOT_END_YEAR   = 2124

# Active Model Variants to simulate ('a' to 'e')
ACTIVE_VARIANTS = ['a', 'b', 'c', 'd', 'e']

# Phase 1: Retention factor for Fountain Model
FOUNTAIN_R = 0.5 

# --- PHASE 2 SETTINGS (Post-2024) ---

# 1. Start Value Mode
# 'absolute' = Fixed value (P2_ABS_START_GTC)
# 'relative' = Percentage of 2024 value (P2_REL_START_PCT)
P2_START_MODE = 'relative' 
P2_ABS_START_GTC = 0.0   
P2_REL_START_PCT = 100.0  

# 2. Growth Mode
# 'linear'      = Fixed annual increment (+/- GtC)
# 'exponential' = Fixed annual growth rate (+/- %)
P2_GROWTH_MODE = 'linear'
P2_LINEAR_INC_GTC = 0.0 
P2_EXP_GROWTH_PCT = 1.0   

# =============================================================================
# CONSTANTS & LOGIC
# =============================================================================
PPM_TO_GTC = 600.0 / 280.0
BERN_A = np.array([0.186, 0.338, 0.259, 0.217])
BERN_TAU = np.array([1.186, 18.51, 172.9, np.inf])

# Initial Inventory 1850 (Atmosphere, Bio_short, Bio_long, Ocean_mix, Ocean_deep)
SEG_INVENTORY = np.array([600.0, 1000.0, 2500.0, 1000.0, 38500.0])
SEG_FLOWS = np.zeros((5, 5))

def set_flow(s1, s2, val):
    """Helper to set bidirectional flows between segments."""
    SEG_FLOWS[s1-1][s2-1] = val
    SEG_FLOWS[s2-1][s1-1] = val

# Define natural flows
set_flow(1, 2, 115) # Atm <-> Bio Short
set_flow(1, 4, 55)  # Atm <-> Ocean Mix
set_flow(2, 3, 115) # Bio Short <-> Bio Long
set_flow(4, 5, 275) # Ocean Mix <-> Ocean Deep

# Variant Definitions
VARIANTS = {
    'a': {'name': '2 Boxen',          'map': [0, 1, 1, 1, 1]},
    'b': {'name': '3 Boxen parallel', 'map': [0, 1, 1, 2, 2]},
    'c': {'name': '3 Boxen seriell',  'map': [0, 1, 2, 1, 2]},
    'd': {'name': '5 Boxen homogen',  'map': [0, 1, 3, 2, 4]},
    'e': {'name': '5 Boxen träge',    'map': [0, 1, 3, 2, 4]} 
}

def get_atm_conc_poly(year):
    """Returns historical atmospheric CO2 target based on polynomial fit."""
    t = year - 1850
    a = 2.949112e-5; b = -5.247702e-4; c = 0.020
    return 280 + a*(t**3) + b*(t**2) + c*t

def build_boxes(variant_key):
    """Constructs box inventories and flow matrices based on variant mapping."""
    mapping = VARIANTS[variant_key]['map']
    num_boxes = max(mapping) + 1
    box_inv = np.zeros(num_boxes); box_flows = np.zeros((num_boxes, num_boxes))
    
    for seg_idx, box_idx in enumerate(mapping): 
        box_inv[box_idx] += SEG_INVENTORY[seg_idx]
        
    for i in range(5):
        for j in range(5):
            if i != j and SEG_FLOWS[i][j] > 0:
                b_from = mapping[i]; b_to = mapping[j]
                if b_from != b_to: box_flows[b_from][b_to] += SEG_FLOWS[i][j]
                
    # Infinite sink condition for Variant 'e'
    if variant_key == 'e' and num_boxes > 3:
        box_inv[3] = 1e15; box_inv[4] = 1e15
        
    return box_inv, box_flows

def bern_pulse_response(t_years):
    """Calculates the Bern model impulse response function."""
    if t_years < 0: return 0.0
    if t_years == 0: return 1.0
    res = 0.0
    for i in range(4):
        if np.isinf(BERN_TAU[i]): res += BERN_A[i]
        else: res += BERN_A[i] * np.exp(-t_years / BERN_TAU[i])
    return res

def get_external_influx_phase2(year, start_val):
    """Calculates external influx for Phase 2 based on growth mode."""
    dt = year - 2024
    if dt <= 0: return 0.0
    if P2_GROWTH_MODE == 'linear': 
        return max(0.0, start_val + (P2_LINEAR_INC_GTC * dt))
    elif P2_GROWTH_MODE == 'exponential': 
        return max(0.0, start_val * ((1 + P2_EXP_GROWTH_PCT/100.0)**dt))
    return start_val 

def calculate_equilibrium_concentration(model, current_inv, box_inv_eq):
    """Calculates theoretical equilibrium concentration."""
    if model == 'Lake':
        return (np.sum(current_inv) / np.sum(box_inv_eq)) * 280.0
    elif model == 'Fountain':
        return 280.0 + ((current_inv[0] / PPM_TO_GTC - 280.0) * 0.217)
    return 280.0

def calculate_tau_phys(ppm_series, years, year_target, c_equilibrium):
    """Calculates physical relaxation time Tau."""
    idx = year_target - 1850
    if idx >= len(ppm_series) - 1: slope = ppm_series[idx] - ppm_series[idx-1]
    else: slope = ppm_series[idx+1] - ppm_series[idx]
    dist = ppm_series[idx] - c_equilibrium
    if abs(dist) < 1e-5: return None 
    if abs(slope) < 1e-12: return 9999.9 
    return -dist / slope

def find_50_percent_year(ppm_series, years):
    """Finds the year where concentration drops to 50% of the peak (2025)."""
    idx_2025 = 2025 - 1850
    target = (ppm_series[idx_2025] - 280.0) * 0.5
    for i in range(idx_2025, len(ppm_series)):
        if (ppm_series[i] - 280.0) <= target: return years[i]
    return "-"

# --- FORMATTERS ---
# _en = Dot decimal (International), _de = Comma decimal (German Excel)
def fmt_en(value): return f"{value:.2f}" if isinstance(value, float) else (str(value) if value is not None else "-")
def fmt_de(value): return f"{value:.2f}".replace('.', ',') if isinstance(value, float) else (str(value) if value is not None else "-")

# --- CORE SIMULATION ---
def run_simulation(model_type, variant_key):
    box_inv_eq, box_flows_eq = build_boxes(variant_key)
    num_boxes = len(box_inv_eq)
    years = np.arange(1850, 2127) 
    n_years = len(years)
    
    history_inv = np.zeros((n_years, num_boxes))
    history_ext_in = np.zeros(n_years)
    current_inv = box_inv_eq.copy()
    history_inv[0] = current_inv
    
    idx_2024 = 2024 - 1850
    idx_2025 = 2025 - 1850

    # --- PHASE 1 (1850 - 2024) ---
    for t_idx in range(0, idx_2025):
        year = years[t_idx]
        target_inv = get_atm_conc_poly(year + 1) * PPM_TO_GTC
        
        if model_type == 'Lake':
            outflows = np.zeros(num_boxes); inflows_int = np.zeros(num_boxes)
            for i in range(num_boxes):
                for j in range(num_boxes):
                    if box_flows_eq[i][j] > 0:
                        f = box_flows_eq[i][j] * (current_inv[i] / box_inv_eq[i])
                        outflows[i] += f; inflows_int[j] += f
            req_ext = target_inv - current_inv[0] - inflows_int[0] + outflows[0]
            next_inv = current_inv.copy(); next_inv[0] = target_inv 
            for i in range(1, num_boxes): next_inv[i] = current_inv[i] + inflows_int[i] - outflows[i]
            
        elif model_type == 'Fountain':
            req_ext = (target_inv - current_inv[0]) / FOUNTAIN_R
            overflow = req_ext * (1.0 - FOUNTAIN_R)
            dist_flux = np.zeros(num_boxes)
            if variant_key in ['a', 'c']: dist_flux[1] = overflow
            else: dist_flux[1] = overflow * (115/170.0); dist_flux[2] = overflow * (55/170.0)
            
            next_inv = current_inv.copy(); next_inv[0] = target_inv
            for i in range(1, num_boxes):
                natural_in = box_flows_eq[0][i] if box_flows_eq[0][i] > 0 else 0
                total_in = natural_in + dist_flux[i]
                out_to_atm = box_flows_eq[i][0]
                out_other = 0; in_other = 0
                for j in range(1, num_boxes):
                    if i==j: continue
                    if box_flows_eq[i][j]>0: out_other += box_flows_eq[i][j]*(current_inv[i]/box_inv_eq[i])
                    if box_flows_eq[j][i]>0: in_other += box_flows_eq[j][i]*(current_inv[j]/box_inv_eq[j])
                next_inv[i] = current_inv[i] + total_in + in_other - out_to_atm - out_other
        
        history_ext_in[t_idx] = req_ext
        current_inv = next_inv; history_inv[t_idx+1] = current_inv

    # --- PHASE 2 (2025 - 2124) ---
    start_flux_p2 = history_ext_in[idx_2024] if P2_START_MODE == 'relative' else P2_ABS_START_GTC
    if P2_START_MODE == 'relative' and P2_REL_START_PCT != 100.0:
        start_flux_p2 *= (P2_REL_START_PCT / 100.0)

    fountain_pulses = []; scaling_factor = None
    
    # Fountain Special Logic: New Baseline Calculation
    if model_type == 'Fountain':
        # 1. Identify connected boxes
        connected = [i for i in range(1, num_boxes) if box_flows_eq[0][i]>0 or box_flows_eq[i][0]>0]
        # 2. Sum inventories (1850 vs 2024)
        inv_base = box_inv_eq[0] + sum(box_inv_eq[i] for i in connected)
        inv_curr = history_inv[idx_2024][0] + sum(history_inv[idx_2024][i] for i in connected)
        # 3. Scaling factor
        scaling_factor = inv_curr / inv_base
        # 4. New Baseline & Initial Pulse
        atm_baseline = box_inv_eq[0] * scaling_factor
        fountain_pulses.append({'year': 2024, 'amount': history_inv[idx_2024][0] - atm_baseline, 'type': 'init'})

    for t_idx in range(idx_2025, n_years):
        year = years[t_idx]
        ext_in_p2 = get_external_influx_phase2(year, start_flux_p2)
        history_ext_in[t_idx] = ext_in_p2
        
        if model_type == 'Lake':
            outflows = np.zeros(num_boxes); inflows_int = np.zeros(num_boxes)
            for i in range(num_boxes):
                for j in range(num_boxes):
                    if box_flows_eq[i][j] > 0:
                        f = box_flows_eq[i][j] * (current_inv[i] / box_inv_eq[i])
                        outflows[i] += f; inflows_int[j] += f
            next_inv = np.zeros(num_boxes)
            next_inv[0] = current_inv[0] + ext_in_p2 + inflows_int[0] - outflows[0]
            for i in range(1, num_boxes): next_inv[i] = current_inv[i] + inflows_int[i] - outflows[i]
            
        elif model_type == 'Fountain':
            if ext_in_p2 > 0: fountain_pulses.append({'year': year, 'amount': ext_in_p2, 'type': 'ext'})
            
            # Apply Bern Model on top of new baseline
            atm_total = atm_baseline 
            for p in fountain_pulses:
                if year - p['year'] >= 0: atm_total += p['amount'] * bern_pulse_response(year - p['year'])
            
            next_inv = np.zeros(num_boxes); next_inv[0] = atm_total
            
            # Heuristic distribution for other boxes (visual consistency)
            dist_flux = np.zeros(num_boxes)
            net_out = (current_inv[0] - next_inv[0]) + ext_in_p2
            if variant_key in ['a', 'c']: dist_flux[1] = net_out
            else: dist_flux[1] = net_out * (115/170.0); dist_flux[2] = net_out * (55/170.0)
            
            for i in range(1, num_boxes):
                bal = dist_flux[i]
                for j in range(1, num_boxes):
                    if i==j: continue
                    if box_flows_eq[i][j]>0: bal -= box_flows_eq[i][j]*(current_inv[i]/box_inv_eq[i])
                    if box_flows_eq[j][i]>0: bal += box_flows_eq[j][i]*(current_inv[j]/box_inv_eq[j])
                next_inv[i] = current_inv[i] + bal
        
        current_inv = next_inv
        if t_idx + 1 < n_years: history_inv[t_idx+1] = current_inv

    inv_start_p2 = history_inv[idx_2025]
    c_final = calculate_equilibrium_concentration(model_type, inv_start_p2, box_inv_eq)
    
    return years, history_inv, history_ext_in, None, c_final, scaling_factor

def write_csv_dual(filename_base, header, data_rows, save_dir):
    """Writes two CSV files: _EN (dot separator) and _DE (semicolon separator)."""
    # English Version (Standard CSV)
    path_en = os.path.join(save_dir, f"{filename_base}_EN.csv")
    path_de = os.path.join(save_dir, f"{filename_base}_DE.csv")
    
    try:
        with open(path_en, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for row in data_rows: writer.writerow([fmt_en(item) for item in row])
                
        with open(path_de, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            for row in data_rows: writer.writerow([fmt_de(item) for item in row])
        
        print(f"  -> Exported: {filename_base}")
    except PermissionError:
        print(f"  !!! ERROR: Could not write {filename_base}. Is the file open in Excel?")

def export_results(results_store, save_dir, timestamp_str):
    print("\n--- STARTING DATA EXPORT ---")
    
    if P2_GROWTH_MODE == 'linear': val = P2_LINEAR_INC_GTC; unit = "GtC/a"
    else: val = P2_EXP_GROWTH_PCT; unit = "%"

    # --- TABLE 0: PARAMETERS ---
    t0_header = ["Parameter", "Value", "Description"]
    t0_rows = [
        ["PLOT_START_YEAR", PLOT_START_YEAR, "Start year"],
        ["PLOT_END_YEAR", PLOT_END_YEAR, "End year"],
        ["FOUNTAIN_R", FOUNTAIN_R, "Retention Factor (Phase 1)"],
        ["P2_START_MODE", P2_START_MODE, "Start Mode Phase 2"],
        ["P2_GROWTH_MODE", P2_GROWTH_MODE, "Growth Mode Phase 2"],
        ["P2_LINEAR_INC_GTC", P2_LINEAR_INC_GTC, "Linear Increment"],
        ["P2_EXP_GROWTH_PCT", P2_EXP_GROWTH_PCT, "Exponential Growth Rate"]
    ]
    write_csv_dual(f"Table_0_Parameters_{timestamp_str}", t0_header, t0_rows, save_dir)

    # --- TABLE 1: OVERVIEW ---
    try:
        t1_header = ["ID", "Model", "Variant", "Growth Value", "Growth Unit", "Scaling Factor",
                     "Ext Influx P1 Cum", "Ext Influx P1 End", "Target Val (ppm)",
                     "Tau 2025", "Tau 2030", "Tau 2050", "Tau 2100", "50% Year"]
        t1_rows = []
        tau_years = [2025, 2030, 2050, 2100]
        idx_2024 = 2024 - 1850
        
        for i, res in enumerate(results_store):
            cum = np.sum(res['ext'][:idx_2024+1])
            last = res['ext'][idx_2024]
            row = [i+1, res['model'], res['variant'], val, unit, res['scaling_factor'], cum, last, res['c_final']]
            for ty in tau_years:
                row.append(calculate_tau_phys(res['ppm'], res['years'], ty, res['c_final']))
            row.append(find_50_percent_year(res['ppm'], res['years']))
            t1_rows.append(row)
        write_csv_dual(f"Table_1_Overview_{timestamp_str}", t1_header, t1_rows, save_dir)
    except Exception as e:
        print(f"!!! ERROR creating Table 1: {e}")
        traceback.print_exc()

    # --- TABLE 3: CONCENTRATIONS ---
    try:
        t3_header = ["Year"]
        for res in results_store:
            num_boxes = res['inv'].shape[1]
            for b in range(num_boxes):
                t3_header.append(f"{res['model']} {res['variant']} Box{b+1}")
        
        t3_rows = []
        num_years = len(results_store[0]['years'])
        for t_idx in range(num_years):
            row = [results_store[0]['years'][t_idx]]
            for res in results_store:
                for b in range(res['inv'].shape[1]):
                    row.append(res['inv'][t_idx, b])
            t3_rows.append(row)
        write_csv_dual(f"Table_3_Concentrations_{timestamp_str}", t3_header, t3_rows, save_dir)
    except Exception as e:
        print(f"!!! ERROR creating Table 3: {e}")

    # --- TABLE 4: EXTERNAL INFLUX ---
    try:
        t4_header = ["Year"]
        for res in results_store:
            t4_header.append(f"{res['model']} {res['variant']} [GtC/a]")
        t4_rows = []
        for t_idx in range(num_years):
            row = [results_store[0]['years'][t_idx]]
            for res in results_store:
                row.append(res['ext'][t_idx])
            t4_rows.append(row)
        write_csv_dual(f"Table_4_External_Influx_{timestamp_str}", t4_header, t4_rows, save_dir)
    except Exception as e:
        print(f"!!! ERROR creating Table 4: {e}")
        
    print("--- EXPORT FINISHED ---\n")

def main():
    try: save_dir = os.path.dirname(os.path.abspath(__file__))
    except: save_dir = os.getcwd()
    os.chdir(save_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Starting Model Simulation... [{timestamp}]")
    print(f"Output Directory: {save_dir}")
    
    plt.rcParams.update({'font.size': 14})
    colors = {'a': 'red', 'b': 'blue', 'c': 'green', 'd': 'orange', 'e': 'purple'}
    results_store = [] 
    
    for m in ['Lake', 'Fountain']:
        for v in ACTIVE_VARIANTS:
            yrs, inv, ext, _, c_fin, sc_f = run_simulation(m, v)
            results_store.append({
                'model': m, 'variant': v, 'years': yrs, 'ppm': inv[:, 0]/PPM_TO_GTC, 
                'ext': ext, 'scaling_factor': sc_f, 'c_final': c_fin, 'color': colors[v], 'inv': inv
            })

    # EXPORT TABLES
    export_results(results_store, save_dir, timestamp)

    # PLOTS
    print("Generating Plots...")
    
    # Plot 1: Concentration
    plt.figure(figsize=(12, 7)) 
    for res in results_store:
        ls = '-' if res['model'] == 'Lake' else '--'
        mask = (res['years'] >= PLOT_START_YEAR) & (res['years'] <= PLOT_END_YEAR)
        plt.plot(res['years'][mask], res['ppm'][mask], linestyle=ls, color=res['color'], linewidth=2)
    
    if PLOT_START_YEAR <= 2124 and PLOT_END_YEAR >= 2124: plt.axvline(x=2124, color='gray', linestyle=':')
    plt.title(f"Atmospheric CO2 ({PLOT_START_YEAR}-{PLOT_END_YEAR}) | Run: {timestamp}", fontsize=12)
    plt.ylabel("Concentration [ppm]"); plt.xlabel("Year")
    plt.xlim(left=PLOT_START_YEAR, right=PLOT_END_YEAR + 2)
    plt.grid(True, linestyle='--')
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[k], lw=2, label=f"Var {k}") for k in ACTIVE_VARIANTS]
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', label='Lake'))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label='Fountain'))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.75) 
    plt.savefig(f"Plot_1_Concentration_{timestamp}.png")

    # Plot 2: Influx
    plt.figure(figsize=(12, 7))
    for res in results_store:
        ls = '-' if res['model'] == 'Lake' else '--'
        mask = (res['years'] >= PLOT_START_YEAR) & (res['years'] <= PLOT_END_YEAR)
        plt.plot(res['years'][mask], res['ext'][mask], linestyle=ls, color=res['color'], linewidth=2)
    
    plt.title(f"External Influx ({PLOT_START_YEAR}-{PLOT_END_YEAR}) | Run: {timestamp}", fontsize=12)
    plt.ylabel("Influx [GtC/a]"); plt.xlabel("Year")
    plt.xlim(left=PLOT_START_YEAR, right=PLOT_END_YEAR + 2)
    plt.grid(True, linestyle='--')
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.75)
    plt.savefig(f"Plot_2_External_Influx_{timestamp}.png")

    print(f"DONE. Please check your folder.")

if __name__ == "__main__":
    main()
