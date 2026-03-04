import numpy as np
from scipy.stats import qmc
from src.core.montecarlo import generate_montecarlo
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Fallback or standalone VLP models if psapy/pyResToolbox complex
# We'll implement a standardized multiphase marching algorithm using basic Beggs & Brill approximation or similar if real correlations aren't trivial to import.

def beggs_and_brill_gradient(P, T, v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D, angle, roughness):
    '''
    Simplified pseudo-Beggs & Brill gradient calculation for demonstration of Marching Algorithm.
    In a real full implementation, this calls psapy.fluid or pyResToolbox Beggs&Brill.
    '''
    # Basic properties
    rho_m_no_slip = (rho_l * v_sl + rho_g * v_sg) / max((v_sl + v_sg), 1e-6)
    v_m = v_sl + v_sg
    
    # Very simplified gradient formula (gravity + basic friction)
    g = 32.174 # ft/s2
    gc = 32.174
    
    # Frictional component approximation
    Re = 1488 * rho_m_no_slip * v_m * D / max(mu_l, 1e-6)
    f = 0.02 # Assuming rough pipe approximation for fast demo
    
    dp_dz_elev = (rho_m_no_slip * g / gc) * np.sin(np.radians(angle))
    dp_dz_fric = (f * rho_m_no_slip * v_m**2) / (2 * gc * D)
    
    dp_dz_total = dp_dz_elev + dp_dz_fric
    
    # Convert from lb/ft2/ft to psi/ft
    return dp_dz_total / 144.0

def calculate_vlp_deterministic(q_liq, q_gas, p_wh, wc, roughness, D, md_total, angle=90, steps=50):
    '''
    Marching Algorithm to calculate Bottomhole Pressure (Pwf) from Wellhead Pressure (Pwh).
    '''
    if q_liq == 0 and q_gas == 0:
        return p_wh
        
    dz = md_total / steps
    p_current = p_wh
    
    # Constant properties assumption for this model iteration:
    # In reality, PVT object updates these per step based on p_current
    rho_o = 50.0  # lb/ft3
    rho_w = 62.4  # lb/ft3
    rho_g = 0.5   # lb/ft3 (highly variable)
    
    mu_o = 2.0 # cp
    mu_w = 1.0 # cp
    mu_g = 0.02 # cp
    
    for _ in range(steps):
        # Update gas density simply by ideal gas law approx
        rho_g_local = rho_g * (p_current / 14.7) 
        
        # Superficial velocities (ft/s)
        # Area in ft2
        A = (np.pi / 4) * (D / 12)**2
        
        q_o_cfs = (q_liq * (1 - wc) * 5.615) / 86400
        q_w_cfs = (q_liq * wc * 5.615) / 86400
        q_g_cfs = (q_gas * 1000) / 86400 / max((p_current / 14.7), 1) # Approx reservoir volume
        
        v_sl = (q_o_cfs + q_w_cfs) / A
        v_sg = q_g_cfs / A
        
        rho_l = rho_w * wc + rho_o * (1 - wc)
        mu_l = mu_w * wc + mu_o * (1 - wc)
        
        dp_dz = beggs_and_brill_gradient(
            P=p_current, T=150, v_sl=v_sl, v_sg=v_sg, 
            rho_l=rho_l, rho_g=rho_g_local, mu_l=mu_l, mu_g=mu_g, 
            D=(D/12), angle=angle, roughness=roughness
        )
        
        p_current += dp_dz * dz
        
    return p_current

def generate_stochastic_vlp(rates, p_wh_dist, wc_dist, roughness_dist, d_dist, md_total, iterations=1000):
    '''
    Generate Monte Carlo VLP curves.
    Returns array of shape (len(rates), iterations) representing Pwf.
    '''
    pwf_matrix = np.zeros((len(rates), iterations))
    
    # If parameters are deterministic scalars, convert them to arrays for vectorization if possible, 
    # but since marching is sequential, we iterate.
    for i, q in enumerate(rates):
        # We assume 0 gas for simplicity in this VLP liquid model, or infer it
        q_gas = q * 500 # GOR of 500 scf/stb roughly
        
        for j in range(iterations):
            pwh = p_wh_dist[j] if isinstance(p_wh_dist, np.ndarray) else p_wh_dist
            wc = wc_dist[j] if isinstance(wc_dist, np.ndarray) else wc_dist
            rough = roughness_dist[j] if isinstance(roughness_dist, np.ndarray) else roughness_dist
            dia = d_dist[j] if isinstance(d_dist, np.ndarray) else d_dist
            
            pwf_matrix[i, j] = calculate_vlp_deterministic(
                q_liq=q, q_gas=q_gas, p_wh=pwh, wc=wc, 
                roughness=rough, D=dia, md_total=md_total
            )
            
    return pwf_matrix

def find_intersection(q_array, ipr_curves, vlp_curves):
    '''
    Find intersection points (Q_eq, Pwf_eq) from probabilistic matrices.
    ipr_curves: (num_rates, iterations)
    vlp_curves: (num_rates, iterations)
    '''
    iterations = ipr_curves.shape[1]
    q_eq = np.zeros(iterations)
    pwf_eq = np.zeros(iterations)
    
    for j in range(iterations):
        ipr_j = ipr_curves[:, j]
        vlp_j = vlp_curves[:, j]
        
        # Find where IPR and VLP cross (IPR - VLP goes from positive to negative)
        diff = ipr_j - vlp_j
        
        # Find roots
        idx = np.where(np.diff(np.sign(diff)))[0]
        
        if len(idx) > 0:
            i = idx[0]
            # Linear interpolation
            x1, x2 = q_array[i], q_array[i+1]
            y1, y2 = diff[i], diff[i+1]
            
            x_root = x1 - y1 * (x2 - x1) / (y2 - y1)
            q_eq[j] = x_root
            
            # Interpolate pwf at x_root
            pwf1, pwf2 = vlp_j[i], vlp_j[i+1]
            pwf_eq[j] = pwf1 + (x_root - x1) * (pwf2 - pwf1) / (x2 - x1)
        else:
            q_eq[j] = 0.0
            pwf_eq[j] = vlp_j[0] # Doesn't flow
            
    return q_eq, pwf_eq
