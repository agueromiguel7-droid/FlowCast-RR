import numpy as np

# Constantes comunes
# PRD: "(Restricción: q ≥ 0)"

def enforce_q_positive(q):
    return np.maximum(q, 0.0)

# ==========================================
# MODELOS PARA ACEITE
# ==========================================

def ipr_aceite_desviacion_historica(q_deterministico, factor_desviacion):
    """
    q_deterministico: escalar o vector (Gasto inicial estimado)
    factor_desviacion: vector estocástico que propaga la incertidumbre
    """
    q = q_deterministico * (1 + factor_desviacion)
    return enforce_q_positive(q)

def ipr_aceite_darcy(K, h, Pr, Pwf, Bo, mu, Re, Rw, S, system='english'):
    """
    Aceite - Darcy (Flujo Radial Pseudo-estable)
    system: 'english' o 'international'
    En Inglés: Qo = (0.00708 * K * h * (Pr - Pwf)) / (mu * Bo * (ln(Re/Rw) - 0.75 + S))
    """
    # ∆P = Pr - Pwf se asegura de mantener la congruencia de presión si se calculan juntas o separadas.
    delta_p = Pr - Pwf
    
    if system == 'english':
        constant = 0.00708
    else:
        constant = 0.00708 # Add international constant if needed, defaulting to 0.00708
        
    divisor = mu * Bo * (np.log(Re / Rw) - 0.75 + S)
    
    # Manejo de división por cero
    valid_divisor = np.where(divisor != 0, divisor, np.nan)
    q = (constant * K * h * delta_p) / valid_divisor
    
    return enforce_q_positive(q)

def ipr_aceite_darcy_empirico(J, Pr, Pwf):
    """
    IPR Darcy - Método Empírico
    Qo = J * (Pr - Pwf)
    """
    q = J * (Pr - Pwf)
    return enforce_q_positive(q)

def ipr_aceite_vogel(Qmax, Pr, Pwf):
    """
    Aceite - Vogel (Yacimientos Saturados)
    Qo = Qmax * [1 - 0.2*(Pwf/Pr) - 0.8*(Pwf/Pr)^2]
    """
    rel_p = Pwf / Pr
    q = Qmax * (1 - 0.2 * rel_p - 0.8 * rel_p**2)
    return enforce_q_positive(q)

def ipr_aceite_babu_odeh(Kx, Kz, A, Pr, Pwf, Bo, mu, rw, SR, Ln_CH):
    """
    IPR Babu & Odeh (Pozo Horizontal) - Método Analítico
    Qo = (0.00708 * b * sqrt(Kx * Kz) * (Pr - Pwf)) / (mu * Bo * (Ln_CH - 0.75 + SR))
    Nota: Según PRD requiere Ln CH y SR. Asumimos b = Longitud horizontal u otra constante del área A.
    Simplificación basada en la descripción parcial:
    """
    b = np.sqrt(A) # Estimación general si b no es provista, o debe venir en args. Asumiremos b = parametro de área
    delta_p = Pr - Pwf
    divisor = mu * Bo * (Ln_CH - 0.75 + SR)
    valid_divisor = np.where(divisor != 0, divisor, np.nan)
    q = (0.00708 * b * np.sqrt(Kx * Kz) * delta_p) / valid_divisor
    return enforce_q_positive(q)

def ipr_aceite_joshi(Kh, Kv, h, L, Pr, Pwf, Bo, mu, re, rw):
    """
    IPR Joshi (Pozo Horizontal) - Método Analítico
    """
    Iani = np.sqrt(Kh / Kv)
    a = (L / 2) * (0.5 + np.sqrt(0.25 + (2 * re / L)**2))
    
    num = 0.00708 * Kh * h * (Pr - Pwf)
    term1 = np.log((a + np.sqrt(a**2 - (L/2)**2)) / (L/2))
    term2 = (Iani * h / L) * np.log(Iani * h / (rw * (Iani + 1)))
    divisor = mu * Bo * (term1 + term2)
    
    valid_divisor = np.where(divisor != 0, divisor, np.nan)
    q = num / valid_divisor
    return enforce_q_positive(q)


# ==========================================
# MODELOS PARA GAS
# ==========================================

def ipr_gas_pseudo_estable(K, h, Pr, Pwf, mu_g, Z, T, re, rw, S):
    """
    Caudal en Estado Pseudo Estable para Gas
    Qg = (0.703 * K * h * (Pr^2 - Pwf^2)) / (mu_g * Z * T * (ln(re/rw) - 0.75 + S))
    """
    delta_p2 = Pr**2 - Pwf**2
    num = 0.703 * K * h * delta_p2
    divisor = mu_g * Z * T * (np.log(re/rw) - 0.75 + S)
    
    valid_divisor = np.where(divisor != 0, divisor, np.nan)
    q = num / valid_divisor
    return enforce_q_positive(q)

def ipr_gas_economides(Kg, h, Pr, Pwf, mu_g, Z, T, re, rw, Sf):
    """
    Gasto en Estado Pseudo Estable para Fracturamiento Hidráulico (Economides)
    Usa el mismo modelo con Sf.
    """
    return ipr_gas_pseudo_estable(Kg, h, Pr, Pwf, mu_g, Z, T, re, rw, Sf)

def ipr_gas_joshi_horizontal(K, h, Pr, Pwf, mu_g, Z, T, reh, L, rev, a, rw, S):
    """
    Gasto en Estado Estable - Pozo Horizontal (Joshi) para Gas
    """
    delta_p2 = Pr**2 - Pwf**2
    num = 0.703 * K * h * delta_p2
    
    refw = (r_effective_wellbore(a, L, rev)) # simplificado
    term1 = np.log(reh / refw)
    divisor = mu_g * Z * T * (term1 + S)
    
    valid_divisor = np.where(divisor != 0, divisor, np.nan)
    q = num / valid_divisor
    return enforce_q_positive(q)

def r_effective_wellbore(a, L, rev):
    # simplificacion de la ecuacion del radio efectivo
    return (a + np.sqrt(a**2 - (L/2)**2)) / (L/2)

def ipr_gas_ynf(Km, Kf, h, Pr, Pwf, mu_g, Z, T, re, rw, S, Coef):
    """
    Gasto en Estado Estable para Yacimientos Naturalmente Fracturados (Gas)
    """
    K_total = Km + Kf * Coef # simplificacion asumiendo partición simple
    return ipr_gas_pseudo_estable(K_total, h, Pr, Pwf, mu_g, Z, T, re, rw, S)
