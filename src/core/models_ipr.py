import numpy as np

def enforce_q_positive(q):
    return np.maximum(q, 0.0)

def ipr_aceite_desviacion_historica(q_deterministico, factor_desviacion):
    return enforce_q_positive(q_deterministico * (1 + factor_desviacion))

def ipr_aceite_darcy(K, h, delta_p, Bo, mu, Re, Rw, S, system='english'):
    divisor = mu * Bo * (np.log(Re / Rw) - 0.75 + S)
    if system == 'english':
        q = (0.00708 * K * h * delta_p) / divisor
    else:
        divisor = 18.42 * divisor
        q = (K * h * delta_p) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))

def ipr_aceite_darcy_empirico(J, delta_p):
    return enforce_q_positive(J * delta_p)

def ipr_aceite_darcy_ynf(Km, Kf, poro_m, poro_f, h, delta_p, Bo, mu, Re, Rw, S, system='english'):
    coef_part = 1.0 - (poro_m / (poro_m + poro_f))
    k_eq = Km * h * (1.0 - coef_part) + Kf * h
    divisor = mu * Bo * (np.log(0.472 * (Re / Rw)) + S)
    c = 0.00708 if system == 'english' else 0.0525
    q = (c * k_eq * delta_p) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))

def ipr_aceite_vogel(Qtest, delta_p_test, Pr_test, delta_p_eval, Pr_eval):
    # En Vogel estrictamente se necesita la presión absoluta para la relación
    Pwf_test = Pr_test - delta_p_test
    Pwf_eval = Pr_eval - delta_p_eval
    
    rel_test = np.where(Pr_test != 0, Pwf_test / Pr_test, 0)
    qmax = Qtest / (1.0 - 0.2 * rel_test - 0.8 * rel_test**2)
    rel_p = np.where(Pr_eval != 0, Pwf_eval / Pr_eval, 0)
    q = qmax * (1.0 - 0.2 * rel_p - 0.8 * rel_p**2)
    return enforce_q_positive(q)

def ipr_aceite_babu_odeh(Kx, Kz, b, A, delta_p, Bo, mu, rw, system='english'):
    divisor = mu * Bo * (np.log(np.sqrt(A) / rw) - 0.75)
    c = 0.00708 if system == 'english' else 0.0525
    q = (c * b * np.sqrt(Kx * Kz) * delta_p) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))

def ipr_aceite_joshi(Kh, Kv, h, L, delta_p, Bo, mu, re, rw, system='english'):
    beta = np.sqrt(Kh / Kv)
    a = (L / 2.0) * (0.5 + np.sqrt(0.25 + (re / (L/2.0))**4))**0.5
    t1 = np.log((a + np.sqrt(a**2 - (L/2.0)**2)) / (L/2.0))
    t2 = (beta * h / L) * np.log(beta * h / (2 * rw))
    divisor = Bo * mu * (t1 + t2)
    c = 0.00708 if system == 'english' else 0.00424
    q = (c * Kh * h * delta_p) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))

def ipr_gas_pseudo_estable(Kg, h, delta_p2, mu_g, Z, T, re, rw, S, system='english'):
    divisor = mu_g * Z * T * (np.log(re / rw) - 0.75 + S)
    c = 0.000703 if system == 'english' else 0.00424
    q = (c * Kg * h * delta_p2) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))

def ipr_gas_economides(Kg, Kf, wf, xf, h, delta_p2, mu_g, Z, T, re, rw, system='english'):
    Sf = -np.log((xf / 2.0) / rw)
    return ipr_gas_pseudo_estable(Kg, h, delta_p2, mu_g, Z, T, re, rw, Sf, system)

def ipr_gas_joshi_horizontal(Kh, Kv, h, L, delta_p2, mu_g, Z, T, re, rw, S=0.0, system='english'):
    beta = np.sqrt(Kh / Kv)
    a = (L / 2.0) * (0.5 + np.sqrt(0.25 + (re / (L/2.0))**4))**0.5
    t1 = np.log((a + np.sqrt(a**2 - (L/2.0)**2)) / (L/2.0))
    t2 = (beta * h / L) * np.log(beta * h / (2 * rw))
    divisor = mu_g * Z * T * (t1 + t2 + S)
    c = 0.000703 if system == 'english' else 0.00424
    q = (c * Kh * h * delta_p2) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))

def ipr_gas_ynf(Km, Kf, poro_m, poro_f, h, delta_p2, mu_g, Z, T, re, rw, S, system='english'):
    coef_part = 1.0 - (poro_m / (poro_m + poro_f))
    k_eq = Km * h * (1.0 - coef_part) + Kf * h 
    divisor = mu_g * Z * T * (np.log(0.472 * (re / rw)) + S)
    c = 0.000703 if system == 'english' else 0.00424
    q = (c * k_eq * delta_p2) / divisor
    return enforce_q_positive(np.where(divisor != 0, q, np.nan))
