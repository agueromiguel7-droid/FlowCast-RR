import numpy as np

# ==========================================
# CONSTANTES
# ==========================================
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4167

# ==========================================
# ECUACIONES DE DECLINACIÓN
# ==========================================

def dca_exponencial(qi, D, t):
    """
    Declinación Exponencial
    q(t) = qi * exp(-D * t)
    t: vector de tiempo
    """
    return qi * np.exp(-D * t)

def dca_hiperbolica(qi, D, b, t):
    """
    Declinación Hiperbólica
    q(t) = qi / (1 + b * D * t)^(1/b)
    """
    return qi / ((1 + b * D * t)**(1 / b))

def dca_armonica(qi, D, t):
    """
    Declinación Armónica (Hiperbólica b=1)
    q(t) = qi / (1 + D * t)
    """
    return qi / (1 + D * t)

def dca_lineal(qi, D, t):
    """
    Declinación Lineal
    q(t) = qi - D * t
    """
    q = qi - D * t
    return np.maximum(q, 0.0)

def dca_gas_doble_exponencial(qi, D1, D2, T1, t):
    """
    Modelo de declinación exponencial de dos tramos para Gas.
    Hasta T1 usa D1. Después usa D2.
    """
    q = np.empty_like(t, dtype=float)
    
    # Fase 1: t <= T1
    mask_1 = t <= T1
    q[mask_1] = qi * np.exp(-D1 * t[mask_1])
    
    # Fase 2: t > T1
    # Gasto inicial para la fase 2 es el q al final del periodo T1
    qi_2 = qi * np.exp(-D1 * T1)
    mask_2 = t > T1
    q[mask_2] = qi_2 * np.exp(-D2 * (t[mask_2] - T1))
    
    return q

# ==========================================
# CÁLCULO DE EUR Y PERFIL DE PRODUCCIÓN
# ==========================================

def calcular_y_truncar_perfil(q_t, q_abandono=0.0):
    """
    Trunca el perfil de producción cuando q(t) < q_abandono.
    Retorna el perfil truncado.
    """
    perfil_truncado = np.where(q_t < q_abandono, 0.0, q_t)
    return perfil_truncado

def calcular_eur_mensual(perfil_q_t_bpd):
    """
    Integra numéricamente el vector de gastos q(t) [bls/día o mmpc/día].
    Considerando que cada índice t es un paso temporal (ej. mes).
    Asume t espaciado por mes. Multiplica por DAYS_PER_MONTH.
    """
    # Suma simple tipo rectángulo o trapecio. Para uso vectorial 2D:
    # perfil_q_t_bpd shape: (iteraciones, tiempos)
    
    # Volumen mensual = gasto_diario_en_ese_mes * dias_por_mes
    vol_mensual = perfil_q_t_bpd * DAYS_PER_MONTH
    
    # EUR es la suma en el tiempo (axis 1)
    eur = np.sum(vol_mensual, axis=1)
    return eur

def generar_perfil_etapa(qi, D, b, t, modelo):
    if modelo.lower() in ["exponencial", "exponential", "declinación exponencial"]:
        return dca_exponencial(qi, D, t)
    elif "hiperb" in modelo.lower() or "hyperb" in modelo.lower():
        # Prevent zero or negative b
        b_safe = np.where(b <= 0.001, 0.001, b)
        return dca_hiperbolica(qi, D, b_safe, t)
    elif "arm" in modelo.lower() or "harm" in modelo.lower():
        return dca_armonica(qi, D, t)
    elif "lineal" in modelo.lower() or "linear" in modelo.lower():
        return dca_lineal(qi, D, t)
    else:
        return np.zeros_like(qi * t)

def generar_perfil_montecarlo(qi_vec, t_steps, 
                              etapas=1, 
                              modelo1="exponencial", D1_vec=None, b1_vec=None, T1_vec=None,
                              modelo2="exponencial", D2_vec=None, b2_vec=None,
                              q_abandono=0.0):
    """
    Aplica el modelo de DCA para Montecarlo. Permite 1 o 2 etapas de declinación.
    qi_vec: vector 1D de tamaño N (iteraciones)
    t_steps: vector 1D de tamaño M (tiempo, ej. meses)
    Retorna:
    - perfil_2d: matriz (N, M) con perfiles q(t)
    - eur_vec: vector 1D de EUR para cada iteración
    """
    N = qi_vec.shape[0]
    M = t_steps.shape[0]
    
    # Usamos numpy broadcasting.
    qi = qi_vec[:, np.newaxis]
    D1 = D1_vec[:, np.newaxis] if D1_vec is not None else np.zeros((N, 1))
    b1 = b1_vec[:, np.newaxis] if b1_vec is not None else np.zeros((N, 1))
    t = t_steps[np.newaxis, :]
    
    if etapas == 1:
        q_t = generar_perfil_etapa(qi, D1, b1, t, modelo1)
    else:
        T1 = T1_vec[:, np.newaxis] if T1_vec is not None else np.full((N, 1), M)
        D2 = D2_vec[:, np.newaxis] if D2_vec is not None else np.zeros((N, 1))
        b2 = b2_vec[:, np.newaxis] if b2_vec is not None else np.zeros((N, 1))
        
        # Fase 1
        q_t_1 = generar_perfil_etapa(qi, D1, b1, t, modelo1)
        
        # Gasto inicial de la Fase 2 es el q_t evaluado en T1 (con el modelo 1)
        qi_2 = generar_perfil_etapa(qi, D1, b1, T1, modelo1)
        
        # Fase 2
        t_fase2 = np.maximum(0.0, t - T1)
        q_t_2 = generar_perfil_etapa(qi_2, D2, b2, t_fase2, modelo2)
        
        # Ensamblar matriz usando máscara
        mask_1 = t <= T1
        q_t = np.where(mask_1, q_t_1, q_t_2)

    q_t = calcular_y_truncar_perfil(q_t, q_abandono)
    eur = calcular_eur_mensual(q_t)
    
    return q_t, eur
