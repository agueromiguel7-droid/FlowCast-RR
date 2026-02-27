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

def generar_perfil_montecarlo(qi_vec, D_vec, b_vec, t_steps, modelo="exponencial", q_abandono=0.0):
    """
    Aplica el modelo de DCA para Montecarlo.
    qi_vec, D_vec, b_vec: vectores 1D de tamaño N (iteraciones)
    t_steps: vector 1D de tamaño M (tiempo, ej. meses)
    Retorna:
    - perfil_2d: matriz (N, M) con perfiles q(t)
    - eur_vec: vector 1D de EUR para cada iteración
    """
    N = qi_vec.shape[0]
    M = t_steps.shape[0]
    
    # Usamos numpy broadcasting.
    # qi_vec (N,) -> (N, 1)
    qi = qi_vec[:, np.newaxis]
    D = D_vec[:, np.newaxis]
    if b_vec is not None:
        b = b_vec[:, np.newaxis]
    # t_steps (M,) -> (1, M)
    t = t_steps[np.newaxis, :]
    
    if modelo.lower() == "exponencial":
        q_t = dca_exponencial(qi, D, t)
    elif modelo.lower() == "hiperbolica":
        q_t = dca_hiperbolica(qi, D, b, t)
    elif modelo.lower() == "armonica":
        q_t = dca_armonica(qi, D, t)
    elif modelo.lower() == "lineal":
        q_t = dca_lineal(qi, D, t)
    else:
        q_t = np.zeros((N, M))
        
    q_t = calcular_y_truncar_perfil(q_t, q_abandono)
    eur = calcular_eur_mensual(q_t)
    
    return q_t, eur
