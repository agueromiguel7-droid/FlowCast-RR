import numpy as np
import scipy.stats as stats

def get_betapert_params(min_val, most_likely, max_val, l=4):
    """
    Calcula los parámetros alpha y beta para una distribución BetaPERT.
    """
    mu = (min_val + l * most_likely + max_val) / (l + 2)
    
    # Manejar el caso de distribución simétrica o donde most_likely == mu o min_val == max_val
    if np.isclose(most_likely, mu) or np.isclose(max_val, min_val) or np.isclose(mu, min_val):
        alpha = 1 + l * (most_likely - min_val) / (max_val - min_val) if not np.isclose(max_val, min_val) else 1.0
        beta = 1 + l * (max_val - most_likely) / (max_val - min_val) if not np.isclose(max_val, min_val) else 1.0
    else:
        try:
            alpha = ((mu - min_val) * (2 * most_likely - min_val - max_val)) / \
                    ((most_likely - mu) * (max_val - min_val))
            beta = (alpha * (max_val - mu)) / (mu - min_val)
        except ZeroDivisionError:
            alpha, beta = np.nan, np.nan

    # Forma alternativa estándar si la anterior presenta inestabilidad:
    if np.isnan(alpha) or np.isnan(beta) or alpha <= 0 or beta <= 0:
         alpha = 1 + l * (most_likely - min_val) / (max_val - min_val) if not np.isclose(max_val, min_val) else 1.0
         beta = 1 + l * (max_val - most_likely) / (max_val - min_val) if not np.isclose(max_val, min_val) else 1.0
         
    return alpha, beta

def generate_montecarlo(iterations: int, dist_type: str, params: dict, min_limit=None, max_limit=None):
    """
    Genera un vector de muestras usando Inverse Transform Sampling para truncamiento.
    dist_type: 'normal', 'lognormal', 'uniforme', 'triangular', 'weibull', 'gamma', 'betapert', 'deterministico'
    params: Diccionario que contiene los parámetros específicos de la distribución.
    min_limit, max_limit: Límites de truncamiento físico.
    """
    if dist_type.lower() in ['deterministico', 'determinístico']:
        val = params.get('value', 0)
        return np.full(iterations, val)
        
    dist_obj = None
    
    if dist_type.lower() == 'normal':
        mu, sigma = params['mu'], params['sigma']
        dist_obj = stats.norm(loc=mu, scale=sigma)
        
    elif dist_type.lower() == 'lognormal':
        # scipy.stats.lognorm usa parametrización: s = shape (sigma), scale=exp(mu)
        mu, sigma = params['mu'], params['sigma'] # media y std del logaritmo
        dist_obj = stats.lognorm(s=sigma, scale=np.exp(mu))
        
    elif dist_type.lower() == 'uniforme':
        min_v, max_v = params['min'], params['max']
        dist_obj = stats.uniform(loc=min_v, scale=max_v - min_v)
        
    elif dist_type.lower() == 'triangular':
        min_v, ml_v, max_v = params['min'], params['most_likely'], params['max']
        if max_v == min_v:
             return np.full(iterations, min_v)
        c = (ml_v - min_v) / (max_v - min_v)
        dist_obj = stats.triang(c=c, loc=min_v, scale=max_v - min_v)
        
    elif dist_type.lower() == 'weibull':
        shape, scale = params['shape'], params['scale']
        dist_obj = stats.weibull_min(c=shape, scale=scale)
        
    elif dist_type.lower() == 'gamma':
        shape, scale = params['shape'], params['scale']
        dist_obj = stats.gamma(a=shape, scale=scale)
        
    elif dist_type.lower() == 'betapert':
        min_v, ml_v, max_v = params['min'], params['most_likely'], params['max']
        if max_v == min_v:
             return np.full(iterations, min_v)
        alpha, beta = get_betapert_params(min_v, ml_v, max_v)
        dist_obj = stats.beta(a=alpha, b=beta, loc=min_v, scale=max_v - min_v)
        
    elif dist_type.lower() in ['exponencial', 'exponential']:
        scale = params['scale']
        dist_obj = stats.expon(scale=scale)
        
    elif dist_type.lower() == 'beta':
        alpha, beta = params['alpha'], params['beta']
        min_v, max_v = params['min'], params['max']
        if max_v == min_v:
             return np.full(iterations, min_v)
        dist_obj = stats.beta(a=alpha, b=beta, loc=min_v, scale=max_v - min_v)
        
    else:
        raise ValueError(f"Distribución {dist_type} no soportada.")

    # Truncamiento físico usando Inverse Transform Sampling
    if min_limit is not None or max_limit is not None:
        p_min = dist_obj.cdf(min_limit) if min_limit is not None else 0.0
        p_max = dist_obj.cdf(max_limit) if max_limit is not None else 1.0
        
        # En caso de que los parámetros generen p_max = p_min
        if np.isclose(p_min, p_max):
            return np.full(iterations, min_limit if min_limit is not None else 0)
            
        u = np.random.uniform(p_min, p_max, iterations)
        return dist_obj.ppf(u)
    else:
        # Sin truncamiento estricto manual
        return dist_obj.rvs(size=iterations)

def characterize_data(data, dist_types=['norm', 'lognorm', 'weibull_min', 'gamma', 'uniform']):
    """
    Prueba de bondad de ajuste de Anderson-Darling para > 15 datos.
    Retorna la mejor distribución y sus parámetros.
    """
    best_dist = None
    best_params = {}
    best_statistic = np.inf
    
    # Asegurar array 1D
    data = np.asarray(data).flatten()
    
    for dist_name in dist_types:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            # Aplicar Anderson-Darling
            stat, critical_values, significance_level = stats.anderson(data, dist_name if dist_name in ['norm','expon','logistic','gumbel'] else 'norm')
            # Scipy anderson solo soporta algunas, para el resto usamos kstest (Kolmogorov-Smirnov) como alternativa general
            kstest_res = stats.kstest(data, dist_name, args=params)
            
            # Buscamos minimizar el estadístico KS para mayor compatibilidad
            if kstest_res.statistic < best_statistic:
                best_statistic = kstest_res.statistic
                best_dist = dist_name
                best_params = params
        except Exception as e:
            continue
            
    mapped_dist = map_scipy_dist_to_internal(best_dist)
    return mapped_dist, best_params

def map_scipy_dist_to_internal(scipy_dist_name):
    dist_map = {
        'norm': 'normal',
        'lognorm': 'lognormal',
        'weibull_min': 'weibull',
        'gamma': 'gamma',
        'uniform': 'uniforme'
    }
    return dist_map.get(scipy_dist_name, 'normal')
