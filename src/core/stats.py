import numpy as np
import pandas as pd
import scipy.stats as stats

# Equivalencia de distribuciones soportadas en FlowCast
DISTRIBUTIONS = {
    "Weibull (2P)": stats.weibull_min,
    "Normal": stats.norm,
    "Lognormal (2P)": stats.lognorm,
    "Exponential (1P)": stats.expon,
    "Gamma (2P)": stats.gamma,
    "Beta": stats.beta,
    "Triangular": stats.triang
}

def calculate_ad_statistic(data, dist, params):
    """Calcula el Anderson-Darling estadístico para una distribución y parámetros dados."""
    n = len(data)
    if n < 3: return np.inf
        
    sorted_data = np.sort(data)
    cdf_values = dist.cdf(sorted_data, *params)
    
    epsilon = 1e-10
    cdf_values = np.clip(cdf_values, epsilon, 1 - epsilon)
    
    S = 0
    for i in range(1, n + 1):
        F_Yi = cdf_values[i-1]
        term1 = np.log(F_Yi)
        term2 = np.log(1 - cdf_values[n - i])
        S += (2 * i - 1) * (term1 + term2)
        
    AD = -n - (1/n) * S
    return AD

def fit_all_distributions(data):
    """Ajusta todas las distribuciones soportadas y retorna un DataFrame ordenado."""
    results = []
    
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    for name, dist in DISTRIBUTIONS.items():
        try:
            if name in ["Weibull (2P)", "Lognormal (2P)", "Exponential (1P)", "Gamma (2P)", "Triangular"]:
                 params = dist.fit(data, floc=0)
            else:
                 params = dist.fit(data)
            
            ad_stat = calculate_ad_statistic(data, dist, params)
            d_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))

            results.append({
                "Distribution": name,
                "AD Statistic": ad_stat,
                "P-Value": p_value,
                "_params_obj": params
            })
        except Exception as e:
            continue
            
    if not results:
        return pd.DataFrame(columns=["Distribution", "AD Statistic", "P-Value", "_params_obj"])
        
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="AD Statistic").reset_index(drop=True)
    return df_results
