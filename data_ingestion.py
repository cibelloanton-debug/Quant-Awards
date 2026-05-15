import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff(series, d, threshold=1e-4):
    weights = get_weights(d, len(series))
    weights = weights[np.abs(weights) >= threshold]
    res = []
    for i in range(len(weights) - 1, len(series)):
        window = series.iloc[i - len(weights) + 1 : i + 1]
        res.append(np.dot(weights.flatten(), window))
    return pd.Series(res, index=series.index[len(weights) - 1:])

def find_optimal_d(series):
    # Cherche le plus petit d qui rend la série stationnaire
    for d in np.arange(0.1, 1.05, 0.05):
        diff_series = frac_diff(series, d=d).dropna()
        if len(diff_series) > 100:
            p_val = adfuller(diff_series)[1]
            if p_val < 0.05:
                return d, diff_series
    # Différenciation classique (rendements) en dernier recours
    return 1.0, series.diff().dropna() 

def get_data():
    start_date = '2015-01-01'
    end_date = '2026-05-01'
    
    tickers = ['IVW', 'IVE', 'MTUM', '^VIX', '^TNX', 'TIP']
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    df = df.dropna()
    df_log = np.log(df)
    
    print("\nRecherche du paramètre de différenciation optimal (d) par actif...")
    df_frac_dict = {}
    for col in df_log.columns:
        opt_d, diff_series = find_optimal_d(df_log[col])
        print(f"{col}: d = {opt_d:.2f}")
        df_frac_dict[col] = diff_series
        
    df_frac = pd.DataFrame(df_frac_dict).dropna()
    return df_frac