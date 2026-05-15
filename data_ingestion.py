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
        res.append(np.dot(weights.T, window)[0])
    return pd.Series(res, index=series.index[len(weights) - 1:])

def get_data():
    start_date = '2015-01-01'
    end_date = '2026-05-01'
    
    tickers = ['IVW', 'IVE', 'MTUM', '^VIX']
    df_market = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Variables macro (FRED en accès direct via CSV)
    df_dgs10 = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10', index_col='DATE', parse_dates=True, na_values='.')
    df_t10yie = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE', index_col='DATE', parse_dates=True, na_values='.')
    df_macro = df_dgs10.join(df_t10yie).loc[start_date:end_date]
    
    df = df_market.join(df_macro, how='inner').dropna()
    df_log = np.log(df)
    
    df_frac = pd.DataFrame({col: frac_diff(df_log[col], d=0.4) for col in df.columns})
    
    print("P-values du test de Dickey-Fuller Augmenté :")
    for col in df_frac.columns:
        p_val = adfuller(df_frac[col].dropna())[1]
        print(f"{col}: {p_val:.4f}")
        
    return df_frac