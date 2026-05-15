import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def align_and_clean(df_raw):
    df_raw.index = pd.to_datetime(df_raw.index)
    df_aligned = df_raw.dropna()
    df_cleaned = df_aligned.copy()
    
    for col in df_cleaned.columns:
        df_cleaned[col] = winsorize(df_cleaned[col], limits=[0.01, 0.01])
    
    print(f"Dimensions après nettoyage : {df_cleaned.shape}")
    return df_cleaned