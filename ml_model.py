import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from data_ingestion import get_data
import warnings

warnings.filterwarnings("ignore")

def prepare_kelly_features():
    print("Ingestion : Modélisation pour le Critère de Kelly...")
    
    X_frac = get_data()
    raw_prices = yf.download(['MTUM', '^TNX', 'IVW', '^VIX'], start='2015-01-01', end='2026-05-01')['Close']
    raw_prices = raw_prices.dropna()
    
    # Cible : Probabilité de hausse à 5 jours
    returns_5d = raw_prices['MTUM'].shift(-5) / raw_prices['MTUM'] - 1
    y_series = pd.Series(np.where(returns_5d > 0.0, 1, 0), index=raw_prices.index)
    
    common_index = X_frac.index.intersection(raw_prices.index)
    X_frac = X_frac.loc[common_index]
    raw_prices = raw_prices.loc[common_index]
    y_clean = y_series.loc[common_index]
    
    X = pd.DataFrame(index=common_index)
    X['TNX_frac_1'] = X_frac['^TNX'].shift(1)
    X['MTUM_frac_1'] = X_frac['MTUM'].shift(1)
    X['VIX_frac_1'] = X_frac['^VIX'].shift(1)
    
    X['MTUM_Vol_20'] = raw_prices['MTUM'].shift(1).pct_change().rolling(20).std() * np.sqrt(252)
    X['VIX_Level'] = raw_prices['^VIX'].shift(1)
    
    X['Target'] = y_clean
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    y_final = X.pop('Target')
    return X, y_final, raw_prices

def backtest_kelly_strategy(raw_prices, signals_series, transaction_cost=0.001):
    print(f"\nCalcul des métriques (Critère de Kelly Continu + Frais {transaction_cost*100}%)...")
    daily_returns = raw_prices['MTUM'].pct_change().shift(-1)
    risk_free_daily = (raw_prices['^TNX'] / 100 / 252).shift(-1)
    
    strat_returns = pd.Series(0.0, index=signals_series.index)
    position_changes = signals_series.diff().abs().fillna(0)
    
    for date in signals_series.index:
        if date not in daily_returns.index or pd.isna(daily_returns.loc[date]):
            continue
            
        weight = signals_series.loc[date]
        turnover = position_changes.loc[date]
        friction = turnover * transaction_cost
        
        if weight > 0:
            borrowing_cost = risk_free_daily.loc[date] * max(0, weight - 1)
            strat_returns.loc[date] = (daily_returns.loc[date] * weight) - borrowing_cost - friction
        else:
            strat_returns.loc[date] = risk_free_daily.loc[date] - friction
            
    market_returns = daily_returns.loc[strat_returns.index]
    
    strat_equity = 100 * (1 + strat_returns.fillna(0)).cumprod()
    bh_equity = 100 * (1 + market_returns.fillna(0)).cumprod()
    
    strat_std = strat_returns.std()
    strat_sharpe = np.sqrt(252) * strat_returns.mean() / strat_std if strat_std > 0 else 0
    bh_sharpe = np.sqrt(252) * market_returns.mean() / market_returns.std()
    
    strat_dd = (strat_equity / strat_equity.cummax()) - 1.0
    bh_dd = (bh_equity / bh_equity.cummax()) - 1.0
    
    print("\nRÉSULTATS DU BACKTEST KELLY")
    print(f"Ratio de Sharpe Stratégie : {strat_sharpe:.2f} | (Marché : {bh_sharpe:.2f})")
    print(f"Maximum Drawdown          : {strat_dd.min()*100:.1f}% | (Marché : {bh_dd.min()*100:.1f}%)")
    print(f"Capital Final (Base 100)  : {strat_equity.iloc[-1]:.1f} | (Marché : {bh_equity.iloc[-1]:.1f})")

def run_kelly_model():
    X, y, raw_prices = prepare_kelly_features()
    
    print("\nExécution : Modèle d'Allocation de Kelly...")
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    raw_signals = pd.Series(0.0, index=X.index) 
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train = y.iloc[train_index]
        
        model.fit(X_train, y_train)
        probs_bull = model.predict_proba(X_test)[:, 1]
        
        for i, p_bull in enumerate(probs_bull):
            current_date = X_test.index[i]
            current_vol = X_test['MTUM_Vol_20'].iloc[i]
            
            # Application du Critère de Kelly Continu : Avantage probabiliste / Variance
            variance = current_vol ** 2
            edge = p_bull - 0.50
            
            if edge > 0.05 and variance > 0:
                # Pondération pour calmer l'agressivité de Kelly
                kelly_weight = (edge / variance) * 0.10
                raw_signals.loc[current_date] = min(kelly_weight, 2.0)
            else:
                raw_signals.loc[current_date] = 0.0
                
    # Filtre d'Hystérésis (économie de frais)
    current_weight = 0.0
    final_signals = pd.Series(0.0, index=X.index)
    
    for idx in X.index:
        target_weight = raw_signals.loc[idx]
        if target_weight == 0.0:
            current_weight = 0.0
        elif abs(target_weight - current_weight) >= 0.25:
            current_weight = target_weight
        final_signals.loc[idx] = current_weight
        
    backtest_kelly_strategy(raw_prices, final_signals)

if __name__ == "__main__":
    run_kelly_model()