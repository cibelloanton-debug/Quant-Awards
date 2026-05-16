import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from data_ingestion import get_data
import warnings

warnings.filterwarnings("ignore")

def prepare_pure_continuous_features():
    print("Ingestion : Modélisation 100% Continue (Z-Scores & Ratios)...")
    
    X_frac = get_data()
    raw_prices = yf.download(['MTUM', '^TNX', 'IVW', '^VIX'], start='2015-01-01', end='2026-05-01')['Close']
    raw_prices = raw_prices.dropna()
    
    # Cible 100% continue : Rendement réel futur à 5 jours
    y_continuous = raw_prices['MTUM'].shift(-5) / raw_prices['MTUM'] - 1
    
    common_index = X_frac.index.intersection(raw_prices.index)
    X_frac = X_frac.loc[common_index]
    raw_prices = raw_prices.loc[common_index]
    y_clean = y_continuous.loc[common_index]
    
    X = pd.DataFrame(index=common_index)
    
    # Indicateurs continus fluides (aucune variable binaire)
    X['Dist_MA200'] = raw_prices['MTUM'].shift(1) / raw_prices['MTUM'].shift(1).rolling(200).mean() - 1
    X['Vol_Ratio'] = raw_prices['MTUM'].shift(1).pct_change().rolling(10).std() / raw_prices['MTUM'].shift(1).pct_change().rolling(60).std()
    X['VIX_ZScore'] = (raw_prices['^VIX'].shift(1) - raw_prices['^VIX'].shift(1).rolling(60).mean()) / raw_prices['^VIX'].shift(1).rolling(60).std()
    X['Corr_TNX_MTUM'] = raw_prices['^TNX'].shift(1).rolling(60).corr(raw_prices['MTUM'].shift(1))
    
    X['MTUM_Vol'] = raw_prices['MTUM'].shift(1).pct_change().rolling(20).std() * np.sqrt(252)
    
    X['Target'] = y_clean
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    y_final = X.pop('Target')
    return X, y_final, raw_prices

def backtest_continuous_strategy(raw_prices, signals_series, transaction_cost=0.001):
    print(f"\nCalcul des métriques (Huber + Allocation Continue + Frais {transaction_cost*100}%)...")
    daily_returns = raw_prices['MTUM'].pct_change().shift(-1)
    risk_free_daily = (raw_prices['^TNX'] / 100 / 252).shift(-1)
    
    strat_returns = pd.Series(0.0, index=signals_series.index)
    position_changes = signals_series.diff().abs().fillna(0)
    
    for date in signals_series.index:
        if date not in daily_returns.index or pd.isna(daily_returns.loc[date]):
            continue
            
        weight = signals_series.loc[date]
        friction = position_changes.loc[date] * transaction_cost
        
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
    
    print("\nRÉSULTATS DU BACKTEST CONTINU ABSOLU")
    print(f"Ratio de Sharpe Stratégie : {strat_sharpe:.2f} | (Marché : {bh_sharpe:.2f})")
    print(f"Maximum Drawdown          : {strat_dd.min()*100:.1f}% | (Marché : {bh_dd.min()*100:.1f}%)")
    print(f"Capital Final (Base 100)  : {strat_equity.iloc[-1]:.1f} | (Marché : {bh_equity.iloc[-1]:.1f})")

def run_continuous_quant_model():
    X, y, raw_prices = prepare_pure_continuous_features()
    
    print("\nExécution : Apprentissage Linéaire Robuste aux Krachs (Huber)...")
    tscv = TimeSeriesSplit(n_splits=5, gap=5)
    
    # RobustScaler gère les valeurs extrêmes des features, Huber gère les extrêmes de la cible
    model = make_pipeline(RobustScaler(), HuberRegressor(epsilon=1.35))
    
    raw_signals = pd.Series(0.0, index=X.index)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train = y.iloc[train_index]
        
        # Isolation de la volatilité pour ne pas l'utiliser comme feature prédictive directe
        vol_test = X_test.pop('MTUM_Vol')
        X_train.pop('MTUM_Vol')
        
        model.fit(X_train, y_train)
        expected_returns = model.predict(X_test)
        
        for i, exp_ret in enumerate(expected_returns):
            current_date = X_test.index[i]
            current_vol = vol_test.iloc[i]
            
            # Allocation mathématique fluide : Plus le rendement espéré est grand face à la volatilité, plus on emprunte.
            if current_vol > 0:
                # Scaler empirique pour transposer la prédiction en levier
                signal_strength = (exp_ret / current_vol) * 10
                raw_signals.loc[current_date] = max(0.0, min(signal_strength, 2.0))
            else:
                raw_signals.loc[current_date] = 0.0
                
    # Filtre de friction pour préserver le capital
    current_weight = 1.0
    final_signals = pd.Series(1.0, index=X.index)
    
    for idx in raw_signals.index:
        target_weight = raw_signals.loc[idx]
        if abs(target_weight - current_weight) >= 0.20:
            current_weight = target_weight
        final_signals.loc[idx] = current_weight

    backtest_continuous_strategy(raw_prices, final_signals)

if __name__ == "__main__":
    run_continuous_quant_model()