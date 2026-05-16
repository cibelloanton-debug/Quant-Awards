import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_ingestion import get_data
from trading_env import TradingEnv
import warnings

warnings.filterwarnings("ignore")

# --- 1. FONCTIONS DE DIFFÉRENCIATION (PHASE 2) ---
def get_weights_ffd(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] * (d - k + 1) / k
        w.append(w_)
    return np.array(w[::-1])

def frac_diff_ffd(series, d, window=100):
    weights = get_weights_ffd(d, window)
    def apply_weights(x):
        return np.dot(x, weights)
    return series.rolling(window).apply(apply_weights, raw=True)

def prepare_rl_environment_data():
    X_frac = get_data()
    raw_prices = yf.download(['MTUM', '^TNX', 'IVW', '^VIX'], start='2015-01-01', end='2026-05-01')['Close']
    raw_prices = raw_prices.dropna()
    
    common_index = X_frac.index.intersection(raw_prices.index)
    X_frac = X_frac.loc[common_index]
    raw_prices = raw_prices.loc[common_index]
    
    X = pd.DataFrame(index=common_index)
    
    X['MTUM_Frac_0.45'] = frac_diff_ffd(raw_prices['MTUM'], d=0.45, window=100)
    X['MTUM_Ret'] = raw_prices['MTUM'].pct_change()
    X['TNX_Ret'] = raw_prices['^TNX'].pct_change()
    X['MTUM_Vol_20'] = raw_prices['MTUM'].pct_change().rolling(20).std() * np.sqrt(252)
    X['VIX_Spike'] = raw_prices['^VIX'] / raw_prices['^VIX'].rolling(50).mean()
    X['Cross_Asset_Momentum'] = raw_prices['MTUM'].pct_change(20) - raw_prices['IVW'].pct_change(20)
    
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    raw_prices = raw_prices.loc[X.index]
    
    return X, raw_prices

# --- 2. ÉVALUATION ET BACKTEST HORS ÉCHANTILLON ---
def evaluate_agent(model, env_test, df_test, raw_prices_test):
    print("\nÉvaluation de l'Agent sur des données inconnues (Out-of-Sample)...")
    
    obs, _ = env_test.reset()
    terminated = False
    truncated = False
    
    portfolio_history = [100.0]
    market_history = [100.0]
    
    daily_returns = raw_prices_test['MTUM'].pct_change().dropna().values
    
    step_idx = 0
    while not (terminated or truncated):
        # L'agent décide de son levier de manière déterministe (sans exploration aléatoire)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        
        portfolio_history.append(info['portfolio_value'])
        if step_idx < len(daily_returns):
            market_history.append(market_history[-1] * (1 + daily_returns[step_idx]))
        
        step_idx += 1

    port_series = pd.Series(portfolio_history)
    market_series = pd.Series(market_history)
    
    # Calcul des métriques de survie
    strat_returns = port_series.pct_change().dropna()
    market_returns = market_series.pct_change().dropna()
    
    strat_std = strat_returns.std()
    strat_sharpe = np.sqrt(252) * strat_returns.mean() / strat_std if strat_std > 0 else 0
    bh_sharpe = np.sqrt(252) * market_returns.mean() / market_returns.std()
    
    strat_dd = (port_series / port_series.cummax() - 1.0).min() * 100
    bh_dd = (market_series / market_series.cummax() - 1.0).min() * 100
    
    print("\nRÉSULTATS DE L'AGENT DRL (NET DE FRAIS 0.1%)")
    print(f"Ratio de Sharpe Stratégie : {strat_sharpe:.2f} | (Marché : {bh_sharpe:.2f})")
    print(f"Maximum Drawdown          : {strat_dd:.1f}% | (Marché : {bh_dd:.1f}%)")
    print(f"Capital Final (Base 100)  : {port_series.iloc[-1]:.1f} | (Marché : {market_series.iloc[-1]:.1f})")

# --- 3. MOTEUR D'ENTRAÎNEMENT PPO ---
def run_drl_model():
    df_state, raw_prices = prepare_rl_environment_data()
    
    # Scission stricte (Train: 80% / Test: 20%) pour interdire la triche temporelle
    split_idx = int(len(df_state) * 0.8)
    df_train = df_state.iloc[:split_idx]
    df_test = df_state.iloc[split_idx:]
    prices_test = raw_prices.iloc[split_idx:]
    
    print(f"\nDonnées d'entraînement : {len(df_train)} jours.")
    print(f"Données de test (Inconnues) : {len(df_test)} jours.")
    
    # Vectorisation de l'environnement d'entraînement
    env_train_fn = lambda: TradingEnv(df_train, transaction_cost=0.001)
    vec_env_train = DummyVecEnv([env_train_fn])
    
    print("\nInitialisation du Réseau de Neurones Profond (PPO)...")
    # Politique MlpPolicy : Architecture dense classique
    model = PPO("MlpPolicy", vec_env_train, learning_rate=0.0003, n_steps=2048, batch_size=64, verbose=0, seed=42)
    
    print("Entraînement en cours (100 000 itérations). La machine apprend la douleur de la friction...")
    model.learn(total_timesteps=100000)
    
    print("Entraînement terminé. L'Agent est prêt.")
    
    # Environnement de test (Out-Of-Sample)
    env_test = TradingEnv(df_test, transaction_cost=0.001)
    evaluate_agent(model, env_test, df_test, prices_test)
    model.save("ppo_quant_awards_model")

if __name__ == "__main__":
    run_drl_model()