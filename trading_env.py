import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, initial_capital=100.0, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        self.df = df
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # Action continue : l'agent décide de son levier de 0.0 (Cash) à 2.0 (Emprunt maximal)
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Observation : Variables de marché + Exposition actuelle (crucial pour que l'IA apprenne l'inertie)
        self.obs_shape = self.df.shape[1] + 1 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weight = 1.0 
        self.portfolio_value = self.initial_capital
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.append(self.df.iloc[self.current_step].values, self.current_weight)
        return obs.astype(np.float32)

    def step(self, action):
        target_weight = action[0]
        
        turnover = abs(target_weight - self.current_weight)
        friction = turnover * self.transaction_cost
        
        market_return = self.df.iloc[self.current_step]['MTUM_Ret']
        risk_free_return = self.df.iloc[self.current_step]['TNX_Ret'] / 252.0
        
        if target_weight > 1.0:
            borrowing_cost = risk_free_return * (target_weight - 1.0)
            step_return = (market_return * target_weight) - borrowing_cost - friction
        else:
            cash_return = risk_free_return * (1.0 - target_weight)
            step_return = (market_return * target_weight) + cash_return - friction
            
        self.portfolio_value *= (1 + step_return)
        
        # Fonction de récompense : l'agent est jugé sur le rendement pur net de frais
        reward = step_return
        
        self.current_weight = target_weight
        self.current_step += 1
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.portfolio_value <= 0.0
        
        info = {"portfolio_value": self.portfolio_value, "turnover": turnover}
        
        return self._get_obs(), reward, terminated, truncated, info