from typing import List

import numpy as np
import pandas as pd
import gym

class SingleStockDiscreteEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, state_cols: List[str], trade_col: str = 'OPEN') -> None:
        super().__init__()
        self.df = df.sort_values('date')
        assert set(['symbol', 'date', trade_col] + state_cols) in set(df.columns), f'some columns are missing.'

        self.state_cols = state_cols
        self.cur_position = 0
        self.price_in = -100
        # self.price_out = -100

        n_state = len(state_cols) + 1
        self.action_space = gym.spaces.Discrete(2) # 0 for buy, 1 for sell
        self.observation_space = gym.spaces.Box(np.zeros((n_state, )), np.ones((n_state, )))

    
    def reset(self):
        sample = self.df.iloc[0]
        self.cur_idx = 0
        self.state = np.append(sample[self.state_cols].values, self.cur_position)
        self.reward = 0.0
        self.done = False
        self.info = {'date': sample.date.item(), 'cur_idx': 0}

        return self.state

    def step(self, action: int):
        assert self.action_space.contains(action)
        if self.done:
            pass
        else:
            if action == 0:
                if self.cur_position == 0:
                    self.cur_position = 1 # buy in
                    self.price_in = self.df[self.df[]]




