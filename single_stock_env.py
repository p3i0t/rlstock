from typing import List

import numpy as np
import pandas as pd
import gym

class Context:
    """Maintain the trading records and compute pnl.
    """
    def __init__(
        self, initial_money: float = 1000000,
        close_tax: float = 0.001,
        open_commision: float = 0.0003,
        close_commision: float = 0.0003) -> None:

        # total_capital = position * holding_price + cash

        self.initial_money = initial_money
        self.close_tax = close_tax
        self.open_commision = open_commision
        self.close_commision = close_commision

        self._initialize()

    def _initialize(self):
        self.position_history = [0.0]
        self.stock_units_history = [0]
        self.holding_price_history = [-np.inf]
        self.holding_capital_history = [0.0]
        self.cash_history = [self.initial_money]
        self.capital_history = [self.initial_money]
        self.pnl_history = []
        self.ret_history = []

    def step(self, position: float, stock_price: float):
        # 结算
        pre_price = self.holding_price_history[-1]
        ## 浮盈
        if pre_price ==  -np.inf:
            stock_delta = 0.0
        else:
            pre_stock_units = self.stock_units_history[-1]
            stock_delta = pre_stock_units * (stock_price - pre_price)
        # print('==========>')
        # print(f'stock delta: {stock_delta:.4f}')
        # 名义总资产
        cur_capital = self.capital_history[-1] + stock_delta

        # 交易
        target_stock_capital = cur_capital * position
        target_stock_units = (target_stock_capital * (1 - self.open_commision)) // stock_price
        # print(f"target_stock_units: {target_stock_units}")
        pre_stock_units = self.stock_units_history[-1]
        # print(f"pre_stock_units: {pre_stock_units}")

        cost = 0.0
        if target_stock_units > pre_stock_units: # Buy
            units_to_buy = target_stock_units - pre_stock_units
            commision_cost = units_to_buy * stock_price * self.open_commision
            close_tax_cost = 0.0
            cost = commision_cost + close_tax_cost

        elif target_stock_units < pre_stock_units: # Sell
            units_to_sell = pre_stock_units - target_stock_units
            cost = (self.close_tax + self.close_commision) * units_to_sell * stock_price

        else:
            pass # do nothing

        holding_capital = target_stock_units * stock_price
        cash = cur_capital - cost - holding_capital

        real_capital = cash + holding_capital
        pnl = real_capital - self.capital_history[-1]
        ret = real_capital / self.capital_history[-1] - 1.0

        self.stock_units_history.append(target_stock_units)
        self.position_history.append(position)
        self.holding_capital_history.append(holding_capital)
        self.cash_history.append(cash)
        self.holding_price_history.append(stock_price)
        self.capital_history.append(real_capital)
        self.pnl_history.append(pnl)
        self.ret_history.append(ret)

        return pnl, ret

    # def get_capital()


class SingleStockEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, state_cols: List[str],
                 trade_col: str = 'OPEN',
                 initial_money: float = 1000000,
                 close_tax: float = 0.001,
                 open_commision: float = 0.0003,
                 close_commision: float = 0.0003) -> None:
        super().__init__()
        self.df = df.sort_values('date')
        assert len(self.df) > 2
        # print(set(['symbol', 'date', trade_col] + state_cols))
        # print(set(df.columns))
        assert set(['symbol', 'date', trade_col] + state_cols).issubset(set(df.columns)), f'some columns are missing.'

        self.state_cols = state_cols
        self.trade_col = trade_col
        self.n_state = len(state_cols) + 1

        self.context = Context(
            initial_money=initial_money, close_tax=close_tax,
            open_commision=open_commision,
            close_commision=close_commision)

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, )) # 0 for buy, 1 for sell, 2 for do nothing
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_state, ))

    def _initialize_state(self):
        self.cur_idx = 0
        sample = self.df.iloc[0]
        self.today = sample.date
        print(self.today)
        cur_position = 0.0
        self.state = np.append(sample[self.state_cols].values, cur_position)
        self.reward = 0.0
        self.done = False
        self.info = {'date': self.today, 'cur_idx': self.cur_idx}

    def reset(self):
        self._initialize_state()
        self.context._initialize()
        return self.state

    def step(self, action: float):
        assert self.action_space.contains(np.array([action], dtype=np.float32))

        sample = self.df.iloc[self.cur_idx]
        stock_price = sample[self.trade_col]
        pnl, ret = self.context.step(position=action, stock_price=stock_price)
        self.today = sample.date

        if self.cur_idx > len(self.df) - 1:
            self.state = np.zeros((self.n_state))
            self.reward = 0.0
            self.done = True
            self.info = {}
        else:
            cur_position = self.context.position_history[-1]
            self.state = np.append(sample[self.state_cols].values, cur_position)
            self.reward = pnl
            self.done = False
            self.info = {'date': self.today, 'cur_idx': self.cur_idx}

        self.cur_idx += 1

        return self.state, self.reward, self.done, self.info

    # def render(self, mode="human"):
    #     return super().render(mode)


if __name__ == '__main__':
    positions = [0.2, 0.5, 0.8, 0.9]

    # c = Context()
    # for pos, p in zip(positions, prices):
    #     print(c.step(pos, p))


    df = pd.DataFrame()

    dates = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']
    # prices = ['2022-01-01', '2022-01-02', '2022-01-03']
    state1 = [-100]*4
    state2 = [-1000]*4
    prices = [25.45, 25.59, 26.34, 26.01]
    df['date'] = dates
    df['OPEN'] = prices
    df['state1'] = state1
    df['state2'] = state2
    df['symbol'] = 'test'

    env = SingleStockEnv(df=df, state_cols=['state1', 'state2'], trade_col='OPEN')

    obs = env.reset()
    print(obs)

    for action in positions:
        next_obs, r, done, _ = env.step(action)
        print(action, r, next_obs)









