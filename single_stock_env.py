from typing import List

import numpy as np
import pandas as pd
import gym

def print_variable(v):
    print(f"{v=}, {type(v)}")

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
        if isinstance(position, np.ndarray):
            position = position.item()
        # 结算
        pre_price = self.holding_price_history[-1]

        ## 浮盈
        if pre_price ==  -np.inf:
            stock_delta = 0.0
        else:
            pre_stock_units = self.stock_units_history[-1]
            stock_delta = pre_stock_units * (stock_price - pre_price)

        # print_variable(stock_delta)
        # print('==========>')
        # print(f'stock delta: {stock_delta:.4f}')
        # 名义总资产
        cur_capital = self.capital_history[-1] + stock_delta
        # print_variable(cur_capital)

        # 交易
        target_stock_capital = cur_capital * position
        # print_variable(target_stock_capital)
        target_stock_units = (target_stock_capital * (1 - self.open_commision)) // stock_price
        # print(f"target_stock_units: {target_stock_units}")
        pre_stock_units = self.stock_units_history[-1]
        # print(f"pre_stock_units: {pre_stock_units}")

        cost = 0.0
        if target_stock_units > pre_stock_units: # Buy
            units_to_buy = target_stock_units - pre_stock_units
            # print(f"{pre_stock_units=}, {type(pre_stock_units)}")
            # print(f"{target_stock_units=}, {type(target_stock_units)}")
            # print(f"{units_to_buy=}, {type(units_to_buy)}")
            commision_cost = units_to_buy * stock_price * self.open_commision
            # print(f"{commision_cost=}, {type(commision_cost)}")
            close_tax_cost = 0.0
            cost = commision_cost + close_tax_cost
            # print(f"{close_tax_cost=}, {type(close_tax_cost)}")

        elif target_stock_units < pre_stock_units: # Sell
            units_to_sell = pre_stock_units - target_stock_units
            cost = (self.close_tax + self.close_commision) * units_to_sell * stock_price

        else:
            pass # do nothing


        # print(f"{cost=}, {type(cost)}")
        holding_capital = target_stock_units * stock_price
        # print(f"{target_stock_units=}, {type(target_stock_units)}")
        # print(f"{stock_price=}, {type(stock_price)}")
        # print(f"{holding_capital=}, {type(holding_capital)}")
        cash = cur_capital - cost - holding_capital

        real_capital = cash + holding_capital
        pnl = real_capital - self.capital_history[-1]
        ret = real_capital / self.capital_history[-1] - 1.0

        # print(f'{pnl=}, {type(pnl)}')
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
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(self.n_state, ))

    def _initialize_state(self):
        self.cur_idx = 0
        sample = self.df.iloc[0]
        self.today = sample.date
        # print('first day: ', self.today)
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
        # print(f"{action=}, {type(action)}")
        if not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.float32)

        assert self.action_space.contains(action)

        if self.cur_idx > len(self.df) - 1:
            self.state = np.zeros((self.n_state))
            self.reward = 0.0
            self.done = True
            self.info = {}
        else:
            sample = self.df.iloc[self.cur_idx]
            stock_price = float(sample[self.trade_col])
            # print(f'+++++++ {type(stock_price)}')
            pnl, ret = self.context.step(position=action, stock_price=stock_price)
            self.today = sample.date

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
    positions = [0.2, 0.5, 0.8, 0.95]
    positions = [0.2, 0.5]

    # c = Context()
    # for pos, p in zip(positions, prices):
    #     print(c.step(pos, p))


    # df = pd.DataFrame()

    # dates = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']
    # # prices = ['2022-01-01', '2022-01-02', '2022-01-03']
    # state1 = [-100]*4
    # state2 = [-1000]*4
    # prices = [25.45, 25.59, 26.34, 26.01]
    # df['date'] = dates
    # df['OPEN'] = prices
    # df['state1'] = state1
    # df['state2'] = state2
    # df['symbol'] = 'test'
    df = pd.read_csv('df.csv')

    symbol = '002932.XSHE'
    df_ = df[df.symbol == symbol].copy()
    # print(df_)

    state_cols = ['k1d', 'd1d', 'j1d', 'k2h', 'd2h', 'j2h']

    df_new = df_[['date', 'symbol', 'open']]
    for col in state_cols:
        df_new[col] = df_[col].shift(1)

    df_new = df_new.dropna()
    print(df_new)
    # exit(0)

    # process
    df_new[state_cols] /= 100
    # for col in df_new
    df_new['kj_1d_diff'] = df_new['j1d'] - df_new['k1d']
    df_new['dj_1d_diff'] = df_new['j1d'] - df_new['d1d']

    df_new['kj_2h_diff'] = df_new['j2h'] - df_new['k2h']
    df_new['dj_2h_diff'] = df_new['j2h'] - df_new['d2h']

    print(df_new)

    state_cols += ['kj_1d_diff', 'dj_1d_diff', 'kj_2h_diff', 'dj_2h_diff']
    env = SingleStockEnv(df=df_new, state_cols=state_cols, trade_col='open')

    obs = env.reset()
    print(obs, type(obs), obs.shape)

    # for action in positions:
    #     next_obs, r, done, _ = env.step(action)
    #     print('================')
    #     print(action, type(action))
    #     print(r, type(r))
    #     print(next_obs, type(next_obs))


    print('==========')
    env = gym.make('CartPole-v0')
    obs = env.reset()
    print(obs, type(obs), obs.shape)

    # action = env.action_space.sample()
    # print(action, type(action))

    # next_obs, r, done, _ = env.step(action)
    # print(action, type(action))
    # print(r, type(r))
    # print(next_obs, type(next_obs))









