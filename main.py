from typing import List, Dict
import os

import numpy as np
import pandas as pd
import gym

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray import tune

from single_stock_env import SingleStockEnv
from models import MyFCNet

ray.shutdown()
ray.init(num_cpus=22)


def process_df():
    df = pd.read_csv('df.csv')
    state_cols = ['k1d', 'd1d', 'j1d', 'k2h', 'd2h', 'j2h']
    df[state_cols] /= 100 # scale to [0, 1]
    # for col in df
    df['kj_1d_diff'] = df['j1d'] - df['k1d']
    df['dj_1d_diff'] = df['j1d'] - df['d1d']

    df['kj_2h_diff'] = df['j2h'] - df['k2h']
    df['dj_2h_diff'] = df['j2h'] - df['d2h']

    df.to_csv('df_processed.csv')
    return df

# process_df()
# exit(0)


def env_creator(env_config: Dict):
    state_cols = ['k1d', 'd1d', 'j1d', 'k2h', 'd2h', 'j2h']
    state_cols += ['kj_1d_diff', 'dj_1d_diff', 'kj_2h_diff', 'dj_2h_diff']

    df = pd.read_csv('df_processed.csv')
    sample = df[df.symbol == env_config['symbol']].sort_values('date').copy()

    df_sample = sample[['date', 'symbol', env_config['trade_col']]].copy()
    for col in state_cols:
        df_sample[col] = sample[col].shift(1)
    df_sample = df_sample.dropna()

    n_days = len(df_sample)

    print(f"game length: {len(df_sample)}")
    # bound = int(0.8 * n_days)

    # if env_config['mode'] == 'train':
    #     df_run = df_sample[:bound]
    # else:
    #     df_run = df_sample[bound:]
    df_run = df_sample
    env = SingleStockEnv(df=df_run, state_cols=state_cols, trade_col=env_config['trade_col'], initial_money=100000)
    return env


register_env('my_env', env_creator=env_creator)

ModelCatalog.register_custom_model(
    'my_fc', MyFCNet
)

config = {
    'env': "my_env",
    'env_config': {
        'dataframe_path': 'df.csv',
        # 'symbol': '601088.XSHG', # single stock symbol
        # 'symbol': '600866.XSHG', # single stock symbol
        # 'symbol': '600466.XSHG', # single stock symbol
        # 'symbol': '600546.XSHG', # single stock symbol
        # 'symbol': '603363.XSHG', # single stock symbol
        # 'symbol': '601615.XSHG', # single stock symbol
        # 'symbol': '603858.XSHG', # single stock symbol
        # 'symbol': '600732.XSHG', # single stock symbol
        # 'symbol': '002932.XSHE', # single stock symbol
        # 'symbol': '000402.XSHE', # single stock symbol
        # 'symbol': '002605.XSHE', # single stock symbol
        # 'symbol': '002815.XSHE', # single stock symbol
        # 'symbol': '002408.XSHE', # single stock symbol
        # 'symbol': '603719.XSHG', # single stock symbol
        # 'symbol': '603650.XSHG', # single stock symbol
        # 'symbol': '600171.XSHG', # single stock symbol
        # 'symbol': '002867.XSHE', # single stock symbol
        # 'symbol': '001965.XSHE', # single stock symbol
        'symbol': '605358.XSHG', # single stock symbol
        # 'trade_col': 'avg', # trading price
        'trade_col': 'avg', # trading price
        'mode': 'train'
    },
    'normalize_actions': True, # clip actions to be in range [low, high]

    # 'env': 'CartPole-v0',
    'num_workers': 20,
    'framework': 'torch',
    # 'framework': 'tf',
    'model': {
        # 'custom_model': 'my_fc',
        # 'fcnet_hiddens': [64, 64],
        'fcnet_hiddens': [128, 128],
        'fcnet_activation': 'tanh',

        ## LSTM
        # 'use_lstm': True,
        # 'max_seq_len': 10,
        # 'lstm_cell_size': 128
    },
    'lr': 1e-3,
    'evaluation_num_workers': 1,
    'evaluation_config': {
        'render_env': False
    },

    'vf_clip_param': 100,
    'log_level': 'WARN',
}

stop = {
    "training_iteration": 100.0,
    "timesteps_total": 100000,
    # "episode_reward_mean": 199
}

# results = tune.run(
#     "PPO",
#     config=config,
#     stop=stop,
#     verbose=2,
#     checkpoint_freq=1,
#     checkpoint_at_end=True,
# )

# checkpoint = results.get_last_checkpoint()


trainer = PPOTrainer(config=config)
save_dir = 'save'

train = False
if train is True:
    for _ in range(100):
        result = trainer.train()
        print(pretty_print(result))
        if result['timesteps_total'] >= stop['timesteps_total']:
            break

    res = trainer.save(save_dir)
    print(res)


else:
    ckpt_path = 'save/checkpoint_000025/checkpoint-25'
    trainer.restore(ckpt_path)

    # Evaluation
    config['env_config']['mode'] = 'test'
    env = env_creator(config['env_config'])
    obs = env.reset()

    n_episodes = 0
    ep_reward = 0.0

    while n_episodes < 1:
        a = trainer.compute_single_action(
            observation=obs,
            explore=False,
            policy_id='default_policy'
        )
        a = np.clip(a, 0.0, 1.0)
        # print(f"{a=}, {type(a)}")
        obs, reward, done, info = env.step(a)

        ep_reward += reward
        if done:
            print(f"Inference episode {n_episodes} done: reward = {ep_reward:.2f}")
            obs = env.reset()
            n_episodes += 1
            ep_reward = 0.0
        else:
            print(f"date: {info['date']}, pnl: {reward:9.2f}, position: {a.item():6.4f}, ret: {info['step_ret']:6.4f}")

ray.shutdown()



# for _ in range(30):
#     print(trainer.train())

# trainer.evaluate()