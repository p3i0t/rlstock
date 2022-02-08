import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune

ray.shutdown()
ray.init(num_cpus=10)
config = {
    'env': "CartPole-v0",
    'num_workers': 4,
    'framework': 'torch',
    'model': {
        'fcnet_hiddens': [64, 64],
        'fcnet_activation': 'relu',
    },
    'evaluation_num_workers': 1,
    'evaluation_config': {
        'render_env': False
    },
    'log_level': 'WARN'
}

stop = {
    "training_iteration": 30,
    "timesteps_total": 100000,
    "episode_reward_mean": 199
}

results = tune.run(
    "PPO",
    config=config,
    stop=stop,
    verbose=2,
    checkpoint_freq=1,
    checkpoint_at_end=True,
)

checkpoint = results.get_last_checkpoint()


trainer = PPOTrainer(config=config)
trainer.restore(checkpoint)


env = gym.make('CartPole-v0')
obs = env.reset()

n_episodes = 0
ep_reward = 0.0

while n_episodes < 10:
    a = trainer.compute_single_action(
        observation=obs,
        explore=False,
        policy_id='default_policy'
    )

    obs, reward, done, _ = env.step(a)
    ep_reward += reward
    if done:
        print(f"Inference episode {n_episodes} done: reward = {ep_reward:.2f}")
        obs = env.reset()
        n_episodes += 1
        ep_reward = 0.0


ray.shutdown()


# for _ in range(30):
#     print(trainer.train())

# trainer.evaluate()