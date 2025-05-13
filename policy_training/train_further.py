from SpotmicroEnv import SpotmicroEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from reward_function import reward_function

TOTAL_STEPS = 2_000_000

env = SpotmicroEnv(use_gui=False, reward_fn=reward_function)
check_env(env, warn=True) #optional

model = PPO.load("ppo_walk3M-6")
model.set_env(env)
model.tensorboard_log = "./logs"
model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False)
model.save("ppo_walk5M-6")