from SpotmicroEnv import SpotmicroEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

TOTAL_STEPS = 300_000

env = SpotmicroEnv(use_gui=False)
check_env(env, warn=True) #optional

model = PPO.load("ppo_walk900k.3")
model.set_env(env)
model.tensorboard_log = "./logs"
model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False)
model.save("ppo_walk1.2M.3")