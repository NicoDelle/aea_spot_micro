from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from stable_baselines3.common.env_checker import check_env
from reward_function import reward_function

TOTAL_STEPS = 4_000_000

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

env = SpotmicroEnv(use_gui=False, reward_fn=reward_function)
check_env(env, warn=True) #optional

model = PPO(
    "MlpPolicy", 
    env, 
    verbose = 1, 
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.002, #previously 0.0015
    clip_range=0.1,
    tensorboard_log="./logs"
)
model.learn(total_timesteps=TOTAL_STEPS)
model.save("ppo_walk4M-7")