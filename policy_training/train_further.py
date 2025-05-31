from SpotmicroEnv import SpotmicroEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from reward_function import reward_function, init_custom_state

TOTAL_STEPS = 2_000_000

env = SpotmicroEnv(
    use_gui=False,
    reward_fn=reward_function, 
    init_custom_state=init_custom_state, 
    src_save_file="states/state5M-2.pkl",
    dest_save_file="states/state7M-2.pkl"
    )
check_env(env, warn=True) #optional

model = PPO.load("policies/ppo_walk5M-2")
model.set_env(env)
model.tensorboard_log = "./logs"
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    )
model.save("policies/ppo_walk7M-2")