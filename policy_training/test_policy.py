import time
import numpy as np
from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from reward_function import reward_function, init_custom_state

env = SpotmicroEnv(use_gui=True, reward_fn=reward_function, init_custom_state=init_custom_state, src_save_file="states/state1.5M-4-11.pkl")
obs, _ = env.reset()

# Load your trained model
model = PPO.load("policies/ppo_stand1.5M-4")  # or path to your .zip

# Run rollout
for _ in range(3001):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Terminated")
        env.plot_reward_components()  # 👈 plot per episode
        obs, _ = env.reset()
    
    time.sleep(1/60.)  # Match simulation step time for real-time playback

env.close()