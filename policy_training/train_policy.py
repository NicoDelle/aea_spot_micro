from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from reward_function import reward_function, init_custom_state

TOTAL_STEPS = 4_000_000

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

def make_env(rank):
    def _init():
        env = SpotmicroEnv(
            use_gui=False,
            reward_fn=reward_function,
            init_custom_state=init_custom_state,
            dest_save_file=f"state4M-{rank}.pkl"
        )
        env.seed(100 + rank)
        return env
    return _init

def main():
    num_envs = 12
    vec_env = SubprocVecEnv([make_env(rank) for rank in range(num_envs)])

    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=0, 
        learning_rate=clipped_linear_schedule(3e-4),
        ent_coef=0.002,
        clip_range=0.1,
        tensorboard_log="./logs"
    )

    model.learn(total_timesteps=TOTAL_STEPS)
    model.save("ppo_walk4M-3-norm")
    vec_env.close()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # safe in Docker
    main()

#400k at 11:55