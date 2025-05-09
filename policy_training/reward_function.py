import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    linear_vel = np.array(env.agent_linear_velocity)
    angular_vel = np.array(env.agent_angular_velocity)
    contacts = env.agent_ground_feet_contacts
    scale = 1 - np.exp(- 0.333 * (env.num_steps / 1_000_000))

    effort = 0.0
    for joint in env.motor_joints:
        effort += abs(joint.effort) / joint.max_torque

    # === 1. Forward Progress ===
    fwd_velocity = np.dot(linear_vel, env._TARGET_DIRECTION)
    fwd_reward = np.clip(fwd_velocity, -1, 0.5)  # m/s, clip for robustness

    # === 2. Uprightness (Pitch & Roll) ===
    max_angle = np.radians(55)
    upright_penalty = (abs(roll) + abs(pitch)) / max_angle
    upright_reward = np.clip(1.0 - upright_penalty, 0.0, 1.0)

    # === 3. Height regulation ===
    height_target = env._TARGET_HEIGHT
    height_error = abs(base_height - height_target)
    height_reward = np.clip(1.0 - height_error / 0.1, 0.0, 1.0)

    # === 4. Energy / Smoothness ===
    # Penalize large actions and penalize deviation from previous action
    action_magnitude_penalty = np.linalg.norm(action)
    action_diff_penalty = np.linalg.norm(action - env._previous_action)
    energy_penalty = action_magnitude_penalty + 0.5 * action_diff_penalty

    # === 5. Contact (optional) ===
    contact_bonus = 1.0 if len(contacts) >= 3 else -0.5

    # === Reward weighting ===
    reward_dict = {
        "fwd_reward": 2 * fwd_reward,
        "uprightness": 2.5 * upright_reward,
        "height": 1.5 * height_reward,
        "energy_penalty": -0.8 * scale * energy_penalty,
        "contact_bonus": 2 * contact_bonus,
        "reward_penalty": -0.25 * effort
    }

    total_reward = sum(reward_dict.values())
    return total_reward, reward_dict