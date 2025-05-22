import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    linear_vel = np.array(env.agent_linear_velocity)
    angular_vel = np.array(env.agent_angular_velocity)
    contacts = env.agent_ground_feet_contacts
    scale = lambda coeff: 1 - np.exp(- coeff * (env.num_steps / 1_000_000)) #0->1

    effort = 0.0
    for joint in env.motor_joints:
        effort += abs(joint.effort) / joint.max_torque
    effort /= len(env.motor_joints)

    # === 1. Forward Progress ===
    fwd_velocity = np.dot(linear_vel, env._TARGET_DIRECTION)
    fwd_reward = np.clip(fwd_velocity, -1, 0.5)  # m/s, clip for robustness
    deviation_velocity = abs(np.dot(linear_vel, np.array([0,1,0])))
    deviation_penalty = np.clip(deviation_velocity, 0, 1)

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
    env.set_custom_state("previous_gfc", env.agent_ground_feet_contacts) 

    #=== Reward weighting ===
    reward_dict = {
        "fwd_reward": 7 * scale(0.2) * fwd_reward,
        "deviation_penalty": -2 * scale(0.2) * deviation_penalty,
        "uprightness": 2.5 * upright_reward,
        "height": 1.5 * height_reward,
        "contact_bonus": 3 * scale(0.45) * contact_bonus,
        "energy_penalty": -0.85 * scale(0.30) * energy_penalty,
        "effort_penalty": -1 * scale(0.30) * effort
    }

    total_reward = sum(reward_dict.values())
    return total_reward, reward_dict