import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

def init_custom_state(env: SpotmicroEnv) -> None:
    """
    Initialize custom defined state variables for the environment.
    """
    env.set_custom_state("previous_gfc", set())

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    linear_vel = np.array(env.agent_linear_velocity)
    angular_vel = np.array(env.agent_angular_velocity)
    contacts = env.agent_ground_feet_contacts

    scale_coefficient = 1.5
    fade_in_at = lambda start: 0 if env.num_steps < start else 1 - np.exp(- scale_coefficient * ((env.num_steps - start) / 1_000_000)) #0->1

    effort = 0.0
    for joint in env.motor_joints:
        effort += abs(joint.effort) / joint.max_torque #normalzing each contributio by max effort
    effort /= len(env.motor_joints) #normalizing by number of joints

    # === 1. Forward Progress ===
    fwd_velocity = np.dot(linear_vel, env._TARGET_DIRECTION)
    fwd_reward = np.clip(fwd_velocity, -1, 1)  # m/s, clip for robustness
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
    action_magnitude_penalty = np.linalg.norm(action) / len(action) #Normalizing, since actions are in range -1,1
    action_diff_penalty = np.linalg.norm(action - env._previous_action) / len(action) #Normalizing, since actions are in range -1,1
    energy_penalty = action_magnitude_penalty + 0.5 * action_diff_penalty #range: [-1.5,1.5]

    # === 5. Contact (optional) ===
    contact_bonus = 0.0
    if len(env.get_custom_state("previous_gfc")) >= 3:
        contact_bonus += 0.25
    contact_bonus += 1.0 if len(contacts) >= 3 else -0.5
    env.set_custom_state("previous_gfc", env.agent_ground_feet_contacts)

    weights_dict = {
        "fwd_reward": 7 * fade_in_at(200_000),
        "deviation_penalty": -3 * fade_in_at(200_000),
        "uprightness": 2.5,
        "height": 2.5,
        "contact_bonus": 4,
        "energy_penalty": -4.5 * fade_in_at(1_000_000),
        "effort_penalty": -3 * fade_in_at(1_000_000)
    }

    #=== Reward weighting ===
    reward_dict = {
        "fwd_reward": fwd_reward,
        "deviation_penalty": deviation_penalty,
        "uprightness": upright_reward,
        "height": height_reward,
        "contact_bonus": contact_bonus,
        "energy_penalty": energy_penalty,
        "effort_penalty": effort
    }

    for k in reward_dict.keys():
        if k in weights_dict:
            reward_dict[k] *= weights_dict[k]
    total_reward = sum(reward_dict.values())
    return total_reward, reward_dict