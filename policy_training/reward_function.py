import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

def init_custom_state(env: SpotmicroEnv) -> None:
    """
    Initialize custom defined state variables for the environment.
    """
    env.set_custom_state("previous_gfc", set())
    env.set_custom_state("last_x", 0)
    env.set_custom_state("record_distance", 0)

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    linear_vel = np.array(env.agent_linear_velocity)
    angular_vel = np.array(env.agent_angular_velocity)
    contacts = env.agent_ground_feet_contacts

    fade_in_at = lambda start, scale_coefficient: 0 if env.num_steps < start else 1 - np.exp(- scale_coefficient * ((env.num_steps - start) / 1_000_000)) #0->1
    fade_out_at = lambda start, scale_coefficient: 1 if env.num_steps < start else np.exp(- scale_coefficient * ((env.num_steps - start) / 1_000_000)) #1->0

    # === 0. Effort ===
    effort = 0.0
    for joint in env.motor_joints:
        effort += abs(joint.effort) / joint.max_torque #normalzing each contribution by max effort
    effort /= len(env.motor_joints) #normalizing by number of joints

    # === 1. Forward Progress ===
    fwd_velocity = np.dot(linear_vel, env.target_direction)
    fwd_reward = np.clip(fwd_velocity, -1, 1)  # m/s, clip for robustness
    deviation_velocity = abs(np.dot(linear_vel, np.array([0,1,0])))
    deviation_penalty = np.clip(deviation_velocity, 0, 1)
    stillness_reward = 1.0 if np.linalg.norm(linear_vel) < 0.05 else 0.0

    # === 2. Uprightness (Pitch & Roll) ===
    max_angle = np.radians(45)
    upright_penalty = (abs(roll) + abs(pitch)) / max_angle
    upright_reward = np.clip(1.0 - upright_penalty, 0.0, 1.0)

    # === 3. Height regulation ===
    height_target = env.TARGET_HEIGHT
    height_error = abs(base_height - height_target)
    height_reward = np.clip(1.0 - height_error / 0.1, 0.0, 1.0)

    # === 4. Energy / Smoothness ===
    # Penalize large actions and penalize deviation from previous action
    action_magnitude_penalty = np.linalg.norm(action) / len(action) #Normalizing, since actions are in range -1,1
    action_diff_penalty = np.linalg.norm(action - env.agent_previous_action) / len(action) #Normalizing, since actions are in range -1,1
    energy_penalty = 0.25 * action_magnitude_penalty + 0.75 * action_diff_penalty

    # === 5. Contact (optional) ===
    contact_bonus = 0.0
    if len(env.get_custom_state("previous_gfc")) >= 3:
        contact_bonus += 0.25
    contact_bonus += 1.0 if len(contacts) >= 3 else -0.5

    new_distance_traveled = env.agent_base_position[0] - env.get_custom_state("last_x")
    env.set_custom_state("last_x", env.agent_base_position[0])
    
    record_bonus = 0
    if env.agent_base_position[0] > env.get_custom_state("record_distance"):
        record_bonus += 1
        env.set_custom_state("record_distance", env.agent_base_position)
    #env.set_custom_state("previous_gfc", env.agent_ground_feet_contacts)
    #distance_penalty = np.linalg.norm(np.array([0, 0, base_height]) - env.agent_base_position)

    weights_dict = {
        "fwd_reward": 3.5 * fade_in_at(300_000, 1.5),
        "deviation_penalty": -1.5 * fade_in_at(1_000_000, 1.5),
        "stillness_reward": 3 * fade_out_at(50_000, 2),
        "uprightness": 3,
        "height": 4.5,
        "contact_bonus": 5.5,
        "energy_penalty": -3.5 * fade_in_at(300_000, 2),
        "effort_penalty": -0.75 * fade_in_at(300_000, 1.75),
        "distance_traveled": 1.5,
        "record_distance_bonus": 1
    }

    #=== Reward weighting ===
    reward_dict = {
        "fwd_reward": fwd_reward,
        "deviation_penalty": deviation_penalty,
        "stilless_reward": stillness_reward,
        "uprightness": upright_reward,
        "height": height_reward,
        "contact_bonus": contact_bonus,
        "energy_penalty": energy_penalty,
        "effort_penalty": effort,
        "distance_traveled": new_distance_traveled,
        "record_distance_bonus": record_bonus
    }

    for k in reward_dict.keys():
        if k in weights_dict:
            reward_dict[k] *= weights_dict[k]
    total_reward = sum(reward_dict.values())
    return total_reward, reward_dict