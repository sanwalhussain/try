from metadrive.envs.metadrive_env import MetaDriveEnv

def create_env():
    """
    Create the MetaDrive environment with optimized configuration for training.
    """
    env_config = {
        "use_render": False,               # Disable rendering for faster training
        "traffic_density": 0.1,            # Moderate traffic density
        "random_traffic": True,            # Random traffic for variability
        "need_inverse_traffic": True,      # Add reverse traffic
        "map": "S",                        # Simple map for stability
        "vehicle_config": {
            "show_dest_mark": True,
            "show_navi_mark": True,
        },
        "horizon": 5000,                   # Max steps per episode
        "success_reward": 1.5,             # Reward for successful arrival
        "driving_reward": 0.1,             # Small incremental reward
        "out_of_road_penalty": -0.7,       # Balanced penalty for going out
        "crash_vehicle_penalty": -1.0,     # Penalty for crashing vehicles
        "crash_object_penalty": -0.5,      # Penalty for crashing objects
        "use_lateral_reward": True,
        "crash_object_done": True,
        "crash_vehicle_done": True,
        "out_of_route_done": True,
    }
    return MetaDriveEnv(config=env_config)
