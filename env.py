from metadrive.envs.metadrive_env import MetaDriveEnv

def create_env(custom_config=None):
    """
    Create the MetaDrive environment configured for PPO training with top-down rendering.
    """
    default_config = {
        "use_render": False,
        "traffic_density": 0.3,  # Increase traffic density for more complexity
        "need_inverse_traffic": True,
        "map": "CrXT",          # Use a diverse and complex map
        "manual_control": False,
        "vehicle_config": {
            "show_dest_mark": True,
            "show_navi_mark": True,
        },
        "success_reward": 20.0,
        "driving_reward": 1.0,  # Reduce driving reward to encourage progress
        "horizon": 3000,        # Increase horizon for longer episodes
        "image_observation": False,
        "top_down_camera_initial_x": 100,
        "top_down_camera_initial_y": 100,
        "top_down_camera_initial_z": 120,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "out_of_route_done": True,
    }
    if custom_config:
        default_config.update(custom_config)
    return MetaDriveEnv(config=default_config)

reward_config = {
    "lane_penalty": -5.0,       # Increase lane penalty
    "crash_penalty": -50.0,     # Increase crash penalty
    "out_of_road_penalty": -30.0, # Increase off-road penalty
}
