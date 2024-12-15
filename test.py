import os
import torch
from env import create_env
from policy import PPOPolicy
import imageio

def test_model():
    # Define the `path` folder
    folder_name = "path"
    if not os.path.exists(folder_name):
        print("No `path` folder found. Please train a model first.")
        return

    # Find the latest `.pth` file in the `path` folder
    model_files = [f for f in os.listdir(folder_name) if f.endswith('.pth')]
    if not model_files:
        print("No .pth files found in the `path` folder. Please train a model first.")
        return
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    model_path = os.path.join(folder_name, latest_model)

    print(f"Loading model: {model_path}...")

    # Create the environment
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = PPOPolicy(obs_dim, act_dim)

    # Load the model
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    # Prepare for GIF recording
    gif_frames = []
    output_gif_path = os.path.join(folder_name, "top_down_simulation.gif")

    # Evaluate the model for 5000 steps
    max_steps = 5000  # Total steps to run
    render_every = 10  # Render every 10 steps for faster performance
    frame_skip = 5     # Skip intermediate frames by simulating 5 steps at once
    action_scale = 1.5 # Scale the agent's actions to make it faster
    step_count = 0     # Initialize step counter

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    done = False

    while step_count < max_steps and not done:
        # Get action from the policy
        with torch.no_grad():
            action, _, _ = policy.get_action(obs)

        # Scale the action to speed up the agent
        scaled_action = action * action_scale
        clipped_action = scaled_action.clamp(
            min=torch.tensor(env.action_space.low),
            max=torch.tensor(env.action_space.high)
        ).numpy()

        # Step the environment multiple times (frame skipping)
        for _ in range(frame_skip):
            obs, reward, done, truncated, info = env.step(clipped_action)
            step_count += 1
            if step_count >= max_steps or done or truncated:
                break

        # Convert observation to torch.Tensor
        obs = torch.tensor(obs, dtype=torch.float32)

        # Render the environment only every `render_every` steps
        if step_count % render_every == 0:
            frame = env.render(mode="top_down")  # Render only occasionally
            gif_frames.append(frame)

    # Save the frames as a GIF
    print(f"Simulation completed for {step_count} steps.")
    print("Saving the simulation as a GIF...")
    imageio.mimsave(output_gif_path, gif_frames, fps=10)
    print(f"Simulation video saved as '{output_gif_path}'")

    env.close()

if __name__ == "__main__":
    test_model()
