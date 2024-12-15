import os
import torch
import imageio
from env import create_env
from policy import PPOPolicy

def test_model():
    model_path = "model_path"
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]

    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
    env = create_env()

    policy = PPOPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    policy.load_state_dict(torch.load(os.path.join(model_path, latest_model)))
    policy.eval()

    obs, _ = env.reset()
    gif_frames = []

    for step in range(2000):
        frame = env.render(mode="top_down")
        gif_frames.append(frame)

        with torch.no_grad():
            action, _, _ = policy.get_action(torch.tensor(obs, dtype=torch.float32))

        obs, _, done, _, _ = env.step(action.numpy())
        if done:
            break

    imageio.mimsave(os.path.join(model_path, "simulation_behavior.gif"), gif_frames, fps=30)
    env.close()

if __name__ == "__main__":
    test_model()
