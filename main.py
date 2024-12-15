import os
import torch
from datetime import datetime
from env import create_env, reward_config
from policy import PPOPolicy, PPOTrainer
import matplotlib.pyplot as plt

def main():
    # Suppress OpenMP runtime warning
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Create the `model_path` folder if it doesn't exist
    model_path = "model_path"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create the environment
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialize the PPO policy
    policy = PPOPolicy(obs_dim, act_dim)

    # Load the latest existing model
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if model_files:
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
        model_file_path = os.path.join(model_path, latest_model)
        print(f"Loading existing model: {model_file_path}")
        policy.load_state_dict(torch.load(model_file_path))
    else:
        print("No existing model found. Starting training from scratch.")

    # Initialize the PPO trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        reward_config=reward_config,
        lr=1e-4,                # Reduced learning rate for stability
        gamma=0.99,             # Discount factor for long-term rewards
        eps_clip=0.2,           # PPO clipping
    )

    # Train the agent
    rewards, losses = trainer.train(
        num_episodes=500,          # Train for a larger number of episodes
        steps_per_update=2048,     # Steps per policy update
        epochs=10                  # Number of epochs per update
    )

    # Save the trained model with a unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_file_path = os.path.join(model_path, f"ppo_metadrive_model_{timestamp}.pth")
    torch.save(policy.state_dict(), new_model_file_path)
    print(f"Final model saved at {new_model_file_path}")

    # Plot training rewards and losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Episodes")
    plt.legend()
    plt.grid(True)

    # Save the plots
    plot_path = os.path.join(model_path, f"training_rewards_and_loss_plot_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Training plots saved at {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
