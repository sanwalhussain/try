import os
import torch
from datetime import datetime
from env import create_env
from policy import PPOPolicy, PPOTrainer
import matplotlib.pyplot as plt

def main():
    model_path = "model_path"
    os.makedirs(model_path, exist_ok=True)

    steps_per_update = 5000
    num_episodes = 100
    hyperparams = {"lr": 1e-4, "eps_clip": 0.2, "gamma": 0.99, "epochs": 10}

    env = create_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialize the policy
    policy = PPOPolicy(obs_dim, act_dim)

    # Initialize PPOTrainer without `epochs`
    trainer = PPOTrainer(env, policy, lr=hyperparams["lr"], gamma=hyperparams["gamma"], eps_clip=hyperparams["eps_clip"])

    # Train the policy
    rewards, losses = trainer.train(
        num_episodes=num_episodes,
        steps_per_update=steps_per_update,
        epochs=hyperparams["epochs"]
    )

    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_path, f"ppo_metadrive_{timestamp}.pth")
    torch.save(policy.state_dict(), model_file)
    print(f"Model saved at: {model_file}")

    # Plot training rewards and losses
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Episode Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Episode Loss", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Episodes")
    plt.legend()

    # Save and display the plot
    plot_file = os.path.join(model_path, f"training_plot_{timestamp}.png")
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Training plot saved at: {plot_file}")
    plt.show()

if __name__ == "__main__":
    main()
