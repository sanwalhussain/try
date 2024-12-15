import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(PPOPolicy, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Learnable log_std

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        mean_action = self.policy_net(obs)
        state_value = self.value_net(obs)
        return mean_action, state_value

    def get_action(self, obs):
        mean_action, _ = self.forward(obs)
        std = (self.log_std.exp() + 0.1)  # Add slight noise to encourage exploration
        dist = Normal(mean_action, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(), dist.entropy().sum()


class PPOTrainer:
    def __init__(self, env, policy, lr, gamma, eps_clip):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def compute_metadrive_reward(self, reward, info):
        total_reward = 0.0

        # Positive reward for staying on the road
        if not info.get("out_of_road", False):
            total_reward += 0.5

        # Penalties
        if info.get("crash_vehicle", False):
            total_reward -= 1.0
        if info.get("crash_object", False):
            total_reward -= 0.5
        if info.get("out_of_road", False):
            total_reward -= 0.7

        # Success reward
        if info.get("arrive_dest", False):
            total_reward += 1.5

        # Incremental progress reward
        total_reward += reward * 0.1

        return total_reward

    def train(self, num_episodes, steps_per_update, epochs):
        all_rewards, all_losses = [], []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            states, actions, log_probs, rewards, values = [], [], [], [], []

            for _ in range(steps_per_update):
                action, log_prob, _ = self.policy.get_action(obs)
                clipped_action = action.clamp(
                    min=self.env.action_space.low[0], max=self.env.action_space.high[0]
                )
                next_obs, reward, done, truncated, info = self.env.step(clipped_action.numpy())
                custom_reward = self.compute_metadrive_reward(reward, info)

                states.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(custom_reward)
                values.append(self.policy.value_net(obs).item())

                obs = torch.tensor(next_obs, dtype=torch.float32)
                if done or truncated:
                    break

            values.append(self.policy.value_net(obs).item())
            advantages = self._compute_advantages(rewards, values)
            returns = advantages + torch.tensor(values[:-1])

            loss = self._ppo_update(states, actions, log_probs, advantages, returns)
            all_rewards.append(sum(rewards))
            all_losses.append(loss.item())

            print(f"Episode {episode + 1}, Reward: {sum(rewards):.2f}, Loss: {loss:.2f}")

        return all_rewards, all_losses

    def _compute_advantages(self, rewards, values):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            advantage = delta + self.gamma * advantage
            advantages.insert(0, advantage)
        advantages = torch.tensor(advantages)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _ppo_update(self, states, actions, log_probs, advantages, returns):
        states = torch.stack(states)
        actions = torch.stack(actions)

        new_log_probs = []
        for state, action in zip(states, actions):
            mean_action, _ = self.policy(state)
            dist = Normal(mean_action, self.policy.log_std.exp())
            new_log_probs.append(dist.log_prob(action).sum())

        new_log_probs = torch.stack(new_log_probs)
        ratio = (new_log_probs - torch.stack(log_probs)).exp()
        loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).mean()
        value_loss = nn.MSELoss()(self.policy.value_net(states).squeeze(), returns)

        total_loss = loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss
