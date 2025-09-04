import gymnasium as gym
import torch
from ppo import PolicyNetwork

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load environment with human render mode
env = gym.make("CartPole-v1", render_mode="human")

# Initialize and load trained weights
policy_net = PolicyNetwork().to(DEVICE)
policy_net.load_state_dict(torch.load("policy_net.pth", map_location=DEVICE))
policy_net.eval()


def select_action(state):
    with torch.no_grad():
        logits = policy_net(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()


state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32,
                     device=DEVICE).unsqueeze(0)

done = False
total_reward = 0
while not done:
    action = select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

    if not done:
        state = torch.tensor(next_state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)

env.close()
