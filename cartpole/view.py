import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

STATE_SIZE = 4
ACTION_SIZE = 2
HIDDEN_SIZE = 128
DEVICE = torch.device("mps")  # or "cpu"/"cuda" depending on your setup


# Define the same network architecture
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)

    def forward(self, states):
        states = F.relu(self.fc1(states))
        states = F.relu(self.fc2(states))
        return self.fc3(states)


# Load environment with human render mode
env = gym.make("CartPole-v1", render_mode="human")

# Initialize and load trained weights
policy_net = DQN().to(DEVICE)
policy_net.load_state_dict(torch.load("pi_net.pth", map_location=DEVICE))
policy_net.eval()


def select_action(state):
    with torch.no_grad():
        q_values = policy_net(state)
        return q_values.argmax(dim=1).view(1, 1)


# Run a few episodes
for i in range(5):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=DEVICE).unsqueeze(0)

    done = False
    while not done:
        action = select_action(state)
        action_scalar = action.item()

        next_state, reward, terminated, truncated, _ = env.step(action_scalar)
        done = terminated or truncated

        if not done:
            state = torch.tensor(next_state, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)

env.close()
