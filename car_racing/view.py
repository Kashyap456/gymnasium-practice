import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")

OBSERVATION_SHAPE = (4, 84, 84)
ACTION_SIZE = 3

env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)


def preprocess_obs(obs):
    x = torch.tensor(obs.squeeze(-1), dtype=torch.float32,
                     device=device) / 255.0
    return x


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.c1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.mu = nn.Linear(256, ACTION_SIZE)
        self.log_std = nn.Linear(256, ACTION_SIZE)
        self.value = nn.Linear(256, 1)

    def forward(self, state):
        state = F.relu(self.c1(state))
        state = F.relu(self.c2(state))
        state = state.view(state.size(0), -1)
        state = F.relu(self.fc1(state))
        mean = self.mu(state)
        log_std = self.log_std(state)
        v = self.value(state).squeeze(-1)
        return mean, log_std, v

    @torch.no_grad()
    def sample_action(self, x):
        mu, log_std, v = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a_tanh = torch.tanh(z)  # [-1,1]
        steer = a_tanh[..., :1]
        gb = (a_tanh[..., 1:] + 1.0) * 0.5  # [0,1]
        action = torch.cat([steer, gb], dim=-1)
        # tanh Jacobian (for all 3 pre-squash coords)
        logp = dist.log_prob(z).sum(-1) - (2 * (F.softplus(2*z) - z)).sum(-1)
        return action, logp.detach(), v


# Load the trained actor
actor = Actor().to(device)
actor.load_state_dict(torch.load("actor.pth", map_location=device))
actor.eval()

obs, info = env.reset()

done = False
while not done:
    state = preprocess_obs(obs).unsqueeze(0)
    action, _, _ = actor.sample_action(state)
    action_np = action.squeeze(0).detach().cpu().numpy()
    next_state, reward, terminated, truncated, _ = env.step(action_np)
    obs = next_state
    done = terminated or truncated
    if done:
        break

env.close()
