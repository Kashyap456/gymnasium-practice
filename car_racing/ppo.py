import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

device = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")

OBSERVATION_SHAPE = (4, 84, 84)
ACTION_SIZE = 3
BATCH_SIZE = 2048
MINIBATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_UPDATES = 50

GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
EPS = 0.2

LOG_STD_MIN = -5.0   # ~ std >= 0.0067
LOG_STD_MAX = 2.0   # ~ std <= 7.389
EPS_STD = 1e-6

env = gym.make("CarRacing-v3", continuous=True)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)  # (84,84,1)
env = gym.wrappers.FrameStackObservation(
    env, stack_size=4)                        # (84,84,4)


def preprocess_obs(obs):
    x = torch.tensor(obs.squeeze(-1), dtype=torch.float32,
                     device=device) / 255.0

    return x


def tanh_logdet_jacobian(z):
    # stable: 2*(log(2) - z - softplus(-2z))
    return (2.0 * (torch.log(torch.tensor(2.0, device=z.device)) - z - F.softplus(-2.0 * z))).sum(-1)


class Actor(nn.Module):
    # model arch is straight from Deepmind's Atari DQN paper
    def __init__(self):
        super(Actor, self).__init__()
        self.c1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)  # (16, 20, 20)
        self.c2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # (32, 9, 9)
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
        raw_std = self.log_std(state)
        log_std = LOG_STD_MIN + \
            (LOG_STD_MAX - LOG_STD_MIN) * torch.sigmoid(raw_std)
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
        logp = dist.log_prob(z).sum(-1) - tanh_logdet_jacobian(z)
        return action, logp.detach(), v

    # this is kinda hacky but basically revert the squash to get the original latent
    def logp_of_action(self, x, action_env):
        """
        action_env: steer in [-1,1], gas/brake in [0,1]
        Convert gas/brake back to [-1,1], then atanh to pre-squash z.
        """
        # unpack & map back
        steer = action_env[..., :1]                 # [-1,1]
        gb = action_env[..., 1:] * 2.0 - 1.0        # [0,1] -> [-1,1]
        a_tanh = torch.cat([steer, gb], dim=-1).clamp(-0.999999, 0.999999)
        z = 0.5 * (torch.log1p(a_tanh) - torch.log1p(-a_tanh))  # atanh(a)

        mu, log_std, _ = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        base = dist.log_prob(z).sum(-1)
        corr = tanh_logdet_jacobian(z)
        return base - corr


def compute_gae(rewards, dones, values, last_value):
    with torch.no_grad():
        T = rewards.size(0)
        adv = torch.zeros(T, device=device)
        last_gae = 0.0
        values_ext = torch.cat([values, last_value], dim=0)
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * \
                values_ext[t + 1] * nonterm - values_ext[t]
            last_gae = delta + GAMMA * LAMBDA * nonterm * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns


def optimize_model(actor, optimizer, states, actions, rewards, dones, old_log_probs, old_values, last_value):
    adv, ret = compute_gae(rewards, dones, old_values, last_value)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = states.size(0)
    idx = torch.randperm(N, device=device)
    for _ in range(NUM_EPOCHS):
        for st in range(0, N, MINIBATCH_SIZE):
            mb = idx[st:st + MINIBATCH_SIZE]
            s_mb = states[mb]
            a_mb = actions[mb]
            adv_mb = adv[mb]
            ret_mb = ret[mb]
            old_lp_mb = old_log_probs[mb]

            new_lp = actor.logp_of_action(s_mb, a_mb)
            _, _, v_pred = actor(s_mb)
            ratio = (new_lp - old_lp_mb).exp()
            clipped_ratio = torch.clamp(ratio, 1 - EPS, 1 + EPS)
            policy_loss = -torch.min(ratio * adv_mb,
                                     clipped_ratio * adv_mb).mean()
            value_loss = F.mse_loss(v_pred, ret_mb)
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()), 1.0)
            optimizer.step()


def train(model_path=None):
    # Initialize actor
    actor = Actor().to(device)

    # Load existing weights if provided
    if model_path:
        print(f"Loading model weights from {model_path}")
        actor.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(actor.parameters(), lr=LR)

    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # second y-axis for returns

    MA = 100  # moving-average window

    def plot_metrics(lengths, returns):
        ax1.clear()
        ax2.clear()
        ax1.set_title("Training")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Episode length", color="tab:blue")
        ax2.set_ylabel("Episode return", color="tab:orange")

        if lengths:
            ax1.plot(lengths, linewidth=1, color="tab:blue")
            if len(lengths) >= MA:
                d = torch.tensor(lengths, dtype=torch.float32)
                ma = d.unfold(0, MA, 1).mean(1)
                ma = torch.cat([torch.full((MA-1,), float('nan')), ma])
                ax1.plot(ma.numpy(), linestyle="--",
                         color="tab:blue", alpha=0.7)

        if returns:
            ax2.plot(returns, linewidth=1, color="tab:orange", alpha=0.9)
            if len(returns) >= MA:
                r = torch.tensor(returns, dtype=torch.float32)
                rma = r.unfold(0, MA, 1).mean(1)
                rma = torch.cat([torch.full((MA-1,), float('nan')), rma])
                ax2.plot(rma.numpy(), linestyle="--",
                         color="tab:orange", alpha=0.7)

        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(0.001)

    durations = []
    cumulative_rewards = []
    for _ in tqdm(range(NUM_UPDATES)):
        states = torch.empty((BATCH_SIZE, *OBSERVATION_SHAPE),
                             dtype=torch.float32, device=device)
        actions = torch.empty((BATCH_SIZE, ACTION_SIZE), device=device)
        rewards = torch.empty(BATCH_SIZE, dtype=torch.float32, device=device)
        dones = torch.empty(BATCH_SIZE, dtype=torch.float32, device=device)
        old_log_probs = torch.empty(
            BATCH_SIZE, dtype=torch.float32, device=device)
        old_values = torch.empty(
            BATCH_SIZE, dtype=torch.float32, device=device)

        ep_len = 0
        cum_reward = 0
        s, _ = env.reset()
        for t in range(BATCH_SIZE):
            st = preprocess_obs(s).unsqueeze(0)
            a, lp, v = actor.sample_action(st)
            ns, r, terminated, truncated, _ = env.step(
                a.squeeze(0).detach().cpu().numpy())
            d = float(terminated or truncated)
            states[t] = st.squeeze(0)
            actions[t] = a.squeeze(0)
            rewards[t] = r
            dones[t] = d
            old_log_probs[t] = lp.squeeze(0).detach()
            old_values[t] = v.squeeze(0).detach()
            cum_reward += r * (GAMMA ** ep_len)
            ep_len += 1
            if terminated or truncated:
                durations.append(ep_len)
                cumulative_rewards.append(cum_reward)
                plot_metrics(durations, cumulative_rewards)
                s, _ = env.reset()
                ep_len = 0
                cum_reward = 0
            else:
                s = ns

        with torch.no_grad():
            _, _, last_value = actor(preprocess_obs(s).unsqueeze(0))

        with torch.autograd.set_detect_anomaly(True):
            optimize_model(actor, optimizer, states, actions, rewards, dones,
                           old_log_probs, old_values, last_value)

    print("Complete")
    torch.save(actor.state_dict(), "actor.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent for CarRacing")
    parser.add_argument("--model", type=str,
                        help="Path to existing model weights to load")
    args = parser.parse_args()

    train(args.model)
