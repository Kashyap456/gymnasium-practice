import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

OBSERVATION_SIZE = 4
ACTION_SIZE = 2
VALUE_HIDDEN_SIZE = 32
POLICY_HIDDEN_SIZE = 128
BATCH_SIZE = 2048
MINIBATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_UPDATES = 50

GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
EPS = 0.2


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBSERVATION_SIZE, VALUE_HIDDEN_SIZE)
        self.fc2 = nn.Linear(VALUE_HIDDEN_SIZE, VALUE_HIDDEN_SIZE)
        self.fc3 = nn.Linear(VALUE_HIDDEN_SIZE, 1)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        return self.fc3(state).squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBSERVATION_SIZE, POLICY_HIDDEN_SIZE)
        self.fc2 = nn.Linear(POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE)
        self.fc3 = nn.Linear(POLICY_HIDDEN_SIZE, ACTION_SIZE)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        return self.fc3(state)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

env = gym.make("CartPole-v1")
value_net = ValueNetwork().to(device)
policy_net = PolicyNetwork().to(device)
value_optimizer = optim.Adam(value_net.parameters(), lr=LR)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=LR)


@torch.no_grad()
def select_action_tensor(state_tensor):
    logits = policy_net(state_tensor)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    value = value_net(state_tensor)
    return action, log_prob, value


def compute_gae(rewards, dones, values, last_value):
    T = rewards.size(0)
    adv = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    values_ext = torch.cat([values, last_value.unsqueeze(0)], dim=0)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * \
            values_ext[t + 1] * nonterm - values_ext[t]
        last_gae = delta + GAMMA * LAMBDA * nonterm * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def optimize_model(states, actions, rewards, dones, old_log_probs, old_values, last_value):
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
            logits = policy_net(s_mb)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(a_mb)
            v_pred = value_net(s_mb)
            ratio = (new_lp - old_lp_mb).exp()
            clipped_ratio = torch.clamp(ratio, 1 - EPS, 1 + EPS)
            policy_loss = -torch.min(ratio * adv_mb,
                                     clipped_ratio * adv_mb).mean()
            value_loss = F.mse_loss(v_pred, ret_mb)
            loss = policy_loss + 0.5 * value_loss
            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(value_net.parameters()), 1.0)
            policy_optimizer.step()
            value_optimizer.step()


def train():
    plt.ion()
    fig, ax = plt.subplots()

    def plot_durations(durations):
        ax.clear()
        ax.set_title("Training...")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Duration")
        if len(durations):
            ax.plot(durations)
            if len(durations) >= 100:
                d = torch.tensor(durations, dtype=torch.float32)
                means = d.unfold(0, 100, 1).mean(1)
                means = torch.cat([torch.zeros(99), means])
                ax.plot(means.numpy())
        fig.canvas.draw()
        plt.pause(0.001)

    durations = []
    for _ in tqdm(range(NUM_UPDATES)):
        states = torch.empty(BATCH_SIZE, OBSERVATION_SIZE,
                             dtype=torch.float32, device=device)
        actions = torch.empty(BATCH_SIZE, dtype=torch.long, device=device)
        rewards = torch.empty(BATCH_SIZE, dtype=torch.float32, device=device)
        dones = torch.empty(BATCH_SIZE, dtype=torch.float32, device=device)
        old_log_probs = torch.empty(
            BATCH_SIZE, dtype=torch.float32, device=device)
        old_values = torch.empty(
            BATCH_SIZE, dtype=torch.float32, device=device)

        ep_len = 0
        s, _ = env.reset()
        for t in range(BATCH_SIZE):
            st = torch.as_tensor(s, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            a, lp, v = select_action_tensor(st)
            ns, r, terminated, truncated, _ = env.step(a.item())
            d = float(terminated or truncated)
            states[t] = st.squeeze(0)
            actions[t] = a.squeeze(0)
            rewards[t] = r
            dones[t] = d
            old_log_probs[t] = lp.squeeze(0)
            old_values[t] = v.squeeze(0)
            ep_len += 1
            if terminated or truncated:
                durations.append(ep_len)
                plot_durations(ax, durations)
                s, _ = env.reset()
                ep_len = 0
            else:
                s = ns

        with torch.no_grad():
            last_value = value_net(torch.as_tensor(
                s, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)

        optimize_model(states, actions, rewards, dones,
                       old_log_probs, old_values, last_value)

    print("Complete")
    torch.save(policy_net.state_dict(), "policy_net.pth")


if __name__ == "__main__":
    train()
