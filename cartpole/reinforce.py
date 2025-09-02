import gymnasium as gym

import random
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

from itertools import count
from collections import namedtuple, deque
from tqdm import tqdm

STATE_SIZE = 4
ACTION_SIZE = 2
HIDDEN_SIZE = 128


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


device = torch.device("mps")

GAMMA = 0.99
LR = 3e-4

env = gym.make("CartPole-v1")
rnet = DQN().to(device)

# why AdamW? what is amsgrad
optimizer = optim.AdamW(rnet.parameters(), lr=LR, amsgrad=True)

log_probs = []
rewards = []


def select_action(state):
    logits = rnet(state)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    log_probs.append(log_prob)
    return action.item()


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # (Question) - How does this torch logic work
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


def compute_returns():
    reward_vec = torch.tensor(rewards, dtype=torch.float32, device=device)
    T = len(rewards)
    gammas = GAMMA ** torch.arange(T, device=device)
    discounted_rewards = gammas * reward_vec
    returns = torch.cumsum(discounted_rewards.flip(0), dim=0).flip(0) / gammas
    return returns


def optimize_model():
    returns = compute_returns()
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    loss = -(torch.stack(log_probs) * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    log_probs.clear()
    rewards.clear()


truncated_count = 0
TRUNCATED_THRESHOLD = 10

rnet.train()
num_episodes = 10000
for i_episode in tqdm(range(num_episodes)):
    state, info = env.reset()
    # (Question) - why unsqueeze here
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
        state = torch.tensor(next_state, dtype=torch.float32,
                             device=device).unsqueeze(0)

    if truncated:
        truncated_count += 1
    else:
        truncated_count = 0
    if truncated_count >= TRUNCATED_THRESHOLD:
        break

    episode_durations.append(t + 1)
    plot_durations()

    optimize_model()


print('Complete')
plot_durations(show_result=True)
plt.show()

torch.save(rnet.state_dict(), "rnet.pth")
