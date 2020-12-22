"""
__author__ = "Minsuk Sung and Hyunseung Lee"

"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.replay_memory import ReplayBuffer
from models.dqn import Qnet

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
n_episodes = 1000

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit)

    print_interval = 20
    score = 0.0
    max_score = -9999
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(n_episodes):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break
            if max_score < score:
                max_score = score

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"[Episode {n_epi:5d}] Score: {score / print_interval:6.2f} | Max score: {max_score / print_interval:6.2f} | Buffer size: {memory.size():5d} | Epsilon: {epsilon * 100:2.1f}%")
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()