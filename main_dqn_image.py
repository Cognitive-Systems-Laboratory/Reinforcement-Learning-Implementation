"""
__author__ = "Minsuk Sung and Hyunseung Lee"

"""
import os
import sys
import argparse
import json
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.replay_memory import ReplayBuffer
from utils.save_tensorboard import *
from models.dqn_image import DQN as Qnet
import cv2
import numpy as np
import torchvision.transforms as T
from utils.env_utils import get_screen



parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
parser.add_argument('--config', default='configs/dqn.json')
args = parser.parse_args()


def preprocess( screen):
    preprocessed= cv2.resize(screen, (150,100),interpolation = cv2.INTER_CUBIC)  # 60 * 40 로 변경
    preprocessed = np.dot(preprocessed[..., :3], [0.299, 0.587, 0.114])  # Gray scale 로 변경
    # preprocessed: np.array = preprocessed.transpose((2, 0, 1))  # (C, W, H) 로 변경
    preprocessed = preprocessed.astype('float32') / 255.

    return torch.tensor(preprocessed)

# Device
if args.device == 'cpu':
    device = torch.device('cpu')
elif args.device == 'cuda':
    device = torch.device('cuda')
else:
    sys.exit()

# Hyperparameters
with open(args.config, "r") as config_json:
    config = json.load(config_json)

learning_rate = config['learning_rate']
gamma = config['gamma']
buffer_limit = config['buffer_limit']
batch_size = config['batch_size']
n_episodes = config['n_episodes']
min_mem_size = config['min_mem_size']



def main():
    env = gym.make('CartPole-v1')
    Summary_Writer=mk_SummaryWriter("experiments",'DQN_Image')
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit, device)

    print_interval = 20
    score = 0.0
    max_score = -9999
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(n_episodes):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        s=env.render(mode='rgb_array')

        s=get_screen(env)
        done = False
        
        #Use the Preprocessed Image: Simple Resizing and use the history
        height,width=40,90
        history_initial=torch.zeros(4,height,width)
        history_initial=torch.cat((s,history_initial[1:4]),0) 
        
        
        while not done:
            a = q.sample_action(history_initial.unsqueeze(0).to(device), epsilon)

            s_prime, r, done, info = env.step(a)

            s_image=env.render(mode='rgb_array')
            #Use the Preprocess Image: Simple Resizing and Use Its's history
            s_image=get_screen(env)
            history_new=torch.cat((s_image,history_initial[1:4]),0)    

            done_mask = 0.0 if done else 1.0
            memory.put((history_initial, a, r / 100.0, history_new, done_mask))
            history_initial=history_new
            s = s_prime
            score += r
            if done:
                break
            if max_score < score:
                max_score = score

        if memory.size() > min_mem_size:
            for i in range(10):
                s, a, r, s_prime, done_mask = memory.sample_b(batch_size)

                q_out = q(s)
                q_a = q_out.gather(1, a)

                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
                target = r + gamma * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"[Episode {n_epi:5d}] Score: {score / print_interval:6.2f} | Max score: {max_score / print_interval:6.2f} | Buffer size: {memory.size():5d} | Epsilon: {epsilon * 100:2.1f}%")
            add_scalar("Score",score/print_interval,n_epi,Summary_Writer)
            add_scalar("Max Score",max_score/print_interval,n_epi,Summary_Writer)
            add_scalar("Buffer Size",memory.size() /print_interval,n_epi,Summary_Writer)
            add_scalar("Epsilon",epsilon ,n_epi,Summary_Writer)
            score = 0.0


    env.close()
    Summary_Writer.close()

if __name__ == '__main__':
    main()
