'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

import math
import argparse
import itertools
from multiprocessing import Process
from multiprocessing.managers import BaseManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal

import gym
import numpy as np
import matplotlib.pyplot as plt

from util import logger
from util import ReplayBuffer

torch.multiprocessing.set_start_method('spawn')


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    soft_update(target, source, 1)


class SAC():
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.soft_q_net1 = SoftQNetwork(
            num_inputs, action_space.shape[0], args.hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(
            num_inputs, action_space.shape[0], args.hidden_dim).to(self.device)
        self.target_soft_q_net1 = SoftQNetwork(
            num_inputs, action_space.shape[0], args.hidden_dim).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(
            num_inputs, action_space.shape[0], args.hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(
            num_inputs, action_space.shape[0], args.hidden_dim).to(self.device)

        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        hard_update(target=self.target_soft_q_net1, source=self.soft_q_net1)
        hard_update(target=self.target_soft_q_net2, source=self.soft_q_net2)

        self.soft_q_optimizer1 = optim.Adam(
            self.soft_q_net1.parameters(), lr=args.lr)
        self.soft_q_optimizer2 = optim.Adam(
            self.soft_q_net2.parameters(), lr=args.lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=args.lr)

        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam(
                self.log_alpha.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state, action, reward, next_state, done = memory.sample(
            batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        mask = 1 - torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # normalize reward
        # reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy_net.sample(
                next_state)
            qf1_next_target = self.target_soft_q_net1(
                next_state, next_state_action)
            qf2_next_target = self.target_soft_q_net2(
                next_state, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = reward + mask * gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = self.soft_q_net1(state, action)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.soft_q_net2(state, action)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.policy_net.sample(state)
        qf1_pi = self.soft_q_net1(state, pi)
        qf2_pi = self.soft_q_net2(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.soft_q_optimizer1.zero_grad()
        qf1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        qf2_loss.backward()
        self.soft_q_optimizer2.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi +
                                             self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if updates % self.target_update_interval == 0:
            soft_update(self.target_soft_q_net1, self.soft_q_net1, soft_tau)
            soft_update(self.target_soft_q_net2, self.soft_q_net2, soft_tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(),
                   path + '_q1')  # have to specify different path name here!
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        # map model on single gpu for testing
        self.soft_q_net1.load_state_dict(
            torch.load(path + '_q1', map_location=self.device))
        self.soft_q_net2.load_state_dict(
            torch.load(path + '_q2', map_location=self.device))
        self.policy_net.load_state_dict(torch.load(
            path + '_policy', map_location=self.device))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward ')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) ')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate ')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA')

args = parser.parse_args()


# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Memory
memory = ReplayBuffer(capacity=args.replay_buffer_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if total_numsteps < args.start_steps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    if i_episode % 10 == 0 and args.eval == True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

env.close()