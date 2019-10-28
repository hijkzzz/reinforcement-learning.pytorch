import time
import os
import argparse
import random

import numpy as np
import gym
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical

from env import SubprocVecEnv
from env import make_atari, wrap_deepmind
from env import Monitor

from util import init_orthogonal_
from util import logger


class ActorCritic(nn.Module):
    def __init__(self, action_size, hidden_size=128, memory_size=128, extra_hidden=True, enlargement='normal', device=torch.device('cuda')):
        super(ActorCritic, self).__init__()

        enlargement = {
            'small': 1,
            'normal': 2,
            'large': 4
        }[enlargement]

        hidden_size *= enlargement
        memory_size *= enlargement
        self.extra_hidden = extra_hidden
        self.hidden_size = hidden_size
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(6 ** 2 * 64, hidden_size),
            nn.ReLU()
        )

        self.rnn = nn.GRUCell(hidden_size, memory_size)

        if self.extra_hidden:
            self.fc2val = nn.Sequential(nn.Linear(memory_size, memory_size),
                                        nn.ReLU())
            self.fc2act = nn.Sequential(nn.Linear(memory_size, memory_size),
                                        nn.ReLU())

        self.action = nn.Sequential(nn.Linear(memory_size, action_size),
                                    nn.Softmax(dim=1))
        self.value = nn.Linear(memory_size, 1)

    def forward(self, states, hiddens=None):
        T = states.shape[0]
        B = states.shape[1]

        if hiddens is None:
            hiddens = self.init_hidden(B)

        states = states.view(T * B, *states.shape[2:])
        states = self.conv(states)
        states = states.view(T, B, 6 ** 2 * 64)
        states = self.fc(states)

        states, hiddens = self.rnn(states, hiddens)

        value = probs = states
        if self.extra_hidden:
            probs = self.fc2act(probs) + probs
            value = self.fc2val(value) + value

        probs = self.action(probs)
        dist = Categorical(probs)
        value = self.value(value)

        return dist, torch.log(probs), value, hiddens

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class PPO():
    def __init__(self, args):
        super(PPO, self).__init__()
        # neural networks
        self.actor_critic = ActorCritic(action_size=args.action_size, hidden_size=args.hidden_size,
                                        memory_size=args.memory_size, extra_hidden=args.extra_hidden,
                                        enlargement=args.enlargement, device=args.device).cuda()
        self.actor_critic.apply(init_orthogonal_)

        # args
        self.device = args.device
        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.num_rollouts = args.num_rollouts
        self.render = args.render
        self.action_size = args.action_size

        self.update_epochs = args.update_epochs
        self.batch_size = args.batch_size
        self.clip_range = args.clip_range
        self.max_grad_norm = args.max_grad_norm
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.coeff_ent = args.coeff_ent

        # optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), args.learning_rate)

        # batch
        self.num_batch = self.num_steps * self.num_envs // self.batch_size
        self.stride_batch = self.num_envs // self.num_batch

    def select_action(self, states, hiddens):
        # H * W * C ==> C * H * W
        states = torch.from_numpy(states).unsqueeze(0).to(
            device=self.device, dtype=torch.float32)

        with torch.no_grad():
            dist, _, _, hiddens = self.actor_critic(
                states, hiddens)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action[0].cpu().tolist(), log_prob[0].cpu().numpy(), hiddens

    def save_param(self, path):
        torch.save(self.actor_critic.state_dict(),
                   path + '/actor_critic.pkl')

    def load_param(self, path):
        self.actor_critic.load(path + '/actor_critic.pkl')

    def update_parameters(self, states, actions, action_log_probs, rewards, next_states, dones):
        # T * B * features
        states = torch.from_numpy(states).to(
            dtype=torch.float32, device=self.device)
        actions = torch.from_numpy(actions).to(
            dtype=torch.int32, device=self.device)
        old_action_log_probs = torch.from_numpy(action_log_probs).to(
            dtype=torch.float32, device=self.device)
        rewards = torch.from_numpy(rewards).to(
            dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(next_states).to(
            dtype=torch.float32, device=self.device)
        masks = 1 - torch.from_numpy(dones).to(dtype=torch.float32,
                                               device=self.device)

        # GENERALIZED ADVANTAGE ESTIMATION
        with torch.no_grad():
            advantages = torch.zeros_like(rewards)
            _, _, values, _ = self.actor_critic(
                torch.cat([states, next_states[-1].unsqueeze(0)], dim=0))

            last_gae_lam = 0
            for t in range(self.num_steps - 1, -1, -1):
                delta = rewards[t] + self.gamma * \
                    masks[t] * values[t + 1] - values[t]
                advantages[t, :] = last_gae_lam = delta + \
                    self.lamda * self.gamma * masks[t] * last_gae_lam
            returns = advantages + values[:-1]

        logger.record_tabular('advantages mean', advantages.mean(dim=(0, 1)))
        logger.record_tabular('advantages std', advantages.std(dim=(0, 1)))
        logger.record_tabular('returns mean',
                              returns.mean(dim=(0, 1)))
        logger.record_tabular('returns std',
                              returns.std(dim=(0, 1)))
        logger.dump_tabular()

        # train epochs
        for epoch_idx in range(self.update_epochs):
            self.training_step += 1

            for batch_idx in range(self.num_batch):
                # samples in batch
                # T * B * features
                start, end = batch_idx * \
                    self.stride_batch, (batch_idx + 1) * self.stride_batch

                state = states[:, start:end, ...]
                action = actions[:, start:end, ...]
                old_action_log_prob = old_action_log_probs[:, start:end, ...]
                return1 = returns[:, start:end, ...]
                advantage = advantages[:, start:end, ...]

                # policy loss
                dist, _, value, _ = self.actor_critic(
                    state)
                action_log_prob = dist.log_prob(action)

                ratio = torch.exp(action_log_prob - old_action_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * advantage
                action_loss = -torch.mean(torch.min(surr1, surr2), dim=(0, 1))

                # value loss
                smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
                value_loss = smooth_l1_loss(return1.flatten(), value.flatten())

                # entropy loss
                entropy = -torch.mean(dist.entropy(), dim=(0, 1))

                # backprop
                loss = action_loss + value_loss + self.coeff_ent * entropy
                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm > 1e-8:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                if epoch_idx == self.update_epochs - 1:
                    logger.record_tabular('training_step',
                        self.training_step)
                    logger.record_tabular('value_loss',
                                          value_loss.item())
                    logger.record_tabular('policy_loss', action_loss.item())
                    logger.record_tabular('entropy', entropy.item())
                    logger.dump_tabular()

    def train(self, envs):
        self.training_step = 0
        self.best_reward = 0
        self.visited_rooms = set()
        self.eplen = 0

        rollout_idx = 0
        state = np.transpose(envs.reset(), (0, 2, 3, 1)) / 255.0
        hidden = self.actor_critic.init_hidden(self.num_envs)

        # rollout
        while rollout_idx < self.num_rollouts:
            current_best_reward = 0

            states = np.zeros(
                (self.num_steps, self.num_envs, 1, 84, 84), np.float32)
            actions = np.zeros((self.num_steps, self.num_envs), np.int32)
            action_log_probs = np.zeros(
                (self.num_steps, self.num_envs), np.float32)
            rewards = np.zeros((self.num_steps, self.num_envs), np.float32)
            next_states = np.zeros(
                (self.num_steps, self.num_envs, 1, 84, 84), np.float32)
            dones = np.zeros((self.num_steps, self.num_envs), np.int32)

            for t in range(self.num_steps):
                action, action_log_prob, hidden = self.select_action(
                    state, hidden)
                next_state, reward, done, info = envs.step(action)
                # TensorFlow format to PyTorch
                next_state = np.transpose(next_state, (0, 2, 3, 1)) / 255.0

                # transitions
                states[t, ...] = state
                actions[t, ...] = action
                action_log_probs[t, ...] = action_log_prob
                rewards[t, ...] = reward
                next_states[t, ...] = next_state
                dones[t, ...] = done

                if self.render:
                    envs.render(0)

                state = next_state

                # done
                for i, dne in enumerate(done):
                    if dne:
                        hidden[0][i] *= 0

                        epinfo = info[i]['episode']
                        if 'visited_rooms' in epinfo:
                            self.visited_rooms.union(
                                list(epinfo['visited_rooms']))

                        self.best_reward = max(epinfo['r'], self.best_reward)
                        current_best_reward = max(
                            epinfo['r'], current_best_reward)
                        self.eplen += epinfo['l']

            # logger
            logger.record_tabular('visited_rooms',
                                  str(len(self.visited_rooms)) + ', ' + str(self.visited_rooms))
            logger.record_tabular('best_reward', self.best_reward)
            logger.record_tabular('current_best_reward', current_best_reward)
            logger.record_tabular('eplen', self.eplen)
            logger.dump_tabular()

            # train neural networks
            self.update_parameters(states, actions, action_log_probs,
                                   rewards, next_states, dones)
            rollout_idx += 1

    def test(self, envs):
        self.training_step = 0
        self.best_reward = 0
        self.visited_rooms = set()
        self.eplen = 0

        rollout_idx = 0
        state = np.transpose(envs.reset(), (0, 2, 3, 1)) / 255.0
        hidden = self.actor_critic.init_hidden(self.num_envs)

        # rollout
        while rollout_idx < self.num_rollouts:
            current_best_reward = 0

            action, _, hidden = self.select_action(
                state, hidden)
            next_state, reward, done, info = envs.step(action)
            # TensorFlow format to PyTorch
            next_state = np.transpose(next_state, (0, 2, 3, 1)) / 255.0
            envs.render(0)
            state = next_state

            # done
            for i, dne in enumerate(done):
                if dne:
                    hidden[0][i] *= 0

                    epinfo = info[i]['episode']
                    if 'visited_rooms' in epinfo:
                        self.visited_rooms.union(
                            list(epinfo['visited_rooms']))

                    self.best_reward = max(epinfo['r'], self.best_reward)
                    current_best_reward = max(
                        epinfo['r'], current_best_reward)
                    self.eplen += epinfo['l']

        # logger
        logger.record_tabular('visited_rooms',
                                str(len(self.visited_rooms)) + ', ' + str(self.visited_rooms))
        logger.record_tabular('best_reward', self.best_reward)
        logger.record_tabular('current_best_reward', current_best_reward)
        logger.record_tabular('eplen', self.eplen)
        logger.dump_tabular()

def main():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--max_episode_steps', type=int, default=4500)
    parser.add_argument('--num_rollouts', type=int, default=30e3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--coeff_ent', type=float, default=1e-3)
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--memory_size', type=int, default=128)
    parser.add_argument('--enlargement', type=str, default='normal')
    parser.add_argument('--extra_hidden', type=bool, default=True)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--clip_range', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--train', default=True)
    args = parser.parse_args()

    # Enable CUDA
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')
    torch.cuda.current_device() # fix init bug in Windows 10

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create envs
    def make_env(rank):
        def _thunk():
            env = make_atari(args.env_name, args.max_episode_steps)
            env.seed(args.seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_deepmind(env)
        return _thunk

    envs = [make_env(i) for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    args.action_size = envs.action_space.n

    agent = PPO(args)

    if args.train:
        agent.train(envs)
    else:
        agent.test(envs)

    # Exit
    envs.close()

if __name__ == '__main__':
    main()