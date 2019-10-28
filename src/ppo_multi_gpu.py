import time
import os
import numpy as np

import gym
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical

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
        self.actor_critic.apply(weight_init)

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
        self.episode_int = args.episode_int

        self.num_batch = self.num_steps * self.num_envs // self.batch_size
        self.stride_batch = self.num_envs // self.num_batch

        # logger info
        self.training_step = 0
        self.best_reward = 0
        self.visited_rooms = set()
        self.eplen = 0

        # optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), args.learning_rate)

    def select_action(self, states, hiddens):
        # T * B * features
        states = torch.from_numpy(states).unsqueeze(0).to(
            device=self.device, dtype=torch.float32).permute([0, 1, 4, 2, 3]) / 255.0

        with torch.no_grad():
            dist_action, _, _, _, hiddens = self.actor_critic(
                states, None, hiddens)
        action = dist_action.sample()
        action_log_prob = dist_action.log_prob(action)
        return action[0].cpu().tolist(), action_log_prob[0].cpu().numpy(), hiddens

    def save_param(self):
        torch.save(self.actor_critic.state_dict(),
                   './saved/actor_critic' + str(time.time())[:10] + '.pkl')

    def train(self, envs):
        state = envs.reset()
        rollout_id = 0

        # rollout
        while rollout_id < self.num_rollouts:
            current_best_reward = 0

            states = np.zeros(
                (self.num_steps, self.num_envs, 84, 84, 1), np.float32)
            actions = np.zeros((self.num_steps, self.num_envs), np.int32)
            action_log_probs = np.zeros(
                (self.num_steps, self.num_envs), np.float32)
            rewards_ext = np.zeros((self.num_steps, self.num_envs), np.float32)
            next_states = np.zeros(
                (self.num_steps, self.num_envs, 84, 84, 1), np.float32)
            dones = np.zeros((self.num_steps, self.num_envs), np.int32)

            hidden = self.actor_critic.init_hidden(self.num_envs)

            for t in range(self.num_steps):
                action, action_log_prob, hidden = self.select_action(
                    state, hidden)
                next_state, reward_ext, done, info = envs.step(action)

                if self.render:
                    envs.render(0)

                # done
                for i, dne in enumerate(done):
                    if dne:
                        hidden[0][i] *= 0

                        # info
                        epinfo = info[i]['episode']
                        if 'visited_rooms' in epinfo:
                            self.visited_rooms.union(
                                list(epinfo['visited_rooms']))

                        self.best_reward = max(epinfo['r'], self.best_reward)
                        current_best_reward = max(
                            epinfo['r'], current_best_reward)
                        self.eplen += epinfo['l']

                # transitions
                states[t, ...] = state
                actions[t, ...] = action
                action_log_probs[t, ...] = action_log_prob
                rewards_ext[t, ...] = reward_ext
                next_states[t, ...] = next_state
                dones[t, ...] = done

            # logger
            logger.info('>>> rollout: {}'.format(rollout_id))
            logger.record_tabular('visited rooms',
                                  str(len(self.visited_rooms)) + ', ' + str(self.visited_rooms))
            logger.record_tabular('best reward', self.best_reward)
            logger.record_tabular('current best reward', current_best_reward)
            logger.record_tabular('total steps', self.eplen)
            logger.dump_tabular()

            # train neural networks
            self.update(states, actions, action_log_probs,
                        rewards_ext, next_states, dones)
            rollout_id += 1

    def update(self, states, actions, action_log_probs, rewards_ext, next_states, dones):
        # T * B * features
        states = torch.from_numpy(states).to(
            dtype=torch.float32, device=self.device).permute([0, 1, 4, 2, 3]) / 255.0
        actions = torch.from_numpy(actions).to(
            dtype=torch.int32, device=self.device).view(self.num_steps, self.num_envs)
        old_action_log_probs = torch.from_numpy(action_log_probs).to(
            dtype=torch.float32, device=self.device).view(self.num_steps, self.num_envs)
        rewards_ext = torch.from_numpy(rewards_ext).to(
            dtype=torch.float32, device=self.device).view(self.num_steps, self.num_envs)
        next_states = torch.from_numpy(next_states).to(
            dtype=torch.float32, device=self.device).permute([0, 1, 4, 2, 3]) / 255.0
        masks = 1 - torch.from_numpy(dones).to(dtype=torch.float32,
                                               device=self.device).view(self.num_steps, self.num_envs)

        # normalize observations (for random networks)
        self.observation_mean_std.update(
            next_states.view(-1, 84, 84).cpu().numpy())
        normalized_observation = (next_states - torch.from_numpy(self.observation_mean_std.mean).to(dtype=torch.float32, device=self.device)
                                  ) / torch.from_numpy(np.sqrt(self.observation_mean_std.var)).to(dtype=torch.float32, device=self.device)

        # get intrinsic rewards
        rewards_int = self.memories.correlation(normalized_observation)
        self.memories.store(normalized_observation)

        # normalize intrinsic rewards
        with torch.no_grad():
            rewards_int_forward = torch.cat(
                [self.reward_int_forward.update(rew) for rew in rewards_int], dim=0)
            self.reward_int_mean_std.update(
                rewards_int_forward.flatten().cpu().numpy())
            rewards_int = rewards_int / np.sqrt(self.reward_int_mean_std.var)

        # gae advantages
        advantage_exts = torch.zeros_like(rewards_ext)
        advantage_ints = torch.zeros_like(rewards_ext)

        with torch.no_grad():
            _, _, value_exts, value_ints, _ = self.actor_critic(
                torch.cat([states, next_states[-1].unsqueeze(0)], dim=0))

            # external reward
            last_gae_lam = 0
            for t in range(self.num_steps - 1, -1, -1):
                delta = rewards_ext[t] + self.gamma_ext * \
                    masks[t] * value_exts[t + 1] - value_exts[t]
                advantage_exts[t, :] = last_gae_lam = delta + \
                    self.lamda_gae * self.gamma_ext * masks[t] * last_gae_lam
            return_exts = advantage_exts + value_exts[:-1]

            # intrinsic reward
            last_gae_lam = 0
            for t in range(self.num_steps - 1, -1, -1):
                if self.episode_int:
                    delta = rewards_int[t] + self.gamma_ext * \
                        masks[t] * value_ints[t + 1] - value_ints[t]
                    advantage_ints[t, :] = last_gae_lam = delta + \
                        self.lamda_gae * self.gamma_ext * \
                        masks[t] * last_gae_lam
                else:
                    delta = rewards_int[t] + self.gamma_ext * \
                        value_ints[t + 1] - value_ints[t]
                    advantage_ints[t, :] = last_gae_lam = delta + \
                        self.lamda_gae * self.gamma_ext * last_gae_lam
            return_ints = advantage_ints + value_ints[:-1]

        advantages = self.coeff_ext * advantage_exts + self.coeff_int * advantage_ints

        logger.info('>>> samples: {}'.format(np.prod(states.shape[0:2])))
        logger.record_tabular('intrinsic reward mean',
                              rewards_int.mean(dim=(0, 1)))
        logger.record_tabular('intrinsic reward std',
                              rewards_int.std(dim=(0, 1)))
        logger.record_tabular('advantage mean', advantages.mean(dim=(0, 1)))
        logger.record_tabular('advantage std', advantages.std(dim=(0, 1)))
        logger.record_tabular('intrinsic return mean',
                              return_ints.mean(dim=(0, 1)))
        logger.record_tabular('intrinsic return std',
                              return_ints.std(dim=(0, 1)))
        logger.record_tabular('external return mean',
                              return_exts.mean(dim=(0, 1)))
        logger.record_tabular('external return std',
                              return_exts.std(dim=(0, 1)))
        logger.dump_tabular()

        # train epochs
        for epoch_id in range(self.update_epochs):
            self.training_step += 1

            for batch_id in range(self.num_batch):
                # samples in batch
                # T * B * features
                start, end = batch_id * \
                    self.stride_batch, (batch_id + 1) * self.stride_batch

                state = states[:, start:end, ...]
                action = actions[:, start:end, ...]
                old_action_log_prob = old_action_log_probs[:, start:end, ...]
                return_ext = return_exts[:, start:end, ...]
                return_int = return_ints[:, start:end, ...]
                advantage = advantages[:, start:end, ...]

                # PPO Loss!!!
                # policy loss
                dist_action, _, value_ext, value_int, _ = self.actor_critic(
                    state)
                action_log_prob = dist_action.log_prob(action)

                ratio = torch.exp(action_log_prob - old_action_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * advantage
                action_loss = -torch.mean(torch.min(surr1, surr2), dim=(0, 1))

                # value loss
                smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
                value_loss_ext = 0.5 * \
                    smooth_l1_loss(return_ext.flatten(), value_ext.flatten())
                value_loss_int = 0.5 * \
                    smooth_l1_loss(return_int.flatten(), value_int.flatten())
                value_loss = value_loss_ext + value_loss_int

                # entropy loss
                entropy = -torch.mean(dist_action.entropy(), dim=(0, 1))

                # backprop
                loss = action_loss + value_loss + self.coeff_ent * entropy
                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm > 1e-8:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                if epoch_id == self.update_epochs - 1 and batch_id == self.num_batch - 1:
                    logger.info('>>> train: {}'.format(self.training_step))
                    logger.record_tabular('external value loss',
                                          value_loss_ext.item())
                    logger.record_tabular('intrinsic value loss',
                                          value_loss_int.item())
                    logger.record_tabular('action loss', action_loss.item())
                    logger.record_tabular('action entropy', entropy.item())
                    logger.dump_tabular()