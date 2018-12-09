import numpy as np
import random
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """Soft update model parameters

    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    target_local_param = zip(
        target_model.parameters(), local_model.parameters())
    for target_param, local_param in target_local_param:
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent():
    "Single agent no learning algorithm"

    def __init__(self, state_size, action_size, random_seed,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=0):
        """Initialize an Agent object

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            lr_actor (float) : learning rate actor network
            lr_critic (float) : learning rate critic network
            weight_decay (float) : weight decay regularizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.noise = OUNoise(action_size, random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=lr_critic,
            weight_decay=weight_decay)

    def act(self, state, add_noise=True):
        "Returns actions for given state as per current policy"
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def load(self, filename, map_location=None):
        "Load weights for actor and critic"
        weights = torch.load(filename, map_location=map_location)
        self.actor_local.load_state_dict(weights['actor'])
        if 'critic' in weights:
            self.critic_local.load_state_dict(weights['critic'])

    def reset(self):
        self.noise.reset()

    def save(self, filename='checkpoint.pth'):
        "Serialize actor and critic weights"
        checkpoint = {
            'actor': self.actor_local.state_dict(),
            'critic': self.critic_local.state_dict()
        }
        torch.save(checkpoint, filename)


class MiADDPG():
    """Multiple independent Agents trained with DDPG

    This class allows to shared experience-buffer and critic network of the
    class::Agent.
    """

    def __init__(self, num_agents, state_size, action_size, random_seed,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,
                 tau=1e-3, gamma=0.99,
                 batch_size=128, buffer_size=int(1e5),
                 share_critic=True, share_buffer=True):
        """Initialize an multi-agent wrapper

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            tau (float): control soft-update
            gamma (float): discount factor
            batch_size (int): size of training batch
            buffer_size (int) : cap on number of experiences
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.agents = [
            Agent(state_size, action_size, random_seed,
                  lr_actor=1e-4, lr_critic=1e-3, weight_decay=0)
            for i in range(num_agents)
        ]
        self.share_critic = share_critic
        if share_critic:
            self.critic_local = Critic(
                state_size, action_size, random_seed).to(device)
            self.critic_target = Critic(
                state_size, action_size, random_seed).to(device)
            self.critic_optimizer = optim.Adam(
                self.critic_local.parameters(), lr=lr_critic,
                weight_decay=weight_decay)
            for agent in self.agents:
                agent.critic_local = None
                agent.critic_target = None
                agent.critic_optimizer = None

        self.share_buffer = share_buffer
        num_buffer = num_agents
        if share_buffer:
            num_buffer = 2
        self.memory = [
            ReplayBuffer(buffer_size, batch_size)
            for i in range(num_buffer)
        ]

    def step(self, state, action, reward, next_state, done):
        "Save experience and random sample from buffer to learn"
        # Save experience / reward in replay memory
        for i in range(len(state)):
            ind = i
            if self.share_buffer:
                ind = 0
            self.memory[i].add(
                state[i, ...], action[i, ...], reward[i],
                next_state[i, ...], done[i])

        # Learn, if enough samples are available in memory
        c_i = random.randint(0, len(self.agents) - 1)
        for i, agent in enumerate(self.agents):
            update_critic = True
            if self.share_critic and i != c_i:
                update_critic = False

            ind = i
            if self.share_buffer:
                ind = 0
            if len(self.memory[ind]) < self.batch_size:
                continue

            experiences = self.memory[ind].sample()
            self.learn(agent, experiences, self.gamma, update_critic)

    def act(self, state, add_noise=True):
        "Returns actions for given state as per current policy"
        state = torch.from_numpy(state).float().to(device)
        action_list = []
        for i, agent in enumerate(self.agents):
            action_list.append(
                agent.act(state[[i], ...])
            )
        return np.concatenate(action_list, axis=0)

    def load(self, filename, map_location=None):
        "Load weights for actor and critic"
        weights = torch.load(filename, map_location=map_location)
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(weights[f'actor_{i}'])
            if self.share_critic:
                self.critic_local.load_state_dict(weights['critic'])
                continue
            agent.load_state_dict(weights[f'critic_{i}'])

    def reset(self):
        self.noise.reset()

    def save(self, filename='checkpoint.pth'):
        "Serialize actor and critic weights"
        checkpoint = {}
        for i, agent in enumerate(self.agents):
            checkpoint[f'actor_{i}'] = agent.actor_local.state_dict()
            if not self.share_critic:
                checkpoint[f'critic_{i}'] = agent.critic_local.state_dict()
        if self.share_critic:
            checkpoint[f'critic'] = self.critic_local.state_dict()
        torch.save(checkpoint, filename)

    def learn(self, agent, experiences, gamma, update_critic=True):
        """Update policy and value parameters with a batch of experiences

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        critic_target = agent.critic_target
        critic_local = agent.critic_local
        critic_optimizer = agent.critic_optimizer
        if self.share_critic:
            critic_target = self.critic_target
            critic_local = self.critic_local
            critic_optimizer = self.critic_optimizer

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = agent.actor_target(next_states)
        Q_targets_next = critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        if update_critic:
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = agent.actor_local(states)
        actor_loss = -critic_local(states, actions_pred).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # Update target networks
        soft_update(critic_local, critic_target, self.tau)
        soft_update(agent.actor_local, agent.actor_target, self.tau)


class OUNoise:
    "Ornstein-Uhlenbeck random process"

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process

        Params
        ======
            size (int): dimension of action state
            seed (int): seed for random number generator.
            mu (float): mean of gaussian noise.
            theta (float): scale of gaussian noise.
            sigma (float): standard deviation of gaussian noise.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        "Reset the internal state (= noise) to mean (mu)"
        self.state = 1 * self.mu

    def sample(self):
        "Update internal state and return it as a noise sample"
        x = self.state
        dx = self.rng.randn(len(x)).astype(x.dtype, copy=False)
        dx = self.theta * (self.mu - x) + self.sigma * dx
        self.state = x + dx
        return self.state


class ReplayBuffer:
    "Fixed-size buffer to store experience tuples"

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        "Add a new experience to memory"
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        "Randomly sample a batch of experiences from memory"
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        dones_np = np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)
        dones = torch.from_numpy(dones_np).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        "Return the current size of internal memory"
        return len(self.memory)


class SADDPG(Agent):
    "Single agent trained with DDPG criteria"

    def __init__(self, *args, tau=1e-3, gamma=0.99,
                 batch_size=128, buffer_size=int(1e5), **kwargs):
        """Initialize an Agent object

        Params
        ======
            tau (float): control soft-update
            gamma (float): discount factor
            batch_size (int): size of training batch
            buffer_size (int) : cap on number of experiences
        """
        super(SADDPG, self).__init__(*args, **kwargs)
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def step(self, state, action, reward, next_state, done):
        "Save experience and random sample from buffer to learn"
        # Save experience / reward in replay memory
        for i in range(len(state)):
            self.memory.add(state[i, ...], action[i, ...], reward[i],
                            next_state[i, ...], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update policy and value parameters with a batch of experiences

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        soft_update(self.critic_local, self.critic_target, self.tau)
        soft_update(self.actor_local, self.actor_target, self.tau)
