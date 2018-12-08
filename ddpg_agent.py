import numpy as np
import random
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    "Interact with and learns from the environment"

    def __init__(self, state_size, action_size, random_seed,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,
                 tau=1e-3, gamma=0.99, batch_size=128,
                 buffer_size=int(1e5)):
        """Initialize an Agent object

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

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

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, buffer_size, batch_size, random_seed)

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

    def act(self, state, add_noise=True):
        "Returns actions for given state as per current policy"
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def save(self, filename='checkpoint.pth'):
        "Serialize actor and critic weights"
        checkpoint = {
            'actor': self.actor_local.state_dict(),
            'critic': self.critic_local.state_dict()
        }
        torch.save(checkpoint, filename)

    def load(self, filename, map_location=None):
        "Load weights for actor and critic"
        weights = torch.load(filename, map_location=map_location)
        self.actor_local.load_state_dict(weights['actor'])
        if 'critic' in weights:
            self.critic_local.load_state_dict(weights['critic'])

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
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
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


class OUNoise:
    "Ornstein-Uhlenbeck random process"

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        "Initialize parameters and noise process"
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

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

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
