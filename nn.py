import torch.nn as nn
from copy import deepcopy
import torch
from collections import deque
import random
import numpy as np


class OUNoise:
    def __init__(self, action_dimention, mu=0, theta=0.15, sigma=0.3):
        self.action_dimention = action_dimention
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimention) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimention) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x+dx
        return self.state


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim, output_tanh):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.layer3 = nn.Linear(layer2_dim, output_dim)
        self.output_tanh = output_tanh
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.layer1(input)
        hidden = self.leaky_relu(hidden)
        hidden = self.layer2(hidden)
        hidden = self.leaky_relu(hidden)
        output = self.layer3(hidden)
        if self.output_tanh:
            return self.tanh(output)
        else:
            return output


class DDPG():
    def __init__(self, state_dim, action_dim, action_scale, noise_decrease,
                 gamma=0.999, batch_size=64, q_lr=1e-4, pi_lr=1e-5, tau=0.001, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pi_model = NeuralNetwork(
            self.state_dim, 400, 300, self.action_dim, output_tanh=True).to(self.device)
        self.q_model = NeuralNetwork(
            self.state_dim + self.action_dim, 400, 300, 1, output_tanh=False).to(self.device)
        self.pi_target_model = deepcopy(self.pi_model).to(self.device)
        self.q_target_model = deepcopy(self.q_model).to(self.device)
        self.noise = OUNoise(self.action_dim)
        self.noise_threshold = 1
        self.noise_decrease = noise_decrease
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_lr)
        self.pi_optimizer = torch.optim.Adam(
            self.pi_model.parameters(), lr=pi_lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state):
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).to(self.device)
        pred_action = self.pi_model(state_tensor).detach().cpu().numpy()
        action = pred_action + self.noise_threshold * self.noise.sample()
        return action

    def fit(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))
        if len(self.memory) >= self.batch_size:
            # Sample a batch of transitions
            batch = random.sample(self.memory, self.batch_size)
            # Convert batch to numpy arrays for efficient tensor creation
            states, actions, rewards, dones, next_states = map(
                np.array, zip(*batch))
            # Convert numpy arrays to tensors and move to the appropriate device
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(
                actions, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).reshape(
                self.batch_size, 1).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).reshape(
                self.batch_size, 1).to(self.device)
            next_states = torch.tensor(
                next_states, dtype=torch.float32).to(self.device)

            # Update Q-Model
            pred_next_actions = self.pi_target_model(next_states)
            next_states_and_pred_next_actions = torch.cat(
                (next_states, pred_next_actions), dim=1)
            targets = rewards + self.gamma * \
                (1 - dones) * self.q_target_model(next_states_and_pred_next_actions)
            actions = actions.unsqueeze(1)

            states_and_actions = torch.cat((states, actions), dim=1)
            q_loss = torch.mean(
                (targets.detach() - self.q_model(states_and_actions)) ** 2)
            self.update_target_model(
                self.q_target_model, self.q_model, self.q_optimizer, q_loss)

            # print(f'Q loss:{q_loss.item()}')

            # Update Pi-Model
            pred_actions = self.pi_model(states)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            pi_loss = -torch.mean(self.q_model(states_and_pred_actions))
            self.update_target_model(self.pi_target_model,
                                     self.pi_model, self.pi_optimizer, pi_loss)

            # print(f'Policy loss:{pi_loss.item()}')

        if self.noise_threshold > 0:
            self.noise_threshold = max(
                0.01, self.noise_threshold - self.noise_decrease)

    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=0.5)

        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data)

    def save_model(self, filepath="ddpg_checkpoint.pth"):
        checkpoint = {
            'pi_model_state_dict': self.pi_model.state_dict(),
            'q_model_state_dict': self.q_model.state_dict(),
            'pi_target_model_state_dict': self.pi_target_model.state_dict(),
            'q_target_model_state_dict': self.q_target_model.state_dict(),
            'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'noise_state': self.noise.state,
            'noise_threshold': self.noise_threshold,
            'tau': self.tau,
            'gamma': self.gamma,
            'memory': self.memory,
        }
        torch.save(checkpoint, filepath)
        print(f"The model is saved in {filepath}")

    def load_model(self, filepath="ddpg_checkpoint.pth"):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.pi_model.load_state_dict(checkpoint['pi_model_state_dict'])
        self.q_model.load_state_dict(checkpoint['q_model_state_dict'])
        self.pi_target_model.load_state_dict(
            checkpoint['pi_target_model_state_dict'])
        self.q_target_model.load_state_dict(
            checkpoint['q_target_model_state_dict'])
        self.pi_optimizer.load_state_dict(
            checkpoint['pi_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.noise.state = checkpoint.get('noise_state', np.ones(
            self.action_dim))
        self.noise_threshold = checkpoint.get('noise_threshold', 1)
        self.tau = checkpoint.get('tau', 1e-3)
        self.gamma = checkpoint.get('gamma', 0.99)
        self.memory = checkpoint.get('memory', deque(maxlen=100000))
        print(f"Model loaded from {filepath}")
