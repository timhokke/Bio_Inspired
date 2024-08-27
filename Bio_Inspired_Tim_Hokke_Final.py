# Author: Tim Hokke
# 5072395
# This code is used for the AE4350 course Bio-Inspired Intelligence and Learning for AE Applications
# offered by the TU Delft.

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Convert the input image to grayscale, resize, and normalize
def preprocess_observation(observation):
    cropped_image = observation[:72, 12:84]  # Focus on the relevant part of the CarRacing-v2 environment
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY) / 255.0
    return grayscale_image

# Custom environment wrapper to handle frame skipping and stacking
class FrameProcessor(gym.Wrapper):
    def __init__(self, env, frame_skip=4, stack_size=4, no_op_max=100, **kwargs):
        super(FrameProcessor, self).__init__(env, **kwargs)
        self.no_op_max = no_op_max
        self.frame_skip = frame_skip
        self.stack_size = stack_size

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        processed_frame = preprocess_observation(obs)
        self.frame_stack = np.concatenate((self.frame_stack[1:], processed_frame[np.newaxis]), axis=0)
        return self.frame_stack, total_reward, done, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(np.random.randint(1, self.no_op_max + 1)):
            obs, _, done, truncated, _ = self.env.step(0)
            if done or truncated:
                obs, info = self.env.reset()
        processed_frame = preprocess_observation(obs)
        self.frame_stack = np.tile(processed_frame, (self.stack_size, 1, 1))
        return self.frame_stack, info

# Define the convolutional neural network for approximating Q-values
class QNetwork(nn.Module):
    def __init__(self, input_channels, action_space, activation_function=F.relu):
        super(QNetwork, self).__init__()
        self.conv_layer_1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc_input_dim = 32 * 7 * 7
        self.fc_layer_1 = nn.Linear(self.fc_input_dim, 256)
        self.output_layer = nn.Linear(256, action_space)
        self.activation = activation_function

    def forward(self, x):
        x = self.activation(self.conv_layer_1(x))
        x = self.activation(self.conv_layer_2(x))
        x = x.view((-1, self.fc_input_dim))
        x = self.activation(self.fc_layer_1(x))
        x = self.output_layer(x)
        return x

# Experience replay buffer for storing and sampling past experiences
class ReplayBuffer:
    def __init__(self, state_shape, action_shape, buffer_size=int(1e5)):
        self.state_memory = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, *action_shape), dtype=np.int64)
        self.reward_memory = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.termination_memory = np.zeros((buffer_size, 1), dtype=np.float32)
        self.pointer = 0
        self.buffer_size = buffer_size
        self.current_size = 0

    def sample_batch(self, batch_size):
        indices = np.random.randint(0, self.current_size, batch_size)
        return (
            torch.FloatTensor(self.state_memory[indices]),
            torch.FloatTensor(self.action_memory[indices]),
            torch.FloatTensor(self.reward_memory[indices]),
            torch.FloatTensor(self.next_state_memory[indices]),
            torch.FloatTensor(self.termination_memory[indices]),
        )

    def store_transition(self, state, action, reward, next_state, done):
        # Store the current transition in the buffer
        self.state_memory[self.pointer] = state
        self.action_memory[self.pointer] = action
        self.reward_memory[self.pointer] = reward
        self.next_state_memory[self.pointer] = next_state
        self.termination_memory[self.pointer] = done
        # Update the pointer and ensure it wraps around when the buffer is full
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

# DQN agent that interacts with the environment and learns from experiences
class DQNAgent:
    def __init__(self, state_dimensions, action_count, learning_rate=0.0001, epsilon_start=1.0, epsilon_min=0.1, discount_factor=0.95, batch_size=32, initial_steps=2000, target_update_frequency=1000, replay_buffer_size=int(5e3)):
        self.discount_factor = discount_factor
        self.num_actions = action_count
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.initial_steps = initial_steps
        self.target_update_frequency = target_update_frequency

        # Define the policy and target networks
        self.policy_network = QNetwork(state_dimensions[0], action_count)
        self.target_network = QNetwork(state_dimensions[0], action_count)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), learning_rate)

        # Create the experience replay buffer
        self.replay_buffer = ReplayBuffer(state_dimensions, (1,), replay_buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network.to(self.device)
        self.target_network.to(self.device)
        print(f"Using device: {self.device}")

        self.total_steps = 0
        self.epsilon_decay = (epsilon_start - epsilon_min) / 1e5

    def update_policy(self):
        # Sample a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = map(lambda x: x.to(self.device), self.replay_buffer.sample_batch(self.batch_size))
        
        # Compute the target Q-values using the target network
        next_q_values = self.target_network(next_states).detach()
        target_q_values = rewards + (1. - dones) * self.discount_factor * next_q_values.max(dim=1, keepdim=True).values
        
        # Calculate the loss between the predicted Q-values and the target Q-values
        current_q_values = self.policy_network(states).gather(1, actions.long())
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Perform a gradient descent step to minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'steps': self.total_steps, 'loss': loss.item()}

    def store_and_learn(self, transition):
        self.total_steps += 1
        self.replay_buffer.store_transition(*transition)

        # Start updating the policy only after enough initial experiences are collected
        if self.total_steps > self.initial_steps:
            self.update_policy()

        # Update the target network periodically
        if self.total_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        # Gradually decay epsilon to decrease exploration over time
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.1)

    @torch.no_grad()
    def select_action(self, state, explore=True):
        self.policy_network.train(explore)
        if explore and (np.random.rand() < self.epsilon or self.total_steps < self.initial_steps):
            # Choose a random action during exploration
            action = np.random.randint(0, self.num_actions)
        else:
            # Choose the action with the highest Q-value
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action

# Function to evaluate the agent's performance across multiple episodes
def assess_agent(agent, env_name='CarRacing-v2', evaluation_runs=5):
    eval_env = gym.make(env_name, continuous=False)
    eval_env = FrameProcessor(eval_env)
    cumulative_reward = 0
    for _ in range(evaluation_runs):
        (state, _), done, episode_reward = eval_env.reset(), False, 0
        while not done:
            action = agent.select_action(state, explore=False)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            episode_reward += reward
            done = terminated or truncated
        cumulative_reward += episode_reward
    return np.round(cumulative_reward / evaluation_runs, 4)

# Main loop for training the DQN agent
env = gym.make('CarRacing-v2', continuous=False)
env = FrameProcessor(env)

max_steps = int(1e5)
eval_interval = 10000
input_shape = (4, 72, 72)
output_dim = env.action_space.n

dqn_agent = DQNAgent(input_shape, output_dim)
episode_count = 0

# Initialize lists to track progress for plotting
evaluation_steps = []
evaluation_rewards = []

while dqn_agent.total_steps < max_steps:
    episode_count += 1
    (state, _), done, episode_reward = env.reset(), False, 0
    while not done:
        action = dqn_agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        dqn_agent.store_and_learn((state, action, reward, next_state, terminated or truncated))
        state = next_state
        episode_reward += reward
        done = terminated or truncated

    if dqn_agent.total_steps % eval_interval == 0:
        eval_reward = assess_agent(dqn_agent)
        evaluation_steps.append(dqn_agent.total_steps)
        evaluation_rewards.append(eval_reward)
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(evaluation_steps, evaluation_rewards, 'b-')
        plt.xlabel("Iteration")
        plt.ylabel("Average Return")
#         plt.title("DQN Training Progress")
        plt.show()
        print(f"Evaluation after {dqn_agent.total_steps} steps: Average Return = {eval_reward}")
    
    if dqn_agent.total_steps >= max_steps:
        break

# Final evaluation and display of results
final_reward = assess_agent(dqn_agent)
print(f"Final Evaluation Return: {final_reward}")

# Plot the final training progress
plt.figure(figsize=(10, 5))
plt.plot(evaluation_steps, evaluation_rewards, 'b-')
plt.xlabel("Iteration")
plt.ylabel("Average Return")
# plt.title("Final DQN Training Progress")
plt.show()
