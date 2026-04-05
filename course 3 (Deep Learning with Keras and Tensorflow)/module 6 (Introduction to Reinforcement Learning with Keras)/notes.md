# Introduction to Reinforcement Learning with Keras

## Learning Objectives
- Provide an overview of reinforcement learning
- Understand approaches and implementation for reinforcement learning
- Introduce reinforcement learning implementation using Python

## 1. Reinforcement Learning Overview

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning where labels are provided, RL learns through rewards obtained from actions.

### Key Components

| Component | Description | Example (Game) | Example (Ads) |
|-----------|-------------|-----------------|---------------|
| **Agent** | The entity that takes actions | The player | Program that decides ad placement |
| **Environment** | The world through which the agent moves | Chess board | Web page |
| **Action** | Available choices the agent can make | All possible moves | Add ad, remove ad, or do nothing |
| **Reward** | Feedback from the environment | Points, winning the game | Clicks, revenue |

### The RL Feedback Loop

1. **Agent** observes the current **state** of the environment
2. Agent selects an **action** from available actions
3. Action impacts the **environment**, changing its state
4. Environment provides **feedback (reward)** to the agent
5. Agent **learns** from the reward - reinforcing good actions, avoiding bad ones
6. Process repeats dynamically

### Key Characteristics

- **Rewards are generally unknown and must be estimated**
- Agents take many steps before reaching a reward
- Agents continuously learn to estimate rewards over time
- Goal: Maximize expected cumulative rewards over time

## 2. Historical Context and Advances

### Major Milestones

- **2013**: DeepMind developed a system to play Atari games and beat humans
- **2017**: AlphaGo defeated the world champion in Go - first time machines beat human champions in complex games using RL

### Challenges

- RL algorithms require significant data and computational resources
- Infinite possibilities at every juncture
- Large amount of data needed to train models

### Business Applications

- **Recommendation Engines** - Reward = correct recommendations
- **Marketing** - Reward = higher revenues or clicks
- **Automated Bidding** - Reward = optimized spending

## 3. Understanding the Problem

### Policy

- Solutions represent a **policy** by which agents choose actions in response to current state
- The policy is what we ultimately try to optimize
- Not directly supervised learning - input is the current state

### Differences from Typical ML

| Aspect | Typical ML | Reinforcement Learning |
|--------|-----------|----------------------|
| Labels | Known | Unknown, uncertain |
| Solutions | Static | Continuously changing |
| Feedback | Immediate | May be delayed |

### Key Challenges

- **Reward Uncertainty**: May not know if actions resulted in immediate rewards
- **Delayed Rewards**: Intermediate rewards may or may not lead to larger goals
- **Exploration vs. Exploitation**: Trade-off between trying new actions and using known good actions
- **State Changes**: As actions impact environment, the problem continuously changes

## 4. Implementation in Python

### OpenAI Gym

The most common library for reinforcement learning in Python is **OpenAI Gym**.

```python
import gym

# Create an environment
env = gym.make('CartPole-v0')  # Specify the game/environment

# Reset the environment to initial state
state = env.reset()

# Render the current state
env.render()

# Take an action (example)
action = env.action_space.sample()  # Random action
next_state, reward, done, info = env.step(action)

# Close the environment
env.close()
```

### Environment Methods

- `gym.make(env_name)` - Create an environment
- `env.reset()` - Reset environment to initial state
- `env.render()` - Display current state
- `env.step(action)` - Execute an action, returns:
  - `next_state`: New state after action
  - `reward`: Reward received
  - `done`: Whether episode is complete
  - `info`: Additional information
- `env.close()` - Close the environment

### Common Environments

- `CartPole-v0` - Balance a pole on a cart
- `MountainCar-v0` - Drive a car up a mountain
- `Pendulum-v0` - Swing a pendulum
- `Atari games` - Various classic Atari games

## Summary

**Key Takeaways:**

- Reinforcement learning involves an **agent** interacting with an **environment**
- Agents choose **actions** from available options
- Actions impact the environment, which provides **rewards** as feedback
- The goal is to maximize expected rewards over time
- Solutions are represented by a **policy** - how agents choose actions based on current state
- Rewards are often unknown and must be estimated
- Unlike typical ML, RL problems continuously change as actions impact the environment
- **OpenAI Gym** is the primary library for RL implementation in Python
- RL has applications in games, recommendation systems, marketing, and automated bidding

## 5. Q-Learning with Keras

### Learning Objectives
- Explain the concept of Q-learning
- Implement Q-learning using Keras
- Describe how to train the Q-network
- Explain how to evaluate the agent

### What is Q-Learning?

Q-learning is a widely used **off-policy** reinforcement learning algorithm that seeks to learn the value of taking a specific action in a given state. It aims to find the optimal action selection policy for an agent.

**Key Characteristics:**
- Type of **value-based** reinforcement learning algorithm
- Learns a policy that tells an agent what action to take under what circumstances
- Goal: Maximize cumulative reward over time

### The Q-Value Function

The essence of Q-learning lies in the **Q-value function, Q(s, a)**:
- Provides a measure of expected utility of taking action `a` in state `s`
- Follows the optimal policy thereafter

### The Bellman Equation

Q-values are updated iteratively using the Bellman equation:

```
Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `s` = current state
- `a` = current action
- `r` = reward received after taking action `a`
- `s'` = new state resulting from taking action `a`
- `α` = learning rate (controls extent to which new info overrides old)
- `γ` = discount factor (importance of future rewards)

### Implementation Steps

1. **Initialize the environment and parameters**
2. **Define the environment** using OpenAI Gym
3. **Initialize the Q-table** (for small state spaces) or use neural network
4. **Set hyperparameters**: learning rate (α), discount factor (γ), exploration rate (ε)
5. **Build the Q-network** using Keras
6. **Train the Q-network**
7. **Evaluate the agent**

### Example: CartPole Environment

```python
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Initialize environment
env = gym.make('CartPole-v0')

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.005
num_episodes = 100

# Get state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
```

### Build the Q-Network

For environments with large or continuous state spaces, use a neural network to approximate the Q-value function:

```python
def build_q_network(state_size, action_size):
    """Build Q-network using Keras."""
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

q_network = build_q_network(state_size, action_size)
```

**Network Architecture:**
- Input layer: matches state size
- Hidden layers: 2 layers with 24 neurons each, ReLU activation
- Output layer: matches action size, linear activation

### Training the Q-Network

```python
def train_q_network(q_network, env, num_episodes=100):
    """Train the Q-network."""
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0
        for step in range(500):
            # Epsilon greedy policy
            if np.random.random() < exploration_rate:
                action = env.action_space.sample()  # Exploration
            else:
                action = np.argmax(q_network.predict(state)[0])  # Exploitation
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Update Q-values using Bellman equation
            target = reward + discount_factor * np.max(q_network.predict(next_state)[0])
            target_f = q_network.predict(state)
            target_f[0][action] = target
            
            # Train the network
            q_network.fit(state, target_f, epochs=1, verbose=0)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Decay exploration rate
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    return q_network

# Train the network
q_network = train_q_network(q_network, env, num_episodes)
```

### Evaluate the Agent

```python
def evaluate_agent(q_network, env, num_episodes=10):
    """Evaluate the trained agent."""
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for step in range(500):
            # Render environment
            env.render()
            
            # Exploit learned Q-values
            action = np.argmax(q_network.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"Evaluation Episode: {episode + 1}, Total Reward: {total_reward}")
    
    env.close()
    return total_rewards

# Evaluate
avg_reward = np.mean(evaluate_agent(q_network, env))
print(f"\nAverage Reward over 10 episodes: {avg_reward}")
```

### Epsilon Greedy Policy

The agent uses an **epsilon greedy policy** to balance exploration and exploitation:

- **Exploration** (with probability ε): Select random action
- **Exploitation** (with probability 1-ε): Select action with highest Q-value

Epsilon decays over time to shift from exploration to exploitation.

### Summary of Q-Learning Steps

1. **Initialize environment** using `gym.make()`
2. **Set hyperparameters**: α (learning rate), γ (discount factor), ε (exploration rate)
3. **Build Q-network** with input=state_size, output=action_size
4. **Training loop**:
   - Reset environment
   - Select action using epsilon greedy policy
   - Take action, get next_state and reward
   - Update Q-values using Bellman equation
   - Train Q-network to minimize loss
   - Decay exploration rate
5. **Evaluate** by letting agent interact with environment using learned policy

### Key Takeaways

- Q-learning is an **off-policy** value-based RL algorithm
- The **Q-value function** Q(s,a) estimates expected utility of actions
- **Bellman equation** updates Q-values iteratively
- For large state spaces, use **Q-network** (neural network) instead of Q-table
- **Epsilon greedy policy** balances exploration vs. exploitation
- **OpenAI Gym** provides environments for RL training and evaluation

## 6. Deep Q-Networks (DQN) with Keras

### Learning Objectives
- Explain what are Deep Q-Networks
- Describe the DQNs key concepts
- Explain and demonstrate the steps to implement DQNs with Keras

### What are Deep Q-Networks?

**Deep Q-Networks (DQNs)** are an extension of Q-Learning that uses deep neural networks to approximate the Q-value function.

**Why DQNs?**
- Traditional Q-Learning uses a **Q-table** which becomes impractical for large state spaces
- The Q-table grows exponentially with state space size
- DQNs use a **neural network** to estimate Q-values, scaling to large/continuous state spaces

**Historical Note:** The DQN algorithm was famously used by DeepMind to achieve human-level performance in Atari games.

### Key Concepts

#### 1. Q-Value Function Approximation
Instead of using a Q-Table, DQNs use a neural network to approximate Q(s, a):
- Input: State
- Output: Q-values for all possible actions

#### 2. Experience Replay
- Store agent experiences (state, action, reward, next_state) in a **replay buffer**
- During training, sample random minibatches from this buffer
- **Benefits:**
  - Breaks correlation between consecutive samples
  - Improves learning stability
  - Allows efficient data reuse

#### 3. Target Network
- A separate target network generates target Q-values
- Target network weights are updated less frequently than the primary network
- **Benefits:**
  - Provides more stable target values
  - Prevents oscillations during training

### Implementation Steps

1. **Initialize environment and parameters**
2. **Build Q-Network and Target Network**
3. **Implement Experience Replay**
4. **Train the Q-Network**
5. **Evaluate the agent**

### Code Implementation

```python
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Initialize environment
env = gym.make('CartPole-v0')

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.005
batch_size = 32
replay_buffer_size = 1000
target_update_freq = 10

# Get state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
```

### Build Q-Network and Target Network

```python
def build_q_network(state_size, action_size):
    """Build Q-network using Keras."""
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

# Build both networks
q_network = build_q_network(state_size, action_size)
target_network = build_q_network(state_size, action_size)
target_network.set_weights(q_network.get_weights())
```

### Implement Experience Replay

```python
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        minibatch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        targets = []
        
        for idx in minibatch:
            state, action, reward, next_state, done = self.buffer[idx]
            
            # Get target Q-value
            target = reward
            if not done:
                target = reward + discount_factor * np.max(target_network.predict(next_state)[0])
            
            # Get current Q-value prediction
            target_f = q_network.predict(state)
            target_f[0][action] = target
            
            states.append(state[0])
            targets.append(target_f[0])
        
        states = np.array(states)
        targets = np.array(targets)
        
        q_network.fit(states, targets, epochs=1, verbose=0)

# Initialize replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)
```

### Training the Q-Network

```python
def train_dqn(q_network, target_network, replay_buffer, num_episodes=100):
    """Train the DQN."""
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0
        for step in range(500):
            # Epsilon greedy policy
            if np.random.random() < exploration_rate:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_network.predict(state)[0])
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience
            replay_buffer.remember(state, action, reward, next_state, done)
            
            # Train with experience replay
            replay_buffer.replay(batch_size)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            target_network.set_weights(q_network.get_weights())
        
        # Decay exploration rate
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    return q_network

# Train the DQN
q_network = train_dqn(q_network, target_network, replay_buffer, num_episodes)
```

### Evaluate the Agent

```python
def evaluate_dqn(q_network, env, num_episodes=10):
    """Evaluate the trained DQN agent."""
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for step in range(500):
            env.render()
            
            # Exploit learned Q-values
            action = np.argmax(q_network.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"Evaluation Episode: {episode + 1}, Total Reward: {total_reward}")
    
    env.close()
    return total_rewards

# Evaluate
avg_reward = np.mean(evaluate_dqn(q_network, env))
print(f"\nAverage Reward over 10 episodes: {avg_reward}")
```

### Summary of DQN Steps

1. **Initialize environment** with gym.make()
2. **Set hyperparameters**: learning rate, discount factor, exploration rate, batch size
3. **Build Q-Network and Target Network** with same architecture
4. **Initialize replay buffer** to store experiences
5. **Training loop**:
   - Select action using epsilon greedy policy
   - Take action, store experience in replay buffer
   - Sample random minibatch from replay buffer
   - Update Q-Network using Bellman equation
   - Periodically update target network weights
   - Decay exploration rate
6. **Evaluate** by letting agent interact with environment

### Key Differences: Q-Learning vs DQN

| Aspect | Q-Learning | DQN |
|--------|-----------|-----|
| Q-Value Storage | Q-Table | Neural Network |
| Experience Replay | Not used | Used |
| Target Network | Not used | Used |
| Stability | Can be unstable | More stable |

### Key Takeaways

- **DQNs** extend Q-Learning using deep neural networks
- **Experience Replay** breaks correlation between samples and stabilizes training
- **Target Network** provides stable Q-value targets
- DQN was used by DeepMind to achieve human-level performance in Atari games
- Key steps: Initialize → Build networks → Implement replay → Train → Evaluate

---

# Module Summary

**Reinforcement Learning Overview:**
- Reinforcement learning involves an agent interacting with an environment
- Agents choose actions that impact the environment and receive rewards as feedback
- The goal is to maximize expected cumulative rewards over time

**Q-Learning:**
- Q-learning is one of the foundational algorithms in reinforcement learning
- The essence of Q-learning lies in the **Q-value function Q(s, a)**
- The Q-values are updated iteratively using the **Bellman equation**, which incorporates both the immediate reward and the estimated future rewards

**Deep Q-Networks (DQNs):**
- The key innovations of DQNs include **experience replay** and **target networks**, which help stabilize training and improve performance
- Experience replay stores agent experiences and samples random minibatches for training
- Target network provides stable Q-value targets by updating weights less frequently

**Implementation Steps:**
- Initialize the environment
- Build the Q-network and target network
- Implement experience replay
- Train the Q-network
- Evaluate the agent

**Key Takeaway:**
Reinforcement learning is a powerful tool for training agents to make decisions in complex environments.
