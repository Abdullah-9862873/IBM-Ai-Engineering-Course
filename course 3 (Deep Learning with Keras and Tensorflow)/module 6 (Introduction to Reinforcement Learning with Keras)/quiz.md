# Module 6 Quiz: Introduction to Reinforcement Learning with Keras

## Question 1
What is Q-learning?

- [ ] A supervised learning algorithm for classification
- [x] **An off-policy reinforcement learning algorithm that learns the value of taking an action in a given state**
- [ ] An unsupervised learning algorithm for clustering
- [ ] A semi-supervised learning algorithm

## Question 2
What does the Q-value function Q(s, a) represent?

- [ ] The probability of taking action a in state s
- [x] **The expected utility of taking action a in state s and following the optimal policy**
- [ ] The immediate reward received for taking action a in state s
- [ ] The number of times action a has been taken in state s

## Question 3
In the Bellman equation for Q-learning, what does the discount factor (γ) represent?

- [ ] The learning rate that controls how quickly the agent learns
- [x] **The importance of future rewards relative to immediate rewards**
- [ ] The probability of exploration
- [ ] The maximum Q-value

## Question 4
What is the purpose of the exploration rate (epsilon) in Q-learning?

- [ ] To control the learning rate
- [ ] To determine the discount factor
- [x] **To balance exploration of new actions and exploitation of learned knowledge**
- [ ] To set the number of episodes

## Question 5
Why is a Q-network (neural network) used instead of a Q-table in some environments?

- [ ] Because neural networks are always faster
- [x] **Because Q-tables become impractical for large or continuous state spaces**
- [ ] Because Q-tables cannot store Q-values
- [ ] Because neural networks are required by OpenAI Gym

## Question 6
What is the "CartPole" environment in OpenAI Gym?

- [ ] A game where an agent catches falling poles
- [x] **A classic control problem where the goal is to balance a pole on a cart**
- [ ] A racing game where a cart collects poles
- [ ] A puzzle game involving pole removal

## Question 7
What is the epsilon greedy policy used for in Q-learning?

- [ ] To initialize Q-values to zero
- [x] **To balance between exploring new actions and exploiting known good actions**
- [ ] To decay the learning rate
- [ ] To update Q-values

## Question 8
What does the Bellman equation incorporate in its update rule?

- [ ] Only the immediate reward
- [x] **Both the immediate reward and the estimated future rewards**
- [ ] Only the future rewards
- [ ] Neither the immediate nor future rewards

## Question 9
In Q-learning, what is the target Q-value used for training?

- [ ] The current Q-value
- [ ] The initial Q-value
- [x] **r + γ * max(Q(s', a')) - the reward plus discounted maximum future Q-value**
- [ ] The difference between two states

## Question 10
How does the exploration rate change during Q-learning training?

- [ ] It stays constant throughout training
- [x] **It decays over time to shift from exploration to exploitation**
- [ ] It increases over time
- [ ] It is reset after each episode

## Question 11
What are Deep Q-Networks (DQNs)?

- [x] **An extension of Q-Learning that uses deep neural networks to approximate the Q-value function**
- [ ] A type of supervised learning algorithm
- [ ] A clustering algorithm for unsupervised learning
- [ ] A data preprocessing technique

## Question 12
What is the main advantage of using experience replay in DQNs?

- [ ] It increases the speed of training
- [x] **It breaks the correlation between consecutive samples and improves learning stability**
- [ ] It reduces the size of the neural network
- [ ] It eliminates the need for a reward function

## Question 13
What is the purpose of the target network in DQNs?

- [ ] To make decisions during exploration
- [x] **To provide stable target Q-values during training, preventing oscillations**
- [ ] To store the replay buffer
- [ ] To render the environment

## Question 14
Why is experience replay important in DQN training?

- [ ] It stores the final Q-values
- [x] **It allows random sampling from the replay buffer to update the network, breaking correlation between consecutive experiences**
- [ ] It speeds up the environment rendering
- [ ] It eliminates the need for a discount factor

## Question 15
In a DQN, how are the target Q-values computed during training?

- [ ] Using only the immediate reward
- [ ] Using the primary Q-network only
- [x] **Using the target network: r + γ * max(Q(s', a'))**
- [ ] Using random initialization

## Question 16
How often are the target network weights updated in the DQN implementation?

- [ ] After every episode
- [x] **Periodically (e.g., every 10 episodes)**
- [ ] Only at the beginning of training
- [ ] Never

## Question 17
What is the primary limitation of traditional Q-Learning that DQNs address?

- [ ] Q-Learning cannot handle continuous actions
- [x] **Q-Learning uses a Q-table that becomes impractical for large state spaces**
- [ ] Q-Learning requires too little data
- [ ] Q-Learning cannot be used with neural networks

## Question 18
Which company famously used the DQN algorithm to achieve human-level performance in Atari games?

- [ ] OpenAI
- [x] **DeepMind**
- [ ] Google Brain
- [ ] Facebook AI Research

## Question 19
What is the key difference between the primary Q-network and the target network in DQNs?

- [ ] They have different architectures
- [x] **Target network weights are updated less frequently than the primary network**
- [ ] The target network is larger
- [ ] They use different activation functions

## Question 20
What is the purpose of the replay buffer in DQNs?

- [ ] To store the final model weights
- [x] **To store agent experiences (state, action, reward, next_state) for training**
- [ ] To render the game graphics
- [ ] To define the action space

## Question 21
What is the main goal of the Q-learning algorithm in reinforcement learning?

- [ ] To predict the next state given the current state and action
- [x] **To learn the value of an action in a particular state to maximize the total reward**
- [ ] To maximize the immediate reward at each step
- [ ] To minimize the number of steps taken to reach the goal

## Question 22
What does the Q-value represent in the Q-Learning algorithm?

- [ ] The difference between the current and next state
- [ ] The immediate reward for taking an action in a given state
- [ ] The probability of a state transition given an action
- [x] **The expected cumulative reward of taking an action in a given state**

## Question 23
In the epsilon-greedy policy used in Q-learning, what does the epsilon parameter control?

- [x] **The probability of choosing a random action**
- [ ] The discount factor for future rewards
- [ ] The learning rate of the algorithm
- [ ] The rate of decay for the reward function

## Question 24
What is the primary purpose of using a replay buffer in deep Q-networks (DQNs)?

- [x] **To sample a batch of experiences for training**
- [ ] To keep track of the agent's current state
- [ ] To store the rewards obtained during each episode
- [ ] To store the model's weights during training

## Question 25
How does a deep Q-network (DQN) differ from traditional Q-learning?

- [ ] DQNs update Q-values using policy gradients
- [x] **DQNs use a neural network to approximate Q-values**
- [ ] DQNs are used only for continuous action spaces
- [ ] DQNs do not require a reward signal

## Question 26
What is the primary objective of Q-learning in reinforcement learning?

- [ ] To minimize the immediate reward for the agent
- [ ] To ignore future rewards in favor of immediate gains
- [x] **To learn a policy that maximizes the cumulative reward over time**
- [ ] To perform unsupervised clustering of data points

## Question 27
What is the role of the Q-value function Q(s, a) in Q-learning?

- [ ] It records the sequence of actions taken by the agent.
- [ ] It calculates the total number of steps taken by the agent.
- [ ] It measures the probability of reaching a terminal state from state s.
- [x] **It provides a measure of the expected utility of taking action a in state s.**

## Question 28
In the context of Q-learning, what does the term "exploration rate" (ε) refer to?

- [ ] The frequency of resetting the environment
- [ ] The speed at which the Q-values are updated
- [x] **The probability of the agent selecting a random action instead of the optimal one**
- [ ] The rate at which the agent's rewards are discounted

## Question 29
Why is a neural network used to approximate the Q-value function in environments with large state spaces?

- [ ] To simplify the action selection process
- [ ] To eliminate the need for a reward signal
- [ ] To increase the computation time
- [x] **To replace the Q-table that becomes impractical for large state spaces**

## Question 30
Which method is commonly used in Q-learning to balance exploration and exploitation?

- [x] **Epsilon-greedy policy**
- [ ] Stochastic gradient descent
- [ ] Backpropagation
- [ ] K-means clustering

## Question 31
What is a key innovation of deep Q-networks (DQNs) that helps stabilize training?

- [ ] Immediate rewards only
- [ ] Continuous action space exploration
- [x] **Experience replay and target networks**
- [ ] Using a single neural network for all computations

## Question 32
What is the main purpose of the replay buffer in deep Q-networks (DQNs)?

- [x] **To store agent experiences for random sampling during training**
- [ ] To reset the environment periodically
- [ ] To decrease the learning rate over time
- [ ] To store Q-values for each action taken

## Question 33
How often are the weights of the target network updated in deep Q-networks (DQNs) compared to the primary Q-network?

- [x] **Less frequently than the primary Q-network**
- [ ] At the same frequency as the primary Q-network
- [ ] More frequently than the primary Q-network
- [ ] They are never updated

## Question 34
What role does the Bellman equation play in training the Q-network?

- [ ] It helps to calculate the immediate reward for each action.
- [x] **It updates the Q-values by incorporating both immediate and future rewards.**
- [ ] It initializes the neural network weights.
- [ ] It determines the number of actions an agent can take.

## Question 35
What is the significance of the discount factor (γ) in the Bellman equation?

- [ ] It normalizes the Q-values.
- [ ] It controls the learning rate.
- [ ] It adjusts the exploration rate.
- [x] **It models the importance of future rewards.**
