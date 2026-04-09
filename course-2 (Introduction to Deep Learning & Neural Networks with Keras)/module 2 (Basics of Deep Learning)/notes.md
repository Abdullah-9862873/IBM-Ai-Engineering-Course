# Gradient Descent

## Overview
- Iterative optimization algorithm for finding the minimum of a function
- Used to determine the best value of weights (w) in neural networks

## Cost/Loss Function

### Purpose
- Measures how well the model fits the data
- Best weights = minimum cost function value

### Example (Simplified)
- Given data where z = 2x
- Cost function: J = Σ(z - wx)²
- Goal: Find w that minimizes J
- Optimal solution: w = 2

### Properties
- Parabola shape with one global minimum
- Unique solution guaranteed

## Gradient Descent Algorithm

### Process
1. Start at a random initial value of w (w0)
2. Compute the gradient (slope) at current w
3. Update w: w_new = w_old - (learning rate × gradient)
4. Repeat until reaching minimum

### Gradient
- The slope of the tangent at current point
- Indicates direction and magnitude to move toward minimum

### Learning Rate
- Controls step size in each iteration
- **Large learning rate:** Big steps, may miss minimum
- **Small learning rate:** Small steps, takes longer to converge

### Update Rule
- w_new = w_old - α × ∂J/∂w
- Where α = learning rate

## Iterations Example

### Starting Point
- w0 = 0, initial line: z = 0
- High cost, bad fit

### Progression
- Iteration 1: w moves toward 2, steep gradient, big step
- Iteration 2: w gets closer to 2, slope decreases
- Iteration 3-4: w approaches 2, line fits data better

### Convergence
- Continue until cost function is within small threshold of minimum
- At w = 2, line fits perfectly

## Key Points

### Direction of Movement
- If starting to the left of minimum: negative gradient moves right
- If starting to the right of minimum: positive gradient moves left

### Learning Rate Considerations
- Must be carefully chosen
- Too large: oscillate or miss minimum
- Too slow: very slow convergence

## Summary
- Gradient descent finds optimal weights by iteratively moving toward minimum of cost function
- Learning rate controls step size
- Process continues until convergence

---

# Backpropagation

## Overview
- Algorithm for training neural networks
- Propagates error back through network to optimize weights and biases
- Uses chain rule to compute gradients

## Training Process (Supervised Learning)

### Steps
1. Calculate error between predicted value and ground truth
2. Propagate error back through network
3. Perform gradient descent to update weights and biases
4. Repeat until convergence

## Error Calculation

### Mean Squared Error (MSE)
- E = (1/n) × Σ(T - a)²
- Where T = ground truth, a = predicted value

### For Single Data Point
- E = (T - a)²

## Chain Rule for Backpropagation

### Updating Weights (w2)
- ∂E/∂w2 = (∂E/∂a2) × (∂a2/∂z2) × (∂z2/∂w2)
- ∂E/∂a2 = -(T - a2)
- ∂a2/∂z2 = a2 × (1 - a2)
- ∂z2/∂w2 = a1 (input to neuron 2)

### Updating Bias (b2)
- Similar to w2, but ∂z2/∂b2 = 1
- ∂E/∂b2 = -(T - a2) × a2 × (1 - a2)

### Updating Weights (w1)
- ∂E/∂w1 = (∂E/∂a2) × (∂a2/∂z2) × (∂z2/∂a1) × (∂a1/∂z1) × (∂z1/∂w1)
- Additional terms: ∂z2/∂a1 = w2, ∂a1/∂z1 = a1 × (1 - a1), ∂z1/∂w1 = x1

### Updating Bias (b1)
- Similar to w1, but ∂z1/∂b1 = 1

## Update Equations

### Weight Update
- w_new = w_old - (learning rate × gradient)

### Bias Update
- b_new = b_old - (learning rate × gradient)

## Training Algorithm

### Initialization
1. Initialize weights and biases to random values

### Iterative Process (repeat until convergence)
1. **Forward Propagation:** Calculate network output
2. **Calculate Error:** Compare predicted output with ground truth
3. **Backpropagation:** Update weights and biases using gradients
4. **Repeat** for n epochs or until error < threshold

## Example (from Forward Propagation Video)

### Given
- Input: x1 = 0.1
- Initial weights: w1 = 0.15, w2 = 0.2
- Initial biases: b1 = 0.4, b2 = 0.5
- Ground truth: T = 0.25
- Learning rate: 0.4

### Forward Propagation Results
- z1 = 0.415, a1 = 0.6023
- z2 = 0.9210, a2 = 0.7153

### After First Iteration (Backpropagation)
- w2 gradient: 0.05706 → w2 updated to 0.427
- b2 gradient: 0.0948 → b2 updated to 0.612
- w1 gradient: 0.001021 → w1 updated to 0.1496
- b1 gradient: 0.01021 → b1 updated to 0.3959

### Continue
- Repeat forward propagation with updated weights
- Calculate new error and propagate back
- Continue until error < threshold or max epochs reached

## Key Takeaways
- Backpropagation propagates error from output back to input
- Uses chain rule to compute gradients for each weight
- Weights updated in direction that reduces error
- Training continues until predictions match ground truth

---

# Vanishing Gradient Problem

## Overview
- Problem with sigmoid activation function
- Prevented neural networks from developing sooner
- Major issue in training deep neural networks

## The Problem

### With Sigmoid Activation
- All intermediate values are between 0 and 1
- During backpropagation, gradients are multiplied by each other
- Each factor is less than 1
- Gradients get smaller and smaller as they propagate backward

### Effect on Training
- Earlier layers (closer to input) learn very slowly
- Later layers (closer to output) learn faster
- Results in:
  - Very long training time
  - Compromised prediction accuracy

## Example (Simple 2-Neuron Network)
- Gradients of error with respect to w2 are small
- Gradients of error with respect to w1 are even smaller
- Earlier weights barely get updated

## Why It Happens
- Sigmoid derivative: a(1-a), maximum value is 0.25
- When multiplying several values < 1, result approaches zero
- Earlier layers receive vanishingly small gradients

## Solution
- Do not use sigmoid activation function
- Use other activation functions less prone to vanishing gradient
- (ReLU and others discussed in next video)

## Key Takeaways
- Sigmoid activation causes vanishing gradient problem
- Gradients shrink exponentially in deep networks
- Earlier layers train much slower than later layers
- This is why sigmoid is not commonly used as activation function

---

# Activation Functions

## Overview
- Play a major role in the learning process of neural networks
- Add nonlinearity to the network
- Enable network to learn complex patterns

## Types of Activation Functions

### 1. Binary Step Function
- Basic threshold-based function

### 2. Linear / Identity Function
- No nonlinearity (not typically used in hidden layers)

### 3. Sigmoid / Logistic Function
- **Formula:** σ(z) = 1/(1 + e^(-z))
- **Range:** 0 to 1
- **At z = 0:** a = 0.5
- **Problems:**
  - Flat beyond +3 and -3 regions → vanishing gradient
  - Not symmetric around origin
  - All values are positive

### 4. Hyperbolic Tangent (tanh)
- **Formula:** tanh(z)
- **Range:** -1 to +1
- **Advantages:**
  - Symmetric around origin
- **Disadvantages:**
  - Still leads to vanishing gradient in deep networks

### 5. Rectified Linear Unit (ReLU) - MOST POPULAR
- **Formula:** a = max(0, z)
- **Advantages:**
  - Does not activate all neurons simultaneously
  - Makes network sparse and efficient
  - Overcame the vanishing gradient problem
  - Computationally efficient
- **How it works:**
  - If input is negative → converted to zero (neuron not activated)
  - If input is positive → passes through unchanged

### 6. Leaky ReLU
- Allows small negative values to pass through
- Addresses "dying ReLU" problem

### 7. Softmax Function
- **Used in:** Output layer of classifiers
- **Purpose:** Converts outputs to probabilities that sum to 1
- **Example:**
  - Raw outputs: [1.6, 0.55, 0.98]
  - After softmax: [0.51, 0.18, 0.31]

## Summary

### Commonly Used
- **Hidden layers:** ReLU (most popular)
- **Output layer (classification):** Softmax
- **Output layer (regression):** Linear/Sigmoid

### Avoided
- Sigmoid and tanh: Lead to vanishing gradient problem

### Best Practice
- Start with ReLU in hidden layers
- Switch to other functions if ReLU doesn't perform well

## Key Takeaways
- Activation functions add nonlinearity to neural networks
- ReLU is the most widely used activation function today
- Softmax is used for classification output layers
- Sigmoid and tanh are generally avoided due to vanishing gradient
