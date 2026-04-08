# Module 4: Deep Networks

## Overview

Deep neural networks contain multiple hidden layers. This module covers implementing deep neural networks in PyTorch.

## Topics Covered

1. Deep Neural Networks with Multiple Hidden Layers
2. Implementing Deep Neural Networks in PyTorch

---

## 1. Deep Neural Networks with Multiple Hidden Layers

### From Shallow to Deep Networks

- A neural network with one hidden layer can generate decision functions to separate non-linearly separable data
- Adding more neurons to the hidden layer creates more complex decision functions
- However, too many neurons leads to overfitting
- Adding more hidden layers increases performance while decreasing overfitting risk

### Deep Neural Network Definition

- A network with more than one hidden layer is called a deep neural network
- Hidden layers can have different numbers of neurons

### Example Architecture

```
Input (D dimensions) → Hidden Layer 1 (3 neurons) → Hidden Layer 2 (5 neurons) → Output Layer
```

---

## 2. Implementing Deep Neural Networks in PyTorch

### Using nn.Module

```python
class Deep_Neural_Network(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Deep_Neural_Network, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
```

Parameters:
- `D_in`: Size of input features
- `H1`: Number of neurons in first hidden layer
- `H2`: Number of neurons in second hidden layer
- `D_out`: Number of output classes

### With Tanh Activation

```python
class Deep_Neural_Network_Tanh(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Deep_Neural_Network_Tanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x
```

### With ReLU Activation

```python
class Deep_Neural_Network_ReLU(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Deep_Neural_Network_ReLU, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
```

### Network Parameters Shape

For a network with:
- Input: 3 dimensions
- Hidden Layer 1: 3 neurons
- Hidden Layer 2: 4 neurons
- Output: 3 classes

Each neuron in first layer has 3 input dimensions.
Each neuron in second layer has 3 inputs.
Each neuron in output layer has 4 inputs.

Use `.parameters()` to check the shape of the network.

### Using nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(D_in, H1),
    nn.Sigmoid(),
    nn.Linear(H1, H2),
    nn.Sigmoid(),
    nn.Linear(H2, D_out)
)
```

### Training with MNIST Dataset

```python
# Create dataset and loaders
train_dataset = ...
val_dataset = ...
train_loader = ...
val_loader = ...

# Create model with two hidden layers
model = Deep_Neural_Network(784, 50, 50, 10)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(num_epochs):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Performance Comparison

- ReLU and Tanh activation functions perform better than Sigmoid
- Better performance in terms of loss
- Better performance in terms of validation accuracy

---

## 3. Building Deep Neural Networks with nn.ModuleList

### Overview

- Using nn.ModuleList allows creating neural networks with an arbitrary number of layers
- Automates the process of constructing layers
- More flexible than manually defining each layer

### Using nn.ModuleList

```python
class Deep_Neural_Network_ModuleList(nn.Module):
    def __init__(self,_layers):
        super(Deep_Neural_Network_ModuleList, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            input_size = layers[i]
            output_size = layers[i + 1]
            self.layers.append(nn.Linear(input_size, output_size))
    
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = torch.relu(x)
        
        x = self.layers[-1](x)
        return x
```

### Example Construction

```python
layers = [2, 3, 4, 3]
# 2: input features
# 3: first hidden layer (3 neurons)
# 4: second hidden layer (4 neurons)
# 3: output classes

model = Deep_Neural_Network_ModuleList(layers)
```

### How It Works

1. First element (2): Input size
2. Second element (3): First hidden layer - 3 neurons, each with input dimension 2
3. Third element (4): Second hidden layer - 4 neurons, each with input dimension 3
4. Fourth element (3): Output layer - 3 classes, each with input dimension 4

### Forward Function

```python
def forward(self, x):
    for i in range(len(self.layers) - 1):
        x = torch.relu(self.layers[i](x))
    x = self.layers[-1](x)
    return x
```

- Apply linear transformation and ReLU activation for all hidden layers
- Apply only linear transformation for the output layer
- ReLU performs better than other activation functions

### Example Network Architecture Diagram

```
Layer 0: Input (2 features)
    ↓
Layer 1: Linear(2, 3) → ReLU → 3 neurons
    ↓
Layer 2: Linear(3, 4) → ReLU → 4 neurons
    ↓
Layer 3: Linear(4, 3) → Output (3 classes)
```

### Training

```python
# Create model
model = Deep_Neural_Network_ModuleList([784, 50, 50, 10])

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Advantages

- Can easily add more layers by updating the layers list
- Can experiment with different combinations of neurons and layers
- Allows testing various architectures to find the best performance

### Visualization

```
Layers: [2, 3, 4, 3]

Input (2 dims)
    ↓
Hidden 1 (3 neurons)
    ↓
Hidden 2 (4 neurons)
    ↓
Output (3 classes)
```

---

## 4. Dropout Regularization

### Overview

- Dropout is a regularization technique to prevent overfitting in neural networks
- Involves two phases: training and evaluation
- During training: randomly "drops out" neurons
- During evaluation: all neurons active

### The Problem

- Real-world data is noisy - samples may fall on wrong side of decision boundary
- Too many parameters (layers/neurons) → overfitting
- Too few parameters → underfinding
- Finding optimal architecture is time-consuming

### Solution: Dropout

- Start with a complex model
- Apply dropout to prevent overfitting

### How Dropout Works

The dropout is implemented by multiplying the activation function with a Bernoulli random variable r:
- r = 0 with probability p
- r = 1 with probability 1-p

Bernoulli distribution is like flipping a coin:
- p = 0.5: 50% chance to drop each neuron

### Implementation

In layer l:
```
output = activation * r
```

If r₁ = 0, the first neuron is shut off.

Each neuron is independent:
- Shutting off one neuron doesn't affect others
- Different neurons dropped each iteration

### PyTorch Implementation

```python
class Neural_Network_Dropout(nn.Module):
    def __init__(self, D_in, H1, H2, D_out, p=0.5):
        super(Neural_Network_Dropout, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x
```

Using nn.Sequential:

```python
model = nn.Sequential(
    nn.Linear(D_in, H1),
    nn.ReLU(),
    nn.Dropout(p),
    nn.Linear(H1, H2),
    nn.ReLU(),
    nn.Dropout(p),
    nn.Linear(H2, D_out)
)
```

### Training vs Evaluation

Training phase:
```python
model.train()  # Enables dropout
```

Evaluation phase:
```python
model.eval()  # Disables dropout
```

### Normalization

PyTorch normalizes activations during training by dividing by (1-p):
- Expected value of each neuron being on is 1-p

Choosing p:
- p = 0.8: majority of neurons removed
- p = 1: all neurons multiplied by zero
- Layers with few neurons: p = 0.1 to 0.05
- Layers with many neurons: p = 0.5

### Performance Comparison

Without Dropout:
- Decision boundary overlaps data
- Validation accuracy: ~77%

With Dropout (p=0.5):
- Smoother decision boundary
- Validation accuracy: ~87%

### Hyperparameter Tuning

- p too small (e.g., 0.01): overfitting
- p too large: underfitting
- Optimal p found through cross-validation

### Cost Comparison Graph

```
Training Cost:
    │
    │   ╭────────────── Blue (no dropout)
    │  ╱
    │ ╱        Green (with dropout)
    │╱
    └────────────────────────────── Iteration
```

Observation:
- Model without dropout: training cost continuously decreases
- Model with dropout: training cost higher but validation cost lower

```
Validation Cost:
    │
    │       Blue (no dropout) ─ increases over time
    │      ╱
    │     ╱  Green (with dropout) ─ lower and stable
    │    ╱
    └────────────────────────────── Iteration
```

### Training Code

```python
# Create model with dropout
model = Neural_Network_Dropout(D_in, H1, H2, D_out, p=0.5)

# Set to training mode
model.train()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(val_data)
```

---

## 5. Weight Initialization

### Overview

- Proper weight initialization is crucial for neural network performance
- Incorrect initialization leads to poor training
- PyTorch handles initialization by default, but understanding it helps

### Problem: Same Weights

If all weights in the same layer have the same value:
- Each neuron will have the same output
- Same gradient updates for all neurons
- Network fails to learn properly

Example:
- Initialize all weights to 1, bias to 0
- Model performs poorly predicting decision function
- All linear weights in first layer are identical

Solution: Random initialization from a distribution

### Uniform Distribution

Sampling from uniform distribution:
- Constant probability in specified range (e.g., -1 to 1)
- Any value sampled with equal probability

Problems with uniform distribution:
1. Range too narrow (e.g., -0.05 to 0.05): Values too close together
2. Range too wide: Large values cause vanishing gradient

### Vanishing Gradient Problem

For tanh activation:
- At large |z| or small |z|: derivative ≈ 0
- Gradient is product of derivative of activation functions
- If derivative is close to 0, gradient approaches 0

### Large z Values

If weights are too large:
- z = sum of (weight × activation) for each input
- With 2 inputs × 0.5 × 1: z = 1 (derivative ≈ 0.7)
- With 4 inputs × 0.5 × 1: z = 2 (derivative much smaller)
- With 100 inputs: z = 50 (derivative ≈ 0)
- Result: Vanishing gradient

### Solution: Scale Distribution

Scale width by inverse of number of inputs:
- 2 neurons: scale by 1/2 → max value = 0.5
- 4 neurons: scale by 1/4 → max value = 0.25
- 6 neurons: scale by 1/6 → max value ≈ 0.167

---

## Initialization Methods in PyTorch

### 1. Default Initialization (PyTorch)

For Linear(D_in, D_out):
- Range: [-1/√D_in, 1/√D_in]

```python
layer = nn.Linear(D_in, D_out)
# Weights already initialized correctly
```

### 2. Xavier Initialization

For use with Tanh activation:
- Considers both input (Lin) and output (Lout) neurons

Range formula:
```
limit = sqrt(6 / (Lin + Lout))
```
Range: [-limit, limit]

```python
# Apply Xavier initialization
nn.init.xavier_uniform_(layer.weight)
```

### 3. He Initialization

For use with ReLU activation:
- Considers number of input neurons

Range formula:
```
limit = sqrt(2 / Lin)
```
Range: [-limit, limit]

```python
# Apply He initialization
nn.init.kaiming_uniform_(layer.weight)
```

### Performance Comparison

```
Validation Accuracy:
    │
    │       ╭──────── Simple uniform
    │      ╱
    │     ╱    ╭─── PyTorch default
    │    ╱    ╱
    │   ╱    ╱  ╭─ Xavier/He method
    │  ╱    ╱  ╱
    │ ╱    ╱  ╱
    │╱    ╱  ╱
    └────────────────────────── Iteration
```

Xavier Method:
- Improves faster than simple uniform or default
- Best for Tanh activation

He Method:
- Best for ReLU activation
- Significantly better than uniform or default

### Implementation Code

```python
import torch.nn as nn

# Create layer
layer = nn.Linear(D_in, D_out)

# Xavier initialization (for Tanh)
nn.init.xavier_uniform_(layer.weight)

# He initialization (for ReLU)
nn.init.kaiming_uniform_(layer.weight)

# Check initialization
print(layer.weight)
```

---

## 6. Gradient Descent with Momentum

### Overview

- Momentum helps overcome saddle points and local minima
- Uses physics analogy: position, velocity, acceleration

### Physics Analogy

Position equation:
```
x(t) = x₀ + v₀t + ½at²
```

- x: position of ball
- v: velocity
- a: acceleration (like gradient)
- t: time

### Gradient Descent with Momentum

Update rules:
1. Velocity: vₖ₊₁ = ∇J(wₖ) + ρvₖ
2. Weight: wₖ₊₁ = wₖ - ηvₖ₊₁

Where:
- ∇J(wₖ): gradient of loss function
- ρ (rho): momentum term (0 < ρ < 1)
- η (eta): learning rate

### Velocity Update

```
v₁ = gradient of loss + ρ × v₀
v₀ = 0 (initial velocity)
```

Current velocity = gradient + (momentum × previous velocity)

### Parameter Update

```
w_new = w_old - learning_rate × velocity
```

### How Momentum Helps

**Saddle Points:**
- Flat region where gradient = 0
- With standard gradient descent: stuck
- With momentum: previous velocity carries ball through flat region
- Even when gradient = 0, velocity ≠ 0

**Local Minima:**
- Smallest cost in local neighborhood
- Momentum helps escape local minima
- Too small ρ: stuck in local minimum
- Good ρ: reaches global minimum
- Too large ρ: overshoots global minimum

### Visualization

```
Cost vs Weight:
          │
    Global│ minimum
     ╭───╯
    ╱
   ╱ Local minimum
  ╱     (ball gets stuck here without momentum)
 │
 └───────────────────────→ Weight
```

With momentum:
```
          │
    Global│ minimum
     ╭───╯
    ╱     ← momentum carries ball through
   ╱
  ╱         Local minimum
 │          (escaped with momentum)
 └───────────────────────→ Weight
```

### Using Momentum in PyTorch

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Common momentum values: 0.5, 0.9, 0.95

### Performance Comparison

```
Cost:
    │
    │   ╭────── SGD without momentum
    │  ╱
    │ ╱
    │╱  ╭── SGD with momentum (escapes saddle point)
    │╱ ╱
    │╱ ╱
    └──────────────────────→ Iteration
```

### Choosing Momentum

- Too small (e.g., 0.1): Gets stuck in saddle points
- Good value (e.g., 0.5-0.9): Escapes saddle points, reaches global minimum
- Too large: May overshoot global minimum

### PyTorch Implementation

```python
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Or with different values
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

### Lab Example (Spiral Dataset)

Each color represents a different class.

```
Spiral Data:
    │
    │   ● ●    ● ● ●
    │  ●        ●
    │ ●          ●
    │●            ●●
    │              ●
    │ ●          ●
    │  ●        ●
    │   ● ●    ● ● ●
    └───────────────────────
```

**Results:**
- Momentum = 0.5 performs best
- SGD without momentum: gets stuck
- Momentum helps reach global minimum

---

## 7. Batch Normalization

### Overview

- Batch normalization normalizes activations within mini-batches
- Applied before activation function
- Learns scale and shift parameters

### How Batch Normalization Works

For each mini-batch:
1. Calculate mean and variance for each neuron
2. Normalize outputs: (x - mean) / √(variance + ε)
3. Apply learnable scale (γ) and shift (β) parameters

### Step-by-Step Example

Input: Mini-batch with M samples, N neurons

For each neuron:
1. Calculate mean: μ = (1/M) Σxᵢ
2. Calculate variance: σ² = (1/M) Σ(xᵢ - μ)²
3. Normalize: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
4. Scale and shift: yᵢ = γx̂ᵢ + β

### Forward Pass with Batch Norm

```
Input → Linear → Batch Norm → Activation → Output
```

### Batch Norm for Multiple Layers

```python
# Layer 1: 3 neurons
batch_norm1 = nn.BatchNorm1d(3)

# Layer 2: 4 neurons
batch_norm2 = nn.BatchNorm1d(4)
```

### Training vs Prediction

Training: Use batch mean and variance
Prediction: Use population mean and variance (computed during training)

---

## Why Batch Normalization Works

### 1. Normalizes Input Ranges

Without batch normalization:
- z = w₁x₁ + w₂x₂ can have very different ranges
- Parameters take on very different values

With batch normalization:
- Inputs normalized to similar ranges
- Parameters take on more balanced values

### 2. Improves Loss Contours

Without batch normalization:
```
Loss Contours:
    │   
    │  elongated ovals
    │ ╭───────╮
    │╱         ╲
    │╲         ╱
    │ ╰───────╯
    └─────────────
```

- Gradient descent takes large steps in some directions
- Small steps in others

With batch normalization:
```
Loss Contours:
    │   
    │  
    │    ●
    │  
    │   
    └─────────────
```

- Contours are more round
- Gradient descent more uniform

### 3. Reduces Vanishing Gradient

Without normalization:
- Large input values → derivative ≈ 0
- Vanishing gradient problem

With normalization:
- Values in reasonable range
- Gradient doesn't vanish

### 4. Reduces Internal Covariance Shift

- Prevents distribution of activations from changing
- Allows higher learning rates

### Additional Benefits

- Acts as regularization
- May eliminate need for dropout
- Bias term not necessary after normalization

---

## Implementing Batch Norm in PyTorch

### Using nn.Module

```python
class Net_BatchNorm(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net_BatchNorm, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.batch_norm1 = nn.BatchNorm1d(H1)
        self.linear2 = nn.Linear(H1, H2)
        self.batch_norm2 = nn.BatchNorm1d(H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.linear1(x)))
        x = torch.relu(self.batch_norm2(self.linear2(x)))
        x = self.linear3(x)
        return x
```

### Using nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(D_in, H1),
    nn.BatchNorm1d(H1),
    nn.ReLU(),
    nn.Linear(H1, H2),
    nn.BatchNorm1d(H2),
    nn.ReLU(),
    nn.Linear(H2, D_out)
)
```

### Training and Evaluation

Training:
```python
model.train()
# Set to training mode
```

Evaluation:
```python
model.eval()
# Uses population statistics
```

### Performance

```
Training Loss:
    │
    │   ╭────── Without Batch Norm
    │  ╱
    │ ╱
    │╱  ╭─ With Batch Norm (converges faster)
    │╱ ╱
    │╱ ╱
    └──────────────────────→ Iteration
```

- Converges much faster
- Better test accuracy
- More stable training

---

## Summary

- Deep neural networks have more than one hidden layer
- Different hidden layers can have different numbers of neurons
- ReLU and Tanh perform better than Sigmoid for deep networks
- Adding more layers generally increases performance while reducing overfitting risk
- nn.ModuleList allows creating networks with arbitrary number of layers
- Can experiment with different layer configurations
- Dropout prevents overfitting by randomly dropping neurons during training
- Use model.train() for training, model.eval() for evaluation
- Proper weight initialization prevents vanishing gradients
- Use Xavier for Tanh, He for ReLU activation
- Momentum helps escape saddle points and local minima
- Use optimizer with momentum parameter in PyTorch
- Batch normalization normalizes activations, improves training stability
- Use nn.BatchNorm1d for fully connected layers