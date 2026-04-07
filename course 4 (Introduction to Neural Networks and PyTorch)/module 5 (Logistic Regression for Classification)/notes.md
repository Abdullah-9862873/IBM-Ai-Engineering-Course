# Module 5: Logistic Regression for Classification

## Video: Linear Classifiers and Logistic Regression

### Overview
- **Linear Classifier**: Predicts class membership based on features
- **Logistic Regression**: A particular type of linear classifier for classification
- Used to predict which class a sample belongs to based on its features

### Data Representation
- **Matrix X**: Features for each sample (rows = samples, columns = features)
- **Vector y**: Class labels (discrete values: 0, 1, 2, etc.)
- Each element of y represents the class of each corresponding row in X

### Two-Class Classification 
- Linear classifier equation: z = wx + b
- For d dimensions: z = w·x + b

### Linearly Separable Data
- Data can be separated by a line (in 1D) or plane (in 2D+)
- All points of one class on one side, other class on the other side

### Threshold Function
- Converts continuous values to discrete classes
- If z > 0: return 1
- If z < 0: return 0

```python
def threshold(z):
    if z > 0:
        return 1
    else:
        return 0
```

### Limitation of Threshold Function
- Returns only 0 or 1
- No probability estimate
- Not smooth near decision boundary

---

## Video: Sigmoid Function (Logistic Function)

### Sigmoid Function (Logistic Function)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Smooth approximation to threshold function
- Output between 0 and 1
- Can be interpreted as probability

### Sigmoid Properties
- If z is very large negative → σ(z) ≈ 0
- If z is very large positive → σ(z) ≈ 1
- If z = 0 → σ(z) = 0.5

### Prediction with Sigmoid
```python
import torch

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# For z = 2 (positive side)
z = torch.tensor(2.0)
y_hat = sigmoid(z)  # > 0.5 → predict class 1

# For z = -3 (negative side)
z = torch.tensor(-3.0)
y_hat = sigmoid(z)  # < 0.5 → predict class 0
```

### Classification Decision
- If sigmoid(z) > 0.5 → ŷ = 1
- If sigmoid(z) < 0.5 → ŷ = 0
- Close to 0.5 → uncertain

### Why Sigmoid is Better Than Threshold
- Provides probability estimates
- Smooth gradient (better for training)
- Gradual transition near decision boundary

---

## Video: Multi-Dimensional Classification

### 2D Classification
- In 2D, use a plane instead of a line
- Bird's eye view: plane appears as a line
- Decision boundary is the line where z = 0

### Classification in Higher Dimensions
- Use hyperplane for classification
- General formula: z = w·x + b

### Probability Interpretation
- σ(z) = P(y = 1 | x)
- 1 - σ(z) = P(y = 0 | x)

```python
# Probability calculation
z = torch.tensor([2.0, -1.0, 0.5])
probs = sigmoid(z)

# probs[0] = P(y=1|x1) ≈ 0.88
# probs[1] = P(y=1|x2) ≈ 0.27
# probs[2] = P(y=1|x3) ≈ 0.62

# Predictions
predictions = (probs > 0.5).int()
```

### Summary of Classification Process
1. **Forward Pass**: Calculate z = w·x + b
2. **Apply Sigmoid**: ŷ = σ(z)
3. **Apply Threshold**: If ŷ > 0.5, predict class 1; else predict class 0

---

## Video: Logistic Regression Model

### Logistic Regression Equation
$$z = w_1x_1 + w_2x_2 + ... + w_dx_d + b$$

### Prediction
$$\hat{y} = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

### Binary Classification
- Output probability between 0 and 1
- Threshold at 0.5 for final prediction

### Training Logistic Regression
- Use cross-entropy loss (log loss) instead of MSE
- Gradient descent to optimize weights

### Complete Logistic Regression Example
```python
import torch
import torch.nn as nn

# Simple Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Create model
model = LogisticRegression(2)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training data
X = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.float32)
y = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_hat = model(X)
    
    # Calculate loss
    loss = criterion(y_hat, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

# Make predictions
predictions = (model(X) > 0.5).float()
print(f'Predictions: {predictions.tolist()}')
```

### Key Differences: Linear Regression vs Logistic Regression
| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| Output | Continuous | Probability (0-1) |
| Activation | None | Sigmoid |
| Loss Function | MSE | Cross Entropy |
| Use Case | Prediction | Classification |

---

## Video: Logistic Regression in PyTorch

### Creating Logistic Function in PyTorch

#### Method 1: Using torch.nn
```python
import torch
import torch.nn as nn

# Create sigmoid object
sigmoid = nn.Sigmoid()

# Input tensor
x = torch.tensor([1.0])

# Apply sigmoid
y_hat = sigmoid(x)
```

#### Method 2: Using torch module
```python
import torch

# Input tensor
x = torch.tensor([1.0])

# Apply sigmoid function
y_hat = torch.sigmoid(x)
```

---

## Video: Using nn.Sequential

### Building Logistic Regression with nn.Sequential
```python
import torch.nn as nn

# Build model using sequential
model = nn.Sequential(
    nn.Linear(1, 1),  # Linear layer: z = wx + b
    nn.Sigmoid()     # Sigmoid activation
)
```

### How nn.Sequential Works
- Takes input through first layer (Linear)
- Passes output to second layer (Sigmoid)
- Returns final prediction

### For 2D Input
```python
# 2D input model
model = nn.Sequential(
    nn.Linear(2, 1),  # 2 input features, 1 output
    nn.Sigmoid()      # Sigmoid activation
)
```

---

## Video: Building Custom Modules

### Custom Logistic Regression Module
```python
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)

# Create model (1D input)
model = LogisticRegression(1)

# Create model (2D input)
model = LogisticRegression(2)
```

### Comparison: Custom Module vs Sequential
```python
# Method 1: Custom module
custom_model = LogisticRegression(1)

# Method 2: Sequential
sequential_model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)

# Both produce same output
```

---

## Video: Making Predictions

### Single Sample Prediction
```python
# Model parameters (example)
b = 0.23
w = -0.23

# Input
x = torch.tensor([1.0])

# Prediction
# z = w*x + b = -0.23*1 + 0.23 = 0
# y_hat = sigmoid(0) = 0.5

model = LogisticRegression(1)
y_hat = model(x)
print(y_hat)  # Output: tensor([...])
```

### Multiple Samples Prediction
```python
# Multiple inputs
X = torch.tensor([[1.0], [2.0], [3.0]])

# Apply model to all samples
y_hat = model(X)

# Each element corresponds to one sample
# y_hat[0] = prediction for sample 1
# y_hat[1] = prediction for sample 2
# y_hat[2] = prediction for sample 3
```

### 2D Input Prediction
```python
# 2D input model
model = LogisticRegression(2)

# Single 2D sample
x = torch.tensor([[1.0, 2.0]])

# Prediction
y_hat = model(x)

# Multiple 2D samples
X = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
    [1.0, 1.0]
])

y_hat = model(X)
```

### Complete Prediction Example
```python
import torch
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Input data
X = torch.tensor([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2]
], dtype=torch.float32)

# Make predictions
predictions = model(X)

# Get class predictions (0 or 1)
class_predictions = (predictions > 0.5).int()

print(f'Probabilities: {predictions.tolist()}')
print(f'Class predictions: {class_predictions.tolist()}')
```

---

## Video: Training Logistic Regression

### Training Process
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training data
X = torch.tensor([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
], dtype=torch.float32)

y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_hat = model(X)
    
    # Calculate loss
    loss = criterion(y_hat, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

# Make predictions
predictions = (model(X) > 0.5).int()
print(f'Predictions: {predictions.tolist()}')
```

---

## Video: Bernoulli Distribution and Maximum Likelihood Estimation

### Bernoulli Distribution
- **Definition**: Probability distribution for a binary outcome (coin flip)
- **Parameter**: θ (theta) = probability of "success" (e.g., heads)
- **Formula**:
  - P(y = 1) = θ (probability of success)
  - P(y = 0) = 1 - θ (probability of failure)

### Example: Biased Coin Flip
```python
# θ = 0.2 means:
# P(heads) = 0.2
# P(tails) = 0.8
theta = 0.2
```

### Likelihood Function
- **Likelihood**: Probability of observing a specific sequence of outcomes
- Calculated by multiplying probabilities of individual events
- **Formula**: L(θ) = P(y₁) × P(y₂) × ... × P(yₙ)

### Example: Calculating Likelihood
```python
# Sequence: Head, Head, Tail
# θ = 0.2
likelihood = 0.2 * 0.2 * 0.8  # = 0.032
```

### Maximum Likelihood Estimation (MLE)
- **Goal**: Find the value of θ that maximizes the likelihood
- We don't know θ, but we observe data
- We try different values of θ and see which gives highest likelihood

### Example: Comparing Two θ Values
```python
# Sequence: Head, Tail, Head, Tail
# Compare θ = 0.5 vs θ = 0.2

# For θ = 0.5:
likelihood_theta_05 = 0.5 * 0.5 * 0.5 * 0.5  # = 0.0625

# For θ = 0.2:
likelihood_theta_02 = 0.2 * 0.8 * 0.2 * 0.8  # = 0.0256

# θ = 0.5 has higher likelihood (more likely to be the true parameter)
```

### Mathematical Representation
- **Bernoulli Distribution**:
$$P(y | \theta) = \theta^y (1 - \theta)^{1-y}$$

- **Likelihood Function**:
$$L(\theta) = \prod_{i=1}^{n} P(y_i | \theta)$$

- **Log Likelihood** (easier to maximize):
$$\log L(\theta) = \sum_{i=1}^{n} [y_i \log(\theta) + (1-y_i) \log(1-\theta)]$$

### Finding Optimal θ
- To maximize likelihood, we find where derivative equals zero
- **Log function is monotonically increasing**: Maximum position stays the same
- This is the basis for logistic regression training

### Connection to Logistic Regression
- Logistic regression uses Maximum Likelihood Estimation
- Instead of directly maximizing likelihood, we maximize log-likelihood
- This leads to the cross-entropy loss function

---

## Video: Cross-Entropy Loss

### Problem with Mean Squared Error (MSE)
- MSE can create flat regions in the loss surface
- Gradient becomes 0 in flat regions → parameters don't update
- Not ideal for classification tasks

### Cross-Entropy Loss
- **Also known as**: Log Loss, Binary Cross-Entropy
- Derived from Maximum Likelihood Estimation
- Formula:
$$L = -\frac{1}{n} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

### Advantages of Cross-Entropy
- **Smoother gradient** than MSE
- No flat regions (except at minimum)
- Better for classification tasks
- Converges faster

### Cross-Entropy vs MSE
| Aspect | MSE | Cross-Entropy |
|--------|-----|---------------|
| Gradient | Can be flat | Always smooth |
| Classification | Not ideal | Ideal |
| Convergence | Slower | Faster |

---

## Video: Training Logistic Regression with Cross-Entropy

### Creating the Model
```python
import torch.nn as nn

# Method 1: Using nn.Sequential
model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)

# Method 2: Custom Module
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(1)
```

### Defining Loss Function
```python
# Binary Cross Entropy Loss
criterion = nn.BCELoss()
```

### Complete Training Code
```python
import torch
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_hat = model(X)
    
    # Calculate loss
    loss = criterion(y_hat, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

# Make predictions
predictions = (model(X) > 0.5).float()
print(f'Predictions: {predictions.tolist()}')
```

### Training Process Summary
1. **Load data**: Get X and y from dataset
2. **Forward pass**: Pass X to model → get ŷ
3. **Calculate loss**: Compare ŷ with y using criterion
4. **Zero gradients**: Clear previous gradients
5. **Backward pass**: Compute gradients
6. **Update parameters**: Use optimizer.step()
7. **Repeat** until convergence

### Converting Probabilities to Classes
```python
# Predictions are probabilities (0 to 1)
y_prob = model(X)

# Convert to binary classes using threshold
y_pred = (y_prob > 0.5).int()
```

---

## Conclusion

Logistic Regression is a fundamental classification algorithm that:
- Uses linear function z = w·x + b
- Applies sigmoid function to get probability
- Uses threshold (0.5) for final prediction
- Can be extended to multi-class classification
- Is the basis for neural network classification
