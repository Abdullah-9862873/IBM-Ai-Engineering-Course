# Module 2: Linear Regression

## Video: Simple Linear Regression 1-D

### What is Linear Regression?
- A method to understand the relationship between two variables:
  - **Predictor (independent) variable x** - also called feature
  - **Target (dependent) variable y**
- Goal: Come up with a linear relationship between the variables
- In 1D, it's just the equation of a line: **y = wx + b**
  - **w** = slope (weight in PyTorch)
  - **b** = bias

### Prediction Step (Forward Step)
- The line maps the dependent variable x to the estimated value of y
- The hat (^) on y indicates it's an estimate: ŷ
- In PyTorch, this is called the **forward step**

### Steps in Linear Regression
1. **Training**: Use training points to fit/train the model and get parameters (w, b)
2. **Prediction**: Use parameters in the model to predict any values of y given x

---

## Video: Linear Regression in PyTorch (Manual)

### Creating Parameters Manually

```python
import torch

# Create tensors for Bias and Weight
# requires_grad=True allows gradient calculation
bias = torch.tensor(-1.0, requires_grad=True)
weight = torch.tensor(2.0, requires_grad=True)

# Forward function for prediction
def forward(x):
    return weight * x + bias

# Make prediction on single value
x = torch.tensor([2.0])
y_hat = forward(x)
```

### Multiple Predictions

```python
# Multiple inputs (2 samples, 1 feature)
x = torch.tensor([[1.0], [2.0]])
y_hat = forward(x)  # Applies linear function to each row
```

---

## Video: Using the Linear Class

### Built-in Linear Class

```python
import torch.nn as nn

# Create linear regression model
# in_features: size of each input sample (number of columns)
# out_features: size of each output sample
model = nn.Linear(in_features=1, out_features=1)

# Get model parameters
print(list(model.parameters()))
# Output: [Parameter containing: tensor([...], requires_grad=True), 
#          Parameter containing: tensor([...], requires_grad=True)]
# First is weight, second is bias
```

### Making Predictions

```python
# Single prediction
x = torch.tensor([[2.0]])
y_hat = model(x)

# Multiple predictions
x = torch.tensor([[1.0], [2.0], [3.0]])
y_hat = model(x)
```

---

## Video: Custom Modules using nn.Module

### Why Custom Modules?
- Allows wrapping multiple objects to make more complex modules
- PyTorch convention for building models

### Creating Custom Linear Regression Module

```python
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, in_features, out_features):
        # Initialize parent class
        super(LR, self).__init__()
        
        # Create linear layer
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # Make prediction
        return self.linear(x)
```

### Using Custom Module

```python
# Create model object
model = LR(in_features=1, out_features=1)

# Get parameters
print(list(model.parameters()))

# Make prediction
x = torch.tensor([[1.0], [2.0]])
y_hat = model(x)
```

### Key Methods
| Method | Description |
|--------|-------------|
| `__init__` | Initialize the model |
| `forward` | Define the forward pass |
| `parameters` | Get model parameters |
| `state_dict` | Get model parameters as dictionary |

### Accessing Parameters

```python
# Using state_dict
print(model.state_dict())
# {'linear.weight': tensor([...]), 'linear.bias': tensor([...])}

# Using parameters()
for param in model.parameters():
    print(param)
```

---

## Video: Training Linear Regression - Dataset and Cost Function

### What is a Dataset?
- A dataset consists of examples with x (predictor/independent variable) and y (target/dependent variable) values
- For n data points: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- Each pair of x-y coordinates are linked together with the same subscript

### Simple Linear Regression Examples
| Example | x (Feature) | y (Target) |
|---------|-------------|------------|
| Housing | House size | House price |
| Stocks | Interest rate | Stock price |
| Cars | Horsepower | Fuel economy |

### The Noise Assumption
- Even if the linear assumption is correct, there is always some error
- A small random value (noise) is added to the point on the line
- For linear regression, the noise is **Gaussian** (normally distributed)
- Gaussian noise:
  - Most values added are near zero (small positive or negative)
  - Sometimes large values are added, but rarely
  - Standard deviation controls how much samples deviate from the line

### Finding the Best Line
- Plot points on Cartesian plane
- Find a linear function that "best" represents the points
- The best fit is determined by **minimizing a cost function**

### Cost Function (Mean Squared Error)
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2$$

- Function of slope (w) and bias (b)
- Different slope and bias values produce different MSE values
- **The line with the best fit has the smallest MSE value**

---

## Video: Training Overview

### Training Process
1. **Initialize parameters**: Start with random values for w and b
2. **Calculate predictions**: Use current parameters to make predictions
3. **Calculate loss**: Compute the cost function (MSE)
4. **Update parameters**: Adjust w and b to reduce the loss (using gradients)
5. **Repeat**: Steps 2-4 until convergence

### Key Training Concepts
- **Forward pass**: Calculate prediction using current parameters
- **Loss calculation**: Measure how far predictions are from actual values
- **Backward pass**: Calculate gradients and update parameters
- **Optimization**: Use gradient descent to find minimum

---

## Video: Loss and Cost Function

### What is Loss?
- A quantity that is **near zero when model provides a good estimate** and **large when estimate is bad**
- Measures how far the prediction is from the actual value

### Calculating Loss
- For a single sample: subtract model estimate from actual value, then square it
- **Loss = (y - ŷ)²** where ŷ is the predicted value

```python
# Example: Single sample
x = torch.tensor([-2.0])  # input
y = torch.tensor([4.0])   # actual value

# Model prediction
w = torch.tensor([1.0], requires_grad=True)
y_hat = w * x

# Calculate loss
loss = (y - y_hat) ** 2
```

### Loss Function
- Loss is a **function of the parameter(s)** you want to learn
- **Goal**: Find the parameter that **minimizes** the loss function
- Also called **criterion function** or **cost function**

### Visualizing Loss
- Loss function is typically shaped like a **concave bowl** (convex)
- Different parameter values produce different loss values
- **Minimum** of loss function corresponds to the best line fit

### Loss Function Behavior
| Parameter Value | Line Position | Loss Value |
|-----------------|--------------|------------|
| Large (e.g., 5) | Far from point | Large |
| Small (e.g., 1) | Close to point | Near minimum |
| Negative (e.g., -1) | Closer to minimum | Near minimum |
| Very negative (e.g., -5) | Far from point | Large |

### Finding the Minimum using Derivatives
- **Derivative of loss function** tells us which direction to go:
  - **Negative derivative**: Left of minimum (increase parameter)
  - **Positive derivative**: Right of minimum (decrease parameter)
  - **Zero derivative**: At the minimum
- **Setting derivative = 0** gives us the optimal parameter value
- For simple cases, we can solve algebraically
- For complex deep learning models, we use gradient descent

### Summary
1. **Loss** measures prediction error for a single sample
2. **Cost** (or Cost function) is the average loss over all samples
3. We minimize the cost function to find the best parameters
4. Derivatives help us find the direction to update parameters

---

## Video: Gradient Descent

### What is Gradient Descent?
- A method to find the **minimum of a function**
- Iteratively updates parameters by moving in the **direction opposite to the derivative**

### Gradient Descent Update Rule
$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

Where:
- $w_{new}$ = new parameter value
- $w_{old}$ = current parameter value
- $\eta$ (eta) = learning rate
- $\frac{\partial L}{\partial w}$ = derivative of loss with respect to parameter

### How Gradient Descent Works
| Position | Derivative Sign | Action |
|----------|-----------------|--------|
| Left of minimum | Negative | Add (move right) |
| Right of minimum | Positive | Subtract (move left) |
| At minimum | Zero | No change |

### Gradient Descent Algorithm
```python
# Pseudocode
w = initial_guess  # Starting parameter value
learning_rate = 0.01

for iteration in range(num_iterations):
    # Calculate derivative
    gradient = compute_gradient(w)
    
    # Update parameter
    w = w - learning_rate * gradient
```

### Example
```python
# Initial guess
w = -4.0

# Learning rate
learning_rate = 0.05

# First iteration
gradient = -112  # derivative at w = -4
w = w - learning_rate * gradient  # w = -4 - 0.05*(-112) = 1.6

# Loss is now lower
```

---

## Video: Learning Rate Problems

### Problem 1: Learning Rate Too Large
- If learning rate is too big, you might **miss the minimum**
- Loss can actually **increase** instead of decrease
- The parameter jumps across the minimum

**Example:**
```python
# Learning rate too large (e.g., 0.2)
w_new = w_old - 0.2 * gradient  # Overshoots minimum
# Loss increases instead of decreasing
```

### Problem 2: Learning Rate Too Small
- If learning rate is too small, training takes **too long**
- Parameter barely changes between iterations

**Example:**
```python
# Learning rate too small (e.g., 0.001)
w_new = w_old - 0.001 * gradient  # Very small update
# Takes many iterations to reach minimum
```

### Choosing Learning Rate
- Start with moderate learning rate (e.g., 0.01)
- Too high → oscillate/miss minimum
- Too low → slow convergence
- Use **learning rate schedulers** for fine-tuning

---

## Video: Stopping Gradient Descent

### Method 1: Fixed Number of Iterations
```python
num_epochs = 1000

for epoch in range(num_epochs):
    # Calculate loss and gradients
    # Update parameters
```
- Simple but may not find optimal value
- Risk: stopping too early or too late

### Method 2: Loss Threshold
```python
# Stop when loss is below threshold
while loss > threshold:
    # Update parameters
```

### Method 3: Early Stopping (Loss Increase)
```python
# Stop when loss starts increasing
previous_loss = float('inf')

for epoch in range(num_epochs):
    current_loss = compute_loss()
    
    if current_loss > previous_loss:
        break  # Stop if loss increases
    
    previous_loss = current_loss
```
- Records loss values
- Stops when loss starts increasing (overshooting minimum)
- Uses the parameter value from the iteration with the **lowest loss**

### Method 4: Small Parameter Change
```python
# Stop when parameter change is very small
while abs(w_new - w_old) > threshold:
    # Update parameters
```

### Summary of Stopping Criteria
| Method | Description |
|--------|-------------|
| Fixed iterations | Run for predetermined number of epochs |
| Loss threshold | Stop when loss goes below a value |
| Early stopping | Stop when loss starts increasing |
| Parameter change | Stop when parameter changes are minimal |

---

## Video: Cost Function with Multiple Samples

### Cost Function for Multiple Samples
- Instead of minimizing loss for one sample, we **minimize for multiple samples**
- **Cost = Sum of losses** for all samples
- Sometimes divided by number of samples (average loss)

$$Cost = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2$$

- **L** is used to represent the cost/loss function in PyTorch

### Cost Function Visualization
- Visualize with squares (residuals) where area = error
- Average area of squares = cost function value
- Goal: Find parameters (w, b) that minimize the total area

### Cost Function Components
- **Slope (w)**: Controls relationship between x and y
- **Bias (b)**: Controls horizontal offset

### Gradient Descent on Cost Function
- Derivative of cost function with respect to slope
- Same gradient descent rule applies

---

## Video: Batch Gradient Descent

### Batch Gradient Descent
- **Batch**: All samples in the training set
- **Batch Gradient Descent**: Use all samples to calculate loss, then find derivative

```python
# Pseudocode for batch gradient descent
for epoch in range(num_epochs):
    # Use ALL samples in training set
    for sample in training_set:
        # Calculate prediction
        y_hat = w * x + b
        
        # Calculate loss
        loss = (y - y_hat) ** 2
    
    # Calculate total/average loss (Cost)
    cost = sum(losses) / n
    
    # Calculate derivative of cost function
    # Update parameters
    w = w - learning_rate * dCost/dw
    b = b - learning_rate * dCost/db
```

### How It Works with Multiple Samples
| Sample Position | Derivative Sign | Parameter Update |
|-----------------|-----------------|-------------------|
| All on same side | Large magnitude | Large update |
| On opposite sides | Near zero | Small update |
| Mixed positions | Cancels out | Balanced update |

### Example: Batch Size = 3
```python
# Training data: 3 samples
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

# Use all 3 samples to calculate cost
# Then perform gradient descent update
```

### Summary
- **Loss**: For a single sample
- **Cost**: Average loss over all samples (or total loss)
- **Batch Gradient Descent**: Uses entire dataset for each update
- PyTorch documentation refers to it as the **loss function** (symbol L)

---

