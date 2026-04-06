# Module 4: Multiple Input Output Linear Regression

## Video: Training Procedure for Multiple Linear Regression

### Overview
- **Multiple Linear Regression**: Linear regression with multiple input features
- Each input sample has multiple features (d dimensions)
- Goal: Find relationship between multiple features and target

### Mathematical Representation
- For input x with d dimensions: x = [x₁, x₂, ..., x_d]
- Weights: w = [w₁, w₂, ..., w_d]
- Bias: b
- **Model**: y = w₁x₁ + w₂x₂ + ... + w_d x_d + b
- Or in vector form: y = w · x + b

### Cost Function
$$Cost = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b))^2$$

- If x has 2 dimensions → 3 parameters (2 weights + 1 bias)
- If x has 3 dimensions → 4 parameters (3 weights + 1 bias)
- Generalizes to d dimensions → d + 1 parameters

### Gradient Descent for Multiple Regression
- **Gradient with respect to weights**: ∂Cost/∂w
- **Gradient with respect to bias**: ∂Cost/∂b
- **Update rule for weights**:
$$w_{new} = w_{old} - \eta \cdot \frac{\partial Cost}{\partial w}$$
- **Update rule for bias**:
$$b_{new} = b_{old} - \eta \cdot \frac{\partial Cost}{\partial b}$$

### Training in PyTorch
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for 2D input
class Data2D(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.float32)
        self.y = torch.tensor([[3], [4], [4], [5]], dtype=torch.float32)
        self.len = len(self.x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

### Creating Model
```python
# Create model with 2 input features and 1 output
model = nn.Linear(2, 1)

# Create dataset and dataloader
dataset = Data2D()
trainloader = DataLoader(dataset, batch_size=2)

# Create criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

### Training Loop
```python
# Training
num_epochs = 100

for epoch in range(num_epochs):
    for x, y in trainloader:
        # Forward pass: make prediction
        y_hat = model(x)
        
        # Calculate loss
        loss = criterion(y_hat, y)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
```

### Making Multiple Predictions
```python
# Create new data for prediction
new_x = torch.tensor([[1, 1.5], [2, 2.5], [3, 3.5]], dtype=torch.float32)

# Make predictions
predictions = model(new_x)
print(predictions)
```

### Model Parameters
```python
# Access model parameters
print(model.weight)  # Shape: (1, 2) for 2 input features
print(model.bias)    # Shape: (1,) 

# Get values
print(f'Weights: {model.weight.data}')
print(f'Bias: {model.bias.data}')
```

### Visualization
- In 2D input case, model represents a **plane** (not a line)
- Training data points in 3D space
- Goal: Plane that best fits all data points

### Training Progress
- **Before training**: Plane not aligned with data points
- **After training**: Plane better tracks data points

### Summary of Steps
1. **Create Dataset**: Define custom Dataset class or use tensors
2. **Create Model**: `nn.Linear(in_features, out_features)`
3. **Create DataLoader**: For batching data
4. **Create Optimizer**: `optim.SGD(model.parameters(), lr=...)`
5. **Training Loop**: Forward → Loss → Zero Grad → Backward → Step
6. **Make Predictions**: Use trained model on new data

### Key Differences from Simple Linear Regression
| Aspect | Simple LR | Multiple LR |
|--------|-----------|-------------|
| Input dimensions | 1 | Multiple (d) |
| Weight shape | Single value | Vector (d) |
| Model representation | Line | Plane (d=2), Hyperplane (d>2) |

---

## Video: Multiple Linear Regression in Multiple Dimensions

### Overview
- **Multiple Linear Regression**: Uses multiple predictor variables
- For 4 predictor variables: y = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b
- Weights and bias are parameters obtained via training

### Representation
- Input: x = [x₁, x₂, ..., x_d] (1×D tensor or vector)
- Weights: w = [w₁, w₂, ..., w_d] (D×1 tensor or vector)
- Bias: b (scalar)
- **Model**: y = w · x + b (dot product + bias)

### Shape Requirements
- Number of columns of x MUST equal number of rows of w
- After dot product, add bias b to get prediction

### Multiple Samples
- For n samples with d features: X is n×d matrix
- Each row represents one sample
- Predictions: y_hat = X · w + b (vector of n predictions)

```python
# Multiple samples example
X = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.float32)  # 4 samples, 2 features
w = torch.tensor([1.0, 2.0], dtype=torch.float32)  # 2 weights
b = torch.tensor(1.0)  # bias

# Predictions for all samples
y_hat = torch.mm(X, w.view(-1, 1)) + b  # Dot product + bias
```

---

## Video: The Linear Class

### Using nn.Linear
```python
import torch.nn as nn

# Create linear regression model
# in_features: number of input features (columns)
# out_features: number of output features
model = nn.Linear(in_features=2, out_features=1)

# Get model parameters
print(list(model.parameters()))
# Output: [Parameter containing: tensor([[...]], requires_grad=True), 
#          Parameter containing: tensor([...], requires_grad=True)]
# First is weight (shape: 1×2), second is bias (shape: 1,)
```

### Making Predictions
```python
# Single sample (1 row, 2 columns)
x_single = torch.tensor([[1, 2]], dtype=torch.float32)
y_hat = model(x_single)  # Output: tensor([[...]])

# Multiple samples (3 rows, 2 columns)
x_multiple = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32)
y_hat = model(x_multiple)  # Output: 3×1 tensor
```

### Visual Representation
```
Input x (1×2) → Linear Layer → Output y_hat (1×1)
     [x₁, x₂]    [w₁, w₂]      y = w₁x₁ + w₂x₂ + b
                  [b]
```

---

## Video: Custom Modules for Multiple Regression

### Creating Custom Module
```python
class MultipleLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Create model with 2 input features and 1 output
model = MultipleLinearRegression(2, 1)
```

### Using Custom Module
```python
# Single sample prediction
x = torch.tensor([[1, 2]], dtype=torch.float32)
y_hat = model(x)

# Multiple samples prediction
x = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32)
y_hat = model(x)

# Access parameters
for param in model.parameters():
    print(param)
```

### Training the Model
```python
# Define model, criterion, optimizer
model = MultipleLinearRegression(2, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training data
X = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.float32)
y = torch.tensor([[3], [4], [4], [5]], dtype=torch.float32)

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
```

---

## Video: Multiple Outputs (Extension)

### Linear Regression with Multiple Outputs
```python
# Model with multiple outputs
model = nn.Linear(3, 2)  # 3 input features, 2 outputs

# Input (1 sample, 3 features)
x = torch.tensor([[1, 2, 3]], dtype=torch.float32)

# Output (1 sample, 2 outputs)
y_hat = model(x)  # Shape: (1, 2)
```

### Custom Module for Multiple Outputs
```python
class MultipleOutputRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultipleOutputRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Create model
model = MultipleOutputRegression(3, 2)
```

### Summary
- Multiple Linear Regression extends simple regression to multiple features
- `nn.Linear(in_features, out_features)` creates linear transformation
- Input shape: (batch_size, in_features)
- Output shape: (batch_size, out_features)
- Same training procedure as simple linear regression

---

## Video: Linear Regression with Multiple Outputs

### Overview
- Multiple Outputs: M linear functions with D inputs
- Each output has its own set of weights and bias
- Can be expressed as matrix operation: y = X · W + b

### Mathematical Representation
- **Input**: X (n×D matrix, n samples, D features)
- **Weights**: W (D×M matrix, D features, M outputs)
- **Bias**: b (M vector, M bias terms)
- **Output**: Y = X·W + b (n×M matrix)

### Visual Representation
```
Input x (1×D) → Linear Layer → Output y (1×M)
     [x₁, x₂, ..., x_D]  W (D×M)   [y₁, y₂, ..., y_M]
                          b (M)    
```

### Creating Model with Multiple Outputs
```python
import torch.nn as nn

# Model with 2 inputs and 3 outputs
model = nn.Linear(2, 3)

# Get parameters
print(model.weight.shape)  # (3, 2) - 3 outputs, 2 inputs
print(model.bias.shape)     # (3,) - 3 bias terms
```

### Making Predictions
```python
# Single sample with 2 features
x = torch.tensor([[1, 2]], dtype=torch.float32)
y_hat = model(x)  # Shape: (1, 3)

# Multiple samples
x = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32)
y_hat = model(x)  # Shape: (3, 3)
```

### Custom Module for Multiple Outputs
```python
class MultipleOutputRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultipleOutputRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Create model with 2 inputs and 3 outputs
model = MultipleOutputRegression(2, 3)
```

### Multiple Outputs with Single Sample
```python
# Input: 1 sample with 2 features
x = torch.tensor([[1, 2]], dtype=torch.float32)

# Weights matrix: 2 rows × 3 columns
# Each column represents weights for one output

# First output: w₁x₁ + w₂x₂ + b₁
# Second output: w₃x₁ + w₄x₂ + b₂
# Third output: w₅x₁ + w₆x₂ + b₃

y_hat = model(x)  # Shape: (1, 3)
```

### Multiple Outputs with Multiple Samples
```python
# Input: 4 samples, each with 2 features
X = torch.tensor([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
], dtype=torch.float32)  # Shape: (4, 2)

# Output: 4 samples, each with 3 outputs
y_hat = model(X)  # Shape: (4, 3)

# First column of y_hat = predictions for output 1
# Second column of y_hat = predictions for output 2
# Third column of y_hat = predictions for output 3
```

### Training Model with Multiple Outputs
```python
import torch.optim as optim

# Create model
model = MultipleOutputRegression(2, 3)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training data (4 samples, 2 features, 3 outputs)
X = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.float32)
Y = torch.tensor([
    [3, 1, 2],
    [4, 2, 3],
    [4, 2, 3],
    [5, 3, 4]
], dtype=torch.float32)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    Y_hat = model(X)
    
    # Calculate loss
    loss = criterion(Y_hat, Y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
```

### Summary of Shapes
| Variable | Shape |
|----------|-------|
| Input X | (batch_size, input_dim) |
| Weights W | (input_dim, output_dim) |
| Bias b | (output_dim,) |
| Output Y | (batch_size, output_dim) |

---

## Video: Training Multiple Output Model

### Complete Training Example
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class MultiOutputDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([
            [1, 1], [1, 2], [2, 1], [2, 2],
            [1, 3], [3, 1], [3, 2], [2, 3]
        ], dtype=torch.float32)
        self.y = torch.tensor([
            [3, 1, 2], [4, 2, 3], [4, 2, 3], [5, 3, 4],
            [5, 2, 3], [5, 2, 3], [6, 3, 4], [6, 4, 5]
        ], dtype=torch.float32)
        self.len = len(self.x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create dataset and dataloader
dataset = MultiOutputDataset()
trainloader = DataLoader(dataset, batch_size=2)

# Create model
model = nn.Linear(2, 3)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for x, y in trainloader:
        # Forward pass
        y_hat = model(x)
        
        # Calculate loss
        loss = criterion(y_hat, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
```

---

## Video: Training Linear Regression with Multiple Outputs

### Overview
- **Target y and prediction y_hat are vectors** (not scalars)
- Cost function: Sum of squared distance between prediction and target
- Weights W is a matrix, bias terms are vectors
- Update rule is similar but performs vector/matrix operations

### Cost Function
$$Cost = \sum_{i=1}^{n} ||y_i - \hat{y}_i||^2$$

- For multiple outputs, we compute MSE across all outputs
- PyTorch's `MSELoss` handles this automatically

### Training Procedure
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Custom Dataset with 2 output targets
class MultiOutputDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([
            [1, 1], [1, 2], [2, 1], [2, 2]
        ], dtype=torch.float32)
        self.y = torch.tensor([
            [3, 1], [4, 2], [4, 2], [5, 3]
        ], dtype=torch.float32)
        self.len = len(self.x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

### Complete Training Code
```python
# Create dataset object
dataset = MultiOutputDataset()

# Create criterion (cost function)
criterion = nn.MSELoss()

# Create train loader with batch size
trainloader = DataLoader(dataset, batch_size=1)

# Create model: 2 input features, 2 outputs
model = nn.Linear(2, 2)

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for x, y in trainloader:
        # Make prediction
        y_hat = model(x)
        
        # Calculate loss
        loss = criterion(y_hat, y)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Differentiate loss with respect to parameters
        loss.backward()
        
        # Update parameters (performs vector operations)
        optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
```

### Key Points
1. **Dataset**: Returns 2 target values per sample
2. **Model**: `nn.Linear(2, 2)` - 2 inputs → 2 outputs
3. **Loss**: Calculates error across all outputs
4. **Update**: Matrix operations update all weights simultaneously

### Training Summary
| Step | Action |
|------|--------|
| 1 | Create dataset with multiple outputs |
| 2 | Create model with multiple outputs |
| 3 | Define loss function (MSELoss) |
| 4 | Create optimizer |
| 5 | Training loop: Forward → Loss → Zero Grad → Backward → Step |
| 6 | Repeat until convergence |

---

## Conclusion

Multiple Linear Regression extends simple linear regression to handle multiple input features. The training procedure in PyTorch is similar, using:
- `nn.Linear(d, 1)` for d input features
- Gradient descent to update all weights and bias simultaneously
- Same optimization patterns learned in previous modules
