## Video: Stochastic Gradient Descent (SGD)

### Overview
- **Stochastic Gradient Descent (SGD)**: Minimizes cost one sample at a time
- Each sample is used to update parameters individually
- One pass through all samples = **one epoch**
- Cost is approximated - may fluctuate rapidly

### How SGD Works
| Sample | Action | Effect |
|--------|--------|--------|
| Sample 1 | Update parameters using sample 1 | Decreases error for sample 1 |
| Sample 2 | Update parameters using sample 2 | Decreases error for sample 2 |
| Sample 3 | Update parameters using sample 3 | May increase error for other samples |

### SGD vs Batch Gradient Descent
| Aspect | Batch GD | Stochastic GD |
|--------|----------|---------------|
| Data used per iteration | All samples | One sample |
| Speed | Slower (all data) | Faster (one sample) |
| Convergence | Stable | Noisy (fluctuates) |
| Memory | High | Low |

### SGD Challenges
- **Outliers**: A single outlier can cause the loss to increase
- **Fluctuation**: The approximate cost fluctuates rapidly with each iteration
- May not find exact minimum but often finds good approximation

### Gradient Calculation in SGD
- Similar to batch gradient descent
- Proportional to the data distance from the line
- Update rule: $w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$

---

## Video: Mini-Batch Gradient Descent

### Overview
- **Mini-batch gradient descent**: Uses a few samples at a time (batch)
- Splits dataset into smaller samples (batches)
- Allows processing larger datasets that don't fit in memory
- Minimizes a "mini-cost function" for each iteration

### Mini-Batch Terminology
- **Batch size**: Number of samples used per iteration
- **Iteration**: One update using a batch
- **Epoch**: One complete pass through all samples

### Relationship Between Batch Size, Iterations, and Epochs
- **Number of iterations per epoch** = Number of samples / Batch size
- **Total iterations** = Number of epochs × (Samples / Batch size)

### Examples (6 samples)
| Batch Size | Iterations per Epoch |
|------------|---------------------|
| 1 (SGD) | 6 |
| 2 | 3 |
| 3 | 2 |
| 6 (Batch GD) | 1 |

### Mini-Batch Gradient Descent in PyTorch
```python
from torch.utils.data import Dataset, DataLoader

# Create dataset
x = torch.arange(-3, 3, 1).float().view(-1, 1)
y = (-3 * x + 1 + torch.randn(6, 1) * 0.5)

# Custom Dataset
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = SimpleDataset(x, y)

# Create DataLoader with batch_size
batch_size = 5
trainloader = DataLoader(dataset, batch_size=batch_size)
```

### Training with Mini-Batch
```python
# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in trainloader:
        # Forward pass
        y_hat = model(batch_x)
        
        # Calculate loss (average over batch)
        loss = criterion(y_hat, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Zero gradients
        optimizer.zero_grad()
```

### Convergence Rate
- Different batch sizes affect how quickly the cost stops decreasing
- Smaller batches: Noisier but can escape local minima
- Larger batches: More stable but may get stuck in local minima
- Common batch sizes: 32, 64, 128, 256

### Advantages of Mini-Batch
1. **Memory efficient**: Process data in chunks
2. **Faster than batch GD**: More frequent updates
3. **Smoother than SGD**: Less noisy than single sample
4. **Better generalization**: Often finds better solutions

---

## Video: SGD in PyTorch (Manual)

### Setting Up Data
```python
import torch
import matplotlib.pyplot as plt

# Create X values
x = torch.arange(-3, 3, 1).float()

# True relationship: y = -3x
w_true = torch.tensor(-3.0)

# Add noise
noise = torch.randn(6) * 0.5
y = w_true * x + noise
```

### Define Model and Loss
```python
# Forward function
def forward(x, w):
    return w * x

# Loss function (for single sample)
def criterion(y_hat, y):
    return (y_hat - y) ** 2
```

### SGD Training Loop
```python
# Initialize parameter
w = torch.tensor(-10.0, requires_grad=True)

learning_rate = 0.1
num_epochs = 4

# Training loop
for epoch in range(num_epochs):
    for i in range(len(x)):
        # Get single sample
        x_single = x[i]
        y_single = y[i]
        
        # Forward pass
        y_hat = forward(x_single, w)
        
        # Calculate loss
        loss = criterion(y_hat, y_single)
        
        # Backward pass
        loss.backward()
        
        # Update parameter
        w.data = w.data - learning_rate * w.grad
        
        # Zero gradients
        w.grad.zero_()
    
    print(f"Epoch {epoch+1}: w={w.item():.4f}")
```

### Visualizing SGD
- Parameter values move towards minimum across epochs
- Each sample causes parameter update
- Line changes with each iteration

---

## Video: SGD with DataLoader

### Custom Dataset
```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(x)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

### Using DataLoader
```python
from torch.utils.data import DataLoader

# Create dataset
x = torch.arange(-3, 3, 1).float().view(-1, 1)
y = (-3 * x + 1 + torch.randn(6, 1) * 0.5)
dataset = SimpleDataset(x, y)

# Create DataLoader
trainloader = DataLoader(dataset, batch_size=1)

# Training with DataLoader
for epoch in range(num_epochs):
    for batch_x, batch_y in trainloader:
        # Forward pass
        y_hat = model(batch_x)
        
        # Calculate loss
        loss = criterion(y_hat, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Zero gradients
        optimizer.zero_grad()
```

### DataLoader Features
- **batch_size**: Number of samples per iteration
- **shuffle**: Randomly shuffle data each epoch
- **num_workers**: Parallel data loading

### Benefits of DataLoader
- Automatic batching
- Efficient data iteration
- Memory efficient
- Supports shuffling and parallel loading

---

## Video: PyTorch Optimizer

### Overview
- **Optimizer**: Standard way to perform gradient descent in PyTorch
- Holds current state and updates parameters based on computed gradients
- Replaces manual parameter update code

### Steps in PyTorch Training
1. Create dataset object
2. Create custom module (subclass of nn.Module)
3. Create criterion/loss function
4. Create DataLoader
5. Create model
6. Create optimizer
7. Training loop

### Creating the Optimizer
```python
import torch.optim as optim

# Create optimizer (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### Optimizer Options
- **lr**: Learning rate (required)
- **momentum**: For momentum-based updates
- **weight_decay**: For L2 regularization
- **And others specific to each optimizer**

### State Dictionary
```python
# Access optimizer state
print(optimizer.state_dict())
```

### Training Loop with Optimizer
```python
# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in trainloader:
        # Forward pass: make prediction
        y_hat = model(batch_x)
        
        # Calculate loss
        loss = criterion(y_hat, batch_y)
        
        # Zero gradients (important!)
        optimizer.zero_grad()
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
```

### Key Differences from Manual Training
| Manual | Optimizer |
|--------|-----------|
| `w.data = w.data - lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

### How Optimizer Works
1. **Initial State**: Optimizer holds model parameters
2. **Forward Pass**: Model produces predictions
3. **Loss Calculation**: Compare predictions with actual values
4. **Zero Gradients**: Clear previous gradients
5. **Backward Pass**: Compute new gradients
6. **Step**: Update parameters based on gradients

### Visual Representation
```
[Data] → [Model] → [Predictions] → [Loss] → [backward()] → [Gradients]
                                                            ↓
[Model Parameters] ← [optimizer.step()] ← [optimizer]
```

### Common Optimizers
```python
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam (Adaptive Learning Rate)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

### Most Training in PyTorch Follows This Pattern
- Standard methodology used across all deep learning models
- Becomes more important as models get more complex
- Cleaner, more maintainable code

---

## Video: Training, Validation, and Test Data

### Overfitting
- **Overfitting**: Model fits training data well but performs poorly on new data
- Usually occurs with complex models that memorize training data
- Model doesn't generalize to data outside training set

### Types of Data
| Type | Purpose |
|------|---------|
| **Training Data** | Used to train the model and get parameters (slope, bias) |
| **Validation Data** | Used to determine hyperparameters (learning rate, batch size) |
| **Test Data** | Shows how model performs in the real world |

### Splitting the Dataset
```python
# Typical split ratios
training_data = 80% of dataset
validation_data = 10% of dataset
test_data = 10% of dataset
```

### Parameters vs Hyperparameters
- **Parameters**: Learned via training (slope, bias)
  - Obtained through gradient descent
- **Hyperparameters**: Set before training
  - Examples: learning rate, batch size, number of epochs

### Using Validation Data
```python
# Try different learning rates
learning_rates = [0.01, 0.1, 1.0]

for lr in learning_rates:
    # Train model with this learning rate
    model = train_model(training_data, lr)
    
    # Evaluate on validation data
    val_loss = evaluate(model, validation_data)
    
    # Select best model
    if val_loss < best_val_loss:
        best_model = model
        best_lr = lr
```

### Validation Cost Formula
$$Validation Cost = \frac{1}{N_v} \sum (y_{pred} - y_{actual})^2$$

### Example: Selecting Best Model
```python
# Train with different hyperparameters
model1 = train(training_data, lr=0.01)
model2 = train(training_data, lr=0.1)

# Evaluate on validation data
val_loss1 = criterion(model1(validation_x), validation_y)
val_loss2 = criterion(model2(validation_x), validation_y)

# Select best model
best_model = model1 if val_loss1 < val_loss2 else model2
```

### Test Data
- Used to evaluate final model performance
- Should only be used once at the end of training
- Represents real-world data the model will encounter

### Important Insights
- Training loss is not always the best indicator of model quality
- Model that minimizes training loss may not minimize validation/test loss
- Validation data helps select hyperparameters
- Test data shows true generalization performance

---

## Video: Train, Validate, and Save Model

### Overview
- One of many ways to train, validate, and save models
- Uses both training and validation data
- Selects best model based on validation loss
- Can save model for later use

### Creating Data with Outliers
```python
class DatasetWithOutliers(Dataset):
    def __init__(self, n_points=6, outlier=False):
        self.x = torch.arange(-3, 3, 1).float().view(-1, 1)
        self.y = -3 * self.x + 1
        
        if outlier:
            # Add outliers to training data
            self.x = torch.cat([self.x, torch.tensor([[-3], [2]]).float()], 0)
            self.y = torch.cat([self.y, torch.tensor([[15], [10]]).float()], 0)
        
        self.len = len(self.x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

### Training Loop with Validation
```python
# Import required modules
from torch import nn

# Create datasets
train_data = DatasetWithOutliers(outlier=True)
val_data = DatasetWithOutliers(outlier=False)

# Create DataLoaders
trainloader = DataLoader(train_data, batch_size=1)
valloader = DataLoader(val_data, batch_size=1)

# Different learning rates to try
learning_rates = [0.01, 0.1, 1.0]

# Store results
train_losses = []
val_losses = []
models = []

# Train for each learning rate
for lr in learning_rates:
    # Create model and optimizer
    model = LinearRegressionModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(10):
        for x, y in trainloader:
            # Forward pass
            y_hat = model(x)
            
            # Calculate loss
            loss = criterion(y_hat, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Calculate training loss
    train_loss = 0
    for x, y in trainloader:
        y_hat = model(x)
        train_loss += criterion(y_hat, y).item()
    train_losses.append(train_loss)
    
    # Calculate validation loss
    val_loss = 0
    for x, y in valloader:
        y_hat = model(x)
        val_loss += criterion(y_hat, y).item()
    val_losses.append(val_loss)
    
    # Store model
    models.append(model)
```

### Selecting Best Model
```python
# Find best learning rate (lowest validation loss)
best_idx = val_losses.index(min(val_losses))
best_model = models[best_idx]
best_lr = learning_rates[best_idx]

print(f"Best learning rate: {best_lr}")
print(f"Best validation loss: {val_losses[best_idx]}")
```

### Saving the Model
```python
# Save the model
torch.save(best_model.state_dict(), 'best_model.pth')

# Load the model
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load('best_model.pth'))

# Make predictions with loaded model
predictions = loaded_model(test_x)
```

### Complete Workflow
```python
# Step 1: Prepare data
train_data = DatasetWithOutliers(outlier=True)
val_data = DatasetWithOutliers(outlier=False)

# Step 2: Create model
model = LinearRegressionModel()

# Step 3: Train with different hyperparameters
learning_rates = [0.01, 0.1, 1.0]
best_model = None
best_val_loss = float('inf')

for lr in learning_rates:
    # Train
    model = train_model(model, train_data, lr)
    
    # Validate
    val_loss = validate(model, val_data)
    
    # Check if best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_lr = lr

# Step 4: Save model
torch.save(best_model.state_dict(), 'model.pth')

# Step 5: Test with test data
test_data = DatasetWithOutliers()
test_loss = test(best_model, test_data)
print(f"Test Loss: {test_loss}")
```

### Visualization
- Plot training loss and validation loss for each learning rate
- Select the learning rate with lowest validation loss
- The optimal model should fit both training and validation data

---

## Conclusion

Applying these best practices can improve the training process and performance of linear regression models in PyTorch. By carefully managing learning rates, standardizing data, using validation, and monitoring the training process, you'll set a strong foundation for building more advanced machine learning models.
