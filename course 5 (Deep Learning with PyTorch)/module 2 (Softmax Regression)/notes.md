# Softmax Regression

## Overview
The softmax function is used for multiclass classification problems (more than 2 classes). Like logistic regression, it uses lines (linear decision boundaries) to classify data.

## Softmax in 1D

### Example Setup
- Three classes with 1D feature vector X
- Class 0 (blue), Class 1 (red), Class 2 (green)
- Each class region separated by decision boundaries

### How Lines Classify Data
- Three different lines with weights and bias terms
- Each line produces output Zi where i corresponds to class index
- For each X value, compute outputs Z0, Z1, Z2
- Classify based on which output is largest

### Classification Logic
- For X in blue region: Z0 > Z1 and Z0 > Z2 → classify as class 0
- For X in red region: Z1 > Z0 and Z1 > Z2 → classify as class 1
- For X in green region: Z2 > Z0 and Z2 > Z1 → classify as class 2

## Argmax Function

### Definition
The argmax function returns the index corresponding to the largest value in a sequence.

### Examples
- Z = [100, 50, 30] → argmax(Z) = 0 (largest value 100 at index 0)
- Z = [2, 5, 10, 1] → argmax(Z) = 2 (largest value 10 at index 2)

### Usage in Softmax
- After computing Z values for each class, apply argmax
- The returned index is the predicted class

## Softmax with Argmax - Examples

### Example 1: X = -0.5
- Compute outputs for each line
- Result: Z0 is largest → y-hat = 0 (class 0)

### Example 2: X = 0.5
- Compute outputs for each line
- Result: Z1 is largest → y-hat = 1 (class 1)

### Example 3: X = 1.5
- Compute outputs for each line
- Result: Z2 is largest → y-hat = 2 (class 2)

## Softmax in General Case (Multidimensional)

### MNIST Dataset Example
- Handwritten digits 0-9 (10 classes)
- Each image has 784 pixels (28×28)
- Each pixel intensity ranges 0-255
- Each image flattened to vector of 784 values

### 2D Visualization
- Consider weight vectors w0, w1, w2 for 3 classes
- Each sample x is a vector
- Compute dot product: x · w_i for each class
- Classify based on largest dot product

### Why Called "Softmax"
- The raw dot products (distances) are converted to probabilities
- Similar to logistic regression's sigmoid function
- Makes outputs interpretable as probabilities

## Key Concepts

### Decision Boundaries
- Linear boundaries (lines in 1D, planes in higher dimensions)
- Each class region determined by nearest weight vector

### Classification Process
1. Input vector X
2. Compute dot products with each weight vector
3. Apply argmax to find predicted class

### Intuition
- Softmax finds points nearest to each parameter vector
- Classifies based on closest weight vector
- Similar to nearest neighbor classification

## Softmax Prediction in PyTorch

### Directed Graph Structure
- Model is identical to multi-output linear regression
- Output is z (logits) instead of y
- For classification, use argmax on z to get class index

### Custom Softmax Module Implementation
```python
class Softmax(nn.Module):
    def __init__(self, input_size, output_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        z = self.linear(x)
        return z
```
- Similar to logistic regression custom module
- Difference: output_size parameter (number of classes)
- No sigmoid function in output (unlike logistic regression)

### Using the Softmax Model

#### Single Sample
```python
model = Softmax(2, 3)  # 2D input, 3 output classes
x = torch.tensor([[...]])  # input tensor
z = model(x)  # raw logits
```
- Output z contains raw scores for each class

#### Getting Predicted Class
```python
prediction = z.max(1)[1]  # argmax along axis 1
```
- `max(1)` finds maximum along columns (axis 1)
- `[1]` gets the indices (not values)
- Returns class with highest logit score

#### Multiple Samples
- Process multiple rows in a batch
- `max(1)` returns max index for each row
- Each element in prediction corresponds to class of each sample

### Example
- Input: 2D tensor with multiple samples
- Output z: matrix of logits
- Apply max function with axis=1
- Result: predicted class indices for each sample

## Errata: Corrections and Clarifications

### Predicted Class Identification
- **Correction**: Use `torch.argmax()` to identify predicted class, not just `max()`
- `max()` returns both values and indices
- `argmax()` directly returns the index of maximum value (correct method for class prediction)
- **Correct code**: `torch.argmax(z, dim=1)`

### Terminology - Logits vs Probabilities
- **z (logits)**: Raw outputs from the model before applying Softmax
- **Softmax output**: Converts logits to normalized probabilities
- **Predictions**: Made by identifying index of highest probability

### Softmax Output Interpretation
- Similar structure to multi-output linear regression
- Unlike regression outputs, Softmax outputs are **class probabilities**, not raw scores
- Probabilities sum to 1 across all classes

### Correct Implementation in PyTorch
```python
# Forward pass produces logits
z = model(x)  # raw scores (logits)

# Apply Softmax to get probabilities
probabilities = torch.softmax(z, dim=1)

# Get predicted class (index of highest probability)
predicted_class = torch.argmax(z, dim=1)
```

### Handling Multiple Inputs
- Each input sample produces a set of logits
- Predicted class for each input is based on index of highest probability
- Use `torch.argmax(z, dim=1)` where dim=1 examines columns (classes) for each row (sample)

## Implementing Softmax in PyTorch (MNIST Example)

### Steps for Classification
1. Load data
2. Create model
3. Train model on training data
4. Test/validate on test data

### Loading Data (MNIST Dataset)
```python
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
validation_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
```
- `transforms.ToTensor()`: Converts images to PyTorch tensors
- Each sample is a tuple: (image tensor, class label)
- Image: 28×28 float tensor
- Label: Long tensor (class 0-9)

### Creating the Model
```python
model = Softmax(784, 10)  # input_dim=784 (28×28), output_dim=10 (classes 0-9)
```
- Input dimension: 28×28 = 784 (flattened image)
- Output dimension: 10 (digits 0-9)
- 10 weight vectors, each 784 dimensions
- 10 bias parameters

### Initial Parameters
- PyTorch initializes weights randomly (looks like noise)
- After training, weights resemble the digits they represent

### Loss Function
```python
criterion = nn.CrossEntropyLoss()
```
- PyTorch's CrossEntropyLoss automatically applies Softmax
- Input labels must be Long tensor (not one-hot)

### Optimizer
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### Data Loaders
```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
```

### Training Loop
```python
for epoch in range(epochs):
    # Training
    for x, y in train_loader:
        x = x.view(-1, 784)  # Flatten: (batch_size, 784)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    correct = 0
    for x_test, y_test in validation_loader:
        x_test = x_test.view(-1, 784)
        y_hat = model(x_test)
        _, predicted = torch.max(y_hat, 1)
        correct += (predicted == y_test).sum().item()
    accuracy = correct / len(validation_dataset)
```

### Key Points
- `x.view(-1, 784)`: Flattens 28×28 images to 1D vectors
- `torch.max(y_hat, 1)`: Returns max values and indices along dimension 1
- Predictions compared with actual labels to calculate accuracy
- After training, weight parameters visualize as digit shapes