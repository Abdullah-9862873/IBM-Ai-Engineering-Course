# Module 3: Shallow Neural Network

## Overview

Neural networks are functions that can be used to approximate most functions using a set of parameters. This module covers Neural Networks with One Hidden Layer featuring Two Neurons.

## Topics Covered

1. Introduction to Neural Networks with One Hidden Layer
2. Creating a Neural Network with One Hidden Layer using nn.Module
3. Creating a Neural Network with One Hidden Layer using nn.Sequential
4. Training the Neural Network model

---

## 1. Introduction to Neural Networks with One Hidden Layer

### Classification Problem as Decision Function

- Classification problems can be represented as a decision function
- Example: When y = 1, value is mapped to one on the vertical axis; when y = 0, value is mapped to zero
- This can be visualized as a box function - any values of x in one region is one, any in another region is mapped to zero

### Building Neural Networks with Linear Classifiers

- A single straight line cannot always separate the data
- Logistic regression can approximate the box function but with limitations

### Activation Functions

- Represented as a node taking input z from a linear function and producing output A
- The function A is a function of z and x
- Called the "activation function" in neural networks
- The output of A is called the activation

### Combining Multiple Sigmoid Functions

- Consider two sigmoid functions: A₁ and A₂
- Subtract the second sigmoid function from the first: A₁ - A₂
- Result approximates the decision function

### Two-Layer Neural Network Architecture

- First, apply two linear functions to x, get two outputs
- Apply sigmoid to each linear function output
- Apply a second linear function to the outputs of the sigmoid
- Apply another function to scale the output
- Apply a threshold (values < 0.5 to 0, > 0.5 to 1)

### Neural Network Diagram

```
Input x → Linear → Sigmoid (A₁) ─┐
       → Linear → Sigmoid (A₂) ─┼→ Linear → Sigmoid → Output
```

### Terminology

- **Hidden Layer**: First layer with linear + activation functions
- **Output Layer**: Second layer producing final output
- **Artificial Neuron**: Each linear function and activation combination
- In this case: Hidden layer has 2 artificial neurons, Output layer has 1

---

## 2. Creating a Neural Network with One Hidden Layer using nn.Module

### Import Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Define Neural Network Class

```python
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x
```

### Parameters

- `D_in`: Size of the input to the network (1 in this example)
- `H`: Number of neurons in the hidden layer (2 in this example)
- `D_out`: Size of the output layer (1 in this example)

### Matrix Interpretation

- First linear: Matrix W with 1 row and 2 columns (input size × neurons)
- Bias has 2 columns representing number of neurons
- Sigmoid applied to each element of linear output
- Second linear: Matrix with 2 rows (input size) and 1 column (output neurons)
- Final sigmoid applied to the output

### For Multiple Samples

- Operation applied to every row in x
- Each row in z represents a sample
- Each column is the output of the artificial neuron to that particular sample

### Making Predictions

```python
model = Net(1, 2, 1)
y_pred = model(x)
```

---

## 3. Creating a Neural Network with One Hidden Layer using nn.Sequential

### Using nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Sigmoid(),
    nn.Linear(H, D_out),
    nn.Sigmoid()
)
```

```python
# Example
model = nn.Sequential(
    nn.Linear(1, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)
```

### Making Predictions

```python
y_pred = model(x)
```

---

## 4. Training the Neural Network Model

### Training Procedure

1. Create the data
2. Create a neural network model
3. Specify the loss function (BCE Loss for classification)
4. Create an optimizer
5. Train the model iteratively

### Example Training Code

```python
# Create data
X = ...
Y = ...

# Create model
model = Net(1, 2, 1)

# Specify loss function
criterion = nn.BCELoss()

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### For Discrete Predictions

Apply a threshold to get discrete values:

```python
predictions = (y_pred >= 0.5).float()
```

### Regression Note

- For regression, remove the last sigmoid function
- Change the loss function accordingly (e.g., MSE Loss)

---

## 5. Adding More Neurons to the Hidden Layer

### Why Add More Neurons?

- Adding more neurons in the hidden layer gives the model more flexibility
- With 2 neurons, the model may not be flexible enough to classify all samples correctly
- Samples in certain regions may be misclassified regardless of shifting or scaling

### Example Problem

- With the decision function, samples between -5 and 5 are misclassified
- Shifting the function doesn't help - most samples are still misclassified
- Scaling doesn't help - samples between -10 and -5 are misclassified

### How More Neurons Help

- Adding more neurons adds more functions to approximate the desired decision boundary
- Each neuron outputs a sigmoid function multiplied by weights
- Combining multiple sigmoid functions allows approximation of complex decision boundaries

### Combining Multiple Neurons

1. First neuron output × weight → sigmoid function
2. Second neuron output × weight → add to first
3. Continue for each neuron
4. Each additional neuron adds more flexibility to approximate the desired shape
5. Final sigmoid applied to fix scaling issues

### Creating Network with More Neurons

Using nn.Module:

```python
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# Create model with 6 neurons in hidden layer
model = Net(1, 6, 1)
```

Using nn.Sequential:

```python
model = nn.Sequential(
    nn.Linear(1, 6),
    nn.Sigmoid(),
    nn.Linear(6, 1),
    nn.Sigmoid()
)
```

For more accurate predictions, use 7 neurons:

```python
model = nn.Sequential(
    nn.Linear(1, 7),
    nn.Sigmoid(),
    nn.Linear(7, 1),
    nn.Sigmoid()
)
```

### Training the Model

```python
# Create data and data loader
train_loader = ...

# Create model with more neurons
model = Net(1, 6, 1)

# Create optimizer (adam optimizer recommended)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Create loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(1000):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Results

- With more neurons (e.g., 6-7 neurons), the model can accurately classify training points
- Graphing the model shows accurate predictions for all training points

---

## 6. Neural Networks with Multidimensional Input

### Overview

- Neural networks can take multidimensional input data
- This section focuses on 2-dimensional input

### Adding More Input Dimensions

- We can add more dimensions to the input
- More input dimensions means more weights between input layer and hidden layer
- Bias terms are typically omitted in diagrams

### 2D Input Example

Consider samples in two dimensions:
- Blue points: y = 0
- Red points: y = 1

A single line cannot separate the two classes in 2D space.

### Visualizing as a Function

- Use color to represent output of function in different regions
- Any point in the red area: function outputs 1
- Any point in the blue area: function outputs 0
- Points in the "wrong" colored region are misclassified

### Number of Neurons and Classification

- With 3 neurons: function is not linear, but some points still misclassified
- With 4 neurons: better classification, but may still have some errors
- With enough neurons: samples correctly classified in their respective regions

### Higher Dimension Visualization

- Add an extra dimension to represent y and ŷ (predicted y)
- Surface has ŷ value of 0 in blue data regions
- Surface has ŷ value of 1 in red data regions

### Building Network in PyTorch

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# Create model with 2D input and 4 neurons in hidden layer
model = Net(2, 4, 1)
```

### Training with 2D Input

```python
# Create dataset and data loader
train_loader = ...

# Create model
model = Net(2, 4, 1)

# Create loss function
criterion = nn.BCELoss()

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

- As total loss decreases, accuracy improves

---

## 7. Overfitting and Underfitting

### Overfitting

**Definition**: Overfitting occurs when the model is too complex for the data.

**Cause**: Too many neurons in the hidden layer

**Symptoms**:
- Decision region is too complex
- Model memorizes training data including noise
- Poor generalization to new data

### Underfitting

**Definition**: Underfitting occurs when the model cannot capture the complexity of the data.

**Cause**: Too few neurons in the hidden layer

**Symptoms**:
- Decision region is too simple
- Cannot correctly classify some data points
- Poor performance on training data

### Adding Noise to Data

- Real-world data often has noise
- Some data points may be on the wrong side of the decision boundary
- This is normal and should be accounted for

### Solutions to Overfitting and Underfitting

1. **Use validation data**: Determine the optimum number of neurons
2. **Get more data**: More training data helps generalization
3. **Regularization**: Reduce model complexity (discussed in later sections)

### Finding the Right Balance

- Not too many neurons (avoid overfitting)
- Not too few neurons (avoid underfitting)
- Use validation data to find the optimal number

---

## 8. Multiclass Neural Networks

### Overview

- Multiclass Neural Networks classify data into multiple classes
- Number of neurons in output layer should match number of classes
- Each neuron has its own set of parameters

### Architecture

- For 3 classes: 3 neurons in output layer
- Each neuron expressed as a row in a matrix
- Each neuron has as many input parameters as neurons in previous layer

### Prediction Process

Similar to Softmax regression:

1. For an input, obtain a value in each neuron of output layer
2. Choose the class with the neuron that has the largest value
3. This is equivalent to applying Softmax

### Example

For 3 classes (colors represent neurons):
- Neuron 0: Red
- Neuron 1: Blue
- Neuron 2: Green

Input produces values for each neuron. If neuron 2 has the largest value, output is class 2.

### Matrix Representation

- Hidden layer: 4 neurons
- Output layer: 3 classes
- Matrix has 3 columns (one per class) and 4 rows (one per hidden neuron)
- 3 bias terms (one per class)

Process:
1. Get activations from hidden layer
2. Apply linear transformation
3. Select index of column with largest value as prediction

### Building Multiclass Neural Network in PyTorch

Using nn.Module:

```python
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        # No activation function in last layer for multiclass
        x = self.linear2(x)
        return x

model = Net(D_in, H, D_out)
```

Parameters:
- `D_in`: Number of input features
- `H`: Number of neurons in hidden layer
- `D_out`: Number of classes

Using nn.Sequential:

```python
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Sigmoid(),
    nn.Linear(H, D_out)
    # No activation function for multiclass
)
```

### Training with MNIST Dataset

The MNIST dataset contains handwritten digits (0-9), so 10 classes.

```python
# Create validation and training dataset
train_dataset = ...
val_dataset = ...

# Create data loaders
train_loader = ...
val_loader = ...

# Create model
model = Net(D_in, H, D_out)

# Use cross entropy loss for multiclass
criterion = nn.CrossEntropyLoss()

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate validation accuracy
    correct = 0
    for x, y in val_loader:
        y_pred = model(x)
        # Select class with highest value
        predicted = torch.argmax(y_pred, dim=1)
        if predicted == y:
            correct += 1
    accuracy = correct / len(val_dataset)
```

### Data Preparation for MNIST

- MNIST images converted to tensor with 784 elements (28×28 = 784)
- Target y represents known classes/labels (0-9)

### Making Predictions

Select output with highest value:

```python
y_pred = model(x)
predicted_class = torch.argmax(y_pred, dim=1)
```

### Note on Adding More Layers

- Can add more hidden layers for complex problems
- These networks are harder to train
- Will be covered in later sections

---

## 9. Backpropagation

### Overview

- Backpropagation is the algorithm used to compute gradients in neural networks
- It reduces the number of computations needed to calculate the gradient
- Uses the chain rule to compute derivatives efficiently

### Chain Rule

A neural network is a function of another function:

- Function z is applied to X: z = f(X)
- Function a is applied to z: a = g(z)
- Chain rule: The derivative of a with respect to X is:
  - d(a)/d(X) = d(a)/d(z) × d(z)/d(X)

### Toy Example: One Hidden Layer

Network architecture:
- Input X → Hidden Layer → Output Layer → Output ŷ

For gradient descent, we need derivatives with respect to:
- Parameters in output layer
- Parameters in hidden layer

### Derivative for Output Layer Parameters

Using chain rule:
1. Derivative of loss with respect to activation
2. Derivative of activation with respect to linear input z⁽²⁾
3. Derivative of z⁽²⁾ with respect to parameter

### Derivative for Hidden Layer Parameters

More complex because of chain rule:
1. Derivative of loss with respect to activation
2. Derivative of activation with respect to linear input z⁽²⁾
3. Derivative of z⁽²⁾ with respect to activation of hidden layer
4. Derivative of activation with respect to linear input z⁽¹⁾
5. Derivative of z⁽¹⁾ with respect to parameter

### Computational Savings with Backpropagation

- Blue terms appear in both output layer and hidden layer derivatives
- Instead of recomputing, reuse the term from output layer
- This provides computational savings

For deeper networks:
1. Compute gradient for output layer
2. Use terms from current layer to compute previous layer gradients
3. Repeat for each preceding layer

### PyTorch Implementation

PyTorch handles backpropagation automatically:

```python
# Forward pass
y_pred = model(x)

# Compute loss
loss = criterion(y_pred, y)

# Backward pass - PyTorch computes all gradients automatically
loss.backward()

# Update parameters
optimizer.step()
```

---

## 10. Vanishing Gradient Problem

### The Problem

- As networks get deeper (more layers), gradients become very small
- This makes learning very slow or stops it entirely

### Cause

The gradient for early layers is a product of many gradients:

```
gradient = gradient₁ × gradient₂ × gradient₃ × ... × gradientₙ
```

Each derivative from sigmoid activation is less than 1 for most inputs.

When any input is too large:
- The gradient will be near zero
- The product of many small numbers approaches zero

Result:
- Early layers receive almost zero gradient updates
- Parameters don't change, learning stops

### Solution

To overcome vanishing gradient:

1. **Change activation functions**: Use ReLU instead of sigmoid
2. **Learn optimization methods**: Techniques that reduce this effect
3. **Use skip connections**: Allow gradient to flow directly
4. **Batch normalization**: Normalize layer inputs

Will be covered in detail in later sections.

---

## 11. Activation Functions

### Overview

Three commonly used activation functions:
1. Sigmoid
2. Tanh (Hyperbolic Tangent)
3. ReLU (Rectified Linear Unit)

### Sigmoid Function

Mathematical formula:

```
σ(z) = 1 / (1 + e^(-z))
```

Characteristics:
- Upper bound: 1
- Lower bound: 0
- S-shaped curve (Sigmoid curve)

Derivative:

```
σ'(z) = σ(z) × (1 - σ(z))
```

Values:
- When z = -10: derivative ≈ 0
- When z = 0: derivative = 0.25
- When z = 2.5: derivative ≈ 0.07

**Drawback**: Suffers from vanishing gradient problem because:
- Derivative values are always less than 1
- Product of many small derivatives approaches zero
- Makes learning slow in deep networks

### Tanh Function (Hyperbolic Tangent)

Mathematical formula:

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Characteristics:
- Upper bound: 1
- Lower bound: -1
- Zero-centered (output can be negative)

**Advantage**: Better than sigmoid because output is zero-centered.

**Drawback**: Also suffers from vanishing gradient problem:
- Derivative less than 1 for all z ≠ 0
- Still has vanishing gradient issues

### ReLU Function (Rectified Linear Unit)

Mathematical formula:

```
ReLU(z) = max(0, z)
```

Characteristics:
- If z > 0: output = z
- If z ≤ 0: output = 0
- Upper bound: ∞
- Lower bound: 0

Derivative:
- If z > 0: derivative = 1
- If z ≤ 0: derivative = 0

**Advantage**: Partial solution to vanishing gradient:
- Derivative is 1 for positive inputs
- Gradient flows without being multiplied by small numbers

### Comparison

| Function | Range | Zero-centered | Vanishing Gradient |
|----------|-------|--------------|-------------------|
| Sigmoid | [0, 1] | No | Yes |
| Tanh | [-1, 1] | Yes | Yes |
| ReLU | [0, ∞) | No | Partial solution |

### Implementation in PyTorch

Sigmoid:

```python
class Net_Sigmoid(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_Sigmoid, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x
```

Tanh:

```python
class Net_Tanh(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_Tanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return x
```

ReLU:

```python
class Net_ReLU(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_ReLU, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return x
```

### Using nn.Sequential

Tanh:

```python
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Tanh(),
    nn.Linear(H, D_out),
    nn.Tanh()
)
```

ReLU:

```python
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
    nn.ReLU()
)
```

### Performance Comparison

- ReLU and Tanh achieve significantly better performance than Sigmoid
- ReLU is preferred: faster convergence, less computationally expensive
- Tanh is preferred when zero-centered output is beneficial
- Sigmoid is mainly used for binary classification output layer

---

## Summary

A neural network with one hidden layer:

1. Applies two linear functions to input x
2. Applies sigmoid activation to each linear output
3. Applies a second linear function to the activation outputs
4. Applies sigmoid to produce final output

The architecture consists of:
- Input Layer: 1 neuron
- Hidden Layer: 2 neurons (with sigmoid activation)
- Output Layer: 1 neuron (with sigmoid activation)

### Key Insights

- Three main activation functions: Sigmoid, Tanh, ReLU
- ReLU is preferred for deep networks (solves vanishing gradient)
- Tanh is zero-centered, better than sigmoid for hidden layers
- Sigmoid mainly used for binary output classification
- Backpropagation uses the chain rule to compute gradients efficiently
- PyTorch handles backpropagation automatically with .backward()
- The vanishing gradient problem occurs when gradients become too small in deep networks
- More neurons in the hidden layer provide more flexibility to approximate complex decision boundaries
- More input dimensions require more weights in the network
- Start with fewer neurons and increase as needed for better classification accuracy
- Balance between overfitting and underfitting is crucial for good model performance
- For multiclass: set output layer neurons equal to number of classes
- Use CrossEntropyLoss for multiclass classification
- Select class with highest output value as prediction