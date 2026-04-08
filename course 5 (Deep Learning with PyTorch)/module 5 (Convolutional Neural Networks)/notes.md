# Module 5: Convolutional Neural Networks

## Overview

Convolutional Neural Networks (CNNs) help identify patterns in images by looking at relative positions of pixels rather than absolute positions. This module covers convolution operations, activation maps, stride, and zero padding.

## Topics Covered

1. Introduction to Convolution
2. Convolution Operation
3. Determining Activation Map Size
4. Stride Parameter
5. Zero Padding
6. Activation Functions in CNNs
7. Max Pooling
8. Convolution with Multiple Channels
9. Building a CNN
10. CNN for MNIST

---

## 1. Introduction to Convolution

### The Problem with Traditional Networks

- Images shifted slightly have intensity values in totally different locations
- Traditional neural networks treat each pixel position as independent
- CNNs solve this by looking at relative positions

### What is Convolution?

Convolution is analogous to a linear equation:
- W: kernel (learnable parameters)
- b: bias
- *: convolution operation

```
Z = W * X + b
```

The result is called an **activation map** or **feature map**.

### Image Representation

- Image converted to a matrix (M × M pixels)
- Black values represent high intensity
- White values represent low intensity
- For grayscale: 1 input channel
- For color images: 3 input channels (RGB)

---

## 2. Convolution Operation

### Creating Convolution in PyTorch

```python
import torch.nn as nn

# Create convolution layer
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

# Image tensor: (batch, channels, height, width)
X = torch.randn(1, 1, 5, 5)
```

### How Convolution Works

1. **Overlay kernel on image**: Place kernel on top-left corner
2. **Element-wise multiplication**: Multiply each kernel element with corresponding image pixel
3. **Sum**: Add all products together
4. **Shift**: Move kernel to the right (stride determines step size)
5. **Repeat**: Continue until entire image is covered

### Convolution Process

```
Image (5×5)         Kernel (3×3)         Activation Map
┌─────┬─────┬─────┐          ┌───┬───┬───┐
│ P₁₁ │ P₁₂ │ ... │    *      │   │   │   │    =
└─────┴─────┴─────┘          ├───┼───┼───┤
│ P₂₁ │ ...  │     │          │   │ z₁│   │
└─────┴─────┴─────┘          └───┴───┴───┘
```

Each position produces one output value in the activation map.

### Adding Bias

Bias is added to every element in the output:
```
Z = convolution(X, W) + b
```

---

## 3. Determining Activation Map Size

### Formula

For an M × M image with K × K kernel:

```
Output size = (M - K + 1) × (M - K + 1)
```

### Example

Image: 4 × 4 (M = 4)
Kernel: 2 × 2 (K = 2)

```
Output size = (4 - 2 + 1) × (4 - 2 + 1)
            = 3 × 3
```

### Visual Explanation

```
Step 1: Kernel at position (0,0) → z₁
Step 2: Shift right → z₂
Step 3: Shift right → z₃
Step 4: Shift down and left → z₄
Step 5: Shift right → z₅
...
```

---

## 4. Stride Parameter

### What is Stride?

Stride determines how many pixels the kernel moves each iteration.

- **Stride = 1**: Move 1 pixel per iteration (default)
- **Stride = 2**: Move 2 pixels per iteration

### Formula with Stride

For M × M image, K × K kernel, stride S:

```
Output size = floor((M - K) / S + 1) × floor((M - K) / S + 1)
```

### Example

Image: 4 × 4 (M = 4)
Kernel: 2 × 2 (K = 2)
Stride: 2

```
Output size = floor((4 - 2) / 2 + 1) �� floor((4 - 2) / 2 + 1)
            = floor(2/2 + 1) × floor(2/2 + 1)
            = floor(1 + 1) × floor(1 + 1)
            = 2 × 2
```

### Implementation in PyTorch

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
```

### Visual with Stride = 2

```
Iteration 1: Kernel at (0,0) → z₁
Iteration 2: Shift 2 right → z₂
Iteration 3: Shift 2 down, return to left → z₃
Iteration 4: Shift 2 right → z₄
```

---

## 5. Zero Padding

### Problem

When stride is too large, we may get invalid dimensions or incomplete coverage.

### Solution: Zero Padding

Add rows and columns of zeros around the image:
- Padding = 1: Add 1 row at top, 1 row at bottom, 1 column at left, 1 column at right

### Image Size with Padding

If original image is M × M and padding = P:

```
New image size = (M + 2P) × (M + 2P)
```

### Example

Original image: 4 × 4
Padding: 1

```
New image size = (4 + 2×1) × (4 + 2×1)
             = 6 × 6
```

### Implementation in PyTorch

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3, padding=1)
```

### Convolution with Padding

```
Original Image (4×4):      With Padding (6×6):
┌───┬───┬───┬───┐         ⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔
│   │   │   │   │         ⊔│ A │ B │ C │ D │⊔
├───┼───┼───┼───┤         ⊔├───┼───┼───┼───┤⊔
│   │   │   │   │         ⊔│ E │ F │ G │ H │⊔
├───┼───┼───┼───┤    →    ⊔├───┼───┼───┼───┤⊔
│   │   │   │   │         ⊔│ I │ J │ K │ L │⊔
├───┼───┼───┼───┤         ⊔├───┼───┼───┼───┤⊔
│   │   │   │   │         ⊔│   │   │   │   │⊔
└───┴───┴───┴───┘         ⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔⊔
(⊔ = zeros)
```

### Summary

| Parameter | Description | Effect |
|-----------|-------------|--------|
| Kernel Size | Size of filter | Smaller kernel = more detailed features |
| Stride | Step size | Larger stride = smaller output |
| Padding | Zero border | Allows control of output size |

---

## 6. Activation Functions in CNNs

### Overview

After convolution, apply activation function element-wise to the activation map.

### Process

1. Apply convolution: Z = W * X + b
2. Apply activation: A = activation(Z)

### ReLU in CNNs

```python
# Method 1: Direct
X = torch.randn(1, 1, 4, 4)
conv = nn.Conv2d(1, 1, kernel_size=2)
Z = conv(X)
A = torch.relu(Z)

# Method 2: Sequential
model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=2),
    nn.ReLU()
)
```

### Example

Input Image:
```
[[1, 2],
 [3, 4]]
```

Kernel: 
```
[[1, 0],
 [0, 1]]
```

Convolution result:
```
[[4, 4],
 [4, 4]]
```

After ReLU activation:
```
[[4, 4],
 [4, 4]]
```

If negative values in result:
```
[[-4, 2],
 [3, 4]]
```

After ReLU:
```
[[0, 2],
 [3, 4]]
```

### Visual

```
Input Image      Convolution      Activation Map    After ReLU
┌───┬───┐       ┌─────┬─────┐    ┌───┬───┐       ┌───┬───┐
│ A │ B │   *   │     │     │ →  │-1 │ 2 │  →   │ 0 │ 2 │
├───┼───┤       ├─────┼─────┤    ├───┼───┤       ├───┼───┤
│ C │ D │       │     │     │    │ 3 │ 4 │      │ 3 │ 4 │
└───┴───┘       └─────┴─────┘    └───┴───┘       └───┴───┘
           (kernel)
```

---

## 7. Max Pooling

### Overview

- Reduces size of activation maps
- Reduces number of parameters
- Makes network invariant to small changes/shifts
- Extracts maximum value from each region

### How Max Pooling Works

1. Divide activation map into regions (K×K)
2. Select maximum value in each region
3. Shift region and repeat

### Parameters

- Kernel size (pool_size): Size of region
- Stride: How far to shift

### Implementation in PyTorch

```python
# Max pooling layer
max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

# Apply as function
output = torch.max_pool2d(input, kernel_size=2)
```

### Visual Example

```
Input (4×4):         Max Pool (2×2, stride=2):
┌───┬───┬───┬───┐    
│ 1 │ 2 │ 3 │ 4 │    
├───┼───┼───┼───┤    ┌───┬───┐
│ 5 │ 6 │ 7 │ 8 │ →  │ 6 │ 8 │  (max of each 2×2 region)
├───┼───┼───┼───┤    ├───┼───┤
│ 9 │ 10│ 11│ 12│    │ 10│ 12│
├───┼───┼───┼───┤    
│ 13│ 14│ 15│ 16│    
└───┴───┴───┴───┘    
```

Region breakdown:
- Top-left (1,2,5,6): max = 6
- Top-right (3,4,7,8): max = 8
- Bottom-left (9,10,13,14): max = 10
- Bottom-right (11,12,15,16): max = 12

### Output Size Formula

Same as convolution:
```
Output size = floor((M - K) / S + 1) × floor((M - K) / S + 1)
```

### Benefits

1. **Reduces parameters**: Smaller output = fewer parameters in next layer
2. **Translation invariance**: Small shifts don't change output
3. **Reduced overfitting**: Less prone to overfitting

### Invariance Example

Two slightly shifted images:
```
Image 1:          Image 2 (shifted):
┌───┬───┐         ┌───┬───┐
│ 1 │ 2 │   →     │ 0 │ 1 │
├───┼───┤         ├───┼───┤
│ 3 │ 4 │         │ 2 │ 3 │
└───┴───┘         └───┴───┘
```

After Max Pooling (2×2, stride=2):
```
Output: 4        Output: 4
```

Same output despite shift!

---

## 8. Convolution with Multiple Channels

### Overview

- Multiple input channels (e.g., RGB: 3 channels)
- Multiple output channels (multiple feature maps)
- Multiple input AND output channels

### Multiple Output Channels

Each kernel produces its own activation map:
- 1 input image → 3 kernels → 3 activation maps

```python
# Create convolution with 3 output channels
conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)

# Input: (batch, 1, height, width)
X = torch.randn(1, 1, 5, 5)

# Output: (batch, 3, height, width)
Z = conv(X)
```

Each output channel has independent kernel and bias.

### Feature Detection

Different kernels detect different features:
- Kernel 1: Vertical line detector
- Kernel 2: Horizontal line detector
- Kernel 3: Edge detector

### Multiple Input Channels

For RGB images: 3 input channels (R, G, B)

Process:
1. Convolve each input channel with its own kernel
2. Add results together
3. Add bias

### Multiple Input AND Output Channels

```python
# 3 input channels → 2 output channels
conv = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
```

Total kernels = out_channels × in_channels = 2 × 3 = 6 kernels

### Summary of Dimensions

| Parameter | Description |
|-----------|-------------|
| in_channels | Number of input channels |
| out_channels | Number of output channels |
| kernel_size | Size of each kernel |
| Total kernels | out_channels × in_channels |

---

## 9. Building a Convolutional Neural Network

### Overview

CNN Architecture:
1. Convolution layers (with kernels)
2. Activation functions
3. Pooling layers
4. Fully connected output layer

### Simple CNN Example

Classify horizontal vs vertical lines (binary classification)

### Architecture Diagram

```
Input Image (grayscale)
    ↓
Conv1 (2 kernels) → Activation Map1 → ReLU → Pool1
    ↓
Conv2 (1 kernel) → Activation Map2 → ReLU → Pool2
    ↓
Flatten
    ↓
Fully Connected Layer → Output
```

### CNN Constructor in PyTorch

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolution layer
        # 1 input channel (grayscale), 2 output channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution layer
        # 2 input channels (from conv1), 1 output channel
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        # Input: 7×7 = 49 (after conv+pool)
        self.fc = nn.Linear(49, 2)  # 2 output classes
    
    def forward(self, x):
        # First conv layer
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # Second conv layer
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 49)
        
        # Fully connected
        x = self.fc(x)
        return x
```

### Calculating Output Size

After each step:
1. Input: (batch, 1, H, W)
2. After conv1 + pool: (batch, 2, H/2, W/2)
3. After conv2 + pool: (batch, 1, H/4, W/4)

For 28×28 input:
- After first conv+pool: 14×14
- After second conv+pool: 7×7
- Flattened: 49

### Training CNN

```python
# Create model
model = CNN()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Summary Flow

1. **Convolution**: Extract features using kernels
2. **Activation**: Apply ReLU (non-linearity)
3. **Pooling**: Reduce size, add invariance
4. **Flatten**: Convert 2D to 1D
5. **Fully Connected**: Make classification

---

## 10. CNN for MNIST

### Overview

MNIST: Handwritten digits (0-9), 10 classes
Image size: 16×16 (for faster training in lab)

### Architecture

```
Input Image (1 channel)
    ↓
Conv1: 1 → 16 channels, kernel=5, padding=2
    ↓ ReLU + MaxPool
    ↓
Conv2: 16 → 32 channels, kernel=5, padding=2
    ↓ ReLU + MaxPool
    ↓
Flatten
    ↓
FC: 512 → 10
```

### Constructor

```python
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        
        # First convolution layer
        # 1 input channel, 16 output channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                            kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolution layer
        # 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                            kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layer
        # Input: 32 channels × 4×4 = 512
        self.fc = nn.Linear(512, 10)
```

### Output Shape Calculation

Input: 16×16
After Conv1 + Pool: 16/2 × 16/2 = 8×8
After Conv2 + Pool: 8/2 × 8/2 = 4×4

- 32 output channels × 4×4 = 512 elements

### Forward Method

```python
def forward(self, x):
    # First conv layer
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.pool1(x)
    
    # Second conv layer
    x = self.conv2(x)
    x = torch.relu(x)
    x = self.pool2(x)
    
    # Flatten
    x = x.view(-1, 512)
    
    # Output layer
    x = self.fc(x)
    return x
```

### Parameter Summary

| Layer | Input | Output | Parameters |
|-------|-------|-------|------------|
| Conv1 | 1 | 16 | 5×5×1×16 + 16 = 416 |
| Conv2 | 16 | 32 | 5×5×16×32 + 32 = 12,832 |
| FC | 512 | 10 | 512×10 + 10 = 5,130 |

---

## 11. Using Pre-trained Models (TorchVision)

### Overview

Pre-trained models are trained by experts on large datasets.
We only need to retrain the output layer for our own classification task.

### Common Pre-trained Models

- ResNet18
- VGG16
- AlexNet
- etc.

### Advantages

- Use expert knowledge
- Less training data needed
- Better performance

### Loading Pre-trained Model

```python
import torchvision.models as models

# Load ResNet18 pretrained model
model = models.resnet18(pretrained=True)
```

### Image Preprocessing

Different models expect different normalization:

```python
from torchvision import transforms

# Standard normalization for ResNet
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Compose transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

### Modifying Output Layer

```python
# Number of input features from pretrained model
num_ftrs = model.fc.in_features

# Replace output layer for your classes (e.g., 7 classes)
model.fc = nn.Linear(num_ftrs, 7)

# Set require_grad to false for pretrained layers
for param in model.parameters():
    param.requires_grad = False
```

### Training Setup

```python
# Only optimize output layer parameters
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Loss function
criterion = nn.CrossEntropyLoss()
```

### Training Loop

```python
# Training
model.train()

# Validation/Evaluation
model.eval()
```

### Process Summary

```
Pretrained Model (ResNet18)
    ↓
(freeze pretrained layers)
    ↓
Replace Output Layer (fc)
    ↓
Train Only Output Layer
    ↓
Classify New Images
```

---

## 12. Using GPUs in PyTorch

### Overview

GPUs (Graphics Processing Units) allow much faster computation for neural networks.

### What is CUDA?

CUDA is a parallel computing platform by NVIDIA that enables using NVIDIA GPUs for computation.
- PyTorch's `torch.cuda` package enables GPU computation

### Checking GPU Availability

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda:0")
else:
    print("Using CPU")
    device = torch.device("cpu")
```

### Sending Tensors to GPU

```python
# Create tensor on CPU
x = torch.randn(3, 3)

# Send tensor to GPU
x_gpu = x.to(device)
```

### Sending Model to GPU

```python
# Create model
model = CNN()

# Send model to GPU
model = model.to(device)
```

### Training with GPU

```python
# Training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        # Send data to GPU
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Testing with GPU

```python
# Testing
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        # Send data to GPU
        x = x.to(device)
        y_pred = model(x)
```

### Note

- Labels don't need to be sent to GPU during testing (no loss calculation)
- Models and tensors must be on same device

---

## Summary

- Convolution helps identify patterns regardless of image position
- Kernel slides over image performing element-wise multiplication
- Activation map shows where patterns are detected
- Stride controls kernel movement step size
- Zero padding allows valid convolutions with large strides
- PyTorch's nn.Conv2d handles all parameters
- For color images, use 3 input channels (see labs)
- Activation functions applied element-wise to activation map
- ReLU sets negative values to zero
- Max pooling reduces size and provides translation invariance
- Multiple input/output channels increase feature detection capability
- CNN architecture: Conv → Activation → Pool → Flatten → FC
- Pre-trained models can be used for transfer learning
- GPUs can significantly speed up training using .to(device)