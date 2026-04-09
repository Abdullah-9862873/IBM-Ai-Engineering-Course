# Deep Learning with PyTorch - Course Overview

## Course Description
Deep learning, a branch of machine learning, is revolutionizing many fields including computer vision, natural language processing, and robotics. PyTorch is an open-source library used for creating deep neural networks. This course is suitable for aspiring AI engineers who want to gain advanced knowledge on deep learning using PyTorch.

## Course Progression
The course advances from fundamental machine learning concepts to more complex models and techniques:
- Softmax regression
- Shallow and deep neural networks
- Specialized architectures (Convolutional Neural Networks)

## Course Structure (6 Modules)

### Module 1: Logistic Regression & Cross-Entropy Loss
- Identify the problem with mean squared error
- Calculate maximum likelihood estimation
- Cross-entropy loss
- Train deep neural networks in PyTorch using nn.Module list

### Module 2: Softmax Regression
- Use lines to classify data
- Understand Softmax and argmax functions
- Create custom module for Softmax using nn.Module package in PyTorch

### Module 3: Neural Networks with Hidden Layer
- Create neural network with hidden layer using nn.Module and nn.Sequential
- Explore overfitting and underfitting
- Multiclass neural networks
- Backpropagation and vanishing gradient
- Implement Sigmoid, Tanh, and ReLU activation functions in PyTorch

### Module 4: Deep Neural Networks
- Dropout
- Layers and weights
- Different initialization methods in PyTorch
- Gradient descent
- Batch normalization

### Module 5: Convolutional Neural Networks (CNNs)
- Convolution with multiple input and output channels
- CNN constructor, forward step, and training in PyTorch
- GPUs, CUDA
- Residual networks and ResNet18

### Module 6: Final Project
- Implement CNN using PyTorch to classify images from MNIST dataset

## Learning Outcomes
- Implement Softmax regression and understand its application in multiclass classification problems
- Develop and train shallow neural networks with various architectures
- Explore deep neural networks including dropout, weight initialization, and batch normalization
- Gain practical experience with CNNs, exploring layers, activation functions, and more

## Prerequisites
- Basic knowledge of Python programming
- Familiarity with PyTorch
- Git and GitHub for code repositories

## Course Components
- Instructional videos
- Hands-on labs
- Practice and graded quizzes
- Reference materials (glossaries, coding cheat sheets)
- Comprehensive final project

## Related Courses in IBM AI Engineering Professional Certificate Program
- Machine Learning with Python
- Introduction to Deep Learning and Neural Networks with Keras
- Deep Learning with Keras and Tensorflow
- Introduction to Neural Networks with PyTorch
- AI Capstone Project with Deep Learning
- Generative AI and LLMs: Architecture and Data Preparation
- Gen AI Model Foundations for NLP and Language Understanding
- Generative AI Language Modeling with Transformers
- AI Engineering with Transformer-Based LLMs
- Project: Generative AI Applications with RAG and LangChain

---

# Lesson: Cross-Entropy Loss

## Overview
Cross-entropy loss is the total loss/cost function for logistic regression and other classification models.

## Problem with Mean Squared Error (MSE)

### Threshold Function Issues
- Loss calculated for misclassified samples
- Example with 3 red samples (misclassified) and 3 blue samples:
  - Yn = 0 for red, Yn = 1 for blue
  - Threshold function value = 1 for all samples
  - Loss = 3 (number of misclassified samples)
- **Problem**: Cost surface has flat regions where gradient = 0
  - Parameters get stuck and don't update
  - Results in misclassified samples

### Sigmoid Function Advantages
- Smooth curve vs. threshold function's step function
- Gradient descent works properly (no flat regions)
- Better parameter updates leading to fewer misclassifications

### MSE for Multiple Parameters
- Cost surface for two parameters (W, B) is flat in many regions
- Algorithm only converges if initialized in "good" region
- Bad initialization leads to no learning

## Maximum Likelihood Estimation (MLE)

### Likelihood Calculation
- Given classification dataset with two classes (red=0, blue=1)
- Logistic function gives probability of Y=0 or Y=1
- Example likelihood calculations:
  - Line 1 (bias=some value): likelihood = 0.445
  - Line 2 (better classifier): likelihood = 0.46
  - Line 3 (best classifier): likelihood = 0.47

### Goal
- Find parameters that maximize the likelihood function

### Log Likelihood
- Taking log doesn't affect position of maximum
- To minimize: multiply by negative sign
- Results in cross-entropy loss expression

## Cross-Entropy Loss

### Formula
Cross-entropy loss minimizes the negative log likelihood, equivalent to maximizing the likelihood.

### Implementation in PyTorch
```python
# Using built-in BCE Loss
criterion = nn.BCEWithLogitsLoss()
```

### Contour Plot
- Contours cover the entire plot surface
- Only flat at the minimum (no flat regions like MSE)

## Logistic Regression in PyTorch

### Method 1: Sequential
```python
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
```

### Method 2: Custom Module (nn.Module)
```python
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)
```

### Loss Functions
- Mean Squared Error: `nn.MSELoss()`
- Binary Cross-Entropy: `nn.BCELoss()` or `nn.BCEWithLogitsLoss()`

### Training Loop
1. Load dataset (x, y)
2. Pass input to model: `y_hat = model(x)`
3. Calculate loss: `loss = criterion(y_hat, y)`
4. Get gradients: `loss.backward()`
5. Update parameters: `optimizer.step()`

### Notes
- Model output values between 0 and 1
- Thresholding needed to get actual class values
- Run for multiple epochs (e.g., 100)
- Use SGD optimizer with learning rate (e.g., 0.01)