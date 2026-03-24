# Course Introduction: Fundamentals of Deep Learning and Neural Networks with Keras

## Course Overview
- Introduction to deep learning basics
- Learn about neurons and neural networks
- Explore artificial neural networks and deep learning libraries
- Focus on deep learning models

## Prerequisites
- Proficiency with Python
- Basic knowledge of machine learning
- High-school level mathematics
- Motivation to learn

## Course Topics

### 1. Neural Networks Basics
- Neurons and neural networks in the brain (biological inspiration)
- Artificial neural networks
- Forward propagation process

### 2. How Neural Networks Learn
- Gradient descent
- Backpropagation process
- Vanishing gradient problem
- Activation functions

### 3. Deep Learning Libraries
- TensorFlow
- PyTorch
- Keras

### 4. Building Models with Keras
- Regression models
- Classification models

### 5. Network Comparisons
- Shallow vs. deep neural networks

### 6. Types of Deep Neural Networks
- **Supervised:**
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)
- **Unsupervised:**
  - Transformers
  - Autoencoders

### 7. Practical Application
- Final project using Keras to build a convolutional neural network

## Expectations
- This is an introductory course for beginners
- Advanced topics will be presented in simplified versions
- No prior deep learning experience required

## Tips for Success
- Watch every video
- Review reading material
- Complete quizzes and activities
- Test skills with assessments
- Complete the final project

## Course Completion
- Receive course certificate and badge upon completing all graded assessments and final project

---

# Deep Learning Applications

## Introduction
- Deep learning is one of the hottest subjects in data science
- Numerous fascinating projects have emerged, making previously impossible tasks achievable

## Recent Applications

### 1. Color Restoration
- Converts grayscale images to colored images automatically
- Uses convolutional neural networks
- Example: Researchers in Japan built such a system

### 2. Speech Enactment (Lip Syncing)
- Synthesizes audio clips with video and syncs lip movements
- Uses recurrent neural networks
- First realistic results by University of Washington researchers
- Case study: Barack Obama
- Can also extract audio from one video and sync lip movements in another video

### 3. Automatic Handwriting Generation
- Uses recurrent neural networks (Alex Graves, University of Toronto)
- Generates messages in realistic cursive handwriting
- Multiple styles available

### 4. Other Applications
- **Automatic Machine Translation:** CNNs translate text in images
- **Adding Sounds to Silent Movies:** Matches scenes with pre-recorded sounds
- **Object Classification/Detection:** Identifies objects in images
- **Self-Driving Cars:** Autonomous navigation
- **Chatbots:** Conversational AI
- **Text-to-Image Generators:** Creates images from text descriptions

## Key Takeaways
- Neural networks form the basis of deep learning applications
- Deep learning enables almost unlimited possibilities
- The field continues to evolve rapidly

---

# Neurons and Neural Networks

## Biological Neuron

### Discovery
- First picture of a neuron drawn in 1899 by Santiago Ramon y Cajal
- Known as the father of modern neuroscience
- Neurons have big bodies in the middle with long arms branching off to connect with other neurons

### Components of a Biological Neuron

1. **Soma (Cell Body)**
   - Main body of the neuron
   - Contains the neuron's nucleus
   - Processes electrical impulses/data

2. **Dendrites**
   - Extensive network of arms sticking out of the soma
   - Receive electrical impulses from sensors or terminal buttons of adjoining neurons
   - Carry impulses to the soma

3. **Axon**
   - Long arm sticking out of the soma in the opposite direction
   - Carries processed information from the soma to the terminal buttons

4. **Terminal Buttons (Synapses)**
   - Whiskers at the end of the axon
   - Output becomes input to thousands of other neurons

### How Biological Neurons Work
1. Dendrites receive electrical impulses (information/data) from other neurons
2. Impulses are carried to the soma
3. In the nucleus, impulses are processed by combining them
4. Processed information is passed to the axon
5. Axon carries information to terminal buttons
6. Output becomes input to other neurons

### Learning in the Brain
- Occurs by repeatedly activating certain neural connections
- Reinforces those connections, making them more likely to produce desired outcomes
- Once desired outcome occurs, neural connections become strengthened

## Artificial Neuron

### Overview
- Behaves the same way as a biological neuron
- Consists of: soma, dendrites, and axon

### Components
- **Soma:** Processes inputs and produces output
- **Dendrites:** Receive inputs from other neurons
- **Axon:** Passes output to other neurons (can branch to connect to many neurons)

### Learning Process
- Resembles how learning occurs in the brain
- Strengthens connections that produce desired outputs

## Key Takeaways
- Deep learning algorithms are inspired by biological neurons
- Artificial neurons mimic the structure and function of biological neurons
- Learning occurs by reinforcing neural connections that produce desired outcomes

---

# Artificial Neural Networks

## Perceptron (Artificial Neuron)

### Overview
- An artificial neuron is also referred to as a **perceptron**
- Calculates the weighted sum of inputs and compares with a threshold
- If sum > threshold, output = 1; otherwise output = 0
- Fundamental building block of artificial neural networks

### Mathematical Formulation
- Inputs: x1, x2, ... (can be integer or float)
- Weights: w1, w2, ... (regulate the flow of data)
- Bias: b (constant added to the sum)
- **z** = linear combination = (x1 × w1) + (x2 × w2) + ... + bias
- **a** = output of the neuron (after activation function)

## Neural Network Layers

### 1. Input Layer
- First layer that feeds input into the network

### 2. Hidden Layers
- Any sets of nodes between input and output layers

### 3. Output Layer
- Set of nodes that provide the network's output

## Activation Functions

### Purpose
- Maps weighted sum to nonlinear space
- Enables neural network to learn complex tasks
- Decides whether a neuron should be activated (relevant or ignored)

### Sigmoid Function
- If weighted sum is very large positive → output close to 1
- If weighted sum is very large negative → output close to 0

### Why Activation Functions Matter
- Without activation function: neural network = linear regression model
- With activation function: nonlinear transformation enables complex tasks (image classification, language translation)

## Forward Propagation

### Definition
- Method by which data flows through neuron's layers from input to output
- Process:
  1. Inputs pass through connections (adjusted by weights)
  2. Calculate weighted sum (z) + bias
  3. Apply activation function
  4. Output becomes input for next layer

### Example (1 neuron, 1 input)
- Input: x1 = 0.1
- Weight: w1 = 0.15
- Bias: b1 = 0.4
- z = (0.1 × 0.15) + 0.4 = 0.415
- Output a = σ(z) = sigmoid(0.415) = 0.6023

### Example (2 neurons)
- First neuron output (a1) becomes input to second neuron
- Second neuron computes: z2 = a1 × w2 + b2
- Output a2 = σ(z2) = 0.7153

### Key Point
- No matter how complicated the network, the process is the same

## Three Main Topics in Neural Networks
1. Forward propagation
2. Backpropagation
3. Activation functions

## Key Takeaways
- First layer = input layer
- Last layer = output layer
- Layers in between = hidden layers
- Forward propagation: data flows from input to output
- Given weights and biases, you can compute output for any input
