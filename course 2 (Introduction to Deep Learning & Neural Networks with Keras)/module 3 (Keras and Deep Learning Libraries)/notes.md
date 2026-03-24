# Deep Learning Libraries

## Overview
- Deep learning libraries provide tools to build and train neural networks
- Most popular: TensorFlow, PyTorch, Keras

## Keras

### What is Keras?
- High-level neural networks API
- Written in Python
- Can run on top of TensorFlow, Theano, or CNTK
- User-friendly, easy to learn
- Great for beginners

### Why Keras?
- Simple and consistent API
- Modular and extensible
- Works with CPUs and GPUs
- Fast prototyping

## TensorFlow

### Overview
- Open-source library by Google
- Low-level and high-level APIs
- Flexible and powerful
- Large community support
- Used in production at scale

### Key Features
- TensorBoard for visualization
- TensorFlow Serving for deployment
- Eager execution
- Keras integration (tf.keras)

## PyTorch

### Overview
- Open-source library by Facebook
- Dynamic computation graphs
- Pythonic and intuitive
- Popular in research
- Growing production use

### Key Features
- TorchScript
- Distributed training
- Mobile support
- Autograd system

## Comparison

| Feature | Keras | TensorFlow | PyTorch |
|---------|-------|------------|---------|
| Level | High-level | Both | Both |
| Ease of Use | Easiest | Moderate | Moderate |
| Flexibility | Less | More | Most |
| Debugging | Easy | Moderate | Easy |
| Research | Limited | Growing | Popular |
| Production | Good | Excellent | Growing |

## When to Use Which?

### Use Keras when:
- Starting with deep learning
- Building standard models quickly
- Prototyping

### Use TensorFlow when:
- Need production deployment
- Using TensorBoard
- Large-scale projects

### Use PyTorch when:
- Research and experimentation
- Need dynamic graphs
- Custom architectures

## Installation

```python
# Keras (included in TensorFlow)
pip install tensorflow

# PyTorch
pip install torch
```

## Basic Keras Workflow

1. Define model (Sequential or Functional API)
2. Compile model (optimizer, loss, metrics)
3. Train model (fit)
4. Evaluate model
5. Make predictions

## Key Concepts

### Models
- Sequential: Linear stack of layers
- Functional: More flexible, can have multiple inputs/outputs

### Layers
- Dense: Fully connected
- Conv2D: Convolutional
- MaxPooling2D: Pooling
- Dropout: Regularization

### Optimizers
- Adam (most popular)
- SGD
- RMSprop

### Loss Functions
- Binary crossentropy (binary classification)
- Categorical crossentropy (multi-class)
- Mean squared error (regression)

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score

---

# Regression Models with Keras

## Overview
- Regression: Predict continuous numerical values
- Keras simplifies building neural networks for regression tasks

## Example Dataset
- Concrete compressive strength dataset
- Features: Cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age
- Target: Compressive strength (MPa)

## Data Preparation

### Step 1: Split Data
- Separate predictors (features) from target variable
- `predictors` = features columns
- `target` = target column

### Step 2: Format
- Ensure data is in correct format (numpy arrays or pandas DataFrames)

## Building a Regression Model in Keras

### Step 1: Import Libraries
```python
from keras.models import Sequential
from keras.layers import Dense
```

### Step 2: Create Sequential Model
```python
model = Sequential()
```

### Step 3: Add Dense Layers
```python
# First hidden layer (must specify input_shape)
model.add(Dense(5, activation='relu', input_shape=(8,)))

# Second hidden layer
model.add(Dense(5, activation='relu'))

# Output layer (1 node for regression)
model.add(Dense(1))
```

### Key Points
- **Dense layer:** Every neuron connected to all neurons in next layer
- **ReLU:** Recommended activation for hidden layers
- **Input shape:** Only needed in first layer
- **Output layer:** 1 node for single value prediction

### Step 4: Compile Model
```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

- **Optimizer:** Adam (automatic learning rate adjustment)
- **Loss:** Mean Squared Error (MSE) for regression

### Step 5: Train Model
```python
model.fit(predictors, target, epochs=100)
```

### Step 6: Make Predictions
```python
predictions = model.predict(new_data)
```

## Complete Example Code

```python
import keras
from keras.models import Sequential
from keras.layers import Dense

# Create model
model = Sequential()

# Add layers
model.add(Dense(5, activation='relu', input_shape=(8,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(predictors, target, epochs=100)

# Predict
predictions = model.predict(new_samples)
```

## Notes
- Number of neurons (5) is small for simplicity; real applications use 50-100+
- Dense networks: All nodes in one layer connected to all nodes in next layer
- Adam optimizer: No need to manually set learning rate

## Key Takeaways
- Keras enables building regression models with few lines of code
- Data must be prepared: predictors and target separate
- Sequential model + Dense layers = simple regression network
- Adam optimizer + MSE loss = standard for regression

---

# Classification Models with Keras

## Overview
- Classification: Predict categorical labels/classes
- Different from regression (predicting continuous values)

## Example Dataset
- Car evaluation dataset
- Features: Price, maintenance cost, number of people (encoded)
- Target: Decision (0=unacceptable, 1=acceptable, 2=good, 3=very good)

## Data Preparation

### Step 1: Split Data
- Separate predictors from target column

### Step 2: One-Hot Encode Target
- Use `to_categorical()` from Keras utilities
- Transforms target into binary array
- Example: Class 2 → [0, 0, 1, 0]

### Why One-Hot Encoding?
- Required for multi-class classification
- Each class becomes a separate output neuron

## Building a Classification Model in Keras

### Step 1: Import Libraries
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
```

### Step 2: Transform Target
```python
target = to_categorical(target)
```

### Step 3: Create Sequential Model
```python
model = Sequential()
```

### Step 4: Add Dense Layers
```python
# First hidden layer
model.add(Dense(5, activation='relu', input_shape=(8,)))

# Second hidden layer
model.add(Dense(5, activation='relu'))

# Output layer (4 neurons for 4 classes)
model.add.Dense(4, activation='softmax')
```

### Key Differences from Regression
- **Output layer:** Multiple neurons (equal to number of classes)
- **Activation:** Softmax (outputs sum to 1 as probabilities)

### Step 5: Compile Model
```python
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

- **Loss:** Categorical crossentropy (for multi-class)
- **Metrics:** Accuracy

### Step 6: Train Model
```python
model.fit(predictors, target, epochs=100)
```

### Step 7: Make Predictions
```python
predictions = model.predict(new_data)
```

## Complete Example Code

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Prepare target (one-hot encoding)
target = to_categorical(target)

# Create model
model = Sequential()

# Add layers
model.add(Dense(5, activation='relu', input_shape=(8,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train
model.fit(predictors, target, epochs=100)

# Predict
predictions = model.predict(new_samples)
```

## Understanding Predictions

### Output Format
- Each row: probabilities for each class
- Probabilities sum to 1
- Highest probability = predicted class

### Example Output
```
[[0.99, 0.01, 0.00, 0.00],  # Class 0 (99% confidence)
 [0.99, 0.01, 0.00, 0.00],  # Class 0
 [0.45, 0.50, 0.03, 0.02],  # Class 1 (50% confidence - less sure)
 [0.10, 0.85, 0.03, 0.02]]  # Class 1
```

## Binary Classification
- Use `sigmoid` activation in output layer
- Use `binary_crossentropy` loss
- Output: Single probability value

## Key Takeaways
- Classification requires one-hot encoding of target
- Output layer uses softmax for multi-class
- Use categorical_crossentropy loss
- Predictions are probabilities (sum to 1)
- Higher probability = more confident prediction
