# Module 1: Deep Learning with TensorFlow and Keras

## Course Overview

### What is Deep Learning?
Deep Learning is revolutionizing many fields, including:
- Computer Vision
- Natural Language Processing
- Robotics

### About Keras
- High-level neural networks API written in Python
- Essential part of TensorFlow 
- Makes deep learning accessible and straightforward

## Course Goals
- Develop deep learning models using TensorFlow and Keras
- Progress from fundamental machine learning concepts to advanced models
- Create custom layers and models in Keras
- Build advanced CNNs and transformer models for sequential data

## Topics Covered

### Advanced Techniques
- Custom layers and models in Keras
- Advanced Convolutional Neural Networks (CNNs)
- Transformer models for sequential data
- Unsupervised learning techniques
- Reinforcement Learning with Deep Q Networks (DQNs)

### Practical Skills
- Tensor operations
- Implementing and training linear regression models from scratch
- Hands-on labs and projects

## Lab Topics
- Creating custom layers and models
- Advanced data augmentation with Keras
- Transfer learning implementation
- Application of transposed convolution
- Advanced transformers for text generation
- Building autoencoders
- Diffusion models
- Developing GANs
- Custom training loops
- Hyperparameter tuning
- Q learning and Deep Q Networks

## Final Project
Build and optimize a classification model using Keras and TensorFlow to showcase on GitHub.

## Prerequisites
- Basic knowledge of Python programming
- Familiarity with mathematical concepts (gradients, matrices)
- Git and GitHub usage

## Recommended Pre-requisite Courses
- Python for data science
- Machine learning with Python
- Deep learning and neural networks with Keras

---

## Introduction to Advanced Keras

### Overview
- **Keras** is widely used in industry and academia for:
  - Image and speech recognition
  - Recommendation systems
  - Natural language processing
- Major companies using Keras: **Google, Netflix, Uber**
- Simplicity and ease of use makes it a favorite among deep learning practitioners

### Sequential API vs Functional API

| Sequential API | Functional API |
|----------------|----------------|
| Simple linear stack of layers | More flexibility and control |
| Layers added sequentially | Non-sequential data flows |
| Limited to single input/output | Multiple inputs/outputs supported |
| Less explicit structure | Explicit and easier to debug |

### Sequential API Example
```python
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### Functional API Example
```python
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
```

### Advantages of Functional API
1. **Flexibility** - Create complex architectures:
   - Multi-branch models
   - Residual connections
   - Non-sequential data flows
2. **Clarity** - Model structure is explicit and easier to debug
3. **Reusability** - Layers and models can be reused across different parts of the architecture

### Complex Model with Multiple Inputs
```python
# Two input layers with different shapes
input_a = Input(shape=(32,))
input_b = Input(shape=(64,))

# Separate branches for each input
branch_a = Dense(64, activation='relu')(input_a)
branch_b = Dense(64, activation='relu')(input_b)

# Concatenate outputs
combined = concatenate([branch_a, branch_b])

# Additional layers
x = Dense(128, activation='relu')(combined)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=[input_a, input_b], outputs=outputs)
```

### Real-World Applications

| Industry | Application |
|----------|-------------|
| **Healthcare** | Medical image analysis, disease detection |
| **Finance** | Market trend prediction |
| **Autonomous Driving** | Object detection, lane detection, perception tasks |

### Key Takeaways
- Functional API provides flexibility needed for complex neural network models
- Supports multiple inputs/outputs, shared layers, and intricate architectures
- Essential for advanced deep learning applications
- Enables tackling a wider range of problems

---

## Keras Functional API and Subclassing API

### Keras Functional API Capabilities
- Builds complex neural networks more flexibly than Sequential API
- Creates models beyond simple stack of layers
- Supports:
  - Multiple inputs and outputs
  - Shared layers
  - Non-linear data flows
- Crucial for research and tackling complex problems requiring custom solutions

### Basic Concepts and Syntax

```python
from tensorflow.keras import Input, Model, layers

# Define input layer with shape
inputs = Input(shape=(input_dim,))

# Add dense layers
x = layers.Dense(units=64, activation='relu')(inputs)
outputs = layers.Dense(units=10, activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)
```

**Key Points:**
- `Input()` defines the shape of input data
- `units` = dimensionality of the output space of the layer
- Layers are connected in a **graph of layers** (not stacked sequentially)
- Final layer uses `softmax` for multiclass classification

### Models with Multiple Inputs and Outputs

Useful for **multitask learning** and complex applications:

```python
# Define two separate input layers with different shapes
input1 = Input(shape=(32,))
input2 = Input(shape=(64,))

# Separate branches processing each input
branch1 = layers.Dense(64, activation='relu')(input1)
branch2 = layers.Dense(64, activation='relu')(input2)

# Merge branches
merged = layers.concatenate([branch1, branch2])

# Output layers
output1 = layers.Dense(10, activation='softmax', name='output1')(merged)
output2 = layers.Dense(5, activation='sigmoid', name='output2')(merged)

# Create model
model = Model(inputs=[input1, input2], outputs=[output1, output2])
```

### Shared Layers

Shared layers apply the **same transformation** to multiple inputs:

```python
# Define shared layer
shared_layer = layers.Dense(64, activation='relu')

# Apply shared layer to both inputs
output1 = shared_layer(input1)
output2 = shared_layer(input2)

# Create model with shared layers
model = Model(inputs=[input1, input2], outputs=[output1, output2])
```

**Use Case:** Siamese networks - process two different inputs through the same layers and compare their outputs.

### Practical Example: Complex Model with Multiple Inputs and Shared Layers

```python
# Input layers
input_a = Input(shape=(64, 64, 3))
input_b = Input(shape=(64, 64, 3))

# Branch A - Convolutional layers
conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_a)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
flat1 = layers.Flatten()(pool1)

# Branch B - Convolutional layers (shared architecture)
conv2 = layers.Conv2D(32, (3, 3), activation='relu')(input_b)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
flat2 = layers.Flatten()(pool2)

# Concatenate branches
combined = layers.concatenate([flat1, flat2])

# Dense layers
x = layers.Dense(128, activation='relu')(combined)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

# Create model
model = Model(inputs=[input_a, input_b], outputs=output)
```

---

## Subclassing API

### Overview
- Offers the **most flexibility** compared to Sequential and Functional APIs
- Allows defining **custom and dynamic models**
- Subclass `tf.keras.Model` and implement custom `call()` method
- Useful when forward pass **cannot be defined statically**
- Widely used in research for custom training loops and non-standard architectures

### Basic Syntax

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers in __init__
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        # Define forward pass in call()
        x = self.dense1(inputs)
        return self.dense2(x)

# Instantiate and use the model
model = MyModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

### When to Use Subclassing API

| Scenario | Description |
|----------|-------------|
| **Dynamic Architectures** | Model architecture changes dynamically (e.g., reinforcement learning models) |
| **Custom Training Loops** | Need more control over training process beyond `model.fit()` |
| **Research & Prototyping** | Experimenting with new layers/architectures not in standard Keras API |

### Custom Training Loop with tf.GradientTape

```python
import tensorflow as tf

# Create model, optimizer, and loss function
model = MyModel()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Custom training loop
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x_batch, training=True)
        # Compute loss
        loss = loss_fn(y_batch, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
```

**Advantages over `keras.fit()`:**
- Full control over training process
- Custom loss computation
- Dynamic behavior during training
- Custom metrics and logging

### Dynamic Graphs

Subclassing API supports **dynamic computation graphs**:
- Architecture can change dynamically during training
- Based on input data or conditions
- More flexible than static graphs in Sequential/Functional APIs

```python
class DynamicModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        
        # Dynamic behavior - skip layer based on condition
        if training and tf.random.uniform(()) > 0.5:
            x = self.dense2(x)
        
        return self.output_layer(x)
```

### Comparison of APIs

| Feature | Sequential API | Functional API | Subclassing API |
|---------|---------------|----------------|-----------------|
| **Flexibility** | Low | Medium | High |
| **Multiple Inputs/Outputs** | ❌ | ✅ | ✅ |
| **Shared Layers** | ❌ | ✅ | ✅ |
| **Dynamic Architecture** | ❌ | ❌ | ✅ |
| **Custom Training Loops** | Limited | Limited | Full Control |
| **Ease of Use** | Very Easy | Easy | Moderate |
| **Debugging** | Easy | Easy | Harder |
| **Model Visualization** | ✅ | ✅ | ❌ |

---

## Summary

| API Type | Best For |
|----------|----------|
| **Sequential** | Simple, linear stack of layers |
| **Functional** | Complex models with multiple inputs/outputs, shared layers |
| **Subclassing** | Dynamic models, custom training loops, research prototyping |

### Key Takeaways
- **Functional API**: Define layers and connect them in a graph; handles multiple inputs/outputs and shared layers
- **Subclassing API**: Maximum flexibility through custom `call()` method and dynamic architectures
- **tf.GradientTape**: Provides fine-grained control over training compared to built-in `keras.fit()`
- Choose the API based on model complexity and flexibility requirements

---

## Creating Custom Layers in Keras

### Why Custom Layers?

Standard layers (Dense, Convolutional, LSTM) cover many use cases, but custom layers are essential when you need:

1. **Specific Functionalities** - Operations not provided by built-in Keras layers
2. **Novel Research Ideas** - Implement new algorithms or techniques directly into models
3. **Performance Optimization** - Tailor layers to specific data or computational constraints
4. **Enhanced Flexibility** - Define unique behaviors not possible with standard layers
5. **Improved Readability & Maintenance** - Encapsulate complex logic in reusable components

**Benefits:**
- Unlock full potential of Keras
- Build more sophisticated and fine-tuned models
- Gain deeper understanding of neural network internals
- Enhance ability to innovate

### Basic Structure of a Custom Layer

To create a custom layer, **subclass `tf.keras.layers.Layer`** and implement three key methods:

| Method | Purpose | Called When |
|--------|---------|-------------|
| `__init__` | Initialize layer's attributes | During layer creation |
| `build` | Create layer's weights | During first call to the layer |
| `call` | Define forward pass logic | Every time the layer is called |

### Implementing a Custom Dense Layer

```python
import tensorflow as tf
from tensorflow.keras import layers

class CustomDenseLayer(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create weights (kernel) and biases
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        # Forward pass: dense operation + activation
        x = tf.matmul(inputs, self.w) + self.b
        return self.activation(x)
```

**Key Points:**
- `__init__`: Store layer parameters (units, activation)
- `build`: Create trainable weights using `add_weight()`
- `call`: Implement the computation (matrix multiplication + bias + activation)

### Using Custom Layers in a Model

Custom layers integrate seamlessly with Keras models:

```python
from tensorflow.keras import Sequential

# Create a sequential model with custom layer
model = Sequential([
    CustomDenseLayer(units=64, activation='relu'),
    CustomDenseLayer(units=32, activation='relu'),
    layers.Dense(10, activation='softmax')  # Can mix with standard layers
])

# Compile and train as usual
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

### Using Custom Layers with Functional API

```python
from tensorflow.keras import Input, Model

# Input layer
inputs = Input(shape=(784,))

# Use custom layer
x = CustomDenseLayer(units=128, activation='relu')(inputs)
x = CustomDenseLayer(units=64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)
```

### Best Practices for Custom Layers

1. **Always call `super().__init__(**kwargs)`** to ensure proper initialization
2. **Use `add_weight()`** for creating trainable parameters
3. **Keep `build()` method for weight creation** - it handles input shape inference
4. **Use TensorFlow operations** in `call()` for GPU/TPU compatibility
5. **Implement `get_config()`** for model serialization (optional but recommended)

```python
def get_config(self):
    config = super(CustomDenseLayer, self).get_config()
    config.update({
        'units': self.units,
        'activation': tf.keras.activations.serialize(self.activation)
    })
    return config
```

### Summary

| Aspect | Description |
|--------|-------------|
| **When to Use** | Need functionality beyond standard layers |
| **Base Class** | `tf.keras.layers.Layer` |
| **Key Methods** | `__init__`, `build`, `call` |
| **Integration** | Works with Sequential, Functional, and Subclassing APIs |
| **Benefits** | Flexibility, innovation, optimization, code organization |

### Key Takeaways
- Custom layers extend Keras functionality for specific needs
- Subclass `Layer` and implement `__init__`, `build`, and `call` methods
- Custom layers integrate seamlessly with all Keras model types
- Practice and experimentation with custom layers deepens neural network understanding

---

## TensorFlow 2.X Overview

### What is TensorFlow?

**TensorFlow** is an open-source platform for machine learning developed by **Google**. It has become one of the most popular frameworks for machine learning and deep learning applications.

**Purpose:** Provides comprehensive tools to build and deploy machine learning models across various environments—from servers to edge devices.

---

### Key Features of TensorFlow 2.X

| Feature | Description |
|---------|-------------|
| **Eager Execution** | Operations execute immediately without building graphs; great for debugging and interactive programming |
| **High-Level APIs** | Keras integrated as the official high-level API for simplified model building |
| **Multi-Platform Support** | Deployment across mobile, web, and embedded devices |
| **Scalability & Performance** | Seamless scaling across CPUs, GPUs, and TPUs |
| **Rich Ecosystem** | Extensive tools and libraries (TensorFlow Lite, TensorFlow JS, TFX, etc.) |

---

### Eager Execution

**Eager Execution** is a major enhancement in TensorFlow 2.X that makes the framework more user-friendly and flexible.

#### How It Works
- Operations execute **immediately** without building static computation graphs
- Similar to standard Python programming
- Ideal for dynamic model behaviors and beginners

#### Benefits of Eager Execution

| Benefit | Description |
|---------|-------------|
| **Improved Debugging** | Immediate feedback makes it easier to identify and fix errors |
| **Simplified Code** | No need for static graphs; code is more straightforward and readable |
| **Interactive Development** | Supports exploratory research and interactive programming |

#### Example: Simple TensorFlow Code with Eager Execution

```python
import tensorflow as tf

# Eager execution is enabled by default in TF 2.X
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a + b  # Executes immediately

print(c)  # Output: tf.Tensor(5.0, shape=(), dtype=float32)

# Dynamic behavior example
def linear_layer(x):
    w = tf.Variable(tf.random.normal([10, 5]))
    b = tf.Variable(tf.zeros([5]))
    return tf.matmul(x, w) + b

# Call function directly
result = linear_layer(tf.random.normal([3, 10]))
print(result.shape)  # Output: (3, 5)
```

---

### Keras Integration with TensorFlow 2.X

TensorFlow 2.X integrates **Keras** as its official high-level API, greatly simplifying the process of building and training deep learning models.

#### Benefits of Keras Integration

| Benefit | Description |
|---------|-------------|
| **User-Friendly** | Simple and concise code for creating models |
| **Modular & Composable** | Layers and blocks can be easily combined in modular fashion |
| **Extensive Documentation** | Comprehensive docs and examples facilitate learning |
| **Streamlined Workflow** | Unified API makes model building and training more accessible |

#### Example: Building a Model with Keras in TensorFlow 2.X

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

---

### TensorFlow Ecosystem Components

TensorFlow has a **rich ecosystem** of tools and libraries that extend its capabilities and support the entire ML lifecycle.

| Component | Description | Use Case |
|-----------|-------------|----------|
| **TensorFlow Lite** | Lightweight solution for mobile and embedded devices | On-device ML with low latency and high performance |
| **TensorFlow JS** | Library for JavaScript environments | Train/deploy models in web browsers and Node.js |
| **TensorFlow Extended (TFX)** | End-to-end platform for production ML pipelines | Model deployment, monitoring, and management |
| **TensorFlow Hub** | Repository of reusable ML modules | Accelerate development with pre-trained models |
| **TensorBoard** | Visualization toolkit | Inspect metrics, graphs, and model training process |

#### TensorFlow Ecosystem Overview

```
┌─────────────────────────────────────────────────────────┐
│                    TensorFlow Ecosystem                  │
├─────────────────────────────────────────────────────────┤
│  Development      │  Deployment       │  Production     │
│  ───────────      │  ──────────       │  ──────────     │
│  • TensorFlow     │  • TensorFlow Lite│  • TFX          │
│  • Keras          │  • TensorFlow JS  │  • TensorBoard  │
│  • TensorFlow Hub │  • TensorFlow     │  • Monitoring   │
│                   │    Serving        │                 │
└─────────────────────────────────────────────────────────┘
```

#### Component Details

**1. TensorFlow Lite**
- Optimized for mobile and embedded devices
- Enables on-device inference
- Reduces latency and preserves privacy

**2. TensorFlow JS**
- Run ML models directly in browsers
- No server-side processing required
- Accessible via JavaScript API

**3. TensorFlow Extended (TFX)**
- Production-ready ML pipelines
- Data validation, model analysis, serving
- Continuous monitoring and management

**4. TensorFlow Hub**
- Pre-trained models and modules
- Transfer learning made easy
- Community-contributed resources

**5. TensorBoard**
```python
# Log training metrics
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir)

# Train with callback
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_cb])

# Launch TensorBoard
# tensorboard --logdir=logs/fit/
```

---

### Summary: TensorFlow 2.X Advantages

| Aspect | TensorFlow 2.X |
|--------|----------------|
| **Ease of Use** | Keras integration + eager execution |
| **Debugging** | Immediate execution with clear error messages |
| **Deployment** | Multi-platform (mobile, web, edge, cloud) |
| **Performance** | Optimized for CPUs, GPUs, and TPUs |
| **Ecosystem** | Comprehensive tools for entire ML lifecycle |
| **Community** | Large community with extensive resources |

---

## Key Takeaways

### TensorFlow 2.X
- Powerful and flexible platform for machine learning
- **Eager execution** enables intuitive, immediate operation execution
- **Keras integration** simplifies model building and training
- **Rich ecosystem** supports development to deployment

### When to Use TensorFlow
- Research and prototyping (eager execution)
- Production deployment (TFX, TensorFlow Serving)
- Mobile/edge deployment (TensorFlow Lite)
- Web applications (TensorFlow JS)
- Large-scale distributed training (TPU/GPU support)

Understanding TensorFlow 2.X features and capabilities helps you build and deploy machine learning models effectively across research, prototyping, and production applications.
