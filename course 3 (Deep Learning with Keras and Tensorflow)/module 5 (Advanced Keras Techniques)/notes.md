# Advanced Keras Techniques

## Learning Objectives
After completing this module, you will be able to:
- List advanced Keras techniques
- Demonstrate each technique with code examples

## Overview
Keras offers a variety of advanced techniques that can significantly enhance your model development process, including:
- Custom training loops
- Specialized layers
- Advanced callback functions
- Model optimization with TensorFlow

## 1. Custom Training Loops

### Learning Objectives
- Describe the basic structure of a custom training loop
- Explain the benefits of a custom training loop
- Implement a custom training loop

### Basic Structure
A custom training loop involves several key components:
- **Dataset** - Training data
- **Model** - Neural network model
- **Optimizer** - Updates model weights
- **Loss Function** - Measures prediction accuracy

### Example: Setting Up the Environment

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and prepare MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
```

### Define Loss Function and Optimizer

```python
# Loss function measures how well model's predictions match true labels
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Optimizer updates model's weights to minimize the loss
optimizer = keras.optimizers.Adam(learning_rate=0.001)
```

### Implement the Custom Training Loop

```python
# Create a simple model
model = keras.Sequential([
    layers.Flatten(input_shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])

# Custom training loop
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}")
    
    for batch_x, batch_y in train_dataset:
        # Compute logits by passing inputs through the model
        logits = model(batch_x, training=True)
        
        # Calculate loss
        loss = loss_fn(batch_y, logits)
        
        # Use tf.GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss_value = loss_fn(batch_y, logits)
        
        # Compute gradients
        gradients = tape.gradient(loss_value, model.trainable_weights)
        
        # Apply gradients to update model weights
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        print(f"Loss: {loss_value.numpy():.4f}")
```

### Understanding tf.GradientTape

`tf.GradientTape` is a TensorFlow API that records operations for automatic differentiation:
- Records operations on watched variables during forward pass
- Computes gradients of loss with respect to each weight
- Enables fine control over each training step

This allows implementation of complex training procedures and custom algorithms.

### Benefits of Custom Training Loops

1. **More Granular Control** - Full control over training process
2. **Custom Loss Functions** - Implement non-standard loss functions
3. **Custom Optimization Strategies** - Use custom optimization algorithms
4. **Advanced Logging and Monitoring** - Custom metrics and logging
5. **Integration with Custom Operations** - Add custom layers and operations
6. **Flexibility for Research** - Implement novel training algorithms

## 2. Specialized Layers

Keras allows you to create specialized layers to extend the functionality of your model, such as custom dense layers and Lambda layers. These layers can implement advanced custom activation functions.

### Example: Custom Dense Layer

```python
class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Initialize weights and biases
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        # Define the forward pass
        return tf.matmul(inputs, self.kernel) + self.bias

# Use custom layer in a Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    CustomDenseLayer(32),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

**Key Benefits:**
- Implement custom layers not available in standard Keras library
- Define custom weights and biases
- Implement advanced custom activation functions

## 3. Advanced Callback Functions

Callback functions in Keras are powerful tools to customize and extend the behavior of your model training process. They allow you to:
- Monitor training
- Implement early stopping
- Save model checkpoints
- Create custom callbacks

### Example: Custom Callback

```python
class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print loss and accuracy at the end of each epoch
        print(f"\nEpoch {epoch + 1}")
        print(f"Loss: {logs.get('loss'):.4f}")
        print(f"Accuracy: {logs.get('accuracy'):.4f}")

# Use custom callback during model training
custom_callback = CustomMetricsCallback()

model.fit(
    x_train, y_train,
    epochs=num_epochs,
    callbacks=[custom_callback]
)
```

**Built-in Callbacks:**
- `EarlyStopping` - Stop training when monitored metric stops improving
- `ModelCheckpoint` - Save model checkpoints during training
- `LearningRateScheduler` - Adjust learning rate during training
- `ReduceLROnPlateau` - Reduce learning rate when metrics plateau

## 4. Model Optimization with TensorFlow

Optimizing models is crucial for achieving the best performance. TensorFlow offers various tools for model optimization, including:

### Mixed Precision Training

Mixed precision training allows TensorFlow to use 16-bit floats where possible, improving performance and reducing memory usage without compromising model accuracy.

```python
# Enable mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs)
```

### TensorFlow Model Optimization Toolkit

The TensorFlow Model Optimization Toolkit provides additional optimization techniques such as:
- Quantization
- Pruning
- Clustering

## 5. Hyperparameter Tuning with Keras Tuner

### Learning Objectives
- Explain hyperparameter tuning using Keras Tuner
- Demonstrate how to set up Keras Tuner
- Define a model with hyperparameters
- Configure and run hyperparameter search
- Analyze results and train the optimized model

### Introduction
Hyperparameter tuning is a crucial step in the machine learning pipeline as it helps optimize model performance by finding the best set of hyperparameters. Keras Tuner simplifies this process by providing an easy-to-use interface for hyperparameter optimization.

**Hyperparameters** are variables that govern the training process of a model. Unlike model parameters which are learned during training, hyperparameters must be set before training begins.

Examples:
- Learning rate
- Batch size
- Number of layers
- Number of units in a neural network

### Keras Tuner Overview

Keras Tuner is a library that helps automate the process of hyperparameter tuning. It provides several search algorithms:
- **Random Search** - Randomly samples hyperparameter combinations
- **Hyperband** - Early stopping-based optimization
- **Bayesian Optimization** - Probabilistic model-based search

### Set Up Keras Tuner

```python
# Install Keras Tuner
!pip install keras-tuner

# Import necessary modules
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

### Define Model with Hyperparameters

```python
def build_model(hp):
    """Build a model with tunable hyperparameters."""
    model = keras.Sequential()
    
    # Tune the number of units in the first dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(units=hp_units, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Tune the learning rate
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Hyperparameter Functions:**
- `hp.Int()` - Integer hyperparameters
- `hp.Float()` - Floating point hyperparameters
- `hp.Choice()` - Categorical choices

### Configure the Search

```python
# Create a Random Search tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_tuner_dir',
    project_name='mnist_tuning'
)
```

### Run Hyperparameter Search

```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Run the hyperparameter search
tuner.search(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)
```

### Analyze Results and Train Optimized Model

```python
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best number of units: {best_hps.get('units')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Build the model with best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the optimized model
model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Summary of Keras Tuner Workflow

1. **Install and import** Keras Tuner
2. **Define model** with hyperparameters using `hp.Int()`, `hp.Float()`, etc.
3. **Configure tuner** (RandomSearch, Hyperband, or Bayesian)
4. **Run search** with `tuner.search()`
5. **Get best hyperparameters** with `tuner.get_best_hyperparameters()`
6. **Build and train** optimized model

## 6. Model Optimization

### Learning Objectives
- Explain the need for model optimization
- Describe key optimization techniques
- Apply optimization techniques to deep learning models

### Introduction
Model optimization aims to improve the performance and efficiency of neural networks. Optimized models lead to:
- Faster training times
- Better utilization of hardware resources
- Improved accuracy

### 1. Weight Initialization

Proper weight initialization can significantly impact the convergence and performance of a neural network. Common methods:
- **Xavier/Glorot Initialization** - Suitable for sigmoid/tanh activations
- **He Initialization** - Suitable for ReLU activations

```python
# He initialization for layers with ReLU activation
from tensorflow.keras.layers import Dense

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 2. Learning Rate Scheduling

Learning rate scheduling adjusts the learning rate dynamically during training to help the model converge more efficiently.

```python
# Define learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Create callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train model with scheduler
model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[lr_callback]
)
```

### 3. Batch Normalization

Batch normalization normalizes the input layer by adjusting and scaling activations. It helps:
- Accelerate training
- Improve model convergence
- Reduce sensitivity to initial learning rate

```python
# Model with Batch Normalization
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 4. Mixed Precision Training

Mixed precision training uses both 16-bit and 32-bit floating-point types to speed up training on modern GPUs.

```python
# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### 5. Model Pruning

Model pruning reduces the number of parameters by removing less significant connections.

```python
import tensorflow_model_optimization as tfmot

prune_model = tfmot.sparsity.keras.prune_low_magnitude

def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

model_for_pruning = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_dense,
)
```

### 6. Quantization

Quantization reduces the precision of weights to lower precision (e.g., 8-bit) for deployment on edge devices.

```python
# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

## 7. TensorFlow for Model Optimization

### Learning Objectives
- Explain importance of TensorFlow for optimizing deep learning models
- List TensorFlow tools and techniques for optimization
- Demonstrate how to use TensorFlow tools for optimization

### Importance of Model Optimization
Optimized models provide:
- Faster training and inference
- Reduced model size
- More effective resource utilization
- Better scalability for deployment
- Lower memory and power consumption

### 1. Mixed Precision Training

Mixed-precision training leverages both 16-bit and 32-bit floating-point types to accelerate model training and reduce memory usage.

Benefits:
- Significantly improves performance on modern GPUs
- Reduces memory bandwidth
- Allows faster training without significant accuracy loss
- Uses loss scaling to maintain accuracy

```python
# Enable mixed precision training
from tensorflow.keras import mixed_precision

# Set global policy to Mixed Float16
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with optimizer and loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model - faster with mixed precision
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### 2. Knowledge Distillation

Knowledge distillation trains a smaller student model to mimic the behavior of a larger teacher model.

Benefits:
- Reduces model size
- Suitable for deployment on resource-constrained devices
- Faster inference time
- Maintains performance of complex model

**Key Concepts:**
- **Teacher Model**: Deep, complex network trained to high accuracy
- **Student Model**: Smaller, simpler network that learns from teacher
- **Temperature Parameter**: Controls softening of softmax outputs

```python
# Train teacher model first (larger model)
teacher_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])

teacher_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

teacher_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Knowledge distillation with student model
class DistillationModel(tf.keras.Model):
    def __init__(self, student_model, teacher_model, temperature=4):
        super(DistillationModel, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
    
    def train_step(self, data):
        x, y = data
        
        # Get teacher predictions (logits)
        teacher_logits = self.teacher_model(x, training=False)
        
        with tf.GradientTape() as tape:
            student_logits = self.student_model(x, training=True)
            
            # Softmax with temperature
            student_soft = tf.nn.softmax(student_logits / self.temperature)
            teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
            
            # Distillation loss
            distillation_loss = tf.keras.losses.categorical_crossentropy(
                teacher_soft, student_soft
            )
            
            # Hard label loss
            hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, student_logits
            )
            
            # Combined loss
            total_loss = distillation_loss + hard_loss
        
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student_model.trainable_variables)
        )
        
        return {'loss': total_loss}

# Create smaller student model
student_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

student_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy(from_logits=True)')

# Train student model with distillation
distillation_model = DistillationModel(student_model, teacher_model, temperature=4)
distillation_model.fit(x_train_small, y_train_small, epochs=5)
```

### 3. TensorFlow Model Optimization Toolkit

The TensorFlow Model Optimization Toolkit provides post-training optimization techniques:

- **Pruning**: Removes less significant connections
- **Quantization**: Reduces precision of weights
- **Weight Clustering**: Groups similar weights

```python
import tensorflow_model_optimization as tfmot

# Model Pruning
prune_model = tfmot.sparsity.keras.prune_low_magnitude

def prunable_layer(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return prune_model(layer)
    return layer

model_for_pruning = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model_for_pruning,
    tfmot.sparsity.keras.PruningSchedule(
        tfmot.sparsity.keras.ConstantSparsity(0.5, 0)
    )
)

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

### Summary of TensorFlow Optimization Tools

| Technique | Use Case |
|-----------|----------|
| Mixed Precision Training | Faster training on GPUs |
| Knowledge Distillation | Create smaller models |
| Pruning | Reduce model size |
| Quantization | Edge device deployment |

## Summary

Advanced Keras techniques include:
1. **Custom Training Loops** - Greater flexibility and control over training process
2. **Specialized Layers** - Create custom layers for specific needs
3. **Advanced Callback Functions** - Monitor, control, and customize training behavior
4. **Model Optimization with TensorFlow** - Mixed precision training and optimization toolkits
5. **Hyperparameter Tuning with Keras Tuner** - Automated hyperparameter optimization
6. **Model Optimization Techniques** - Weight initialization, learning rate scheduling, batch normalization, pruning, quantization
7. **TensorFlow Model Optimization** - Mixed precision, knowledge distillation, pruning, quantization

---

# Module Summary

**Advanced Keras techniques** include:
- Custom training loops
- Specialized layers
- Advanced callback functions
- Model optimization with TensorFlow

These techniques will help you create more flexible and efficient deep learning models.

**Custom Training Loops:**
- A custom training loop consists of: dataset, model, optimizer, and loss function
- To implement: iterate over dataset, compute loss, and apply gradients to update model's weights
- Benefits include: custom loss functions and metrics, advanced logging and monitoring, flexibility for research, integration with custom operations and layers

**Hyperparameters:**
- Hyperparameters are variables that govern the training process of a model
- Examples: learning rate, batch size, number of layers, number of units in a neural network
- Keras Tuner is a library that helps automate hyperparameter tuning
- Workflow: define model with hyperparameters, configure search, run search, analyze results, train optimized model

**Model Optimization Techniques:**
- Weight initialization
- Learning rate scheduling
- Batch normalization
- Mixed precision training
- Model pruning
- Quantization

These techniques can significantly improve the performance, efficiency, and scalability of deep learning models.

**TensorFlow Optimization Tools:**
- Mixed precision training
- Model pruning
- Quantization
- TensorFlow Model Optimization Toolkit
