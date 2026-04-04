# Module 4: Unsupervised Learning and Generative Models in Keras

## Introduction to Unsupervised Learning

### Overview

**Unsupervised Learning** is a type of machine learning where the algorithm is used to find patterns in data **without any labels or predefined outcomes**.

### Supervised vs. Unsupervised Learning

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|----------------------|
| **Data** | Labeled data with target variable | Unlabeled data without targets |
| **Goal** | Predict target/outcome | Find patterns and structure |
| **Examples** | Classification, Regression | Clustering, Association, Dimensionality Reduction |
| **Evaluation** | Accuracy, Precision, Recall | Silhouette Score, Reconstruction Error |

**In unsupervised learning:**
- No target variable to predict
- Goal is to understand the **underlying structure of the data**
- Algorithm discovers hidden patterns on its own

---

## Categories of Unsupervised Learning

Unsupervised learning can be **broadly categorized into three types**:

### 1. Clustering

**Definition:** Grouping data points into clusters such that data points in the same cluster are more similar to each other than to those in other clusters.

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **K-Means Clustering** | Partitions data into K clusters based on distance | Customer segmentation, image compression |
| **Hierarchical Clustering** | Builds a tree of clusters (dendrogram) | Biological taxonomy, document clustering |
| **DBSCAN** | Density-based clustering | Anomaly detection, spatial data |
| **Gaussian Mixture Models** | Probabilistic clustering | Soft clustering, overlapping clusters |

**Example Applications:**
- Customer segmentation in marketing
- Image segmentation in computer vision
- Document clustering in NLP
- Anomaly detection in cybersecurity

### 2. Association

**Definition:** Finding relationships between variables in large data sets.

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Apriori Algorithm** | Finds frequent itemsets | Market basket analysis |
| **Eclat Algorithm** | Equivalent Clustering for Transactions | Retail analytics |
| **FP-Growth** | Frequent Pattern Growth | Efficient association rule mining |

**Example Applications:**
- **Market Basket Analysis**: Identify products that frequently co-occur in transactions
- **Recommendation Systems**: "Customers who bought X also bought Y"
- **Web Usage Mining**: Find pages frequently visited together
- **Medical Diagnosis**: Find symptoms that co-occur with diseases

**Common Metrics:**
- **Support**: Frequency of itemset in dataset
- **Confidence**: Likelihood of Y given X
- **Lift**: Strength of association between items

### 3. Dimensionality Reduction

**Definition:** Reducing the number of random variables under consideration by obtaining a set of principal variables.

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **PCA** (Principal Component Analysis) | Linear transformation to orthogonal components | Feature extraction, visualization |
| **t-SNE** (t-Distributed Stochastic Neighbor Embedding) | Non-linear dimensionality reduction | Data visualization, clustering |
| **UMAP** (Uniform Manifold Approximation and Projection) | Non-linear dimensionality reduction | Visualization, preserving global structure |
| **LDA** (Linear Discriminant Analysis) | Supervised dimensionality reduction | Classification preprocessing |

**Example Applications:**
- Data visualization (reducing to 2D/3D)
- Feature extraction for machine learning
- Noise reduction in data
- Compressing high-dimensional data

**Benefits:**
- Reduces computational complexity
- Removes redundant features
- Helps visualize high-dimensional data
- Can improve model performance

---

## Autoencoders

### Overview

**Autoencoders** are a type of neural network used to learn efficient representations of data for the purpose of **dimensionality reduction** or **feature learning**.

### Architecture

Autoencoders consist of **two main parts**:

```
┌─────────────────────────────────────────────────────────┐
│                    Autoencoder                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input → [Encoder] → Latent Space → [Decoder] → Output  │
│          (Compression)  (Bottleneck)  (Reconstruction)   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

| Component | Purpose | Description |
|-----------|---------|-------------|
| **Encoder** | Compresses input | Reduces input into a latent space representation |
| **Decoder** | Reconstructs input | Reconstructs the input from the latent space representation |
| **Latent Space** | Compressed representation | Lower-dimensional encoding of input data |

### Key Idea

The autoencoder is trained to **minimize the difference between the input and the reconstructed output**, forcing the network to learn meaningful representations of the data.

**Loss Function:**
```
Loss = ||Input - Reconstructed Output||²
```

### Types of Autoencoders

| Type | Description | Use Case |
|------|-------------|----------|
| **Vanilla Autoencoder** | Basic encoder-decoder structure | Simple dimensionality reduction |
| **Denoising Autoencoder** | Trained to remove noise from input | Image denoising, robust features |
| **Sparse Autoencoder** | Adds sparsity constraint on latent layer | Feature learning |
| **Variational Autoencoder (VAE)** | Probabilistic latent space | Generative modeling |
| **Convolutional Autoencoder** | Uses CNN layers | Image compression, denoising |

---

## Implementing Autoencoders in Keras

### Simple Autoencoder for MNIST

```python
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize data to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(f"Training data shape: {x_train.shape}")  # (60000, 784)
print(f"Test data shape: {x_test.shape}")  # (10000, 784)

# Define autoencoder for dataset with 784 features
# Encoder compresses input into 64 features
# Decoder reconstructs original 784 features

# Input layer
input_img = Input(shape=(784,))

# Encoder: Compress input to 64-dimensional latent space
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)  # Latent representation

# Decoder: Reconstruct input from latent space
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)  # Output layer

# Create autoencoder model
autoencoder = Model(input_img, decoded)

# Compile model
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'  # or 'mse' for reconstruction error
)

# Print model summary
autoencoder.summary()

# Train the autoencoder
# Use same data for both input and output
history = autoencoder.fit(
    x_train, x_train,  # Input = Output for autoencoder
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.1
)

# Evaluate on test data
test_loss = autoencoder.evaluate(x_test, x_test)
print(f"Test reconstruction loss: {test_loss:.4f}")

# Generate reconstructions
reconstructed = autoencoder.predict(x_test)

# Visualize original vs reconstructed
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# Display original images
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

# Display reconstructed images
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.suptitle('Original vs Reconstructed MNIST Digits')
plt.show()
```

### Autoencoder Architecture Summary

| Layer | Type | Units | Activation | Purpose |
|-------|------|-------|------------|---------|
| Input | Input | 784 | - | Original image |
| Encoded 1 | Dense | 128 | ReLU | First compression |
| Encoded 2 | Dense | 64 | ReLU | Latent representation |
| Decoded 1 | Dense | 128 | ReLU | First decompression |
| Output | Dense | 784 | Sigmoid | Reconstructed image |

### Training Autoencoders

**Key Points:**
- Use **same data for input and output** (x_train, x_train)
- Loss measures **reconstruction error**
- Latent space captures **essential features** of data
- Can be used for:
  - Dimensionality reduction
  - Denoising
  - Anomaly detection
  - Feature learning

---

## Building Autoencoders in Keras - Detailed Guide

### Autoencoder Concepts

**Autoencoders** are a type of neural network used for **unsupervised learning tasks**. They are powerful tools for:

| Application | Description |
|-------------|-------------|
| **Dimensionality Reduction** | Compress data to lower-dimensional representation |
| **Data Denoising** | Remove noise from corrupted input data |
| **Feature Learning** | Learn efficient representations of data |
| **Anomaly Detection** | Identify outliers based on reconstruction error |
| **Data Compression** | Efficient encoding of information |

### Autoencoder Architecture

The basic architecture of an autoencoder includes **three main components**:

```
┌─────────────────────────────────────────────────────────┐
│                    Autoencoder                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input → [Encoder] → [Bottleneck] → [Decoder] → Output  │
│          (Compression)  (Latent)     (Reconstruction)    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

| Component | Purpose | Description |
|-----------|---------|-------------|
| **Encoder** | Compresses input data | Reduces input into a smaller representation |
| **Bottleneck** | Compressed representation | Contains the most important features (latent space) |
| **Decoder** | Reconstructs input data | Reconstructs input from compressed representation |

**Goal:** Minimize the difference between the input data and the reconstructed data.

```
Loss = ||Input - Reconstructed Output||²
```

---

### Types of Autoencoders

| Type | Description | Use Case |
|------|-------------|----------|
| **Basic Autoencoder** | Simple structure with one hidden layer in both encoder and decoder | Simple dimensionality reduction |
| **Variational Autoencoder (VAE)** | Introduces probabilistic elements | Generating new data samples |
| **Convolutional Autoencoder** | Uses convolutional layers | Highly effective for image data |
| **Denoising Autoencoder** | Trained on corrupted input | Remove noise from data |
| **Sparse Autoencoder** | Adds sparsity constraints | Feature learning |

---

## Building a Basic Autoencoder in Keras

### Using the Functional API

```python
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import numpy as np

# ============ Step 1: Define the Autoencoder Architecture ============

# Input layer: 784 neurons for flattened 28x28 images
input_img = Input(shape=(784,))

# Encoder: Reduces input to 64 dimensions
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)

# Bottleneck: 32 dimensions (compressed representation)
bottleneck = Dense(32, activation='relu')(encoded)

# Decoder: Reconstructs input back to 784 dimensions
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(128, activation='relu')(decoded)
output_img = Dense(784, activation='sigmoid')(decoded)

# Create autoencoder model
autoencoder = Model(input_img, output_img)

# Compile the model using Adam optimizer and binary crossentropy loss
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Print model summary
autoencoder.summary()
```

### Autoencoder Architecture Summary

| Layer | Type | Units | Activation | Purpose |
|-------|------|-------|------------|---------|
| **Input** | Input | 784 | - | Flattened 28×28 image |
| **Encoded 1** | Dense | 128 | ReLU | First compression |
| **Encoded 2** | Dense | 64 | ReLU | Second compression |
| **Bottleneck** | Dense | 32 | ReLU | Latent representation |
| **Decoded 1** | Dense | 64 | ReLU | First decompression |
| **Decoded 2** | Dense | 128 | ReLU | Second decompression |
| **Output** | Dense | 784 | Sigmoid | Reconstructed image |

---

## Preparing and Training the Autoencoder

### MNIST Dataset Preparation

```python
# ============ Step 2: Load and Preprocess MNIST Dataset ============

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to single vector (flatten 28x28 to 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(f"Training data shape: {x_train.shape}")  # (60000, 784)
print(f"Test data shape: {x_test.shape}")  # (10000, 784)

# ============ Step 3: Train the Autoencoder ============

# Train autoencoder using training data as BOTH input and output
# Goal: Reconstruct the input data
history = autoencoder.fit(
    x_train, x_train,  # Input = Output for autoencoder
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# ============ Step 4: Evaluate the Autoencoder ============

# Evaluate on test data
test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"Test reconstruction loss: {test_loss:.4f}")

# ============ Step 5: Generate and Visualize Reconstructions ============

# Generate reconstructions
reconstructed = autoencoder.predict(x_test)

# Visualize original vs reconstructed images
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Display original images
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')

# Display reconstructed images
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed')
    plt.axis('off')

plt.suptitle('Original vs Reconstructed MNIST Digits', fontsize=16)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Fine-Tuning Autoencoders

### Freezing and Unfreezing Layers

In addition to using the autoencoder as is, you can **fine-tune** it by training some layers while keeping others frozen. This helps in:
- Adapting the autoencoder to new data
- Improving performance on specific tasks
- Transfer learning applications

### Fine-Tuning Implementation

```python
# ============ Fine-Tuning the Autoencoder ============

# Unfreeze the last 4 layers of the autoencoder
# This allows fine-tuning of decoder layers
for layer in autoencoder.layers[-4:]:
    layer.trainable = True

# Recompile the model after changing trainable status
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Print updated model summary to verify trainable layers
print("Fine-tuning configuration:")
for i, layer in enumerate(autoencoder.layers):
    print(f"{i}: {layer.name} - Trainable: {layer.trainable}")

# Train the model again for a few more epochs
# This fine-tunes the autoencoder
fine_tune_history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Evaluate fine-tuned model
fine_tune_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"Fine-tuned test reconstruction loss: {fine_tune_loss:.4f}")
```

### When to Fine-Tune

| Scenario | Approach |
|----------|----------|
| **New Dataset** | Unfreeze decoder layers, fine-tune for adaptation |
| **Improved Performance** | Unfreeze bottleneck and decoder, train longer |
| **Transfer Learning** | Freeze encoder, fine-tune decoder on new data |
| **Domain Adaptation** | Partial unfreezing based on layer relevance |

---

## Complete Autoencoder Example with Visualization

### Encoding and Decoding Visualization

```python
# ============ Visualize Latent Space Representations ============

# Create encoder model (input to bottleneck)
encoder = Model(input_img, bottleneck)

# Encode test data
encoded_data = encoder.predict(x_test)

print(f"Encoded data shape: {encoded_data.shape}")  # (10000, 32)

# Visualize encoded representations using PCA
from sklearn.decomposition import PCA

# Reduce 32D latent space to 2D for visualization
pca = PCA(n_components=2)
encoded_2d = pca.fit_transform(encoded_data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], 
                      c=np.argmax(autoencoder.predict(x_test), axis=1),
                      cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('Latent Space Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# ============ Reconstruction Error Analysis ============

# Calculate reconstruction errors
reconstruction_errors = np.mean(np.square(x_test - reconstructed), axis=1)

plt.figure(figsize=(10, 4))
plt.hist(reconstruction_errors, bins=50, edgecolor='black')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')
plt.grid(True, alpha=0.3)
plt.show()

# Find worst reconstructions (highest errors)
worst_indices = np.argsort(reconstruction_errors)[-5:][::-1]

plt.figure(figsize=(12, 4))
for i, idx in enumerate(worst_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Original (Error: {reconstruction_errors[idx]:.4f})')
    plt.axis('off')
    
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed')
    plt.axis('off')

plt.suptitle('Worst Reconstructions (Highest Error)', fontsize=16)
plt.tight_layout()
plt.show()
```

---

## Summary: Building Autoencoders in Keras

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Autoencoder** | Neural network for unsupervised learning tasks |
| **Encoder** | Compresses input data into smaller representation |
| **Bottleneck** | Compressed representation with most important features |
| **Decoder** | Reconstructs input from compressed representation |
| **Reconstruction Loss** | Difference between input and reconstructed output |

### Autoencoder Types

| Type | Description |
|------|-------------|
| **Basic Autoencoder** | Simple structure with hidden layers in encoder and decoder |
| **Variational Autoencoder (VAE)** | Probabilistic elements for generating new data |
| **Convolutional Autoencoder** | Uses convolutional layers for image data |

### Architecture Components

| Component | Purpose |
|-----------|---------|
| **Input Layer** | Accepts original data (e.g., 784 neurons for MNIST) |
| **Encoder Layers** | Reduce dimensions (e.g., 784 → 64 → 32) |
| **Bottleneck** | Latent space representation (e.g., 32 dimensions) |
| **Decoder Layers** | Reconstruct dimensions (e.g., 32 → 64 → 784) |
| **Output Layer** | Reconstructed data (same shape as input) |

### Key Takeaways

- **Autoencoders** are versatile tools for data denoising, dimensionality reduction, and feature learning
- **Basic architecture** includes three main components: encoder, bottleneck, and decoder
- **Different types** available: basic autoencoder, VAEs, convolutional autoencoders
- **Training** uses same data for both input and output
- **Fine-tuning** by unfreezing layers can improve performance
- **Applications** include dimensionality reduction, denoising, anomaly detection, and feature learning

Autoencoders are powerful unsupervised learning tools that enable efficient data representation and reconstruction for various machine learning tasks.

---

## Generative Adversarial Networks (GANs)

### Overview

**Generative Adversarial Networks (GANs)** are a class of neural networks designed by **Ian Goodfellow in 2014**.

### Architecture

GANs consist of **two networks** that compete against each other in a **zero-sum game**:

```
┌─────────────────────────────────────────────────────────┐
│              Generative Adversarial Network              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Noise → [Generator] → Fake Data ─┐                     │
│                                   │→ [Discriminator] →  │
│  Real Data ──────────────────────┘    (Real vs Fake)    │
│                                                          │
│  Generator: Tries to fool Discriminator                 │
│  Discriminator: Tries to distinguish real from fake     │
└─────────────────────────────────────────────────────────┘
```

| Network | Purpose | Description |
|---------|---------|-------------|
| **Generator** | Generates new data | Creates data instances that resemble the training data |
| **Discriminator** | Evaluates authenticity | Distinguishes between real and generated (fake) data |

### How GANs Work

**Adversarial Process:**

1. **Generator** takes random noise as input
2. **Generator** produces fake data (images, text, etc.)
3. **Discriminator** receives both real and fake data
4. **Discriminator** tries to classify data as real or fake
5. **Generator** tries to fool the discriminator
6. **Discriminator** tries to correctly identify real vs fake
7. This competition leads to **increasingly realistic generated data**

**Loss Functions:**
- **Generator Loss**: Minimize discriminator's ability to detect fakes
- **Discriminator Loss**: Maximize ability to distinguish real from fake

### GAN Training Process

```
┌─────────────────────────────────────────────────────────┐
│                  GAN Training Loop                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  For each training iteration:                           │
│                                                          │
│  1. Train Discriminator:                                │
│     - Generate fake images from noise                   │
│     - Train on real images (label=1)                    │
│     - Train on fake images (label=0)                    │
│     - Update discriminator weights                      │
│                                                          │
│  2. Train Generator:                                    │
│     - Generate fake images from noise                   │
│     - Train discriminator on fake images (label=1)      │
│     - Update generator weights                          │
│     - Keep discriminator weights frozen                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Implementing GANs in Keras

### Simple GAN for MNIST

```python
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize data to [-1, 1] for better GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 784)

print(f"Training data shape: {x_train.shape}")

# ============ Generator Network ============

def build_generator():
    """
    Generator takes 100-dimensional noise vector as input
    and produces 784-dimensional image.
    """
    model = tf.keras.Sequential([
        # Input: 100-dimensional noise vector
        Input(shape=(100,)),
        
        # Dense layers to upsample
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        
        # Output: 784-dimensional image (28x28 flattened)
        Dense(784, activation='tanh')  # tanh for [-1, 1] output
    ])
    
    return model

# ============ Discriminator Network ============

def build_discriminator():
    """
    Discriminator evaluates whether image is real or generated.
    """
    model = tf.keras.Sequential([
        # Input: 784-dimensional image
        Input(shape=(784,)),
        
        # Dense layers with LeakyReLU for better gradient flow
        Dense(512, use_bias=False),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(256, use_bias=False),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        # Output: Probability of being real
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Print model summaries
print("Generator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

# Compile discriminator
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Create GAN model (generator + discriminator)
discriminator.trainable = False  # Freeze discriminator for GAN training

gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)
gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

print("\nGAN Model Summary:")
gan.summary()
```

### GAN Training Loop

```python
def train_gan(generator, discriminator, gan, x_train, 
              epochs=10000, batch_size=32, latent_dim=100):
    """
    Training loop for GAN.
    Trains discriminator and generator alternately.
    """
    
    # Labels for real and fake data
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # ============ Train Discriminator ============
        
        # Select random real images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise, verbose=0)
        
        # Combine real and fake images
        X = np.vstack([real_images, generated_images])
        y = np.vstack([real_labels, fake_labels])
        
        # Train discriminator
        d_loss = discriminator.train_on_batch(X, y)
        
        # ============ Train Generator ============
        
        # Generate new noise for generator training
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Train generator (via GAN model)
        # Goal: Make discriminator classify generated images as real
        g_loss = gan.train_on_batch(noise, real_labels)
        
        # Print progress
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: D Loss = {d_loss[0]:.4f}, "
                  f"D Acc = {d_loss[1]:.4f}, G Loss = {g_loss:.4f}")
    
    return generator, discriminator

# Train the GAN
generator, discriminator = train_gan(
    generator, discriminator, gan, x_train,
    epochs=10000, batch_size=32, latent_dim=100
)

# Generate new images after training
def generate_images(generator, n_images=5):
    """Generate new images using trained generator."""
    noise = np.random.normal(0, 1, (n_images, 100))
    generated_images = generator.predict(noise)
    
    # Rescale from [-1, 1] to [0, 1] for display
    generated_images = (generated_images + 1) / 2
    
    return generated_images

# Generate and display images
generated = generate_images(generator, n_images=5)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(generated[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle('Generated MNIST Digits')
plt.show()
```

### GAN Architecture Summary

| Component | Layer | Units | Activation | Purpose |
|-----------|-------|-------|------------|---------|
| **Generator** | Input | 100 | - | Noise vector |
| | Dense | 256 | ReLU | Upsample |
| | Dense | 512 | ReLU | Upsample |
| | Dense | 1024 | ReLU | Upsample |
| | Output | 784 | Tanh | Generated image |
| **Discriminator** | Input | 784 | - | Image input |
| | Dense | 512 | LeakyReLU | Feature extraction |
| | Dense | 256 | LeakyReLU | Feature extraction |
| | Output | 1 | Sigmoid | Real/Fake probability |

---

## Generative Adversarial Networks (GANs) - Detailed Guide

### Overview

**Generative Adversarial Networks (GANs)** are a revolutionary type of neural network architecture designed to generate **synthetic data that closely resembles real data**.

GANs have gained significant attention for their ability to generate:
- High-quality images
- Music
- Text
- Video

### GAN Components

GANs consist of **two main components**:

| Component | Goal | Description |
|-----------|------|-------------|
| **Generator** | Create realistic data | Takes random noise as input and generates synthetic data |
| **Discriminator** | Distinguish real from fake | Receives both real and generated data, classifies as real or fake |

**Key Concept:** These two networks are trained simultaneously through a process of **adversarial training**.

### How GANs Work - Step by Step

```
┌─────────────────────────────────────────────────────────┐
│                    GAN Training Process                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Generator takes random noise as input               │
│     ↓                                                    │
│  2. Generator generates synthetic data                  │
│     ↓                                                    │
│  3. Discriminator receives:                             │
│     - Real data (from training set)                     │
│     - Synthetic data (from generator)                   │
│     ↓                                                    │
│  4. Discriminator attempts to classify as real or fake  │
│     ↓                                                    │
│  5. Generator is trained to fool the discriminator      │
│  6. Discriminator is trained to classify accurately     │
│     ↓                                                    │
│  7. Adversarial training loop continues...              │
│     ↓                                                    │
│  Until generator produces data that discriminator       │
│  can no longer distinguish from real data               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Through this adversarial process, both networks improve over time, resulting in the generation of highly realistic data.**

---

### Applications of GANs

GANs have numerous applications across various fields:

| Application | Description | Example |
|-------------|-------------|---------|
| **Image Generation** | Creates realistic images from random noise | Face generation, artwork, landscapes |
| **Image-to-Image Translation** | Converts images from one domain to another | Sketches → Photographs, Day → Night |
| **Text-to-Image Synthesis** | Generates images from textual descriptions | "A red bird on a tree" → Image |
| **Data Augmentation** | Generates synthetic data to augment training datasets | Medical imaging, rare event simulation |
| **Super-Resolution** | Enhances low-resolution images | Old photos → High-resolution |
| **Style Transfer** | Applies artistic styles to images | Photo → Van Gogh style |
| **Video Generation** | Creates realistic video content | Deepfakes, animation |
| **Music Generation** | Composes new music pieces | AI-composed songs |

**These applications demonstrate the versatility and power of GANs in different domains.**

---

## Building a Basic GAN in Keras - Complete Guide

### Step 1: Define Generator and Discriminator Models

```python
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize data to [-1, 1] for better GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 784)

print(f"Training data shape: {x_train.shape}")  # (60000, 784)

# ============ Define the Generator Model ============

def build_generator():
    """
    Generator takes random noise vector as input
    and generates a synthetic image.
    """
    model = tf.keras.Sequential([
        # Input: 100-dimensional noise vector
        Input(shape=(100,)),
        
        # Dense layers to upsample
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        
        # Output: 784-dimensional image (28x28 flattened)
        Dense(784, activation='tanh')  # tanh for [-1, 1] output
    ])
    
    return model

# ============ Define the Discriminator Model ============

def build_discriminator():
    """
    Discriminator takes an image as input and
    outputs a probability indicating whether the image is real or fake.
    """
    model = tf.keras.Sequential([
        # Input: 784-dimensional image
        Input(shape=(784,)),
        
        # Dense layers with LeakyReLU for better gradient flow
        Dense(512, use_bias=False),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(256, use_bias=False),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        # Output: Probability of being real
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Build the two separate models
generator = build_generator()
discriminator = build_discriminator()

# Print model summaries
print("Generator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

# Compile the discriminator
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Step 2: Create the Combined GAN Model

```python
# ============ Create the GAN by combining Generator and Discriminator ============

# Set discriminator to non-trainable when compiling the GAN
# This ensures that only the generator is updated during adversarial training
discriminator.trainable = False

# GAN input: random noise vector
gan_input = Input(shape=(100,))

# Generator creates synthetic image from noise
generated_image = generator(gan_input)

# Discriminator evaluates the generated image
gan_output = discriminator(generated_image)

# Create the combined GAN model
gan = Model(gan_input, gan_output)

# Compile the GAN (discriminator weights are frozen)
gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

print("\nGAN Model Summary:")
gan.summary()
```

### GAN Model Architecture

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| **Generator** | 100D noise vector | 784D image | Generate synthetic data |
| **Discriminator** | 784D image | Probability (0-1) | Classify real vs fake |
| **Combined GAN** | 100D noise vector | Probability (0-1) | Train generator only |

---

## Training the GAN

### Alternating Training Process

The training process involves **updating the discriminator and generator in alternating steps**:

```python
def train_gan(generator, discriminator, gan, x_train,
              epochs=10000, batch_size=32, latent_dim=100):
    """
    Training loop for GAN.
    Trains discriminator and generator in alternating steps.
    """
    
    # Labels for real and fake data
    real_labels = np.ones((batch_size, 1))  # Real = 1
    fake_labels = np.zeros((batch_size, 1))  # Fake = 0
    
    print("Starting GAN training...")
    
    for epoch in range(epochs):
        # ============ Step 1: Train Discriminator ============
        
        # Select random batch of real images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        
        # Generate fake images from random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise, verbose=0)
        
        # Combine real and fake images for discriminator training
        X = np.vstack([real_images, generated_images])
        y = np.vstack([real_labels, fake_labels])
        
        # Train discriminator on both real and generated data
        # Update discriminator weights to improve ability to distinguish real vs fake
        d_loss = discriminator.train_on_batch(X, y)
        
        # ============ Step 2: Train Generator ============
        
        # Generate new random noise for generator training
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Train generator via the combined GAN model
        # Goal: Make discriminator classify generated images as real (label=1)
        # Discriminator weights are frozen during this step
        g_loss = gan.train_on_batch(noise, real_labels)
        
        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: "
                  f"D Loss = {d_loss[0]:.4f}, "
                  f"D Acc = {d_loss[1]:.4f}, "
                  f"G Loss = {g_loss:.4f}")
    
    print("Training complete!")
    return generator, discriminator

# Train the GAN
generator, discriminator = train_gan(
    generator, discriminator, gan, x_train,
    epochs=10000, batch_size=32, latent_dim=100
)
```

### Training Process Summary

| Step | Action | Weights Updated |
|------|--------|-----------------|
| **1. Generate Fake Data** | Generator creates images from noise | - |
| **2. Train Discriminator** | On real images (label=1) and fake images (label=0) | Discriminator |
| **3. Freeze Discriminator** | Set discriminator.trainable = False | - |
| **4. Train Generator** | Via GAN model, fool discriminator (label=1) | Generator |
| **5. Repeat** | Continue adversarial training | Alternating |

**This adversarial training process continues for a specified number of epochs.**

---

## Evaluating and Visualizing GAN Results

### Generate and Visualize Images During Training

To evaluate the GAN, you can **periodically generate and visualize images during training**. This helps monitor the quality of generated images and assess the model's progress.

```python
def generate_images(generator, n_images=5):
    """Generate new images using trained generator."""
    # Generate random noise
    noise = np.random.normal(0, 1, (n_images, 100))
    
    # Generate images from noise
    generated_images = generator.predict(noise, verbose=0)
    
    # Rescale from [-1, 1] to [0, 1] for display
    generated_images = (generated_images + 1) / 2
    
    return generated_images

# Generate images after training
generated = generate_images(generator, n_images=10)

# Display generated images
plt.figure(figsize=(15, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated[i].reshape(28, 28), cmap='gray')
    plt.title(f'Generated Image {i+1}')
    plt.axis('off')

plt.suptitle('Generated MNIST Digits - Trained GAN', fontsize=16)
plt.tight_layout()
plt.show()
```

### Monitoring Training Progress

```python
def plot_training_progress(generator, epoch, noise_dim=100):
    """Generate and plot images at different training stages."""
    noise = np.random.normal(0, 1, (10, noise_dim))
    generated_images = generator.predict(noise, verbose=0)
    generated_images = (generated_images + 1) / 2
    
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.show()

# Example: Monitor at different epochs
# Call this function during training at intervals
# plot_training_progress(generator, epoch=0)      # Initial random noise
# plot_training_progress(generator, epoch=1000)   # Early training
# plot_training_progress(generator, epoch=5000)   # Mid training
# plot_training_progress(generator, epoch=10000)  # Final results
```

**By visualizing these images at different stages of training, you can observe how the quality of the generated images improves over time.**

---

## Summary: Generative Adversarial Networks (GANs)

### Key Concepts

| Concept | Description |
|---------|-------------|
| **GANs** | Revolutionary neural network architecture for generating synthetic data |
| **Generator** | Creates realistic data from random noise |
| **Discriminator** | Distinguishes between real and generated data |
| **Adversarial Training** | Two networks trained simultaneously in competition |

### Training Process

| Step | Description |
|------|-------------|
| **1** | Generator takes random noise as input |
| **2** | Generator generates synthetic data |
| **3** | Discriminator receives real and synthetic data |
| **4** | Discriminator classifies data as real or fake |
| **5** | Generator trained to fool discriminator |
| **6** | Discriminator trained to classify accurately |
| **7** | Loop continues until generator produces indistinguishable data |

### Applications

| Domain | Application |
|--------|-------------|
| **Image Generation** | Create realistic images from random noise |
| **Image-to-Image Translation** | Convert between domains (sketch → photo) |
| **Text-to-Image** | Generate images from text descriptions |
| **Data Augmentation** | Generate synthetic training data |

### Key Takeaways

- **GANs** are revolutionary neural networks designed for generating synthetic data that closely resembles real data
- GANs consist of **two main components**: generator and discriminator
- **Generator's goal**: Create realistic data
- **Discriminator's goal**: Distinguish between real and generated data
- Two networks are trained simultaneously through **adversarial training**
- **Adversarial training loop** continues until generator produces data that discriminator can no longer distinguish from real data
- **Applications** include image generation, image-to-image translation, text-to-image synthesis, and data augmentation
- **Visualization during training** helps monitor quality and assess progress

GANs represent a significant breakthrough in generative modeling, enabling the creation of highly realistic synthetic data across multiple domains.

---

## Applications of Unsupervised Learning

### Autoencoder Applications

| Application | Description | Example |
|-------------|-------------|---------|
| **Dimensionality Reduction** | Compress data to lower dimensions | Feature extraction for classification |
| **Denoising** | Remove noise from data | Image denoising, audio cleanup |
| **Anomaly Detection** | Identify outliers | Fraud detection, defect detection |
| **Data Compression** | Efficient data representation | Image compression |
| **Pretraining** | Initialize weights for supervised tasks | Transfer learning |

### GAN Applications

| Application | Description | Example |
|-------------|-------------|---------|
| **Image Generation** | Create realistic images | Face generation, art creation |
| **Image-to-Image Translation** | Convert between domains | Day→Night, Sketch→Photo |
| **Super-Resolution** | Enhance image resolution | Low-res to high-res conversion |
| **Data Augmentation** | Generate training data | Medical imaging, rare events |
| **Style Transfer** | Apply artistic styles | Photo→Painting style |
| **Text-to-Image** | Generate images from text | DALL-E, Stable Diffusion |

---

## Diffusion Models

### Overview

**Diffusion Models** are a class of generative models that have recently gained popularity for their ability to produce **high-quality synthetic data**. They are used in various applications, including image generation and enhancement.

### What are Diffusion Models?

**Diffusion Models** are **probabilistic models** that generate data by iteratively refining a noisy initial sample.

**Key Concept:** They start with random noise and gradually apply a series of transformations to produce a coherent data sample.

**Physical Analogy:** The process is akin to simulating the physical process of **diffusion**, where particles spread out from regions of high concentration to regions of low concentration.

---

### How Diffusion Models Work

Diffusion models work by defining **two processes**:

```
┌─────────────────────────────────────────────────────────┐
│              Diffusion Model Process                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Forward Process (Adding Noise):                        │
│  Original Data → Noisy Data (gradually increasing noise)│
│                                                          │
│  Reverse Process (Denoising):                           │
│  Random Noise → Denoised Data (gradually removing noise)│
│                                                          │
└─────────────────────────────────────────────────────────┘
```

| Process | Description | Purpose |
|---------|-------------|---------|
| **Forward Process** | Adds noise to the data over a series of steps | Simulates diffusion process |
| **Reverse Process** | Learns to denoise the data step by step | Reconstructs original data from noise |

**The Reverse Denoising Process:**
- This is what allows diffusion models to generate high-quality samples from random noise
- Model learns to reverse the noise addition process
- Iteratively refines noisy input until clean data is produced

---

### Applications of Diffusion Models

| Application | Description | Example |
|-------------|-------------|---------|
| **Image Generation** | Creating realistic images from random noise | Art generation, face synthesis |
| **Image Denoising** | Removing noise from images to enhance quality | Photo restoration, medical imaging |
| **Data Augmentation** | Generating synthetic data to augment training datasets | Training data expansion, rare event simulation |
| **Image Inpainting** | Filling in missing parts of images | Photo editing, restoration |
| **Super-Resolution** | Enhancing image resolution | Low-res to high-res conversion |
| **Text-to-Image** | Generating images from text descriptions | DALL-E 2, Stable Diffusion |

**These applications demonstrate the versatility and power of diffusion models in various domains.**

---

## Building a Basic Diffusion Model in Keras

### Model Architecture

```python
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# ============ Step 1: Define the Diffusion Model Architecture ============

def build_diffusion_model(input_shape=(28, 28, 1)):
    """
    Define a simple Convolutional Neural Network (CNN) as diffusion model.
    Takes noisy 28x28 image as input, outputs denoised image of same size.
    """
    
    # Input: Noisy 28x28 image
    inputs = Input(shape=input_shape)
    
    # Encoder: Process through convolutional and pooling layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)  # 14x14
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)  # 7x7
    
    # Bottleneck
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    # Decoder: Reconstruct through dense and upsampling layers
    x = Dense(7 * 7 * 64, activation='relu')(x)
    x = Reshape((7, 7, 64))(x)
    
    x = UpSampling2D((2, 2))(x)  # 14x14
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = UpSampling2D((2, 2))(x)  # 28x28
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output: Denoised 28x28 image
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create model
    model = Model(inputs, outputs)
    
    return model

# Build the diffusion model
diffusion_model = build_diffusion_model()

# Compile the model using Adam optimizer and binary crossentropy loss
diffusion_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['mse']  # Mean Squared Error for image quality
)

# Print model summary
diffusion_model.summary()
```

### Diffusion Model Architecture Summary

| Layer | Type | Output Shape | Purpose |
|-------|------|--------------|---------|
| **Input** | Input | (28, 28, 1) | Noisy image input |
| **Conv1** | Conv2D | (28, 28, 32) | Feature extraction |
| **Pool1** | MaxPooling | (14, 14, 32) | Downsample |
| **Conv2** | Conv2D | (14, 14, 64) | Feature extraction |
| **Pool2** | MaxPooling | (7, 7, 64) | Downsample |
| **Flatten** | Flatten | (3136,) | Flatten to vector |
| **Dense1** | Dense | (128,) | Bottleneck |
| **Dense2** | Dense | (3136,) | Expand |
| **Reshape** | Reshape | (7, 7, 64) | Reshape for upsampling |
| **UpSample1** | UpSampling | (14, 14, 64) | Upsample |
| **Conv3** | Conv2D | (14, 14, 64) | Feature refinement |
| **UpSample2** | UpSampling | (28, 28, 64) | Upsample |
| **Conv4** | Conv2D | (28, 28, 32) | Feature refinement |
| **Output** | Conv2D | (28, 28, 1) | Denoised image |

---

## Preparing MNIST Dataset and Adding Noise

```python
# ============ Step 2: Load and Preprocess MNIST Dataset ============

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to add channel dimension (28, 28) → (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Training data shape: {x_train.shape}")  # (60000, 28, 28, 1)
print(f"Test data shape: {x_test.shape}")  # (10000, 28, 28, 1)

# ============ Step 3: Add Random Noise to Images ============

def add_noise(images, noise_factor=0.5):
    """
    Add random Gaussian noise to images.
    
    Args:
        images: Input images array
        noise_factor: Controls the amount of noise to add
    
    Returns:
        Noisy images clipped to [0, 1] range
    """
    # Generate random noise
    noise = np.random.normal(0, noise_factor, images.shape)
    
    # Add noise to images
    noisy_images = images + noise
    
    # Clip values to valid range [0, 1]
    noisy_images = np.clip(noisy_images, 0, 1)
    
    return noisy_images

# Add noise to training and test data
noise_factor = 0.5
x_train_noisy = add_noise(x_train, noise_factor)
x_test_noisy = add_noise(x_test, noise_factor)

# Visualize noisy vs original images
plt.figure(figsize=(12, 4))

# Display original images
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')

# Display noisy images
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title(f'Noisy')
    plt.axis('off')

plt.suptitle('Original vs Noisy MNIST Images', fontsize=16)
plt.tight_layout()
plt.show()
```

---

## Training the Diffusion Model

```python
# ============ Step 4: Train the Diffusion Model ============

# Train model using noisy images as input and original images as target
# This teaches the model to denoise the images
history = diffusion_model.fit(
    x_train_noisy, x_train,  # Input: Noisy, Target: Clean
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# ============ Step 5: Evaluate the Diffusion Model ============

# Evaluate on test data
test_loss = diffusion_model.evaluate(x_test_noisy, x_test, verbose=0)
print(f"Test Loss: {test_loss[0]:.4f}")
print(f"Test MSE: {test_loss[1]:.4f}")

# ============ Step 6: Generate Denoised Images ============

# Generate predictions (denoised images)
denoised_images = diffusion_model.predict(x_test_noisy)

# ============ Step 7: Visualize Results ============

# Compare original, noisy, and denoised images side by side
plt.figure(figsize=(15, 6))

for i in range(5):
    # Original image
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')
    
    # Noisy image
    plt.subplot(3, 5, i + 6)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title(f'Noisy')
    plt.axis('off')
    
    # Denoised image
    plt.subplot(3, 5, i + 11)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Denoised')
    plt.axis('off')

plt.suptitle('Diffusion Model: Original vs Noisy vs Denoised', fontsize=16)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Fine-Tuning the Diffusion Model

### Improving Model Performance

You can further improve the diffusion model by **fine-tuning** it. Fine-tuning involves:
- Adjusting the model's parameters
- Retraining for additional epochs
- Unfreezing specific layers for targeted training

### Fine-Tuning Implementation

```python
# ============ Fine-Tuning the Diffusion Model ============

# Unfreeze the last 4 layers of the diffusion model
# This allows fine-tuning of the decoder layers
for layer in diffusion_model.layers[-4:]:
    layer.trainable = True

# Recompile the model after changing trainable status
diffusion_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['mse']
)

# Print fine-tuning configuration
print("Fine-tuning configuration:")
for i, layer in enumerate(diffusion_model.layers):
    print(f"{i}: {layer.name} - Trainable: {layer.trainable}")

# Train the model again for a few more epochs
# This fine-tunes the model for better denoising results
fine_tune_history = diffusion_model.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Evaluate fine-tuned model
fine_tune_loss = diffusion_model.evaluate(x_test_noisy, x_test, verbose=0)
print(f"Fine-tuned Test Loss: {fine_tune_loss[0]:.4f}")
print(f"Fine-tuned Test MSE: {fine_tune_loss[1]:.4f}")

# Generate denoised images with fine-tuned model
denoised_fine_tuned = diffusion_model.predict(x_test_noisy)

# Compare before and after fine-tuning
plt.figure(figsize=(15, 8))

for i in range(5):
    # Original
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')
    
    # Noisy
    plt.subplot(4, 5, i + 6)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title(f'Noisy')
    plt.axis('off')
    
    # Denoised (before fine-tuning)
    plt.subplot(4, 5, i + 11)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Denoised (Before)')
    plt.axis('off')
    
    # Denoised (after fine-tuning)
    plt.subplot(4, 5, i + 16)
    plt.imshow(denoised_fine_tuned[i].reshape(28, 28), cmap='gray')
    plt.title(f'Denoised (After)')
    plt.axis('off')

plt.suptitle('Diffusion Model: Before vs After Fine-Tuning', fontsize=16)
plt.tight_layout()
plt.show()
```

---

## Summary: Diffusion Models

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Diffusion Models** | Probabilistic generative models that produce high-quality synthetic data |
| **Forward Process** | Adds noise to data over a series of steps |
| **Reverse Process** | Learns to denoise data step by step |
| **Denoising** | Core mechanism for generating samples from noise |

### Applications

| Application | Description |
|-------------|-------------|
| **Image Generation** | Creating realistic images from random noise |
| **Image Denoising** | Removing noise from images to enhance quality |
| **Data Augmentation** | Generating synthetic data to augment training datasets |

### Model Architecture

| Component | Purpose |
|-----------|---------|
| **Input Layer** | Accepts noisy image (e.g., 28×28×1) |
| **Convolutional Layers** | Extract features from noisy input |
| **Pooling Layers** | Downsample spatial dimensions |
| **Dense Layers** | Bottleneck representation |
| **Upsampling Layers** | Restore spatial dimensions |
| **Output Layer** | Produces denoised image |

### Training Process

| Step | Description |
|------|-------------|
| **1. Prepare Data** | Load and normalize dataset |
| **2. Add Noise** | Generate noisy versions of images |
| **3. Train Model** | Use noisy images as input, clean images as target |
| **4. Evaluate** | Assess denoising performance |
| **5. Fine-Tune** | Unfreeze layers and retrain for improvement |

### Key Takeaways

- **Diffusion models** are powerful tools for generative tasks
- Capable of producing **high-quality data samples** and enhancing image quality
- **Probabilistic models** that generate data by iteratively refining noisy initial samples
- Process simulates physical **diffusion** (particles spreading from high to low concentration)
- Work by defining **forward process** (adding noise) and **reverse process** (denoising)
- **Reverse denoising process** allows generation of high-quality samples from random noise
- **Applications** include image generation, denoising, and data augmentation
- **Fine-tuning** by unfreezing layers can improve denoising performance

Diffusion models represent a significant advancement in generative modeling, offering high-quality sample generation and versatile applications across various domains.

---

## Summary: Unsupervised Learning in Keras

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Unsupervised Learning** | Finding patterns in data without labels |
| **Clustering** | Grouping similar data points together |
| **Association** | Finding relationships between variables |
| **Dimensionality Reduction** | Reducing number of variables while preserving information |
| **Autoencoders** | Neural networks for dimensionality reduction via encoder-decoder |
| **GANs** | Two networks (generator/discriminator) competing in zero-sum game |

### Autoencoder Components

| Component | Purpose |
|-----------|---------|
| **Encoder** | Compresses input into latent space representation |
| **Decoder** | Reconstructs input from latent space |
| **Latent Space** | Compressed, meaningful representation |

### GAN Components

| Component | Purpose |
|-----------|---------|
| **Generator** | Generates new data instances resembling training data |
| **Discriminator** | Evaluates authenticity of generated data |
| **Adversarial Training** | Generator tries to fool discriminator; discriminator tries to detect fakes |

### Key Takeaways

- **Unsupervised learning** finds patterns in data without labels or predefined outcomes
- **Three categories**: Clustering, Association, Dimensionality Reduction
- **Autoencoders** consist of encoder (compression) and decoder (reconstruction)
- **GANs** consist of generator and discriminator competing in zero-sum game
- **Generator** creates new data instances that resemble training data
- **Discriminator** evaluates authenticity of generated data
- **Adversarial process** leads to increasingly realistic generated data
- Both autoencoders and GANs are powerful tools for **generative modeling** and **feature learning**

---

## TensorFlow for Unsupervised Learning

### Overview

**Unsupervised learning** is a type of machine learning where the model is trained on **data without labeled responses**. This approach is essential for discovering hidden patterns and structures within the data.

### Common Applications of Unsupervised Learning

| Application | Description | Use Cases |
|-------------|-------------|-----------|
| **Clustering** | Grouping similar data points together | Customer segmentation, image compression |
| **Dimensionality Reduction** | Reducing number of features while retaining important information | Data visualization, feature extraction |
| **Anomaly Detection** | Identifying unusual data points that don't fit the general pattern | Fraud detection, defect detection |

**These applications are widely used in various domains such as:**
- **Customer Segmentation**: Group customers by behavior patterns
- **Image Compression**: Reduce image size while preserving quality
- **Fraud Detection**: Identify suspicious transactions

### TensorFlow for Unsupervised Learning

TensorFlow provides powerful tools for implementing unsupervised learning models:

| Feature | Benefit |
|---------|---------|
| **Flexible Architecture** | Handle various unsupervised learning tasks efficiently |
| **Extensive Libraries** | Access to clustering, autoencoders, and more |
| **GPU Acceleration** | Fast training on large datasets |
| **Integration with Keras** | Easy model building and training |

---

## Building a K-Means Clustering Model with TensorFlow

### K-Means Algorithm

**K-Means** is a popular clustering technique that groups data points into K clusters based on similarity.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# ============ Step 1: Load and Preprocess MNIST Dataset ============

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to vectors (flatten 28x28 to 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(f"Training data shape: {x_train.shape}")  # (60000, 784)

# ============ Step 2: Apply K-Means Clustering ============

# Number of clusters
K = 10  # Group images into 10 clusters (digits 0-9)

# Initialize centroids randomly
def initialize_centroids(data, k):
    """Randomly select k data points as initial centroids."""
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

# Assign data points to nearest centroid
def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    distances = tf.norm(data[:, tf.newaxis] - centroids, axis=2)
    return tf.argmin(distances, axis=1)

# Update centroids based on cluster assignments
def update_centroids(data, assignments, k):
    """Calculate new centroids as mean of assigned points."""
    centroids = tf.Variable(tf.zeros_like(centroids_init))
    for i in range(k):
        cluster_points = tf.boolean_mask(data, assignments == i)
        if len(cluster_points) > 0:
            centroids[i].assign(tf.reduce_mean(cluster_points, axis=0))
    return centroids

# K-Means training loop
def kmeans(data, k, max_iterations=100):
    """Perform K-Means clustering."""
    centroids = initialize_centroids(data, k)
    centroids = tf.Variable(centroids, dtype=tf.float32)
    
    for iteration in range(max_iterations):
        # Assign clusters
        assignments = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(data, assignments, k)
        
        # Check for convergence
        if tf.reduce_all(tf.equal(centroids, new_centroids)):
            print(f"Converged at iteration {iteration}")
            break
        
        centroids.assign(new_centroids)
    
    return centroids, assignments

# Run K-Means clustering
print("Running K-Means clustering...")
centroids, cluster_assignments = kmeans(x_train, K, max_iterations=50)

print(f"Clustering complete! Found {K} clusters.")

# ============ Step 3: Visualize Clustering Results ============

# Display a few images from each cluster
plt.figure(figsize=(15, 10))

for cluster in range(K):
    # Get indices of images in this cluster
    cluster_indices = np.where(cluster_assignments == cluster)[0]
    
    # Display first 5 images from each cluster
    for i, idx in enumerate(cluster_indices[:5]):
        plt.subplot(K, 5, cluster * 5 + i + 1)
        plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        if i == 0:
            plt.title(f'Cluster {cluster}', fontsize=12, fontweight='bold')

plt.suptitle('K-Means Clustering Results on MNIST Dataset', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate cluster statistics
print("\nCluster Statistics:")
for cluster in range(K):
    count = np.sum(cluster_assignments == cluster)
    print(f"Cluster {cluster}: {count} images ({count/len(x_train)*100:.1f}%)")
```

### K-Means Clustering Summary

| Step | Description |
|------|-------------|
| **1. Initialize** | Randomly select K data points as initial centroids |
| **2. Assign** | Assign each data point to nearest centroid |
| **3. Update** | Recalculate centroids as mean of assigned points |
| **4. Repeat** | Continue until convergence or max iterations |

---

## Building an Autoencoder for Dimensionality Reduction

### Autoencoder Architecture

An **autoencoder** is a type of neural network used to learn efficient representations of data by:
1. **Compressing** the input into a lower-dimensional space (encoding)
2. **Reconstructing** it back to original dimensions (decoding)

This process effectively **reduces the dimensionality of the data**.

```python
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ============ Step 1: Load and Preprocess MNIST Dataset ============

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(f"Training data shape: {x_train.shape}")  # (60000, 784)

# ============ Step 2: Define the Autoencoder Model ============

def build_autoencoder(input_dim=784, encoding_dim=32):
    """
    Define an autoencoder with:
    - Input layer
    - Encoding layer
    - Bottleneck layer (compressed representation)
    - Decoding layer
    - Output layer
    """
    
    # Input layer
    input_img = Input(shape=(input_dim,))
    
    # Encoding layer: Compress to lower dimension
    encoded = Dense(256, activation='relu')(input_img)
    encoded = LayerNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    
    encoded = Dense(128, activation='relu')(encoded)
    encoded = LayerNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    
    # Bottleneck layer: Compressed representation
    bottleneck = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoding layer: Reconstruct from compressed representation
    decoded = Dense(128, activation='relu')(bottleneck)
    decoded = LayerNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)
    
    decoded = Dense(256, activation='relu')(decoded)
    decoded = LayerNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)
    
    # Output layer: Reconstruct original input
    output_img = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Create autoencoder model
    autoencoder = Model(input_img, output_img)
    
    # Create encoder model (for extracting compressed representations)
    encoder = Model(input_img, bottleneck)
    
    return autoencoder, encoder

# Build autoencoder with 32-dimensional bottleneck
autoencoder, encoder = build_autoencoder(input_dim=784, encoding_dim=32)

# Compile the model
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['mse']
)

# Print model summary
print("\nAutoencoder Summary:")
autoencoder.summary()

# ============ Step 3: Train the Autoencoder ============

# Train autoencoder to reconstruct input data from compressed representation
history = autoencoder.fit(
    x_train, x_train,  # Input = Output for autoencoder
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Evaluate on test data
test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"\nTest Loss: {test_loss[0]:.4f}")
print(f"Test MSE: {test_loss[1]:.4f}")

# ============ Step 4: Visualize Compressed Representations ============

# Extract compressed representations from bottleneck layer
print("\nExtracting compressed representations...")
compressed_data = encoder.predict(x_test)

print(f"Compressed data shape: {compressed_data.shape}")  # (10000, 32)

# ============ Step 5: Use t-SNE for 2D Visualization ============

# t-SNE (t-Distributed Stochastic Neighbor Embedding)
# Technique for visualizing high-dimensional data in 2D

print("Applying t-SNE for 2D visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
data_2d = tsne.fit_transform(compressed_data)

print(f"2D data shape: {data_2d.shape}")  # (10000, 2)

# ============ Step 6: Plot 2D Representation ============

plt.figure(figsize=(12, 10))

# Create scatter plot showing how data points are grouped in lower dimensional space
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                      c=y_test, cmap='viridis', alpha=0.6, s=10)

# Add colorbar to show digit labels
plt.colorbar(scatter, label='Digit Label')
plt.title('t-SNE Visualization of Autoencoder Compressed MNIST Data', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# ============ Step 7: Visualize Reconstruction Quality ============

# Generate reconstructions
reconstructed = autoencoder.predict(x_test)

# Compare original vs reconstructed images
plt.figure(figsize=(15, 6))

for i in range(10):
    # Original image
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original: {y_test[i]}')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed')
    plt.axis('off')

plt.suptitle('Original vs Reconstructed MNIST Images', fontsize=16)
plt.tight_layout()
plt.show()

# ============ Step 8: Plot Training History ============

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Reconstruction MSE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Autoencoder for Dimensionality Reduction Summary

| Component | Purpose | Output Dimension |
|-----------|---------|------------------|
| **Input Layer** | Accept original data | 784 dimensions |
| **Encoding Layers** | Compress data progressively | 784 → 256 → 128 |
| **Bottleneck** | Compressed representation | 32 dimensions |
| **Decoding Layers** | Reconstruct from compressed | 32 → 128 → 256 |
| **Output Layer** | Reconstruct original input | 784 dimensions |

### t-SNE Visualization

| Parameter | Description | Value Used |
|-----------|-------------|------------|
| **n_components** | Target dimension for visualization | 2 |
| **perplexity** | Balance between local/global structure | 30 |
| **n_iter** | Number of iterations | 1000 |
| **random_state** | For reproducibility | 42 |

**The resulting scatter plot shows how data points are grouped in the lower-dimensional space, revealing clusters of similar digits.**

---

## Summary: TensorFlow for Unsupervised Learning

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Unsupervised Learning** | Model trained on data without labeled responses |
| **Clustering** | Grouping similar data points together (K-Means) |
| **Dimensionality Reduction** | Reducing features while retaining information (Autoencoders) |
| **Anomaly Detection** | Identifying unusual data points |

### Applications

| Domain | Application | Technique |
|--------|-------------|-----------|
| **Customer Segmentation** | Group customers by behavior | K-Means Clustering |
| **Image Compression** | Reduce image size | Autoencoders |
| **Fraud Detection** | Identify suspicious transactions | Anomaly Detection |
| **Data Visualization** | Visualize high-dimensional data | t-SNE + Autoencoders |

### TensorFlow Tools

| Tool | Purpose |
|------|---------|
| **K-Means** | Clustering similar data points |
| **Autoencoders** | Learn efficient data representations |
| **t-SNE** | Visualize high-dimensional data in 2D |
| **TensorFlow Clustering** | Built-in clustering algorithms |

### Key Takeaways

- **Unsupervised learning** is a powerful approach for discovering hidden patterns in data
- **TensorFlow provides robust tools** to facilitate unsupervised learning tasks
- **Common applications** include clustering, dimensionality reduction, and anomaly detection
- **K-Means clustering** groups similar data points into K clusters
- **Autoencoders** compress data into lower-dimensional space and reconstruct it
- **t-SNE** enables visualization of high-dimensional compressed data in 2D
- **Applications** are widely used in customer segmentation, image compression, and fraud detection

TensorFlow's comprehensive unsupervised learning capabilities make it an excellent choice for discovering hidden patterns and structures in unlabeled data.

---

*End of Module 4: Unsupervised Learning and Generative Models in Keras*
