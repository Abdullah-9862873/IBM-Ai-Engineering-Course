# Shallow vs Deep Neural Networks

## Overview
- Understanding shallow networks is the foundation for understanding deep networks
- Key differences in architecture and capabilities

## Shallow Neural Networks

### Definition
- Neural network with only **one or two hidden layers**
- Simpler architecture
- Easier to understand and train

### Input Type
- Takes only **vectors** as input
- Requires pre-processed/feature-engineered data

## Deep Neural Networks

### Definition
- Neural network with **three or more hidden layers**
- Large number of neurons in each layer
- More complex architecture

### Input Type
- Can take **raw data** directly:
  - Images
  - Text
  - Audio
- Automatically extracts necessary features

## Key Differences

| Aspect | Shallow NN | Deep NN |
|--------|-----------|---------|
| Hidden Layers | 1-2 | 3+ |
| Neurons | Fewer | Many |
| Input Type | Vectors only | Raw data |
| Feature Extraction | Manual | Automatic |
| Training Complexity | Lower | Higher |

## Why Deep Learning Boomed Recently

### Factor 1: Advancements in the Field
- **ReLU activation function**克服了梯度消失问题，帮助创建非常深的神经网络

### Factor 2: Availability of Data
- Large amounts of data are readily available
- Deep networks work best with **large datasets**
- Conventional ML improves only up to a point with more data
- Deep learning continues to improve with more data

### Factor 3: Computational Power
- **GPUs** enable training deep networks quickly
- Training time: hours instead of days/weeks
- Faster experimentation and prototyping

## Key Takeaways
- Shallow NN: 1-2 hidden layers, vector input only
- Deep NN: 3+ hidden layers, can process raw data
- Deep learning boom due to: ReLU, big data, and GPU computing

---

# Convolutional Neural Networks (CNN)

## Overview
- CNNs are similar to regular neural networks
- Made up of neurons with weights and biases to optimize
- **Key assumption:** Inputs are images
- Best for: Image recognition, object detection, computer vision

## Why CNNs?

### Advantages
- Explicit assumption about image inputs
- More efficient forward propagation
- Fewer parameters than regular NNs
- Prevents overfitting

## CNN Architecture

### Layer Types (in order):
1. Convolutional Layer
2. ReLU Layer
3. Pooling Layer
4. Fully Connected Layer (repeat as needed)
5. Output Layer

## Input Data Format

### Grayscale Images
- n × m × 1
- (height × width × 1)

### Color Images
- n × m × 3
- (height × width × 3)
- 3 = RGB channels

## Convolutional Layer

### Process
1. Define **filters** (kernels)
2. Compute convolution between filters and each image channel
3. Slide filter over image (stride by stride)
4. Compute dot product at each position
5. Apply ReLU activation

### Filters
- Small matrix (e.g., 2×2, 3×3)
- Multiple filters can be used
- More filters = better spatial preservation

### Why Not Flatten Images?
- Flattening creates massive number of parameters
- Computational expensive
- More parameters = more overfitting risk

## ReLU Layer
- Part of convolutional layer
- Passes only positive values
- Turns negative values to zero

## Pooling Layer

### Purpose
- Reduce spatial dimensions
- Provide spatial variance
- Helps recognize objects even if slightly different

### Types

#### Max Pooling (Most Common)
- Keep the highest value in each scanned area

#### Average Pooling
- Compute the average of each scanned area

### Parameters
- Pool size (e.g., 2×2)
- Stride (e.g., 2)

## Fully Connected Layer

### Process
1. Flatten output from previous layer
2. Connect every node to every other node
3. Output n-dimensional vector (n = number of classes)

### Example
- Digit classification: n = 10 (digits 0-9)

## Building CNN with Keras

### Example Code
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create model
model = Sequential()

# Input layer (128x128 color images)
model.add(Conv2D(16, (2,2), strides=(1,1), activation='relu', input_shape=(128,128,3)))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Second conv + pooling
model.add(Conv2D(32, (2,2), strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Flatten
model.add(Flatten())

# Fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10)

# Predict
predictions = model.predict(X_test)
```

### Key Keras Layers
- `Conv2D`: Convolutional layer
- `MaxPooling2D`: Max pooling
- `AveragePooling2D`: Average pooling
- `Flatten`: Flatten for fully connected
- `Dense`: Fully connected layer

## Key Takeaways
- CNNs assume inputs are images
- Convolution extracts features using filters
- Pooling reduces dimensions and adds robustness
- Fully connected layer produces final classification
- Keras makes building CNNs straightforward

---

# Recurrent Neural Networks (RNN)

## Overview
- RNNs handle sequential/pattern data
- Unlike traditional NNs, data points are NOT independent
- Has loops to remember previous outputs

## The Problem RNN Solves

### Traditional Neural Networks
- Treat data points as independent instances
- Cannot handle sequential data (movie scenes, text, time series)

### RNN Solution
- Takes current input + output from previous time step
- Maintains "memory" of previous data
- Considers temporal dimension

## RNN Architecture

### How It Works
- At time t=0: Input x₀ → Output a₀
- At time t=1: Input (x₁ + a₀) → Output a₁
- At time t=2: Input (x₂ + a₁) → Output a₂
- Continues sequentially

### Key Feature
- Loops allow information to persist
- Previous outputs become current inputs

## What RNNs Can Model
- Text/sequences
- Genomes
- Handwriting
- Stock markets
- Time series data
- Video/movie scenes

## Long Short-Term Memory (LSTM)

### Overview
- Popular type of RNN
- Special architecture for long-term dependencies
- Solves vanishing gradient problem in standard RNNs

### Applications
- **Image generation:** Generate new images from training data
- **Handwriting generation:** Create realistic cursive text
- **Image captioning:** Describe images automatically
- **Video captioning:** Describe video content automatically
- **Machine translation**
- **Speech recognition**

## Key Takeaways
- RNNs handle sequential/dependent data
- Loops allow memory of previous outputs
- LSTM is a powerful RNN variant
- Used for text, genomes, handwriting, time series, video

---

# Transformers

## Overview
- Neural network architecture that revolutionized NLP and beyond
- Powers AI tools like ChatGPT, Gemini
- Excels at capturing long-range dependencies in sequential data

## Types of Transformers

### Generative Pre-trained Transformers (GPT)
- Used in ChatGPT

### Bidirectional Transformers (BERT)
- Used in Google Search and Google Translate

### Image Transformers
- Used in Adobe Photoshop

## Key Innovation: Attention Mechanism

### Self-Attention Mechanism (for text)
Allows model to weigh importance of different parts of input when processing each token

#### Three Parts:
1. **Query (Q), Key (K), Value (V) vectors**
   - Q: Word we're focusing on
   - K: All other words
   - V: Information to pass to next layer

2. **Attention Scores**
   - Dot product of query and key vectors
   - Indicates relevance of each token to current token

3. **Weighted Sum**
   - Normalize scores with softmax
   - Compute weighted sum of value vectors
   - Creates contextualized vector representation

### Self-Attention Process (Example: "the dog runs")
1. Each word → vector/embedding
2. Add position vector for positional info
3. Generate Q, K, V from each embedding
4. Calculate dot product (Q × K) for attention scores
5. Scale and apply softmax
6. Compute weighted sum of value vectors
7. Create new contextualized representation

## Cross-Attention Mechanism (for text-to-image)

### Purpose
- Allows one type of data (text) to influence generation of another type (image)

### Process (e.g., DALL-E)
1. Self-attention learns contextualized embeddings from text
2. Pass through transformer encoder → get queries (Q)
3. Cross-attention uses Q to guide image generation
4. Auto-regressive model predicts next image parts

### Capabilities
- Synthesize new images (not just stored images)
- Create novel combinations (e.g., horse with bamboo legs)
- Combine unrelated concepts (e.g., turtle driving a car)
- Generate multiple variants from same prompt

## Advantages Over RNNs

### Parallelization
- Can process data in parallel (unlike sequential RNNs)
- Significantly faster training

### Long-Range Dependencies
- Excel at handling complex relationships across long sequences
- Better than RNNs for long contexts

### Versatility
- Machine translation
- Text summarization
- Question answering
- Text-to-image generation
- Text generation

## Limitations

### Data Requirements
- Require huge amount of training data
- Need large datasets to generalize well

### Bias
- Inherit bias from training data
- Data-driven learning without explicit rules

## Key Takeaways
- Transformers revolutionized NLP
- Self-attention: Q, K, V vectors + weighted sum
- Cross-attention: Text-to-image generation
- Parallel processing = faster training
- Limitations: Need large data, can inherit bias

---

# Autoencoders

## Overview
- **Unsupervised** deep learning model
- Data compression algorithm
- Learns compression/decompression automatically from data
- Built using neural networks

## Key Properties

### Data-Specific
- Only compress data similar to training data
- Example: Car-trained autoencoder won't compress buildings well
- Learns features specific to training data

### Loss Function
- Uses backpropagation
- Target = input (trying to reconstruct itself)
- Learns approximation of identity function

## Architecture

### Encoder
- Compresses input into smaller representation
- Finds optimal compressed representation

### Decoder
- Reconstructs original input from compressed representation

### Example Flow
1. Input image → Encoder → Compressed representation → Decoder → Reconstructed image

## Advantages Over PCA

### Nonlinear Transformations
- Uses nonlinear activation functions
- Can learn more interesting projections than PCA
- PCA handles only linear transformations

## Applications

### Data Denoising
- Remove noise from data
- Learn clean representations

### Dimensionality Reduction
- Reduce data dimensions
- Useful for data visualization

## Restricted Boltzmann Machines (RBM)

### Overview
- Popular type of autoencoder
- Unsupervised neural network

### Applications

#### Fix Imbalanced Data
- Learn distribution of minority class
- Generate more data points of minority class
- Transform imbalanced → balanced dataset

#### Estimate Missing Values
- Learn input distribution
- Fill in missing values in datasets

#### Feature Extraction
- Automatic feature extraction
- Especially useful for unstructured data

## Key Takeaways
- Autoencoders are unsupervised (target = input)
- Data-specific compression
- Encoder → Compressed → Decoder
- Applications: denoising, dimensionality reduction
- RBMs: handle imbalanced data, missing values, feature extraction

---

# Pretrained Models

## Overview
- Neural network models previously trained on large datasets
- Can be used for new tasks without retraining
- Leverage learned features from large datasets (e.g., ImageNet)

## Common Pretrained Models

### ImageNet Models
- VGG16
- ResNet
- Inception

## Using Pretrained Models as Feature Extractors

### Process
1. Load pretrained model (e.g., VGG16)
2. Remove top layers (fully connected layers)
3. Use model to extract features from new data
4. Use extracted features for downstream tasks

### Applications of Extracted Features
- Clustering
- Visualization
- Feed into simpler ML models
- Dimensionality reduction

## Benefits

### No Additional Training Required
- Faster to implement
- Less computational power needed

### Efficient Use of Learned Features
- Rich hierarchical representations from convolutional layers
- Useful for various tasks

### Suitable for Limited Resources
- Ideal when computational resources are limited
- Works when new dataset is too small for full training

## Fine-Tuning

### What is Fine-Tuning?
- Unfreeze few top layers of pretrained model
- Jointly train newly added layers with top layers
- Adjust pretrained weights to match new data

### When to Use
- New dataset differs from original training data
- Need better performance than feature extraction alone

### Transfer Learning
- Adapting pretrained model to new related task
- Useful when you don't have enough data to train from scratch

## Keras Implementation Example

### Step 1: Load Pretrained Model
```python
from keras.applications import VGG16

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

### Step 2: Add Custom Layers
```python
from keras.models import Model
from keras.layers import Flatten, Dense

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)
```

### Step 3: Compile
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
```

### Step 4: Train
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('data/', target_size=(224, 224), batch_size=32, class_mode='binary')

model.fit(train_generator, epochs=10)
```

### Step 5: Fine-Tune (Optional)
```python
# Unfreeze top layers
for layer in base_model.layers[:5]:
    layer.trainable = False

# Retrain
model.fit(train_generator, epochs=5)
```

## Key Takeaways
- Pretrained models leverage features from large datasets
- Can be used as fixed feature extractors
- Fine-tuning adapts model to new tasks
- Transfer learning: use less data for better performance
- Benefits: faster, less computational power, good for limited data





