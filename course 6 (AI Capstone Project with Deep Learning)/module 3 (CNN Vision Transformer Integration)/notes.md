# Video: Transfer Learning and Vision Transformers

## Learning Objectives

After watching this video, you'll be able to:
- Explain the working of vision transformers
- Describe the use of vision transformers in transfer learning using hybrid approaches
- Explain the mechanistic details of vision transformers
- Compare and contrast how Keras and PyTorch implement vision transformers
- List the advantages and limitations of vision transformers

---

## Introduction to Vision Transformers

Vision transformers (ViTs) leverage self-attention mechanisms originally developed for natural language processing for computer vision applications. They treat images as sequences of patches and model relationships between them.

Vision transformers demonstrate remarkable performance on a wide range of visual tasks, particularly when large datasets and computational resources are available.

However, a hybrid model combining CNNs and vision transformers could be useful if the training dataset is small.

---

## How Vision Transformers Work

### Step 1: Patchification
Vision transformers divide an input image into fixed-size patches (e.g., 16x16 pixels).

### Step 2: Flatten and Project
Each patch is flattened and projected into a higher-dimensional embedding space using a dense (fully-connected) layer.

This process transforms the image into a sequence of patch embeddings analogous to word embeddings in NLP.

### Step 3: Positional Encoding
Positional encodings provide information about the position of each patch in the original image, enabling the model to maintain spatial awareness throughout the processing pipeline.

### Step 4: Transformer Encoder Blocks
The sequence of patch embeddings, along with positional information, is passed through a stack of transformer encoder blocks.

Each block consists of:
- **Multi-head self-attention**: Allows the model to weigh the importance of each patch relative to others, capturing both local and global dependencies across the image
- **Feed-forward network**: Applies a small neural network to each patch embedding independently
- **Residual connections**: Help stabilize training
- **Layer normalization**: Enable deeper architectures

### Step 5: Classification Token (CLS Token)
A special learnable token, often called the "classification" or "CLS" token, is prepended to the sequence of patches.

After passing through the transformer blocks, the output corresponding to this token is used for classification by a final dense layer, producing the model's predictions.

---

## Transfer Learning with Vision Transformers

Vision transformers are very powerful in transfer learning scenarios. They often require large datasets to train from scratch.

However, transfer learning using pre-trained weights from large datasets (such as ImageNet) or other pre-trained network models enables visual transformers to perform well even on smaller datasets.

This requires **fine-tuning** on the target task.

---

## Hybrid Approach: CNN + Vision Transformer

A common and effective strategy is to combine CNNs and vision transformers in a hybrid architecture:

1. A **pre-trained CNN** processes the input image to extract local features, generating high-level feature maps
2. The feature maps are **divided into patches** and fed into a vision transformer
3. The vision transformer models **global relationships** and context using self-attention
4. The output is **pooled** and passed through a **dense layer** for final classification

### Benefits of Hybrid Architecture
- Leverages the local feature extraction strengths of CNNs
- Leverages the global context modeling capabilities of vision transformers
- Often results in improved performance, especially when data is limited or computational efficiency is a concern

---

## Mechanistic Details of Vision Transformers

### Global Context Modeling
Vision transformers excel at capturing long-range dependencies and global relationships within an image, making them robust to changes in scale, rotation, or perspective.

### Parallel Processing
All patches are processed in parallel, leading to efficient computation especially for large datasets.

### Minimal Inductive Bias
Unlike CNNs, vision transformers show minimal inductive bias:
- They do not assume locality or translation invariance
- Makes them highly flexible

However, this also means:
- They require more data
- Their output is highly dependent on training data

### Scalability
Vision transformers can scale to very large models and datasets and thus benefit from advances in hardware and distributed training.

---

## Implementation in Keras and PyTorch

### Keras
- Offers high-level Application Processing Interfaces (APIs)
- Ready-to-use vision transformer implementations via **KerasCV**
- The **functional API** allows easy chaining of CNNs and ViTs
- Pre-trained ViT weights available via **TensorFlow Hub**
- Transfer learning workflows are well-supported

### PyTorch
- Provides maximum flexibility for custom architectures
- Libraries like **torchvision** and **timm** supply:
  - Pre-trained ViT models
  - Utilities for patch extraction
  - Embedding and transformer blocks
- PyTorch's **dynamic computation graph** is particularly suited for research and experimentation

### Comparison
While these frameworks differ in syntax and abstraction level, both enable efficient construction, training, and fine-tuning of ViT models for a variety of computer vision tasks.

---

## Advantages of Vision Transformers

| Advantage | Description |
|-----------|-------------|
| Local and Global Context | Ability to capture both local and global image context |
| Large Datasets | Powerful for large datasets and tasks requiring global reasoning |
| Parallelizability | Highly parallelizable |
| Scalability | Scalable to very large models |

---

## Limitations of Vision Transformers

| Limitation | Description |
|-----------|-------------|
| Large Datasets | Require large datasets and significant computational resources to train from scratch |
| Small Datasets | Minimal inductive bias can make them less effective on small datasets without transfer learning or hybridization |
| Complexity | More complex and computationally intensive than lightweight CNNs for simple tasks |

---

## Summary

### Vision Transformers Working:
1. **Patchification** - Divide image into fixed-size patches
2. **Embedding** - Flatten and project patches to embedding space
3. **Positional Encoding** - Add positional information
4. **Transformer Encoder Blocks** - Process through self-attention and feed-forward networks
5. **Classification** - Use CLS token output for final classification

### Key Takeaways:
- ViTs leverage self-attention mechanisms from NLP for computer vision
- Hybrid CNN + ViT architecture is effective for limited data
- Both Keras and PyTorch support ViT implementation
- ViTs excel with large datasets; need transfer learning for small datasets
- Advantages: global context, parallelism, scalability
- Limitations: data requirements, computational complexity

