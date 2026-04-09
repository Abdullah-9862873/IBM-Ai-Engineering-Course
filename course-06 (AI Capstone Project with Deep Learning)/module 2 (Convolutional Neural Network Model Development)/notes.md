# Video: Building a CNN Classifier

## Learning Objectives

After watching this video, you'll be able to:
- Explain the advantages of using Convolutional Neural Networks (CNNs) for image classification
- Discuss the reasons for using frameworks such as Keras and PyTorch
- Compare and contrast the implementation of CNNs in Keras and PyTorch
- Identify the various metrics for evaluating CNNs

---

## Why CNNs for Image Classification?

For image classification tasks, CNNs offer greater accuracy and robustness than any other manually defined set of rules or traditional machine learning pipelines.

### CNN Development Pipeline

To create a CNN classifier:
1. Curate and label the dataset
2. Use a lazy data loader with on-the-fly augmentation
3. Design and train a CNN
4. Evaluate with accuracy, loss, precision, and recall metrics

---

## Keras vs PyTorch

The most widely used frameworks for implementing CNNs are Keras and PyTorch.

### Advantages of Using Frameworks

These frameworks take care of:
- Low-level tensor operations
- Gradient calculations
- Graphic Processing Unit (GPU) management

This allows you to focus on architecture and experimentation instead of writing the backpropagation mathematics.

### Embedded Optimization Research

They embed decades of optimization research including:
- Efficient GPU kernels
- Automatic differentiation
- Mixed-precision arithmetic
- Distributed training routines

### Keras vs PyTorch Comparison

| Feature | Keras | PyTorch |
|---------|-------|--------|
| Level | High-level, user-friendly API | Lower-level, granular control |
| Use Case | Rapid prototyping | Low-level network optimization |
| Popularity | General use | Research and customization |

**Keras** is a high-level, user-friendly API that targets rapid prototyping with minimal boilerplate code.

**PyTorch** offers a more detailed, lower-level interface and provides granular control over every gradient step. PyTorch is popular for low-level network optimization.

---

## Data Loading

### Keras
- Uses the `tf.data.dataset` subsystem
- A helper routine walks the directory tree, infers class labels, and yields batches as needed

### PyTorch
- Uses a two-piece design:
  - **Dataset object**: Returns one sample at a time
  - **DataLoader object**: Batches samples and uses multiple CPU workers to prefetch upcoming batches

---

## Data Augmentation

### Keras
Offers two mechanisms:
1. **Preprocessing layers** - Can be chained into a small sequential block
2. **Image data generator** utility

Each batch is augmented on the CPU, or even the GPU, before reaching the training loop.

### PyTorch
Uses the `torchvision.transforms` module:
- Transformations are declared in a functional list
- Executes inside each data loading worker process

**In both frameworks**: Augmentation happens on the fly - you never store augmented copies on disk, and the model sees a fresh rendition of each image during every epoch.

---

## Model Architecture

### Keras
- Streamlines CNN construction using the **Sequential API**
- Manages the entire training loop via `model.fit()` method
- Contains: forward pass, backpropagation, optimizer steps, and metrics analysis

### PyTorch
- Architecture defined as a subclass of **nn.Module**
- Training loop must be implemented explicitly, including:
  - Moving data to the GPU
  - Zeroing gradients
  - Computing forward pass
  - Calculating loss
  - Calling `.backward()`
  - Updating the optimizer

---

## Model Evaluation Metrics

### 1. Accuracy
- The proportion of correct predictions across all classes
- Can be misleading on skewed datasets

### 2. Precision
- Ratio of true positives to (true positives + false positives)
- Answers: "Of all positive predictions, how many were correct?"
- Crucial when false positives incur logistical costs

### 3. Recall (Sensitivity)
- Ratio of true positives to (true positives + false negatives)
- Answers: "How many [actual positives] were correctly identified?"
- Important when absence of a specific class can impact projections

### 4. F1 Score
- Harmonic mean of precision and recall
- Provides a single value that balances both errors

### 5. Confusion Matrix
- Tabular summary of TP, FP, TN, FN
- Quickly reveals if the model systematically confuses different classes

### 6. ROC Curve and AUC
- **ROC**: Receiver Operating Characteristic curve
- **AUC**: Area Under the Curve
- Summarizes tradeoff between true positive and false positive rates across all possible cutoffs

### 7. IoU and mAP (Object Detection/Segmentation)
- **IoU**: Intersection Over Union - measures overlap between predicted and ground truth bounding boxes
- **mAP**: Mean Average Precision - measures average precision across all classes in object detection tasks

---

## Implementing Metrics

### Keras
- Metrics can be passed into the `compile()` step
- Logging them during `fit()`

### PyTorch
- Calculate metrics using **Scikit-learn** or **Torch metrics**

---

## Comparing Keras vs PyTorch Models

To fairly compare CNNs trained in Keras versus PyTorch, ensure:
- Similar batch sizes
- Similar learning rates
- Same optimizer choices
- Same number of epochs
- Matching weight initialization schemes
- Matching placement of softmax layer

**Note**: 
- Keras typically includes softmax inside the model
- PyTorch omits it and feeds raw logits into the loss function

---

## Summary

### CNN Advantages for Image Classification
- Great accuracy and robustness

### Framework Choice
- **Keras**: Concise, high-level interface, perfect for rapid prototyping
- **PyTorch**: Hands-on transparency, customized research, in-depth debug tools

### Data Loading
- Keras: `tf.data.dataset` subsystem
- PyTorch: Dataset object + DataLoader

### Data Augmentation
- Keras: Preprocessing layers + Image data generator
- PyTorch: `torchvision.transforms` module

### Model Building
- Keras: Sequential API
- PyTorch: `nn.Module` subclass

### Training
- Keras: `model.fit()` method
- PyTorch: Explicit training loop

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC/AUC, IoU/mAP



______________
