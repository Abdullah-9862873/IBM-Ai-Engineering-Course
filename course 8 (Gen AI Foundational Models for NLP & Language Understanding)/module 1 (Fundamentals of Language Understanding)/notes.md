# Course 8: AI Foundational Models for NLP & Language Understanding

## Course Overview

- Learn about various aspects of NLP and AI model development
- Suitable for data scientists, ML engineers, deep learning engineers, AI engineers

## Prerequisites

- Basic Python knowledge (advantage)
- Basic PyTorch knowledge (advantage)
- Awareness of machine learning and neural networks (advantage)

## Learning Outcomes

- Describe fundamentals of language understanding
  - Converting words to features
  - Document categorization prediction
- Explain NLP models and techniques
  - N-Gram
  - Word2Vec
  - Sequence-to-sequence
- Use PyTorch to build, train, and implement NLP models

---

# Module 1: Fundamentals of Language Understanding

## Course Introduction

- AI models for Natural Language Processing (NLP)
- Learn various aspects of NLP and AI model development

## Target Audience

- Existing and aspiring data scientists
- Machine learning engineers
- Deep learning engineers
- AI engineers

## Prerequisites

- Basic Python knowledge (advantage)
- Basic PyTorch knowledge (advantage)
- Awareness of machine learning and neural networks (advantage)

## Learning Outcomes

1. **Describe fundamentals of language understanding**
   - Converting words to features
   - Document categorization prediction

2. **Explain NLP models and techniques**
   - N-Gram
   - Word2Vec
   - Sequence-to-sequence

3. **Use PyTorch** to build, train, and implement NLP models

---

## Module 1 Topics

### Converting Words to Features

- **One Hot Encoding**: Basic method for converting words to numerical features
- **Bag of Words**: Represent text as word frequency vectors
- **Embedding**: Dense vector representations of words
- **Embedding Bags**: Efficient embedding lookup for variable-length sequences

### Neural Networks for NLP

- Document categorization
- Prediction tasks
- Training and optimization

### N-Gram Language Models

- Applications in language modeling
- Hands-on lab exercises with PyTorch
- Build and train simple language model with neural network

---

## Module 2 Topics

### Word2Vec Embedding Models

- Types of Word2Vec models
- Features of Word2Vec models

### Sequence-to-Sequence Models

- Purpose in NLP
- Sequence transformation tasks
- Process of evaluating quality of generated text

### Hands-on Labs

- Develop and integrate pretrained embedding models

---

## Course Structure

- **Videos**: Short and focused on main topics
- **Readings**: Detailed content in text format
- **Labs**: Technical environment with detailed instructions and code snippets
- **Practice Quizzes**: Ungraded self-assessment
- **Graded Quizzes**: Apply and assess knowledge

## Tips for Success

- Watch all videos
- Complete all labs to practice new skills
- Attempt all quizzes

---

## Converting Words to Features

### Overview

- NLP applications classify text (e.g., emails) based on:
  - Presence of specific words
  - Frequency of words
  - Contextual meaning of words
- Convert words to numerical features for ML models

### Example Dataset

- Sample sentences:
  - "I like cats."
  - "I hate dogs."
  - "I'm impartial to hippos."
- **Corpus**: Set of sentences, documents, or sequences
- Task: Neural network to identify subject (cat, dog, or hippo)

---

## One-Hot Encoding

### Definition

- Method to convert categorical data into feature vectors
- Neural network can understand these vectors

### How It Works

- Table with columns: token index, token, one-hot encoded vector
- Vector dimension = number of words in vocabulary
- Token index = element position in vector
- Each token represented as vector with 1 at its position, 0 elsewhere

### Example

| Token | One-Hot Vector |
|-------|---------------|
| I | [1, 0, 0, 0, 0, 0, 0] |
| like | [0, 1, 0, 0, 0, 0, 0] |
| cats | [0, 0, 1, 0, 0, 0, 0] |

---

## Bag of Words

### Definition

- Represents document as aggregate or average of one-hot encoded vectors

### How It Works

- For "I like cats":
  - Add one-hot vectors for "I", "like", "cats"
  - Result = bag of words vector for the sentence
- Represents entire document as single vector

---

## Embeddings

### Why Use Embeddings?

- One-hot vectors have high dimensionality (equal to vocabulary size)
- Embedding vectors have lower dimensionality
- Reduces computational requirements

### How It Works

1. Input token index (instead of one-hot vector)
2. Embedding layer accepts index and outputs embedding vector
3. Embedding weights form an embedding matrix
   - Rows = words
   - Columns = embedding dimension

### In PyTorch

```python
# Initialize embedding layer
embedding = nn.Embedding(num_embeddings, embedding_dim)

# Get embedding for token index
embed = embedding(token_index)
```

- Output: Tensor with embedding vectors for each word

---

## Embedding Bags

### Definition

- Efficient way to get sum/average of embeddings for multiple tokens
- Input: token indexes
- Output: sum or average of word embeddings

### How It Works

- Instead of feeding bag of words vector to hidden layer
- Directly input token indexes
- Embedding bag layer computes sum/average of embeddings

### In PyTorch

```python
# Initialize embedding bag layer
embedding_bag = nn.EmbeddingBag(num_embeddings, embedding_dim)

# Get embedding bag with offset
output = embedding_bag(input_ids, offsets)
```

### Offset Parameter

- Used when dataset is one-dimensional tensor containing multiple documents
- Captures starting position of each document
- Helps identify positions of different documents in combined tensor

---

## Document Categorization with Neural Networks

### Overview

- **Document Classifier**: Categorizes articles by analyzing text content
- Input: Raw text → Output: Classification (science, sports, business, etc.)

---

## Neural Network Basics

### Definition

- Mathematical function consisting of sequence of matrix multiplications with other functions

### Structure

1. **Input Layer**: Accepts input (e.g., bag of words vector)
2. **Hidden Layer**: Matrix multiplication + bias = logits
3. **Activation Function**: Applied to each logit (each element = neuron)
4. **Output Layer**: Produces final predictions

### Key Concepts

- **Logits**: Output values before activation
- **Neurons**: Elements after activation
- **Learnable Parameters**: Weights network adjusts during training

### Classification Process

1. Input embedding vector into network
2. Network outputs vector of logits (one per class)
3. Apply **argmax function** to find highest logit
4. Highest logit index = predicted class

### Example

- Classes: [World, Sports, Business, Science & Technology]
- Logits: [2, 7, 3, 1]
- argmax → index 1 → "Sports"

---

## Neural Network Architecture

### Visualization

- Circles = neurons
- Leftmost = input layer (elements of input vector)
- Connecting lines = weight matrix
- Subsequent layers = hidden and output layers
- Final layer neurons = output classes

### Processing Steps

1. Input → first hidden layer → activation → hidden values (z)
2. Hidden layer → output layer
3. Use argmax to find class with highest score

---

## Hyperparameters

### Definition

- Externically set configurations of a neural network

### Types

1. **Number of Hidden Layers**
   - Single hidden layer (common)
   - Multiple hidden layers

2. **Number of Neurons**
   - Each hidden layer can have different neuron counts
   - If first hidden layer = embedding layer, neurons = vocabulary size
   - Output layer neurons = number of classes

3. **Selection**
   - Number of layers and neurons selected via empirical validation

---

## Creating Neural Network in PyTorch

### Using AG News Dataset with torchtext

```python
# Define categories
category = {1: "World", 2: "Sports", 3: "Business", 4: "Science & Technology"}
```

### Text Processing Pipeline

```python
# Batch function for embedding bags
def batch_function(data):
    # Add code to append labels for each sample to a batch
    return labels, token_indices, offsets
```

### Data Loader

```python
# Create data loader with batch size
dataloader = DataLoader(dataset, batch_size=3)
```

### Sample Output

- Labels for each sample
- Token indices (basis for bag of words model)
- Relative positions of indices

---

## Model Architecture

```python
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, text, offsets):
        embedded = self.embedding_bag(text, offsets)
        return self.fc(embedded)
```

### Key Components

- **Embedding Bag Layer**: First layer
- **Fully Connected Layer**: Output layer
- **Weights initialized** during forward pass

### Forward Pass

1. Input text and offsets to embedding bag
2. No activation applied to embedding bag output
3. Feed into fully connected layer
4. Output final predictions

---

## Making Predictions

### Logits

- Each row = different sample
- Columns = logit values for each class
- Shape: (batch_size, num_classes)

### Using argmax

```python
# Find maximum value in each row
predictions = torch.argmax(logits, dim=1)
```

### Prediction Function

```python
def predict(text):
    # Tokenize text
    tokenized = tokenizer(text)
    # Process through pipeline
    # Model predicts category
    # Return label with highest output value
```

### Example

- Input: Article about sports
- Output: Predicted class = Sports

---

## Summary

- **Document classifier**: Categorizes articles by analyzing text content
- **Neural network**: Mathematical function with matrix multiplications
- **argmax**: Identifies highest logit for predicted class
- **Hyperparameters**: Externally set configurations (layers, neurons)
- **Prediction function**: Tokenizes text → pipeline → predicts category

---

## Neural Networks for Document Categorization

### Overview

- Neural networks can classify documents based on text features
- Process: Input features → Hidden layers → Output predictions

### Architecture

1. **Input Layer**: Accepts text features (bag of words, embeddings)
2. **Hidden Layers**: Process features through weights and activations
3. **Output Layer**: Produces classification probabilities

### Training Process

1. **Forward Pass**: Input → predictions
2. **Loss Calculation**: Compare predictions with actual labels
3. **Backward Pass**: Compute gradients
4. **Weight Update**: Optimize using gradient descent

### Optimization

- **Loss Functions**: Cross-entropy, MSE
- **Optimizers**: SGD, Adam, AdamW
- **Regularization**: Dropout, weight decay

---

## N-Gram Language Models

### Definition

- N-gram: Sequence of n words from a given text
- N-gram model predicts probability of next word based on previous n-1 words

### Types of N-Grams

- **Unigram (n=1)**: Single word
- **Bigram (n=2)**: Two-word sequence
- **Trigram (n=3)**: Three-word sequence

### Applications

- Text generation
- Speech recognition
- Machine translation
- Autocomplete

### Example

- Sentence: "I like cats"
  - Unigrams: I, like, cats
  - Bigrams: I like, like cats

### Probability Calculation

- P(cats | "I like") = count("I like cats") / count("I like")

---

## PyTorch Implementation Example

### Simple Neural Network for Text Classification

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)  # Average pooling
        out = self.relu(self.fc1(pooled))
        out = self.fc2(out)
        return out
```

### Training Loop

```python
# Initialize model, loss, optimizer
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Cross-Entropy Loss and Optimization

### Learnable Parameters

- Neural networks function via matrix and vector operations
- Parameters collectively represented as **Θ (Theta)**
- Networks can have millions to trillions of parameters
- Parameters are fine-tuned during training to enhance model performance

### Loss Function

- Serves as measure of accuracy
- Goal: Find optimal **Θ** that minimizes discrepancy between:
  - Predicted output (ŷ)
  - Actual label (y)
- Loss function expressed as function of Θ

### Cross-Entropy Loss

#### Overview

- Used to find best parameters
- Based on comparing true and predicted distributions

#### Softmax Function

- Transforms logits to probabilities
- For each class, compute: P(class|x, Θ) = exp(logit) / Σ(exp(logits))
- Creates conditional distribution for predicted class
- Enhances distinction between scores for different classes

#### Process

1. Input embedding vector → neural network
2. Network outputs logits for each category
3. Softmax transforms logits to probabilities
4. Cross-entropy compares true distribution (y) with predicted (P(y|x, Θ))

#### True vs Predicted Distribution

- **True distribution (y)**: One-hot encoded actual labels
- **Predicted distribution**: Probability from softmax

#### KL Divergence

- Measures difference between two distributions
- Only second term (cross-entropy) depends on Θ
- Cross-entropy loss = -Σ y * log(P(y|x, Θ))

#### Monte Carlo Sampling

- For unknown distributions, estimate by averaging function over samples
- Approximates true cross-entropy loss

---

## Optimization

### Gradient Descent

- Method to minimize loss
- Equation: Θ_(k+1) = Θ_k - η * ∇L(Θ_k)
- **η (eta)**: Learning rate (step size)
- **∇L(Θ)**: Gradient of loss function (direction of greatest increase)
- Move in reverse direction of gradient to decrease loss

### Iterative Process

1. **k=0**: Start with random initial parameters
2. **k=1**: Adjust parameters using gradient × learning rate, compute loss
3. **k=2**: Update parameters again, loss decreases further
4. Continue until convergence

### Loss Surface

- 2D: Visualize as surface with minimum point
- High-dimensional: Complex surface with many minima
- Neural networks have millions of parameters

### Data Split

- **Training data**: For learning parameters
- **Validation data**: For hyperparameter tuning
- **Test data**: For real-world performance evaluation

---

## PyTorch Implementation

### Cross-Entropy Loss

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Calculate loss
logits = model(input)
loss = criterion(logits, true_labels)
```

### Optimization (SGD)

```python
# Initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1)
```

### Training Loop

```python
for epoch in range(num_epochs):
    # Reset gradients
    optimizer.zero_grad()
    
    # Forward pass
    output = model(text, offsets)
    
    # Calculate loss
    loss = criterion(output, labels)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    optimizer.step()
    
    # Update learning rate
    scheduler.step()
```

### Key Steps

1. **optimizer.zero_grad()**: Reset gradients
2. **model()**: Make prediction
3. **criterion()**: Calculate loss
4. **loss.backward()**: Compute derivatives
5. **optimizer.step()**: Update parameters

---

## Summary

- **Learnable parameters**: Fine-tuned during training (Θ)
- **Loss function**: Measures accuracy, minimizes discrepancy
- **Cross-entropy**: Finds best parameters by comparing distributions
- **Softmax**: Transforms logits to probabilities
- **Optimization**: Minimizes loss using gradient descent
- **Data split**: Training, validation, test sets

---

## Training the Model in PyTorch

### Overview

- Use cross-entropy loss and optimization concepts to train model
- Work with tokenized and indexed news dataset

### Step 1: Split Dataset

```python
# Create iterators for AG News dataset
train_iter, test_iter = AG_NEWS(split='train'), AG_NEWS(split='test')

# Split training data into training and validation
train_data, valid_data = split_dataset(train_iter)
```

### Step 2: Create Data Loaders

```python
# Training data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Validation data loader
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# Test data loader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
```

### Key Parameters

- **batch_size**: Number of samples for gradient approximation
- **shuffle**: Randomizes data for better optimization

### Step 3: Define Model

```python
# Create model instance
model = TextClassificationModel(vocab_size, embedding_dim, num_classes)

# Initialize weights (helps with optimization)
model.apply(init_weights)
```

### Step 4: Initialize Optimizer and Loss

```python
# Initialize SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initialize cross-entropy loss
criterion = torch.nn.CrossEntropyLoss()
```

### Step 5: Training Loop

```python
# Number of epochs
num_epochs = 10

# Record metrics
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    
    total_loss = 0
    
    # Iterate over batches
    for batch in train_loader:
        # Get text, offsets, labels
        text, offsets, labels = batch
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(text, offsets)
        
        # Calculate loss
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient descent (update parameters)
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = evaluate(model, valid_loader)
    
    # Record metrics
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    # Save best model
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_model.pt')
    
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
```

### Training Process

1. **Iterate over each epoch**: Pass through entire dataset
2. **Set training mode**: model.train()
3. **Calculate total loss**: Iteratively process batches
4. **Perform gradient descent**: Adjust model parameters
5. **Update loss**: After each batch is processed

### Monitoring

- Record loss and accuracy for each epoch
- Save model parameters if validation accuracy improves
- Plot loss and accuracy over time

### Expected Results

- **Loss decreases** over epochs
- **Accuracy increases** over epochs
- Model improves its classification performance

---

## Summary: Training

- **Data split**: Training → validation → testing
- **Data loaders**: Batch size for gradient approximation, shuffle for optimization
- **Model definition**: init_weights helps with optimization
- **Training loop**: Epoch iteration → training mode → batch processing → gradient descent → loss update
- **Monitoring**: Save best model based on validation accuracy
- **Training loop**: Forward pass → loss → backward pass → parameter update

---

# Module 1 Summary: Key Takeaways

## Feature Extraction

- **One-hot encoding**: Converts categorical data into feature vectors
- **Bag of words**: Document as aggregate/average of one-hot encoded vectors
- **Embeddings**: Sum of embeddings when fed to neural network hidden layer
- **PyTorch classes**: `Embedding` and `EmbeddingBag`

## Document Classification

- **Document classifier**: Categorizes articles by analyzing text content
- **Neural network**: Mathematical function with sequence of matrix multiplications
- **Argmax function**: Identifies highest logit for most likely class
- **Hyperparameters**: Externally set configurations (layers, neurons)
- **Prediction function**: Tokenizes text → pipeline → predicts category

## Training Process

- **Learnable parameters**: Fine-tuned during training (Θ)
- **Loss function**: Measures accuracy, minimizes discrepancy between predicted and actual
- **Cross-entropy**: Used to find best parameters
- **Monte Carlo sampling**: Estimates unknown distribution by averaging over samples

## Optimization

- **Gradient descent**: Minimizes loss by updating parameters
- **Data split**: Training (learning), validation (hyperparameter tuning), test (evaluation)
- **Training setup**: Data loaders for training, validation, testing
- **Batch size**: Sample count for gradient approximation
- **Shuffle**: Promotes better optimization

## Training Loop

1. Iterate over each epoch
2. Set model to training mode
3. Calculate total loss (divide into batches)
4. Perform gradient descent
5. Update loss after each batch

---

## Language Modeling with N-Grams

### Overview

- Language modeling predicts next word based on preceding words
- Word choices are influenced by context provided by preceding words
- Examples: "I like ___" → vacation; "I hate ___" → surgery

### N-Gram Definition

- Sequence of n words from a given text
- Example: "I like vacations"
  - ω₁ = "I", ω₂ = "like", ω₃ = "vacations"

---

## Bi-Gram Model

### Definition

- Conditional probability model
- **Context size**: 1 (considers only immediate previous word)
- **Formula**: P(word₃ | word₂)

### How It Works

- Given word₂, predict word₃
- Count occurrences of follow-up words in corpus
- Calculate probability: P(word₃=vacation | word₂=like) = count("like vacation") / count("like")

### Example

- "I like ___" → vacation (probability = 1)
- "I hate ___" → surgery (probability = 1)
- "I like surgery" → probability = 0 (never occurs after "like")

---

## Tri-Gram Model

### Definition

- Conditional probability model
- **Context size**: 2 (considers two previous words)
- **Formula**: P(word₃ | word₁, word₂)

### Improvement over Bi-Gram

- Uses more context to make predictions
- Example: "I like ___" vs "surgeons like ___"

### Example

- Context: word₁ = "I", word₂ = "like" → word₃ = "vacation" (P = 1)
- Context: word₁ = "surgeons", word₂ = "like" → word₃ = "surgery" (P = 1)

### Prediction with Argmax

- Given context (word₁, word₂), find word₃ that maximizes P(word₃ | word₁, word₂)
- Example: "surgeons like ___" → "surgery" (highest probability)

---

## N-Gram Generalization

### Arbitrary Context Size

- Tri-gram can be generalized to n-gram with context size = t
- **Formula**: P(word_t | word_{t-1}, word_{t-2}, ..., word_1)

### Stationarity

- Assume relationships remain constant over time
- Can apply bi-gram or tri-gram models at any point in time

### Complexity

- Calculating probabilities for larger context sizes becomes complex
- Neural networks provide solution by approximating these probabilities

---

## Neural Network for N-Gram

### Context Vector

- Context size = number of previous words to consider
- Vocabulary size = number of words in dictionary
- **Context vector size** = context_size × vocabulary_size

### Construction

- Not computed directly from one-hot vectors
- Constructed by concatenating embedding vectors

### Architecture

- **Input**: Concatenated embeddings (context_size × embedding_dim)
- **Output**: Probability distribution over vocabulary (softmax)
- **Hidden layers**: Process context to predict next word

### Example

- Vocabulary: {I, hate, like, surgeons, surgery, vacations} = 6 words
- Context size: 2 (previous 2 words)
- Input dimension: 6 × 2 = 12
- Output: 6 neurons (one per vocabulary word)

### Feedforward Network

- Ignores dependence on position t
- Does not capture order/position of words like modern RNNs/Transformers

---

## Summary

- **Bi-gram**: Context size = 1, uses immediate previous word
- **Tri-gram**: Context size = 2, uses two previous words
- **N-gram**: Arbitrary context size t
- **Neural networks**: Approximate n-gram probabilities using embeddings
- **Context vector**: Concatenation of embedding vectors (context_size × vocab_size)

---

## N-Grams as Neural Networks with PyTorch

### Creating Embedding Layer

```python
# Create embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

# Context size = 2
context_size = 2
```

### Architecture

- **Input**: Two indices representing context (2 words)
- **Embedding layer output**: Two embedding vectors of dimension = embedding_dim
- **Context vector**: Reshape embeddings to (batch_size, context_size × embedding_dim)
- **Next layer input dimension**: context_size × embedding_dim

### Model Structure

- Essentially a classification model
- Uses context vector + extra hidden layer for performance
- Predicts words using sliding window approach

---

## Sliding Window Approach

### How It Works

- N-gram model predicts word at position t based on previous context
- Window shifts incrementally through sequence

### Bi-gram Example

- Predict word at position t using positions t-1 and t-2
- Start at t = 3 to avoid negative indices

### Data Representation

| t | Context | Target |
|---|---------|--------|
| 3 | word₁, word₂ | word₃ |
| 4 | word₂, word₃ | word₄ |

### Example: "I like vacations"

- Word indices: I=0, like=1, vacations=2
- t=3: Context = [I, like], Target = vacations
- t=4: Context = [like, vacations], Target = next word

---

## Implementing in PyTorch

### Batch Function for Windowing

```python
def batch_function(data, context_size):
    contexts = []
    targets = []
    for i in range(context_size, len(data)):
        # Get context (previous context_size words)
        context = data[i-context_size:i]
        # Get target (current word)
        target = data[i]
        contexts.append(context)
        targets.append(target)
    return contexts, targets
```

### Creating Toy Dataset

```python
# Create pipeline to convert text to indexes
# Use list object instead of dataset object
```

### Training

- **KPI**: Prioritize loss over accuracy
- Similar to classification model training
- Pad tokens for consistent shape if needed

---

## Making Predictions

### Index to Token Mapping

```python
# Get index to token mapping
index_to_token = vocab.get_itos()

# List: index → word (acts as decoder)
```

### Prediction Pipeline

```python
# Input text
text = "never gonna"

# Process through pipeline
token_indices = tokenizer(text)
input_tensor = torch.tensor(token_indices)

# Make prediction
output = model(input_tensor)
predicted_index = torch.argmax(output)

# Convert to word
predicted_word = index_to_token[predicted_index]
```

### Generating Sequences

- Use prediction function repeatedly
- Feed predicted word back as context
- Generate sequence of words

---

## Summary

- **N-gram model**: Allows arbitrary context size
- **PyTorch implementation**: Classification model with context vector + hidden layer
- **Sliding window**: Predicts words by shifting context window
- **Training**: Prioritize loss over accuracy as KPI
- **Prediction**: Use index_to_token mapping to convert output to readable words

---

# Module 1 Summary: Key Takeaways

## N-Gram Language Models

### Bi-Gram Model

- Conditional probability model
- Context size = 1 (only immediate previous word)
- Predicts next word based on one previous word

### Tri-Gram Model

- Conditional probability function
- Context size = 2 (two previous words)
- Improves on bi-gram by using more context

### N-Gram Generalization

- Arbitrary context size (t)
- Formula: P(word_t | word_{t-1}, word_{t-2}, ..., word_1)

## Neural Network Implementation

### Input Representation

- One-hot: vocabulary_size × context_size (high dimensional)
- Embeddings: Concatenate embedding vectors (avoid high dimensionality)
- Context vector = concatenate(embeddings of preceding words)

### PyTorch Model

- Classification model using context vector
- Extra hidden layer to enhance performance
- Uses sliding window to predict words

### Sliding Window

- Incremental shifting through sequence
- t=3: context = [word₁, word₂], target = word₃
- t=4: context = [word₂, word₃], target = word₄

## Training & Prediction

### Training

- Prioritize loss over accuracy as KPI
- Similar to classification model training
- Pad tokens for consistent shape

### Prediction

- Use index_to_token (vocab.get_itos()) as decoder
- Convert output indices to readable words
- Generate sequences by feeding predictions back as context

---

# Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know that:

- **Bi-gram model**: Conditional probability model with context size one (consider only immediate previous word to predict next one)
- **Tri-gram model**: Conditional probability function that improves on bi-gram by increasing context size to two
- **N-gram model**: Generalization of trigram that allows arbitrary context size
- **Input vector dimensionality**: Product of vocabulary size and context size if one-hot encodings are used; in practice, embeddings avoid high-dimensional representation by concatenating embedding vectors of preceding words
- **PyTorch implementation**: Classification model using context vector with extra hidden layer to enhance performance
- **Sliding window**: N-gram model predicts words surrounding target by incrementally shifting
- **Training KPI**: Prioritize loss over accuracy as key performance indicator