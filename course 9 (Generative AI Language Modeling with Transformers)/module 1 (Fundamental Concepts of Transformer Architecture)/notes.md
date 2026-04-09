# Course 9: Generative AI Language Modeling with Transformers

## Course Overview

- Learn fundamental and advanced concepts of transformer-based models for NLP
- Suitable for existing and aspiring data scientists, ML engineers, deep learning engineers, AI engineers

## Prerequisites

- Basic Python knowledge (advantage)
- Basic PyTorch knowledge (advantage)
- Awareness of machine learning and neural networks (advantage)

## Learning Outcomes

- Apply positional encoding and attention mechanisms in transformer-based architectures to process sequential data
- Use and implement decoder-based models (GPT) and encoder-based models (BERT) for language modeling
- Implement a transformer model to translate text from one language to another

---

# Module 1: Fundamental Concepts of Transformer Architecture

## Topics Covered

### Positional Encoding

- What is positional encoding
- Implementation in PyTorch
- Why positional information is needed for transformers

### Attention Mechanism

- How attention mechanism works in language translation
- Self-attention mechanisms in language modeling
- Scaled dot product attention mechanism

### Transformer Applications

- Text classification using transformers
- Processing sequential data

### Hands-on Labs

- Implement basic self-attention mechanism in PyTorch
- Implement positional encoding in PyTorch
- Apply transformers for text classification using data loader

---

# Module 2: Decoder and Encoder Models

## Topics Covered

### Decoder Models (GPT)

- GPT architecture
- Training and PyTorch implementation
- Building and training GPT-like model

### Encoder Models (BERT)

- BERT architecture
- Pre-training with Masked Language Modeling (MLM)
- Pre-training with Next Sentence Prediction (NSP)
- Data preparation for BERT
- Building and training BERT model

### Transformer for Translation

- Transformer architecture understanding
- Implementation for language translation
- Construct transformer model from scratch using PyTorch

---

# Course Structure

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

## Positional Encoding

### Why Positional Encoding is Needed

- **Problem**: Transformers process all tokens simultaneously (in parallel)
- Tokens are processed independently, so position information is lost
- **Example**: "King and Queen are awesome" vs "Queen and King are awesome"
  - Without positional encoding: embeddings are identical
  - With positional encoding: vector representations differ

### Definition

- Incorporates information about position of each embedding within sequence
- Added to input embeddings so model can differentiate positions
- Enables model to understand word order and sequence

### Technique: Sine and Cosine Waves

#### Parameters

- **pos**: Position of the sine wave over time (word position in sequence)
- **i**: Dimension index (controls number of oscillations for each wave)
- **d_model**: Total length of each word vector (e.g., 256, 512, 768, 1024)

#### Formula

For even dimensions (i = 2k):
```
PE(pos, 2k) = sin(pos / 10000^(2k/d_model))
```

For odd dimensions (i = 2k+1):
```
PE(pos, 2k+1) = cos(pos / 10000^(2k/d_model))
```

#### Key Properties

- Unique and periodic values for sequence positions
- Cosine waves never intersect at same points
- Range between -1 and 1 (won't overshadow embeddings)
- Differentiable (can be optimized)
- Supports relative positioning

### Implementation in PyTorch

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to embeddings
        return x + self.pe[:, :x.size(1)]
```

### Learnable Positional Encodings

- Used in models like GPT
- Parameters represented as tensors
- Added to embedding vectors and optimized during training
- More flexible than fixed sine/cosine encodings

### Segment Embeddings

- Used in models like BERT
- Related to positional encodings
- Provide additional positional information
- Combined with token embeddings and positional encodings

```python
# Combined embedding in BERT
total_embedding = token_embedding + positional_embedding + segment_embedding
```

---

## Transformer Architecture Overview

### Building Blocks

1. **Word Embeddings**: Convert words to numerical vectors
   - Example: 'cat' → [0.21, -0.45, 0.87, ...]
   - Captures semantic meaning

2. **Positional Encoding**: Track word order
   - Sine/cosine waves or learnable parameters

3. **Encoder**: Processes input (bi-directional self-attention)
   - Used in BERT, for understanding text

4. **Decoder**: Generates output (masked self-attention)
   - Used in GPT, for generating text

5. **Linear Layer**: Maps predictions back to words

### Key Components

- **Attention Mechanism**: Focus on relevant words
- **Query (Q), Key (K), Value (V)**: Matrices for attention
- **Scaled Dot-Product Attention**: 
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k)V
  ```

### Optimization Techniques

- Gradient accumulation
- Mixed precision training
- Distributed training
- Efficient optimizers (AdamW, LAMB)

---

## Summary: Positional Encoding

- **Problem**: Transformers process all tokens simultaneously, losing position information
- **Solution**: Add positional encoding to embeddings
- **Methods**: Sine/cosine waves (fixed) or learnable parameters
- **Formula**: sin/cos with position and dimension index
- **Properties**: Unique, periodic, differentiable, range [-1, 1]
- **PyTorch**: Implement as module with forward method
- **Extensions**: Segment embeddings (BERT), learnable parameters (GPT)

---

## Attention Mechanism

### Overview

- Inspired by human focus in noisy environments
- Focus on most relevant parts of input data and their relationships
- Uses Query (Q), Key (K), Value (V) matrices

### Analogy: Python Dictionary

- **Keys**: French words (one-hot encoded)
- **Values**: English translations (one-hot encoded)
- **Query**: Word to translate
- Process: Query × K^T × V → Translated word

### Attention Formula

```
Attention(Q, K, V) = softmax(Q × K^T) × V
```

### Applying to Word Embeddings

- Replace one-hot vectors with word embeddings
- Keys and Values aligned with translations
- Query embedding × Key matrix → Similar embeddings
- Apply softmax to get probabilities
- Multiply by Value matrix → Translated embedding

### Softmax Function

- Accentuates largest value (→ 1)
- Diminishes smaller values (→ 0)
- Transforms output to one-hot-like vector

### Attention for Sequences

- Consolidate all query vectors into matrix Q
- Process concurrently via single matrix operation
- Input: Sequence of embeddings
- Output: Refined embeddings (same length)

---

## Self-Attention Mechanism

### Overview

- Heart of language transformer
- Each word attends to every other word in parallel
- Generates contextual embeddings

### Simple Language Modeling

- Predicts next word in sequence
- Meaning changes with context
- Example: "not like" → "hate" vs "do like" → "like"

### Query, Key, Value Generation

```python
Q = X × W_Q + b_Q  # Query matrix
K = X × W_K + b_K  # Key matrix
V = X × W_V        # Value matrix
```

Where X is the input sequence matrix (word embeddings)

### Self-Attention Output

```python
H' = softmax(Q × K^T / √d_k) × V
```

Where d_k is the embedding dimension

- **H'**: Contextual embeddings (columns = enhanced word embeddings)
- Each column corresponds to input word
- Captures relationships between words

### Additional Layer

```python
H = H' × W_O  # Output projection
```

- More nuanced representation of initial embeddings

### Prediction Process

1. Self-attention → contextual embeddings
2. Average pooling → Z¹
3. Feed-forward network → logits
4. Softmax → probability distribution
5. Argmax → predicted word index

### Advantages over RNNs

- Parallel processing (GPU-friendly)
- Captures long-range dependencies
- Faster training

### Attention Scores

- Computed as: Q × K^T
- Normalized with softmax
- Shows relationships between tokens
- Example: "makes" depends on "snow", "driving", "difficult"

---

## Scaled Dot-Product Attention with Multiple Heads

### Scaled Dot-Product Attention

```python
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

- **Scaling factor** √d_k: Prevents dot product from becoming too large
- **Masking**: Optional, for decoder (prevent looking at future tokens)

### Multi-Head Attention

- Execute multiple attention processes in parallel
- Each head attends to different segments of input

### Process

1. Split input into h heads
2. Each head: Attention(Q_i, K_i, V_i)
3. Concatenate all head outputs
4. Linear layer to combine

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
```

### Example

- Input: 7 word embeddings (dimension 4)
- Split into 2 heads → each embedding becomes 2 vectors (dimension 2)
- 14 vectors processed by individual attention mechanisms
- Concatenate outputs → 7 vectors (dimension 4)
- Final linear layer

### PyTorch Implementation

```python
from torch.nn import MultiheadAttention

multihead_attn = MultiheadAttention(
    embed_dim=4, 
    num_heads=2, 
    batch_first=False
)

# Create random Q, K, V tensors
Q = torch.randn(10, 5, 4)  # seq_len, batch, embed_dim
K = torch.randn(10, 5, 4)
V = torch.randn(10, 5, 4)

output, _ = multihead_attn(Q, K, V)
```

### Use Case Example

- Sentence: "heavy snow makes driving more difficult"
- Different heads focus on different dependencies:
  - Head 1: "makes" → "more difficult" (distant)
  - Head 2: "snow" → "driving" (local)
  - Head 3: "driving" → "difficult" (semantic)

---

## Transformer Architecture

### Encoder Components

1. **Multi-head Self-attention**: Process input embeddings
2. **Add & Norm**: Residual connection + Layer normalization
3. **Feed-forward Network**: Position-wise fully connected layer
4. **Add & Norm**: Another residual + normalization

### Decoder Components

1. **Masked Multi-head Self-attention**: Prevent looking at future tokens
2. **Cross-attention**: Keys and Values from encoder
3. **Feed-forward Network**: Position-wise FC layer
4. **Linear Layer**: Map to vocabulary for prediction

### Encoder Layer in PyTorch

```python
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Parameters
nhead = 2
d_model = 4
num_layers = 2

# Create encoder layer
encoder_layer = TransformerEncoderLayer(
    d_model=d_model, 
    nhead=nhead
)

# Create transformer encoder
transformer_encoder = TransformerEncoder(
    encoder_layer, 
    num_layers=num_layers
)

# Input
X = torch.randn(7, 5, 4)  # seq_len, batch, embed_dim

# Forward pass
output = transformer_encoder(X)
# Output size matches input
```

### Key Properties

- **Input dimension** must be divisible by number of heads
- **Stacking layers**: Models complex relationships
- **Residual connections**: Gradient flow, deeper networks
- **Layer normalization**: Stabilizes training

---

## Transformers for Text Classification

### Overview

- Traditional NN loses contextual relationships between words
- Transformers process entire sequence collectively, retaining context
- Integrate transformer attention layers for document classification

### Data Pipeline

```python
from torchtext.datasets import AG_NEWS

# Create iterators
train_iter, test_iter = AG_NEWS(split='train'), AG_NEWS(split='test')

# Create tokenizer
tokenizer = Tokenizer(language='en')

# Generate tokens and build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter))
```

### Custom Collate Function

- Handle variable length sequences
- Apply zero padding for standardization
- Output: Sequence index (not single label)

### Model Architecture

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classifier
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x) * math.sqrt(embed_dim)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Mean pooling (aggregate sequence info)
        x = x.mean(dim=0)
        
        # Classification
        x = self.fc(x)
        return x
```

### Input/Output Dimensions

- Input: (seq_len, batch) → tokens
- Embedding: (seq_len, batch, embed_dim)
- Positional encoding: Same dimensions
- Transformer encoder: Same dimensions (contextual embeddings)
- Mean pooling: (batch, embed_dim)
- Output: (batch, num_classes)

### Training

```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch.text)
        loss = criterion(output, batch.label)
        loss.backward()
        optimizer.step()
```

### Results

- Training loss decreases over epochs
- Validation accuracy increases
- Transformers perform better than traditional NNs on larger datasets

---

## Optimization Techniques for Transformers

### Gradient Accumulation

- Accumulate gradients over multiple steps before updating
- Simulates larger batch sizes with limited memory
- Stabilizes training on complex datasets

### Mixed-Precision Training

- Use FP16 (half-precision) for certain calculations
- Reduces memory usage and speeds up training
- PyTorch: Automatic Mixed Precision (AMP)

### Distributed Training

- **Data Parallelism**: Each device processes portion of data
- **Model Parallelism**: Split model across devices (for large models)

### Efficient Optimizers

- **AdamW**: Adam with weight decay, better generalization
- **LAMB**: Layer-wise Adaptive Moments, for large-batch training

---

## Module 1 Summary: Key Takeaways

### Positional Encoding

- Transformers process tokens in parallel, losing position info
- Add positional encoding using sine/cosine waves or learnable parameters
- Formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))

### Attention Mechanism

- Uses Query (Q), Key (K), Value (V) matrices
- Formula: Attention(Q, K, V) = softmax(QK^T/√d_k)V

### Self-Attention

- Each word attends to every other word in parallel
- Generates contextual embeddings
- Predicts next word based on context

### Multi-Head Attention

- Multiple attention processes in parallel
- Each head focuses on different dependencies

### Transformer for Classification

- Encoder + mean pooling + linear classifier
- Retains context while classifying text
- Training similar to standard classification

### Optimization

- Gradient accumulation, mixed precision, distributed training
- AdamW, LAMB optimizers