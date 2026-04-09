# Module 3: Transformers in Keras

## Introduction to Transformers

### Overview

**Transformers** have revolutionized the field of natural language processing and are now being applied to a wide range of tasks, including:
- Image processing
- Time series prediction
- Natural language understanding
- Machine translation
- Text generation

### Historical Context

**Transformers** were introduced by **Vaswani et al.** in the landmark paper:

> **"Attention Is All You Need"** (2017)

**Key Innovation:** Unlike traditional sequence models such as RNNs (Recurrent Neural Networks), transformers leverage **self-attention mechanisms** to process sequential data **in parallel**, making them highly efficient and powerful.

### Modern Applications

Transformers are now the backbone of state-of-the-art models like:

| Model | Application | Description |
|-------|-------------|-------------|
| **BERT** | Language Understanding | Bidirectional Encoder Representations from Transformers |
| **GPT** | Text Generation | Generative Pre-trained Transformer |
| **T5** | Text-to-Text | Text-To-Text Transfer Transformer |
| **ViT** | Image Classification | Vision Transformer |
| **Whisper** | Speech Recognition | OpenAI's speech model |

---

## Transformer Architecture

### Main Components

The transformer model consists of **two main parts**:

```
┌─────────────────────────────────────────────────────────┐
│                    Transformer Model                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐         ┌─────────────────┐        │
│  │    ENCODER      │         │    DECODER      │        │
│  │                 │         │                 │        │
│  │  • Self-Attention│        │  • Self-Attention│        │
│  │  • Feed Forward  │        │  • Cross-Attention│       │
│  │  • Layer Norm    │        │  • Feed Forward  │        │
│  │  • Residual Conn │        │  • Layer Norm    │        │
│  └─────────────────┘         └─────────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Encoder-Decoder Structure

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Encoder** | Processes input sequence | Self-attention + Feed Forward |
| **Decoder** | Generates output sequence | Self-attention + Cross-attention + Feed Forward |

Both encoder and decoder are composed of layers that include:
- **Self-attention mechanisms**
- **Feed forward neural networks**
- **Residual connections**
- **Layer normalization**

---

## Self-Attention Mechanism

### Overview

**Self-attention** is the **core component** of the transformer architecture.

**Key Features:**
- Allows each word in the input to attend to **every other word**
- Captures contexts and relationships more effectively
- Enables understanding of long-range dependencies

### How Self-Attention Works

Each word is represented by **three vectors**:

| Vector | Purpose |
|--------|---------|
| **Query (Q)** | Represents what the word is looking for |
| **Key (K)** | Represents what the word contains |
| **Value (V)** | Contains the actual content/information |

### Attention Score Calculation

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**Process:**
1. Compute **dot product** of Query and Key vectors
2. **Scale** by √d_k (dimension of key vectors)
3. Apply **Softmax** to get attention weights
4. **Weight** the Value vectors to get output

```
┌─────────────────────────────────────────────────────────┐
│              Self-Attention Process                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Words → [Q, K, V] Vectors                        │
│       ↓                                                  │
│  Compute Q · K^T (Attention Scores)                     │
│       ↓                                                  │
│  Apply Softmax (Attention Weights)                      │
│       ↓                                                  │
│  Weight Values → Output                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Benefits of Self-Attention

| Benefit | Description |
|---------|-------------|
| **Parallel Processing** | All positions processed simultaneously |
| **Long-range Dependencies** | Captures relationships regardless of distance |
| **Contextual Understanding** | Each word understands context from all other words |
| **Efficiency** | More efficient than sequential RNN processing |

---

## Implementing Self-Attention in Keras

### Self-Attention Class

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Initialize dense layers for Q, K, V projections
        self.values = Dense(self.head_dim * self.heads)
        self.keys = Dense(self.head_dim * self.heads)
        self.queries = Dense(self.head_dim * self.heads)
        self.fc_out = Dense(embed_size)
    
    def call(self, values, keys, query, mask):
        # Project inputs through dense layers
        V = self.values(values)
        K = self.keys(keys)
        Q = self.queries(query)
        
        # Split into multiple heads
        V = self.split_heads(V)
        K = self.split_heads(K)
        Q = self.split_heads(Q)
        
        # Compute attention scores
        attention = tf.matmul(Q, K, transpose_b=True)
        attention = attention / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Apply mask (optional, for padding or lookahead)
        if mask is not None:
            attention = attention + (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention = tf.nn.softmax(attention)
        
        # Apply attention weights to values
        out = tf.matmul(attention, V)
        
        # Concatenate heads
        out = self.combine_heads(out)
        
        # Final projection
        out = self.fc_out(out)
        
        return out
    
    def split_heads(self, x):
        batch_size, seq_length, _ = x.shape
        x = tf.reshape(x, (batch_size, seq_length, self.heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.shape
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, seq_length, self.embed_size))
```

### Key Methods Explained

| Method | Purpose |
|--------|---------|
| `__init__` | Initializes dense layers for Q, K, V projections |
| `call` | Computes attention weights and applies them to value vectors |
| `split_heads` | Splits embeddings into multiple attention heads |
| `combine_heads` | Combines multiple attention heads back together |

---

## Transformer Encoder

### Architecture

The **transformer encoder** is composed of **multiple layers**, each containing:
1. **Multi-head self-attention mechanism**
2. **Feed forward neural network**
3. **Residual connections** (around each sub-layer)
4. **Layer normalization** (after each residual connection)

### Input Processing

```
Input Tokens → Embedding → Positional Encoding → Encoder Layers → Output
```

**Positional Encoding:**
- Adds information about the **position of words** in the sequence
- Helps the model understand the **order of words**
- Since transformers don't have inherent sequential processing like RNNs

### Transformer Encoder Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, forward_expansion, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        
        # Multi-head attention
        self.attention = SelfAttention(embed_size, heads)
        
        # Feed forward network
        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation='relu'),
            Dense(embed_size)
        ])
        
        # Layer normalization
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
    
    def call(self, query, key, value, mask, training):
        # Apply multi-head attention
        attention_output = self.attention(value, key, query, mask)
        
        # Residual connection + Layer normalization
        x = self.norm1(query + self.dropout(attention_output, training=training))
        
        # Apply feed forward network
        ff_output = self.feed_forward(x)
        
        # Residual connection + Layer normalization
        out = self.norm2(x + self.dropout(ff_output, training=training))
        
        return out
```

### Encoder Layer Components

| Component | Purpose |
|-----------|---------|
| **Multi-head Attention** | Applies self-attention with multiple attention heads |
| **Feed Forward Network** | Transforms input data after attention mechanism |
| **Residual Connection** | Helps gradient flow, prevents vanishing gradients |
| **Layer Normalization** | Stabilizes training, speeds up convergence |
| **Dropout** | Prevents overfitting |

---

## Transformer Decoder

### Architecture

The **transformer decoder** is similar to the encoder, but with an **additional cross-attention mechanism**.

**Key Difference:**
- Decoder has **three** sub-layers instead of two:
  1. **Masked self-attention** (attends to previous positions only)
  2. **Cross-attention** (attends to encoder output)
  3. **Feed forward network**

### Decoder Flow

```
Target Sequence → Masked Self-Attention → Cross-Attention → Feed Forward → Output
                        ↓                        ↓
                    (Decoder)              (Encoder Output)
```

**Cross-Attention:**
- Allows decoder to **attend to encoder's output**
- Enables decoder to generate sequences based on **context provided by encoder**
- Query comes from decoder, Key and Value come from encoder

### Transformer Decoder Implementation

```python
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, forward_expansion, dropout_rate):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        
        # Multi-head attention (for self-attention and cross-attention)
        self.attention = SelfAttention(embed_size, heads)
        
        # Feed forward network
        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation='relu'),
            Dense(embed_size)
        ])
        
        # Layer normalization
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
    
    def call(self, query, key, value, encoder_output, mask, training):
        # Apply self-attention (masked)
        attention_output = self.attention(value, key, query, mask)
        
        # Residual connection + Layer normalization
        x = self.norm1(query + self.dropout(attention_output, training=training))
        
        # Apply cross-attention with encoder output
        cross_attention_output = self.attention(
            encoder_output, encoder_output, x, None
        )
        
        # Residual connection + Layer normalization
        x = self.norm2(x + self.dropout(cross_attention_output, training=training))
        
        # Apply feed forward network
        ff_output = self.feed_forward(x)
        
        # Residual connection + Layer normalization
        out = self.norm3(x + self.dropout(ff_output, training=training))
        
        return out
```

### Decoder Layer Components

| Component | Purpose |
|-----------|---------|
| **Masked Self-Attention** | Attends only to previous positions (prevents cheating) |
| **Cross-Attention** | Attends to encoder output for context |
| **Feed Forward Network** | Transforms combined information |
| **Residual Connections** | Three residual connections for stable training |
| **Layer Normalization** | Three normalization layers |

---

## Complete Transformer Model

### Putting It All Together

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer, Dropout, Dense

class Transformer(tf.keras.Model):
    def __init__(self, embed_size, num_heads, num_encoder_layers, 
                 num_decoder_layers, forward_expansion, dropout_rate, max_length):
        super(Transformer, self).__init__()
        
        self.embed_size = embed_size
        
        # Encoder and decoder stacks
        self.encoder_layers = [
            TransformerEncoder(embed_size, num_heads, forward_expansion, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        
        self.decoder_layers = [
            TransformerDecoder(embed_size, num_heads, forward_expansion, dropout_rate)
            for _ in range(num_decoder_layers)
        ]
        
        self.dropout = Dropout(dropout_rate)
        
    def call(self, src, trg, src_mask, trg_mask, training):
        # Embedding and positional encoding would be added here
        
        # Pass through encoder layers
        encoder_output = src
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(
                encoder_output, encoder_output, encoder_output, src_mask, training
            )
        
        # Pass through decoder layers
        decoder_output = trg
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output, decoder_output, decoder_output,
                encoder_output, trg_mask, training
            )
        
        return decoder_output
```

---

## Positional Encoding

### Why Positional Encoding?

Since transformers process all positions **in parallel**, they have no inherent understanding of sequence order. Positional encoding adds this information.

### Implementation

```python
import numpy as np

def positional_encoding(max_length, embed_size):
    """Generate positional encoding matrix."""
    
    position = np.arange(max_length)[:, np.newaxis]
    i = np.arange(embed_size)[np.newaxis, :]
    
    # Different frequencies for different dimensions
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_size))
    angle_rads = position * angle_rates
    
    # Apply sin to even positions, cos to odd positions
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Add positional encoding to embeddings
embeddings = Embedding(vocab_size, embed_size)(input_tokens)
embeddings *= tf.math.sqrt(tf.cast(embed_size, tf.float32))
position_encoding = positional_encoding(max_length, embed_size)
encoded_output = embeddings + position_encoding
```

---

## Multi-Head Attention

### Concept

**Multi-head attention** allows the model to attend to information from different representation subspaces simultaneously.

```
┌─────────────────────────────────────────────────────────┐
│              Multi-Head Attention                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input → Split into N heads → Parallel Attention → Concat │
│                                                          │
│  Head 1: Q1, K1, V1 → Attention1                        │
│  Head 2: Q2, K2, V2 → Attention2                        │
│  ...                                                     │
│  Head N: QN, KN, VN → AttentionN                        │
│                                                          │
│  Concatenate → Linear Projection → Output               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Multiple Perspectives** | Each head learns different attention patterns |
| **Rich Representations** | Combines information from multiple subspaces |
| **Improved Performance** | Better than single attention mechanism |

---

## Building Transformers for Sequential Data

### Overview

**Sequential data** is prevalent in many real-world applications. Examples include:

| Data Type | Characteristics | Examples |
|-----------|-----------------|----------|
| **Text** | Word meaning depends on context from preceding words | Natural language, documents |
| **Time Series** | Each data point influenced by past values | Stock prices, weather, sensor data |
| **Audio** | Sound waves depend on previous samples | Speech, music, environmental sounds |

### Characteristics of Sequential Data

- **Order matters**: Elements have a specific sequence
- **Dependencies**: Each element depends on previous elements
- **Context**: Meaning/value is influenced by surrounding elements

---

### Limitations of Traditional Models (RNNs/LSTMs)

| Limitation | Description |
|------------|-------------|
| **Long-term Dependencies** | Struggle to capture dependencies over long sequences |
| **Sequential Processing** | Must process data one element at a time (no parallelization) |
| **Vanishing Gradients** | Gradients diminish over long sequences |
| **Training Speed** | Slow due to sequential nature |

**Traditional Approach:**
```
RNN/LSTM: Input[0] → Input[1] → Input[2] → ... → Input[n]
          (Sequential - must wait for previous step)
```

---

### How Transformers Address These Limitations

**Transformers use self-attention mechanisms** that allow the model to:

| Advantage | Description |
|-----------|-------------|
| **Attend to All Positions** | Focus on all positions in input sequence simultaneously |
| **Handle Long-range Dependencies** | Capture relationships regardless of distance |
| **Efficient Parallelization** | Process all positions in parallel during training |
| **Faster Training** | No sequential dependency enables GPU acceleration |

**Transformer Approach:**
```
Transformer: All Input Positions → Self-Attention → Output
             (Parallel - all positions processed at once)
```

---

## Building a Transformer Model in Keras

### Multi-Head Self-Attention Class

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Dense layers for Q, K, V projections
        self.Wq = Dense(embed_size)
        self.Wk = Dense(embed_size)
        self.Wv = Dense(embed_size)
        self.Wo = Dense(embed_size)
    
    def attention(self, query, key, value, mask=None):
        """Compute attention scores and weighted sum of values."""
        
        # Compute attention scores (dot product of Q and K)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale by sqrt of head dimension
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add mask if provided (for padding or lookahead)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Weighted sum of values
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        """Split input into multiple heads for parallel attention computation."""
        
        # Reshape to (batch_size, seq_length, num_heads, head_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        
        # Transpose to (batch_size, num_heads, seq_length, head_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None):
        """Apply self-attention mechanism and combine heads."""
        
        batch_size = tf.shape(inputs)[0]
        
        # Project inputs to Q, K, V
        query = self.Wq(inputs)
        key = self.Wk(inputs)
        value = self.Wv(inputs)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Compute attention
        scaled_attention, attention_weights = self.attention(query, key, value, mask)
        
        # Combine heads (transpose and reshape)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                       (batch_size, -1, self.embed_size))
        
        # Final projection
        output = self.Wo(concat_attention)
        
        return output
```

### Key Methods Explained

| Method | Purpose |
|--------|---------|
| `attention()` | Computes attention scores and weighted sum of values |
| `split_heads()` | Splits input into multiple heads for parallel attention computation |
| `call()` | Applies self-attention mechanism and combines heads |

---

### Transformer Block Class

```python
class TransformerBlock(Layer):
    def __init__(self, embed_size, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_size)
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        """Apply self-attention followed by feed-forward network with residual connections and layer normalization."""
        
        # Apply multi-head self-attention
        attention_output = self.attention(inputs, mask=mask)
        
        # Dropout and residual connection
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)  # Residual connection
        
        # Apply feed-forward network
        ffn_output = self.ffn(out1)
        
        # Dropout and residual connection
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2
```

### Transformer Block Structure

```
Input → Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm → Output
              ↓                           ↑            ↓              ↑
         (Attention)                 (Residual)   (FF Network)   (Residual)
```

---

### Encoder Layer Implementation

```python
class EncoderLayer(Layer):
    def __init__(self, embed_size, num_heads, ff_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention mechanism
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        
        # Feed-forward neural network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_size)
        ])
        
        # Layer normalization (applied after each sub-layer)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout for regularization
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        """Apply self-attention followed by feed-forward network with residual connections and layer normalization."""
        
        # Sub-layer 1: Multi-head self-attention
        attention_output = self.attention(inputs, mask=mask)
        attention_output = self.dropout1(attention_output, training=training)
        # Residual connection around sub-layer 1
        out1 = self.layernorm1(inputs + attention_output)
        
        # Sub-layer 2: Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection around sub-layer 2
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
```

### Encoder Layer Components

| Component | Purpose |
|-----------|---------|
| **Multi-head Self-Attention** | Captures relationships between all positions in sequence |
| **Feed-Forward Network** | Transforms attention output with non-linear activation |
| **Residual Connections** | Helps gradient flow, prevents vanishing gradients |
| **Layer Normalization** | Stabilizes training, applied after each sub-layer |
| **Dropout** | Prevents overfitting during training |

---

### Complete Encoder Stack

```python
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_size, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # Stack of encoder layers
        self.encoder_layers = [
            EncoderLayer(embed_size, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        """Pass input through all encoder layers."""
        
        x = inputs
        
        # Apply each encoder layer sequentially
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training, mask=mask)
        
        return x

# Example usage
encoder = TransformerEncoder(
    num_layers=6,
    embed_size=512,
    num_heads=8,
    ff_dim=2048,
    dropout_rate=0.1
)
```

---

## Applications for Sequential Data

### Natural Language Processing

| Task | Description |
|------|-------------|
| **Machine Translation** | Translate text from one language to another |
| **Text Summarization** | Generate concise summaries of long documents |
| **Question Answering** | Answer questions based on context |
| **Sentiment Analysis** | Classify text sentiment |

### Time Series Forecasting

| Task | Description |
|------|-------------|
| **Stock Price Prediction** | Predict future stock prices |
| **Weather Forecasting** | Predict weather patterns |
| **Demand Forecasting** | Predict product demand |
| **Anomaly Detection** | Detect unusual patterns in data |

### Audio Processing

| Task | Description |
|------|-------------|
| **Speech Recognition** | Convert speech to text |
| **Music Generation** | Generate music sequences |
| **Sound Classification** | Classify audio clips |

---

## Summary: Building Transformers for Sequential Data

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Sequential Data** | Characterized by order and dependency of elements |
| **Self-Attention** | Allows model to attend to all positions simultaneously |
| **Multi-Head Attention** | Multiple attention heads for parallel computation |
| **Encoder-Decoder** | Typical transformer architecture |
| **Residual Connections** | Help gradient flow through deep networks |
| **Layer Normalization** | Stabilizes training |

### Transformers vs. RNNs/LSTMs

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| **Processing** | Sequential | Parallel |
| **Long-range Dependencies** | Struggles | Excellent |
| **Training Speed** | Slow | Fast |
| **Memory Efficiency** | Lower | Higher |

### Transformer Components for Sequential Data

| Component | Purpose |
|-----------|---------|
| **Multi-Head Self-Attention** | Capture relationships across sequence |
| **Feed-Forward Network** | Transform representations |
| **Residual Connections** | Enable deep architectures |
| **Layer Normalization** | Stabilize training |

### Key Takeaways

- Sequential data is characterized by **order and dependencies** between elements
- Transformers address **limitations of RNNs/LSTMs** with self-attention mechanisms
- Self-attention allows model to **attend to all positions simultaneously**
- Typical transformer consists of **encoder and decoder** components
- Both components use **self-attention and feed-forward neural networks**
- Transformers enable **efficient parallelization** during training
- State-of-the-art approach for **NLP and time series forecasting**

Understanding how to build transformers for sequential data enables you to handle complex real-world applications more effectively than traditional recurrent models.

---

## Complete Module 3 Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Transformer Architecture** | Encoder-Decoder structure with self-attention |
| **Self-Attention** | Allows each word to attend to all other words |
| **Query, Key, Value** | Three vectors used to compute attention |
| **Multi-Head Attention** | Multiple attention heads for richer representations |
| **Positional Encoding** | Adds sequence order information |
| **Residual Connections** | Helps gradient flow through deep networks |
| **Layer Normalization** | Stabilizes training |

### Transformer Components

| Component | Purpose |
|-----------|---------|
| **Encoder** | Processes input sequence with self-attention |
| **Decoder** | Generates output with self and cross-attention |
| **Feed Forward** | Transforms data after attention |
| **Layer Norm** | Normalizes activations |
| **Dropout** | Prevents overfitting |

### Applications

- **Natural Language Processing**: Translation, summarization, question answering
- **Computer Vision**: Image classification, object detection (Vision Transformers)
- **Time Series**: Forecasting, anomaly detection
- **Multi-modal**: Text-to-image, video understanding

### Key Takeaways

- Transformers leverage **self-attention** for parallel processing
- **Encoder-Decoder** architecture enables sequence-to-sequence tasks
- **Multi-head attention** captures multiple relationship types
- **Positional encoding** provides sequence order information
- **Residual connections and layer normalization** stabilize deep networks
- Transformers are the foundation of **BERT, GPT**, and other state-of-the-art models
- Sequential data is characterized by **order and dependencies** between elements
- Transformers address **limitations of RNNs/LSTMs** with self-attention mechanisms
- Self-attention allows model to **attend to all positions simultaneously**
- Transformers enable **efficient parallelization** during training
- State-of-the-art approach for **NLP and time series forecasting**

Understanding and implementing transformers enables you to build powerful models for various tasks in NLP and beyond.

---

## Advanced Transformer Applications

### Overview

Although transformers have revolutionized **natural language processing**, their versatile architecture makes them applicable to a wide range of domains including:

| Domain | Application | Key Models |
|--------|-------------|------------|
| **Computer Vision** | Image classification, object detection | Vision Transformer (ViT) |
| **Speech Recognition** | Speech-to-text, audio processing | Wav2Vec, Speech Transformer |
| **Reinforcement Learning** | Action prediction, decision making | Decision Transformer |

---

## Vision Transformers (ViT)

### Overview

**Vision Transformers (ViTs)** have shown that self-attention mechanisms can be applied to image data, often outperforming traditional **Convolutional Neural Networks (CNNs)**.

**Key Concept:** ViTs divide an image into **patches** and treat them as a **sequence**, similar to words in a sentence.

### How ViT Works

```
Input Image → Split into Patches → Embed Patches → Add Positional Encoding → 
Transformer Encoder → Classification Head → Output
```

**Process:**
1. **Divide image** into fixed-size patches (e.g., 16×16 pixels)
2. **Flatten each patch** into a vector
3. **Embed patches** into desired dimension
4. **Add positional encoding** to retain spatial information
5. **Pass through transformer encoder** layers
6. **Classify** using the output

### Vision Transformer Implementation in Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    """Core transformer block with multi-head self-attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Self-attention
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        x1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x1)
        ffn_output = self.dropout2(ffn_output, training=training)
        x2 = self.layernorm2(x1 + ffn_output)
        
        return x2


class PatchEmbedding(layers.Layer):
    """Embeds image patches into the desired dimension."""
    
    def __init__(self, image_size, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = layers.Conv2D(
            embed_dim, 
            kernel_size=patch_size, 
            strides=patch_size, 
            padding='valid'
        )
    
    def call(self, images):
        x = self.projection(images)
        x = tf.reshape(x, shape=(-1, self.num_patches, self.embed_dim))
        return x


class VisionTransformer(Model):
    """Vision Transformer model for image classification."""
    
    def __init__(self, image_size, patch_size, num_classes, embed_dim, 
                 num_heads, ff_dim, num_layers, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        
        self.patch_embedding = PatchEmbedding(image_size, patch_size, embed_dim)
        
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )
        
        self.pos_embedding = self.add_weight(
            shape=(1, self.patch_embedding.num_patches + 1, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='pos_embedding'
        )
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.classifier = layers.Dense(num_classes, activation='softmax')
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, images, training=False):
        batch_size = tf.shape(images)[0]
        
        # Embed patches
        x = self.patch_embedding(images)
        
        # Add class token
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        x = tf.concat([cls_tokens, x], axis=1)
        
        # Add positional encoding
        x = x + self.pos_embedding
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Use class token for classification
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits


def extract_patches(images, patch_size):
    """Extract patches from images for the transformer model."""
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    return patches


# Example Usage
if __name__ == "__main__":
    image_size = 224
    patch_size = 16
    num_classes = 10
    embed_dim = 768
    num_heads = 12
    ff_dim = 3072
    num_layers = 12
    
    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers
    )
    
    sample_images = tf.random.normal((4, image_size, image_size, 3))
    output = vit(sample_images)
    print(f"Output shape: {output.shape}")  # (4, num_classes)
```

### ViT Architecture Components

| Component | Purpose |
|-----------|---------|
| **Patch Embedding** | Converts image patches into embeddings |
| **Class Token** | Special token for classification output |
| **Positional Encoding** | Retains spatial information of patches |
| **Transformer Blocks** | Process patch sequences with self-attention |
| **Classification Head** | Maps final representation to class labels |

### ViT vs. CNN Comparison

| Aspect | CNN | Vision Transformer |
|--------|-----|-------------------|
| **Feature Extraction** | Local receptive fields | Global attention |
| **Inductive Bias** | Translation invariance | Minimal (learns from data) |
| **Data Requirements** | Works with less data | Needs large datasets |
| **Performance** | Good on standard tasks | State-of-the-art on large datasets |

---

## Speech Transformers

### Overview

Transformers are being used in **speech recognition** by converting audio signals into **spectrograms**. Transformers can then process the sequential nature of speech data.

**Key Models:**
- **Wav2Vec**: Self-supervised learning for speech
- **Speech Transformer**: Direct transformer-based speech recognition

### How Speech Transformers Work

```
Audio Signal → Spectrogram → Feature Extraction → Transformer Encoder → Text Output
```

### Speech Transformer Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class SpeechTransformer(Model):
    """Speech Transformer model for speech recognition."""
    
    def __init__(self, num_classes, embed_dim=256, num_heads=8, 
                 ff_dim=1024, num_layers=6, dropout_rate=0.1):
        super(SpeechTransformer, self).__init__()
        
        # Convolutional front-end for feature extraction
        self.conv1 = layers.Conv1D(256, kernel_size=7, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(256, kernel_size=5, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv1D(embed_dim, kernel_size=3, strides=2, padding='same', activation='relu')
        
        self.pos_encoding = self.add_weight(
            shape=(1, 500, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='pos_encoding'
        )
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.output_layer = layers.Dense(num_classes, activation='softmax')
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, spectrograms, training=False):
        # Convolutional feature extraction
        x = self.conv1(spectrograms)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Generate output
        logits = self.output_layer(x)
        
        return logits


# Example Usage
if __name__ == "__main__":
    num_classes = 29  # 26 letters + space + apostrophe + <blank>
    max_frames = 500
    feature_dim = 80
    
    speech_transformer = SpeechTransformer(
        num_classes=num_classes,
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=6
    )
    
    sample_spectrograms = tf.random.normal((4, max_frames, feature_dim))
    output = speech_transformer(sample_spectrograms)
    print(f"Output shape: {output.shape}")  # (4, max_frames, num_classes)
```

### Speech Processing Pipeline

| Step | Description |
|------|-------------|
| **Audio Input** | Raw audio waveform |
| **Preprocessing** | Convert to spectrogram/MFCC |
| **Feature Extraction** | Convolutional layers extract features |
| **Transformer Encoding** | Self-attention processes sequence |
| **Output Generation** | CTC loss or attention-based decoding |

---

## Decision Transformers for Reinforcement Learning

### Overview

Transformers have found applications in **reinforcement learning (RL)** where they can be used to model complex dependencies in sequences of **states and actions**.

**Decision Transformer** leverages the transformer architecture to predict actions based on **past trajectories**, enabling more efficient learning in complex environments.

### How Decision Transformers Work

```
State-Action-Reward Sequence → Embedding → Transformer → Predict Next Action
```

**Key Idea:** Treat RL as a sequence modeling problem where the model predicts actions conditioned on:
- Past states
- Past actions
- Desired return (reward-to-go)

### Decision Transformer Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class DecisionTransformer(Model):
    """Decision Transformer model for reinforcement learning."""
    
    def __init__(self, state_dim, action_dim, embed_dim=256, 
                 num_heads=8, ff_dim=512, num_layers=4, 
                 max_timestep=100, dropout_rate=0.1):
        super(DecisionTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # State embedding
        self.state_embedding = layers.Dense(embed_dim)
        
        # Action embedding
        self.action_embedding = layers.Dense(embed_dim)
        
        # Reward-to-go embedding
        self.reward_embedding = layers.Dense(embed_dim)
        
        # Timestep embedding
        self.timestep_embedding = layers.Embedding(max_timestep, embed_dim)
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.action_predictor = layers.Dense(action_dim, activation='tanh')
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, states, actions, rewards, timesteps, training=False):
        batch_size = tf.shape(states)[0]
        seq_len = tf.shape(states)[1]
        
        # Embed states, actions, and rewards
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        reward_emb = self.reward_embedding(rewards)
        timestep_emb = self.timestep_embedding(timesteps)
        
        # Combine embeddings
        x = tf.stack([state_emb, action_emb, reward_emb], axis=2)
        x = tf.reshape(x, (batch_size, seq_len * 3, self.embed_dim))
        x = x + timestep_emb
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Extract action predictions
        action_predictions = self.action_predictor(x[:, 1::3, :])
        
        return action_predictions


# Example Usage
if __name__ == "__main__":
    state_dim = 17
    action_dim = 6
    seq_len = 20
    batch_size = 4
    
    decision_transformer = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_layers=4,
        max_timestep=100
    )
    
    sample_states = tf.random.normal((batch_size, seq_len, state_dim))
    sample_actions = tf.random.uniform((batch_size, seq_len, action_dim), -1, 1)
    sample_rewards = tf.random.normal((batch_size, seq_len, 1))
    sample_timesteps = tf.random.uniform(
        (batch_size, seq_len), 0, 100, dtype=tf.int32
    )
    
    predicted_actions = decision_transformer(
        sample_states, sample_actions, sample_rewards, sample_timesteps
    )
    
    print(f"Predicted actions shape: {predicted_actions.shape}")
```

### Decision Transformer Components

| Component | Purpose |
|-----------|---------|
| **State Embedding** | Encodes environment states |
| **Action Embedding** | Encodes past actions |
| **Reward Embedding** | Encodes desired return (reward-to-go) |
| **Timestep Embedding** | Provides temporal context |
| **Transformer Blocks** | Models dependencies in trajectory |
| **Action Predictor** | Outputs predicted actions |

### RL Applications

| Application | Description |
|-------------|-------------|
| **Offline RL** | Learn from fixed dataset without environment interaction |
| **Multi-task RL** | Single model for multiple tasks |
| **Long-horizon Planning** | Model long-term dependencies in trajectories |
| **Imitation Learning** | Learn from expert demonstrations |

---

## Summary: Advanced Transformer Applications

### Vision Transformers (ViT)

| Aspect | Description |
|--------|-------------|
| **Input** | Image divided into patches |
| **Processing** | Patches treated as sequence |
| **Advantage** | Global attention, state-of-the-art performance |
| **Use Cases** | Image classification, object detection, segmentation |

### Speech Transformers

| Aspect | Description |
|--------|-------------|
| **Input** | Audio converted to spectrograms |
| **Processing** | Sequential frames processed by transformer |
| **Advantage** | Captures long-range dependencies in speech |
| **Use Cases** | Speech-to-text, speaker identification |

### Decision Transformers (RL)

| Aspect | Description |
|--------|-------------|
| **Input** | State-action-reward sequences |
| **Processing** | Trajectories modeled as sequences |
| **Advantage** | Unified approach to RL as sequence modeling |
| **Use Cases** | Offline RL, multi-task learning, planning |

### Key Takeaways

- **Transformers are versatile**: Applicable beyond NLP to vision, speech, and RL
- **Vision Transformers (ViT)**: Apply self-attention to image patches, competing with CNNs
- **Speech Transformers**: Process spectrograms for speech recognition tasks
- **Decision Transformers**: Model RL trajectories for action prediction
- **Unified Architecture**: Same transformer architecture adapted for different domains
- **State-of-the-art**: Transformers achieve top performance across multiple domains

Transformers' flexible architecture enables them to be adapted to various domains, making them a powerful tool for modern machine learning applications.

---

## Transformers for Time Series Prediction

### Overview

**Time series data** is a sequence of data points collected or recorded at successive points in time.

**Traditional Methods:**
| Method | Description | Limitations |
|--------|-------------|-------------|
| **ARIMA** | AutoRegressive Integrated Moving Average | Linear assumptions, struggles with complex patterns |
| **RNN** | Recurrent Neural Network | Vanishing gradients, sequential processing |
| **LSTM** | Long Short-Term Memory | Better than RNN but still sequential, slow training |

**Transformers** have shown great promise in capturing long-term dependencies in sequential data, making them highly effective for time series forecasting.

---

### Advantages of Transformers for Time Series

| Advantage | Description |
|-----------|-------------|
| **Long-Range Dependencies** | Self-attention captures relationships across entire sequence |
| **Parallel Processing** | Process entire sequence at once (unlike RNNs/LSTMs) |
| **Faster Training** | Parallelization enables GPU acceleration |
| **Variable Length Sequences** | Handle different sequence lengths gracefully |
| **Missing Data Handling** | More robust to missing values than traditional methods |

### Why Transformers Excel at Time Series

```
Traditional RNN/LSTM:
Input[t0] → Input[t1] → Input[t2] → ... → Input[tn]
(Sequential - must wait for previous step, loses long-range context)

Transformer:
All Time Steps → Self-Attention → Output
(Parallel - all time steps processed simultaneously, captures global dependencies)
```

---

## Building a Transformer Model for Time Series

### Model Architecture

```
Input Sequence → Embedding Layer → Transformer Blocks → Dense Layer → Output Prediction
```

**Key Components:**
1. **Embedding Layer**: Converts input sequences into dense vectors
2. **Transformer Blocks**: Multiple blocks with self-attention and feed-forward networks
3. **Dense Output Layer**: Predicts the next value(s) in the series

### Transformer Block Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head self-attention and feed-forward layers."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )
        
        # Feed-forward neural network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        """Apply self-attention and feed-forward layers with residual connections and layer normalization."""
        
        # Self-attention
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        x1 = self.layernorm1(inputs + attention_output)  # Residual connection
        
        # Feed-forward network
        ffn_output = self.ffn(x1)
        ffn_output = self.dropout2(ffn_output, training=training)
        x2 = self.layernorm2(x1 + ffn_output)  # Residual connection
        
        return x2
```

---

## Data Preparation for Time Series

### Stock Price Dataset Example

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('stock_prices.csv')

# Select the 'Close' prices
data = df[['Close']].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences of data points
def create_sequences(data, seq_length):
    """Create sequences and corresponding labels for training."""
    sequences = []
    labels = []
    
    for i in range(len(data) - seq_length):
        # Input sequence
        seq = data[i:i + seq_length]
        # Label: next value in the series
        label = data[i + seq_length]
        
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# Define sequence length (number of time steps)
SEQ_LENGTH = 60  # Use 60 days of data to predict next day

# Create sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Print shapes for debugging
print(f"Input shape: {X.shape}")  # (num_samples, seq_length, 1)
print(f"Output shape: {y.shape}")  # (num_samples, 1)

# Check for missing values
print(f"Missing values in X: {np.isnan(X).sum()}")
print(f"Missing values in y: {np.isnan(y).sum()}")

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

### Data Preparation Steps

| Step | Description | Purpose |
|------|-------------|---------|
| **Load Dataset** | Read stock prices or time series data | Get raw data |
| **Select Feature** | Choose 'Close' prices or relevant column | Focus on target variable |
| **Normalize Data** | Scale to [0, 1] using MinMaxScaler | Improve model convergence |
| **Create Sequences** | Generate input-output pairs | Prepare for supervised learning |
| **Split Data** | Train/test split (e.g., 80/20) | Evaluate model performance |

---

## Building and Training the Model

### Complete Transformer Model for Time Series

```python
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

# Model parameters
SEQ_LENGTH = 60
FEATURE_DIM = 1
EMBED_DIM = 64
NUM_HEADS = 8
FF_DIM = 128
NUM_LAYERS = 4
DROPOUT_RATE = 0.1

# Define input shape
inputs = Input(shape=(SEQ_LENGTH, FEATURE_DIM))

# Embedding layer: Convert input sequences into dense vectors
x = Dense(EMBED_DIM)(inputs)  # Project to embedding dimension

# Add positional encoding (optional but recommended)
# For time series, you can use simple positional encoding
pos_encoding = tf.range(SEQ_LENGTH, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
x = x + pos_encoding

# Stack multiple transformer blocks
for _ in range(NUM_LAYERS):
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)(x)

# Global average pooling (optional)
x = layers.GlobalAveragePooling1D()(x)

# Add dropout for regularization
x = Dropout(DROPOUT_RATE)(x)

# Final dense layer to predict next value in the series
outputs = Dense(1)(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Compile model using Adam optimizer and Mean Squared Error loss
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']  # Mean Absolute Error as additional metric
)

# Print model summary
model.summary()

# Train the model on prepared dataset
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)
```

### Model Architecture Summary

| Layer | Purpose | Output Shape |
|-------|---------|--------------|
| **Input** | Accept sequence input | (batch, 60, 1) |
| **Dense Embedding** | Project to higher dimension | (batch, 60, 64) |
| **Transformer Blocks (×4)** | Process with self-attention | (batch, 60, 64) |
| **Global Average Pooling** | Aggregate temporal information | (batch, 64) |
| **Dropout** | Regularization | (batch, 64) |
| **Dense Output** | Predict next value | (batch, 1) |

---

## Evaluation and Prediction

### Making Predictions and Visualization

```python
import matplotlib.pyplot as plt

# Make predictions on test set
predictions = model.predict(X_test)

# Inverse transform predictions to original scale
predictions_original = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot true values vs predictions
plt.figure(figsize=(14, 6))
plt.plot(y_test_original, label='True Stock Price', linewidth=2)
plt.plot(predictions_original, label='Predicted Stock Price', linewidth=2, linestyle='--')
plt.xlabel('Time Steps (Days)')
plt.ylabel('Stock Price (USD)')
plt.title('Time Series Prediction: True vs Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test_original, predictions_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, predictions_original)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Predict future values (multi-step forecasting)
def predict_future(model, last_sequence, num_days, scaler):
    """Predict future values iteratively."""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(num_days):
        # Reshape for model input
        input_seq = current_seq[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
        
        # Predict next value
        next_pred = model.predict(input_seq, verbose=0)[0, 0]
        
        # Add to predictions
        predictions.append(next_pred)
        
        # Update sequence
        current_seq = np.append(current_seq, next_pred)
    
    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions

# Example: Predict next 30 days
last_sequence = scaled_data[-SEQ_LENGTH:]
future_predictions = predict_future(model, last_sequence, num_days=30, scaler=scaler)

# Plot future predictions
plt.figure(figsize=(14, 6))
plt.plot(df['Close'].values[-60:], label='Historical Data', linewidth=2)
plt.plot(
    range(len(df['Close']), len(df['Close']) + 30),
    future_predictions,
    label='Future Predictions',
    linewidth=2,
    linestyle='--'
)
plt.xlabel('Time Steps (Days)')
plt.ylabel('Stock Price (USD)')
plt.title('Stock Price Forecasting: Next 30 Days')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | Mean Squared Error | Penalizes large errors more |
| **RMSE** | √MSE | Same units as target variable |
| **MAE** | Mean Absolute Error | Average absolute error |

---

## Summary: Transformers for Time Series Prediction

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Time Series Data** | Sequence of data points at successive time points |
| **Self-Attention** | Captures long-term dependencies in sequential data |
| **Embedding Layer** | Converts input sequences into dense vectors |
| **Transformer Blocks** | Process sequences with multi-head attention |
| **Dense Output** | Predicts next value(s) in the series |

### Advantages Over Traditional Methods

| Method | Advantage of Transformers |
|--------|--------------------------|
| **ARIMA** | Handles non-linear patterns, no stationarity requirement |
| **RNN** | Parallel processing, no vanishing gradient problem |
| **LSTM** | Faster training, better long-range dependency capture |

### Model Components

| Component | Purpose |
|-----------|---------|
| **Embedding Layer** | Convert input to dense vectors |
| **Transformer Blocks** | Multiple blocks with self-attention and FFN |
| **Dense Output Layer** | Final prediction layer |

### Key Takeaways

- **Time series data** is a sequence of data points collected at successive time points
- **Self-attention mechanism** effectively captures long-term dependencies in time series data
- **Transformers** are powerful tools for forecasting due to parallel processing and global context
- **Key components**: Embedding layer, multiple transformer blocks, final dense layer
- **Data preparation** includes normalization, sequence creation, and train/test split
- **Evaluation** uses metrics like MSE, RMSE, and MAE
- **Visualization** helps understand model performance and forecast trends

By leveraging transformers for time series prediction, you can build more accurate and efficient forecasting models compared to traditional methods.

---

## TensorFlow for Sequential Data

### Overview

**Sequential data** is a type of data where the **order of the data points is crucial**. TensorFlow offers a range of tools and functionalities that make it well-suited for processing and analyzing sequential data.

### Characteristics of Sequential Data

| Characteristic | Description |
|----------------|-------------|
| **Temporal/Sequential Nature** | Order of data points is important |
| **Dependencies** | Each data point depends on previous values |
| **Patterns** | Contains temporal patterns and trends |

### Examples of Sequential Data

| Data Type | Examples | Applications |
|-----------|----------|--------------|
| **Time Series** | Stock prices, temperature readings, sensor data | Forecasting, anomaly detection |
| **Text** | Sentences, paragraphs, documents | NLP, translation, sentiment analysis |
| **Audio** | Speech, music, environmental sounds | Speech recognition, music generation |

---

### TensorFlow Layers for Sequential Data

TensorFlow provides several layers specifically designed for sequential data:

| Layer | Description | Use Case |
|-------|-------------|----------|
| **RNN** | Recurrent Neural Network | Basic sequential processing |
| **LSTM** | Long Short-Term Memory | Long-term dependencies |
| **GRU** | Gated Recurrent Unit | Efficient alternative to LSTM |
| **Conv1D** | 1D Convolutional Layer | Pattern recognition in sequences |
| **SimpleRNN** | Simple RNN implementation | Basic sequence modeling |

**These layers help capture:**
- Temporal dependencies
- Sequential patterns
- Long-range relationships

**Making TensorFlow powerful for:**
- Time series forecasting
- Natural language processing
- Speech recognition

---

## Building RNN Models for Time Series

### Simple RNN Model

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple sine wave dataset for demonstration
def generate_sine_wave(length=1000, periods=50):
    """Generate sine wave data for demonstration."""
    x = np.linspace(0, periods * 2 * np.pi, length)
    data = np.sin(x)
    return data

# Generate data
data = generate_sine_wave()

# Define sequence dimensions
SEQ_LENGTH = 50  # Use 50 time steps to predict next value

# Prepare dataset by creating sequences and corresponding labels
def create_sequences(data, seq_length):
    """Create sequences and labels for training."""
    sequences = []
    labels = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# Create sequences
X, y = create_sequences(data, SEQ_LENGTH)

# Reshape for RNN input (batch_size, timesteps, features)
X = X.reshape(-1, SEQ_LENGTH, 1)
y = y.reshape(-1, 1)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build RNN model using TensorFlow's SimpleRNN and Dense layers
model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    SimpleRNN(50, activation='relu'),  # RNN layer with 50 units
    Dense(1)  # Dense layer for output prediction
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='True Values', linewidth=2)
plt.plot(predictions, label='RNN Predictions', linewidth=2, linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Simple RNN: Time Series Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Simple RNN Architecture

| Layer | Units | Purpose |
|-------|-------|---------|
| **Input** | (SEQ_LENGTH, 1) | Accept sequence input |
| **SimpleRNN** | 50 | Process sequential data |
| **Dense** | 1 | Output prediction |

---

## Building LSTM Models for Time Series

### LSTM Model Implementation

**LSTMs (Long Short-Term Memory)** are a type of RNN capable of learning **long-term dependencies**, making them suitable for sequential data with long-term patterns.

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense

# Build LSTM model (replace SimpleRNN with LSTM layer)
model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    LSTM(50, activation='relu'),  # LSTM layer instead of SimpleRNN
    Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Print model summary
model.summary()

# Train the LSTM model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions
predictions = model.predict(X_test)

# Plot results to compare with true data
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='True Values', linewidth=2)
plt.plot(predictions, label='LSTM Predictions', linewidth=2, linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('LSTM: Time Series Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compare RNN vs LSTM performance
print(f"RNN Test Loss: {rnn_model.evaluate(X_test, y_test, verbose=0)[0]:.4f}")
print(f"LSTM Test Loss: {model.evaluate(X_test, y_test, verbose=0)[0]:.4f}")
```

### LSTM vs. Simple RNN

| Aspect | Simple RNN | LSTM |
|--------|------------|------|
| **Long-term Dependencies** | Struggles | Excellent |
| **Vanishing Gradient** | Susceptible | Resistant |
| **Complexity** | Lower | Higher |
| **Training Speed** | Faster | Slower |
| **Performance** | Basic tasks | Complex patterns |

### LSTM Cell Structure

```
LSTM Cell Components:
┌─────────────────────────────────────────┐
│  Forget Gate: What to remove from cell │
│  Input Gate: What new info to store    │
│  Output Gate: What to output           │
│  Cell State: Long-term memory          │
│  Hidden State: Short-term memory       │
└─────────────────────────────────────────┘
```

---

## Handling Text Data with TensorFlow

### Text Preprocessing Requirements

Text data requires specific pre-processing steps:
- **Tokenization**: Convert text to numerical tokens
- **Padding**: Ensure uniform sequence lengths
- **Vectorization**: Convert to format suitable for model training

### Text Vectorization Layer

TensorFlow's **TextVectorization** layer helps convert text data into numerical format suitable for model training.

```python
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np

# Define sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing enables text analysis",
    "TensorFlow is a powerful machine learning framework"
]

# Create TextVectorization layer to tokenize and pad text sequences
# Parameters:
# - max_tokens: Maximum vocabulary size
# - output_mode: 'int' for integer encoding, 'multi_hot' for one-hot
# - output_sequence_length: Fixed output length (padding/truncation)
vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=20
)

# Adapt the vectorizer to the text data
# This builds the vocabulary from the training data
vectorizer.adapt(texts)

# Transform text into numerical format
vectorized_text = vectorizer(texts)

# Print results
print("Original texts:")
for i, text in enumerate(texts):
    print(f"\n{i+1}. {text}")
    print(f"   Vectorized: {vectorized_text[i].numpy()}")

# Get vocabulary
vocab = vectorizer.get_vocabulary()
print(f"\nVocabulary size: {len(vocab)}")
print(f"First 10 words: {vocab[:10]}")

# Inverse lookup (numbers to text)
def inverse_vectorize(vectorized_seq):
    """Convert vectorized sequence back to text."""
    words = [vocab[idx] for idx in vectorized_seq]
    return ' '.join(words)

# Example inverse transformation
print(f"\nInverse transformation example:")
print(inverse_vectorize(vectorized_text[0].numpy()))
```

### TextVectorization Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| **max_tokens** | Maximum vocabulary size | 10000 |
| **output_mode** | Encoding type ('int', 'multi_hot', 'count') | 'int' |
| **output_sequence_length** | Fixed output sequence length | 20 |
| **pad_to_max_length** | Pad sequences to max length | True |

### Text Preprocessing Pipeline

```
Raw Text → TextVectorization → Tokenized Sequences → Embedding → Model
     ↓              ↓                  ↓                ↓
  "Hello"      [1, 5, 23]        Padded to         Embedded
  "World"      [2, 8, 45]        fixed length      to dense vectors
```

---

## Complete Sequential Data Pipeline Example

### Multi-Layer LSTM for Complex Sequences

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build a more complex LSTM model with multiple layers
model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    
    # First LSTM layer with return_sequences for stacking
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(32, activation='relu'),
    Dropout(0.2),
    
    # Dense layers for output
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae', 'mse']
)

# Train with callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)
```

---

## Summary: TensorFlow for Sequential Data

### Sequential Data Types

| Type | Characteristics | TensorFlow Layers |
|------|-----------------|-------------------|
| **Time Series** | Numerical, temporal order | RNN, LSTM, GRU, Conv1D |
| **Text** | Tokens, sentences | TextVectorization, Embedding, LSTM |
| **Audio** | Waveforms, spectrograms | Conv1D, LSTM, GRU |

### Key TensorFlow Tools

| Tool | Purpose |
|------|---------|
| **SimpleRNN** | Basic recurrent processing |
| **LSTM** | Long-term dependency learning |
| **GRU** | Efficient recurrent processing |
| **Conv1D** | Pattern recognition in sequences |
| **TextVectorization** | Text tokenization and encoding |

### Key Takeaways

- **Sequential data** is characterized by its temporal or sequential nature where order matters
- **TensorFlow provides specialized layers** for sequential data: RNNs, LSTMs, GRUs, Conv1D
- **RNN models** can be built using SimpleRNN for basic tasks
- **LSTM models** excel at learning long-term dependencies in sequential data
- **Text data requires preprocessing**: tokenization and padding using TextVectorization layer
- **TextVectorization** converts text into numerical format suitable for model training
- **Multiple layer architectures** can be stacked for complex pattern recognition

TensorFlow's comprehensive tools for sequential data make it a powerful framework for time series forecasting, natural language processing, and speech recognition tasks.

---

# Module 3 Summary and Highlights: Transformers in Keras

## Congratulations! You have completed this module.

At this point in the course, you know:

### Transformer Architecture

| Concept | Key Takeaway |
|---------|-------------|
| **Encoder-Decoder Structure** | The transformer model consists of two main parts: the encoder and the decoder |
| **Layer Composition** | Both the encoder and decoder are composed of layers that include self-attention mechanisms and feedforward neural networks |
| **Importance in NLP** | Transformers have become a cornerstone in deep learning, especially in natural language processing |
| **Practical Skills** | Understanding and implementing transformers will enable you to build powerful models for various tasks |

### Transformers for Sequential Data

| Concept | Key Takeaway |
|---------|-------------|
| **Sequential Data Characteristics** | Sequential data is characterized by its order and the dependency of each element on previous elements |
| **Addressing RNN/LSTM Limitations** | Transformers address the limitations of RNNs and LSTMs by using self-attention mechanisms |
| **Parallel Processing** | Self-attention allows the model to attend to all positions in the input sequence simultaneously |
| **Efficiency** | Parallelizable architecture enables faster training and better long-range dependency capture |

### Advanced Transformer Applications

| Domain | Application | Key Insight |
|--------|-------------|-------------|
| **Computer Vision** | Vision Transformers (ViT) | Self-attention mechanisms can be applied to image data by dividing images into patches |
| **Speech Recognition** | Speech Transformers | By converting audio signals into spectrograms, transformers can process the sequential nature of speech data |
| **Reinforcement Learning** | Decision Transformers | Transformers can model complex dependencies in sequences of states and actions for action prediction |

### Time Series Forecasting

| Concept | Key Takeaway |
|---------|-------------|
| **Time Series Data** | A sequence of data points collected or recorded at successive points in time |
| **Self-Attention for Forecasting** | By leveraging the self-attention mechanism, transformers can effectively capture long-term dependencies in time series data |
| **Model Components** | The key components include an embedding layer, multiple transformer blocks, and a final dense layer for output prediction |
| **Advantage** | Powerful tool for forecasting compared to traditional methods like ARIMA, RNN, and LSTM |

### TensorFlow for Sequential Data

| Concept | Key Takeaway |
|---------|-------------|
| **Sequential Data Nature** | Characterized by its temporal or sequential nature, meaning that the order in which data points appear is important |
| **Specialized Layers** | TensorFlow provides several layers and tools specifically designed for sequential data |
| **Available Tools** | RNNs, LSTMs, Gated Recurrent Units (GRUs), Convolutional layers for sequence data (Conv1D) |
| **Text Preprocessing** | Text data requires specific preprocessing steps, such as tokenization and padding |
| **TextVectorization** | TensorFlow's TextVectorization layer helps in converting text data into numerical format suitable for model training |

---

## Complete Module 3 Topic Summary

| Topic | Description | Key Skills Gained |
|-------|-------------|-------------------|
| **Transformer Architecture** | Encoder-decoder structure with self-attention and feedforward networks | Understand and implement transformer models |
| **Self-Attention Mechanism** | Query, Key, Value vectors for computing attention | Implement multi-head self-attention |
| **Positional Encoding** | Adding sequence order information to embeddings | Enable transformers to understand sequence order |
| **Building for Sequential Data** | Transformers for time series and sequential data | Capture long-term dependencies efficiently |
| **Vision Transformers** | Applying self-attention to image patches | Build image classification models with ViT |
| **Speech Transformers** | Processing spectrograms with transformers | Implement speech-to-text models |
| **Decision Transformers** | RL as sequence modeling problem | Apply transformers to reinforcement learning |
| **Time Series Prediction** | Transformer models for forecasting | Build forecasting models with embedding and transformer blocks |
| **TensorFlow Sequential Tools** | RNN, LSTM, GRU, Conv1D, TextVectorization | Process various types of sequential data |

---

## What You Can Do Now

After completing this module, you are able to:

1. ✅ **Explain transformer architecture** including encoder and decoder components
2. ✅ **Implement self-attention mechanisms** with Query, Key, Value vectors
3. ✅ **Build transformer models** using Keras and TensorFlow
4. ✅ **Apply transformers to sequential data** for time series forecasting
5. ✅ **Implement Vision Transformers (ViT)** for image classification
6. ✅ **Build Speech Transformers** for speech recognition tasks
7. ✅ **Create Decision Transformers** for reinforcement learning applications
8. ✅ **Use TensorFlow's sequential data tools** including RNN, LSTM, GRU, and Conv1D
9. ✅ **Preprocess text data** using TextVectorization layer
10. ✅ **Capture long-term dependencies** more efficiently than traditional RNNs/LSTMs

---

## Key Takeaways Summary

### Transformer Fundamentals
- Transformer model consists of **encoder and decoder** parts
- Both use **self-attention mechanisms** and **feedforward neural networks**
- **Cornerstone of deep learning** especially in NLP

### Sequential Data Processing
- Sequential data has **order and dependencies** between elements
- Transformers use **self-attention** to attend to all positions simultaneously
- Addresses **RNN/LSTM limitations** with parallel processing

### Cross-Domain Applications
- **Computer Vision**: Vision Transformers apply self-attention to image patches
- **Speech Recognition**: Transformers process spectrograms for speech-to-text
- **Reinforcement Learning**: Decision Transformers model state-action sequences
- **Time Series**: Self-attention captures long-term dependencies for forecasting

### TensorFlow Tools
- **RNN, LSTM, GRU**: Recurrent layers for sequential processing
- **Conv1D**: 1D convolutional layers for pattern recognition
- **TextVectorization**: Converts text to numerical format for training

---

## Next Steps

Continue building on your transformer knowledge by:
- Experimenting with pre-trained transformer models (BERT, GPT)
- Exploring attention mechanisms in more depth
- Building production-ready transformer applications
- Investigating efficient transformer variants (DistilBERT, EfficientFormer)

**Key Resources:**
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- Hugging Face Transformers: https://huggingface.co/
- Attention Is All You Need Paper: https://arxiv.org/abs/1706.03762

---

*End of Module 3: Transformers in Keras*
