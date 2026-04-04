# Module 3: Transformers in Keras - Quiz

## Question 1
**What is the main advantage of using Transformers over RNNs and LSTMs for sequential data?**

- [ ] Transformers do not require GPUs for training.
- [ ] Transformers have fewer parameters.
- [x] Transformers can handle longer sequences more efficiently.
- [ ] Transformers are only suitable for text data.

**Answer:** Transformers can handle longer sequences more efficiently.

**Explanation:** Transformers use self-attention mechanisms that allow them to capture long-range dependencies in sequences more effectively than RNNs/LSTMs. Unlike RNNs which process data sequentially and suffer from vanishing gradients over long sequences, transformers process all positions in parallel and can directly attend to any position in the sequence regardless of distance.

---

## Question 2
**Which mechanism is central to the functionality of Transformers?**

- [ ] Pooling layers
- [ ] Convolutional layers
- [x] Self-attention
- [ ] Recurrent connections

**Answer:** Self-attention

**Explanation:** Self-attention is the core component of the transformer architecture. It allows each position in the input sequence to attend to all other positions, enabling the model to capture contextual relationships and dependencies regardless of their distance in the sequence. This mechanism replaces the recurrent connections used in RNNs/LSTMs.

---

## Question 3
**In a Transformer, what are the three main vectors used in the self-attention mechanism?**

- [x] Query, Key, Value
- [ ] Forward, Backward, Center
- [ ] Mean, Median, Mode
- [ ] Input, Output, Memory

**Answer:** Query, Key, Value

**Explanation:** In self-attention, each word/position is represented by three vectors:
- **Query (Q)**: Represents what the word is looking for in other positions
- **Key (K)**: Represents what the word contains/offers
- **Value (V)**: Contains the actual content/information

The attention score is computed as the dot product of Query and Key vectors, which is then used to weight the Value vectors to produce the output.

---

## Question 4
**What role does positional encoding play in Transformers?**

- [ ] It enhances the contrast of the input images.
- [x] It adds information about the order of the elements in the sequence.
- [ ] It reduces the dimensionality of the input data.
- [ ] It normalizes the input data.

**Answer:** It adds information about the order of the elements in the sequence.

**Explanation:** Since transformers process all positions in parallel (unlike RNNs which process sequentially), they have no inherent understanding of sequence order. Positional encoding adds information about the position of each element in the sequence, helping the model understand the order of words/tokens. This is typically done using sine and cosine functions of different frequencies.

---

## Question 5
**How do multi-head self-attention mechanisms enhance the performance of Transformers?**

- [ ] By simplifying the training process
- [ ] By ensuring the model uses fewer parameters
- [ ] By reducing the model complexity
- [x] By allowing the model to focus on different parts of the input simultaneously

**Answer:** By allowing the model to focus on different parts of the input simultaneously

**Explanation:** Multi-head attention splits the embedding into multiple heads, each learning different attention patterns. This allows the model to:
- Attend to information from different representation subspaces simultaneously
- Capture multiple types of relationships (e.g., syntactic, semantic, long-range, short-range)
- Learn richer and more diverse representations
- Improve overall model performance compared to single attention mechanism

---

## Quiz Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | Transformers vs RNNs/LSTMs | Handle longer sequences efficiently |
| 2 | Core Mechanism | Self-attention |
| 3 | Self-Attention Vectors | Query, Key, Value |
| 4 | Positional Encoding | Adds order information |
| 5 | Multi-Head Attention | Focus on different parts simultaneously |

**Total Points:** 5 points

---

## Key Concepts Reviewed

| Concept | Description |
|---------|-------------|
| **Transformer Advantage** | Parallel processing, handles long-range dependencies |
| **Self-Attention** | Core mechanism allowing all positions to attend to each other |
| **Q, K, V Vectors** | Used to compute attention scores and weighted outputs |
| **Positional Encoding** | Provides sequence order information |
| **Multi-Head Attention** | Multiple attention heads for richer representations |

---

*End of Module 3 Quiz*

---

## Module 3 Graded Quiz - Advanced Transformers

## Question 1
**What is the primary purpose of using the `MultiHeadAttention` layer in a transformer block?**

- [ ] To perform a single attention operation on the input data.
- [ ] To normalize the input data before feeding it into the network.
- [x] To perform multiple parallel attention operations on the input data.
- [ ] To apply dropout to the input data for regularization.

**Answer:** To perform multiple parallel attention operations on the input data.

**Explanation:** The `MultiHeadAttention` layer performs multiple attention operations in parallel, each with different learned linear transformations. This allows the model to attend to information from different representation subspaces simultaneously, capturing various types of relationships (syntactic, semantic, short-range, long-range) in the input sequence.

---

## Question 2
**Which function is used to create positional encodings in a transformer model?**

- [ ] `get_angles`
- [ ] `embedding`
- [x] `positional_encoding`
- [ ] `transformer_block`

**Answer:** `positional_encoding`

**Explanation:** The `positional_encoding` function generates positional encodings that are added to the input embeddings to provide information about the position/order of elements in the sequence. Since transformers process all positions in parallel, they need this explicit positional information to understand sequence order. The function typically uses sine and cosine functions of different frequencies.

---

## Question 3
**What is the main advantage of using a transformer model for sequential data over traditional RNN models?**

- [ ] Transformers are always faster to train than RNNs, regardless of the sequence length.
- [x] Transformers can handle longer sequences more efficiently due to their parallelizable architecture.
- [ ] Transformers require less data preprocessing compared to RNNs.
- [ ] Transformers do not use any form of attention mechanism.

**Answer:** Transformers can handle longer sequences more efficiently due to their parallelizable architecture.

**Explanation:** Unlike RNNs which process sequences sequentially (one element at a time), transformers process all positions in parallel. This enables:
- **GPU acceleration** through parallelization
- **Better long-range dependency** capture without vanishing gradients
- **Faster training** for long sequences
- **More efficient** use of computational resources

---

## Question 4
**What does the term "self-attention" refer to in the context of transformer models?**

- [ ] A process of normalizing the input data before feeding it into the network.
- [ ] A dropout technique used to prevent overfitting in transformer models.
- [ ] A mechanism where each input element only attends to the previous element in the sequence.
- [x] A mechanism where each input element attends to all other elements in the sequence.

**Answer:** A mechanism where each input element attends to all other elements in the sequence.

**Explanation:** Self-attention allows each position in the input sequence to attend to (compute relationships with) all other positions in the same sequence. This enables the model to:
- Capture **contextual relationships** regardless of distance
- Understand **dependencies** between any two elements
- Build **rich representations** by aggregating information from the entire sequence
- Process elements **in parallel** rather than sequentially

---

## Question 5
**Which layer in a transformer block is responsible for adding nonlinearity to the model?**

- [x] `Dense`
- [ ] `LayerNormalization`
- [ ] `MultiHeadAttention`
- [ ] `Dropout`

**Answer:** `Dense`

**Explanation:** The `Dense` (fully connected) layer in the feed-forward network (FFN) sub-layer adds nonlinearity to the transformer model through activation functions like ReLU. This nonlinearity is essential for:
- Learning **complex patterns** in the data
- Enabling the model to approximate **non-linear functions**
- Increasing the **representational capacity** of the network

The typical FFN structure is: Dense(relu) → Dense(linear)

---

## Complete Module 3 Quiz Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | Transformers vs RNNs/LSTMs | Handle longer sequences efficiently |
| 2 | Core Mechanism | Self-attention |
| 3 | Self-Attention Vectors | Query, Key, Value |
| 4 | Positional Encoding | Adds order information |
| 5 | Multi-Head Attention | Focus on different parts simultaneously |
| 6 | MultiHeadAttention Purpose | Multiple parallel attention operations |
| 7 | Positional Encoding Function | `positional_encoding` |
| 8 | Transformer Advantage | Parallelizable architecture |
| 9 | Self-Attention Definition | Each element attends to all others |
| 10 | Nonlinearity Source | `Dense` layer |

**Total Points:** 10 points (5 questions × 2 sections)

---

## Key Concepts Reviewed

| Concept | Description |
|---------|-------------|
| **Transformer Advantage** | Parallel processing, handles long-range dependencies |
| **Self-Attention** | Core mechanism allowing all positions to attend to each other |
| **Q, K, V Vectors** | Used to compute attention scores and weighted outputs |
| **Positional Encoding** | Provides sequence order information |
| **Multi-Head Attention** | Multiple attention heads for richer representations |
| **MultiHeadAttention Layer** | Performs parallel attention operations |
| **Feed-Forward Network** | Adds nonlinearity through Dense layers |

---

*End of Module 3 Quizzes*

---

## Module 3 Final Quiz - Transformers in Keras

## Question 1
**What is the primary purpose of the multi-head self-attention mechanism in a Transformer model?**

- [ ] To reduce the model's training time by using fewer parameters
- [ ] To sequentially process each word in the input sentence
- [x] To process different parts of the input sequence in parallel
- [ ] To ensure that all outputs are the same size

**Answer:** To process different parts of the input sequence in parallel

**Explanation:** Multi-head self-attention allows the transformer to process different parts of the input sequence simultaneously (in parallel). Each attention head focuses on different representation subspaces, enabling the model to capture various types of relationships (syntactic, semantic, short-range, long-range) at the same time. This parallel processing is a key advantage over sequential models like RNNs.

---

## Question 2
**What is the purpose of the feedforward neural network layers in a transformer model?**

- [ ] To weigh the importance of different words in a sentence
- [x] To transform the input data after the self-attention mechanism
- [ ] To focus on different parts of the input sequence
- [ ] To compute the attention weights

**Answer:** To transform the input data after the self-attention mechanism

**Explanation:** The feedforward neural network (FFN) layers in a transformer block transform the input data after it has been processed by the self-attention mechanism. The FFN typically consists of two Dense layers with a ReLU activation in between, adding nonlinearity and increasing the model's representational capacity.

---

## Question 3
**How do Transformers handle temporal dependencies in time series data?**

- [ ] By normalizing the input data to zero mean
- [ ] By using a convolutional layer in the input
- [x] By using positional encodings to maintain the order of input data
- [ ] By applying recurrent connections across the layers

**Answer:** By using positional encodings to maintain the order of input data

**Explanation:** Since transformers process all positions in parallel (unlike RNNs which process sequentially), they have no inherent understanding of sequence order. Positional encodings are added to the input embeddings to provide information about the position/temporal order of each element in the sequence, enabling the model to understand temporal dependencies.

---

## Question 4
**In the following code snippet, which loss function has been used to compile the model?**

```python
model.compile(optimizer='adam', loss='mse')
```

- [ ] Adam
- [x] mean squared error
- [ ] MinMaxScaler
- [ ] MultiHeadAttention

**Answer:** mean squared error

**Explanation:** `mse` stands for **Mean Squared Error**, which is a common loss function for regression tasks including time series prediction. Adam is the optimizer, not the loss function. MinMaxScaler is for data normalization, and MultiHeadAttention is a layer type.

---

## Question 5
**What is the role of the softmax function in the attention mechanism of Transformers?**

- [x] To normalize the attention scores to probabilities
- [ ] To reduce the dimensionality of the attention vectors
- [ ] To compute the dot product between query and key vectors
- [ ] To provide non-linearity to the model

**Answer:** To normalize the attention scores to probabilities

**Explanation:** The softmax function is applied to the attention scores (computed from the dot product of Query and Key vectors) to normalize them into a probability distribution. This ensures that the attention weights sum to 1, allowing them to be used as weights for the Value vectors in the weighted sum.

---

## Question 6
**Which mechanism is used by transformers to convert speech into text?**

- [x] Spectrograms
- [ ] Images
- [ ] Layers
- [ ] Patches

**Answer:** Spectrograms

**Explanation:** Speech transformers convert audio signals into **spectrograms** (time-frequency representations of sound). The spectrogram is then processed as a sequence of frames by the transformer model, which can learn to map the audio features to text output for speech-to-text tasks.

---

## Question 7
**Which method in a transformer model applies the self-attention mechanism and combines the heads?**

- [ ] TransformerBlock class
- [ ] MultiHeadSelfAttention class
- [ ] split_heads method
- [x] Call method

**Answer:** Call method

**Explanation:** The `call` method in the MultiHeadSelfAttention class is responsible for applying the self-attention mechanism and combining the heads. It orchestrates the entire process: projecting inputs to Q/K/V, splitting into heads, computing attention, and combining the heads back together for the final output.

---

## Question 8
**Which of the following converts text data into a numerical format suitable for model training in TensorFlow?**

- [x] TextVectorization
- [ ] Sequential
- [ ] lstm.model
- [ ] Vectorizer

**Answer:** TextVectorization

**Explanation:** `TextVectorization` is TensorFlow's layer specifically designed to convert text data into numerical format (integer-encoded or multi-hot encoded sequences) suitable for model training. It handles tokenization, vocabulary building, and padding/truncation automatically.

---

## Question 9
**Which of the following code snippet builds an RNN model using TensorFlow's SimpleRNN and Dense layers?**

- [ ] `model = Sequential([RNN(50, activation='relu', input_shape=(time_window, 1))`
- [ ] `model = Sequential([RNN(50, activation='relu', input_shape=(time_window, 1)), Dense(1)`
- [ ] `model = Sequential([lstm(50, activation='relu', input_shape=(time_window, 1)), Dense(1)`
- [x] `model = Sequential([SimpleRNN(50, activation='relu', input_shape=(time_window, 1)), Dense(1)])`

**Answer:** `model = Sequential([SimpleRNN(50, activation='relu', input_shape=(time_window, 1)), Dense(1)])`

**Explanation:** The correct syntax uses `SimpleRNN` (the proper TensorFlow/Keras class name) with proper bracket closing. The model should have both the SimpleRNN layer AND a Dense output layer, with properly closed brackets `])` at the end.

---

## Question 10
**What is the purpose of the following method: `def attention(self, query, key, value)`?**

- [ ] It applies the self-attention mechanism and combines the heads.
- [ ] It defines the multi-head self-attention mechanism.
- [ ] It splits the input into multiple heads for parallel attention computation.
- [x] It computes the attention scores and weighted sum of the values.

**Answer:** It computes the attention scores and weighted sum of the values.

**Explanation:** The `attention` method specifically computes the attention scores (dot product of query and key vectors), applies scaling and softmax, then uses these scores to compute the weighted sum of the value vectors. This is the core attention computation, separate from head splitting/combining which happens in other methods.

---

## Complete Module 3 Quiz Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | Transformers vs RNNs/LSTMs | Handle longer sequences efficiently |
| 2 | Core Mechanism | Self-attention |
| 3 | Self-Attention Vectors | Query, Key, Value |
| 4 | Positional Encoding | Adds order information |
| 5 | Multi-Head Attention | Focus on different parts simultaneously |
| 6 | MultiHeadAttention Purpose | Multiple parallel attention operations |
| 7 | Positional Encoding Function | `positional_encoding` |
| 8 | Transformer Advantage | Parallelizable architecture |
| 9 | Self-Attention Definition | Each element attends to all others |
| 10 | Nonlinearity Source | `Dense` layer |
| 11 | Multi-Head Attention | Process parts in parallel |
| 12 | Feedforward Network | Transform data after attention |
| 13 | Temporal Dependencies | Positional encodings |
| 14 | Loss Function | mean squared error |
| 15 | Softmax Role | Normalize attention scores |
| 16 | Speech-to-Text | Spectrograms |
| 17 | Self-Attention Method | Call method |
| 18 | Text Preprocessing | TextVectorization |
| 19 | RNN Model Code | SimpleRNN + Dense |
| 20 | Attention Method | Computes attention scores |

**Total Points:** 20 points (10 questions × 2 sections)

---

## Key Concepts Reviewed

| Concept | Description |
|---------|-------------|
| **Transformer Advantage** | Parallel processing, handles long-range dependencies |
| **Self-Attention** | Core mechanism allowing all positions to attend to each other |
| **Q, K, V Vectors** | Used to compute attention scores and weighted outputs |
| **Positional Encoding** | Provides sequence order information |
| **Multi-Head Attention** | Multiple attention heads for richer representations |
| **MultiHeadAttention Layer** | Performs parallel attention operations |
| **Feed-Forward Network** | Adds nonlinearity through Dense layers |
| **Softmax** | Normalizes attention scores to probabilities |
| **Spectrograms** | Audio representation for speech transformers |
| **TextVectorization** | Converts text to numerical format |

---

*End of Module 3 Quizzes*
