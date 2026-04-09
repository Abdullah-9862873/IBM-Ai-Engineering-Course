# Module 2: Word2Vec and Sequence to Sequence Models

## Word2Vec Introduction

### Overview

- **Word2Vec**: Short for "word to vector"
- Group of models that produce word embeddings (numerical representations)
- Embeddings capture the essence of words

### Word Relationships

- **Example**: "king" is closer to "man", "queen" is closer to "woman"
- **Vector arithmetic**: king - man + woman ≈ queen
- Effective at capturing semantic and syntactic relationships

### Applications

- Enhance NLP tasks by replacing randomly generated embeddings
- Used for similarity detection, analogies, text classification

---

## Word2Vec Neural Network Architecture

### Model Structure

1. **Input Layer**: Number of neurons = vocabulary size
2. **Embedding Layer**: User-defined dimensions (word vector size)
3. **Output Layer**: Number of neurons = vocabulary size

### How It Works

- Words fed into embedding layer
- Embedding layer interacts with softmax output layer
- Predicts context words from target words

### Training

- **Weights**: W (hidden layer) and W' (output layer)
- **Goal**: Tune weights to refine word vector representations
- **After training**: 
  - "queen" embedding closer to "woman" than "man" if predicted with higher probability
  - "king" embedding closer to "man" than "woman"

### Parameters

- **Input/Output neurons**: Vocabulary size
- **Embedding size**: User-defined (determines word vector dimensions)

---

## Continuous Bag of Words (CBOW) Model

### Overview

- Uses context words to predict a target word
- Generates embedding for the target word

### How It Works

- Given context words, predict target word
- Input: One-hot encoded context words
- Output: Probability distribution over vocabulary

### Example

- **Sentence**: "she exercises every morning"
- **Window width**: 1

| t | Context | Target |
|---|---------|--------|
| 1 | [she, every] | exercises |
| 2 | [exercises, morning] | every |

### Architecture

- **Input dimension**: Number of unique words in corpus
- **Output dimension**: Same as input (vocabulary size)
- **Hidden layer**: Contains word embeddings
- **Output**: Highest logit for target word

### Prediction Process

1. Context words encoded as one-hot vectors
2. Pass through embedding layer
3. Calculate average of context embeddings
4. Pass through output layer
5. Highest logit = predicted target word

---

## CBOW Model in PyTorch

### Model Definition

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = F.relu(embedded)
        output = self.fc(embedded)
        return output
```

### Key Components

- **Embedding layer**: `nn.EmbeddingBag` calculates average of context embeddings
- **Fully connected layer**: `self.fc` with input size = embed_dim, output size = vocab_size
- **Forward method**: Input text + offsets → embedding → ReLU → FC layer

### Training Setup

1. Initialize tokenizer
2. Create vocabulary from tokenized data
3. Set context size (e.g., 2)
4. Slide over text to form context-target pairs
5. Set up data processing pipeline and data loader
6. Batch size for training (e.g., 64)

---

## Summary

- **Word2Vec**: Produces word embeddings capturing word relationships
- **Neural network**: Input layer → Embedding layer → Output layer
- **CBOW model**: Uses context words to predict target word
- **PyTorch implementation**: EmbeddingBag + Linear layer + ReLU activation

---

## Skip-Gram Model

### Overview

- Reverse of CBOW model
- Predicts surrounding context words from a specific target word
- Given target word at position t, predict context words at t-1 and t+1

### Example

- **Sentence**: "she exercises every morning"
- **Window width**: 1

| t | Target | Context |
|---|--------|---------|
| 1 | exercises | [she, every] |
| 2 | every | [exercises, morning] |

### How It Works

1. Input: One-hot encoded target word (e.g., "exercises")
2. Output: Predict context words (e.g., "she", "every")
3. Simplifies task by predicting one context word at a time

### Prediction Process

- For target "exercises" at t=1:
  - Predict "she" (t-1)
  - Predict "every" (t+1)
- Breaks complex context prediction into smaller tasks

### Architecture

- **Input**: One-hot vector for target word
- **Output**: Probability distribution for each context position
- **Goal**: Highest logit values for actual context words

---

## Skip-Gram Model in PyTorch

### Model Definition

```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output = self.fc(embedded)
        return output
```

### Key Components

- **Embedding layer**: `nn.Embedding` creates word embeddings
- **Fully connected layer**: Linear transformation from embed_dim to vocab_size

### Sequence Generation

- Similar to CBOW but with switched order: target first, then context
- Break full context into smaller discrete parts

### Data Preparation

```python
# Each sample: (target, context)
# Nested loop iterates through context
# Appends each context word to target
```

### Training

```python
# Define learning rate
learning_rate = 0.01

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
```

### Training Function

- Train for specified number of epochs
- Returns trained model and list of average losses per epoch

### Retrieving Embeddings

```python
# After training
embeddings = model.embedding.weight.data

# Get embedding for specific word by index
word_embedding = embeddings[word_index]
```

---

## Pre-trained Word Embeddings (GloVe)

### Overview

- Stanford's pre-trained GloVe (Global Vectors) embeddings
- Leverage large-scale data for word embeddings
- Improve NLP tasks like classification

### Using GloVe in PyTorch

```python
from torchtext.vocab import GloVe

# Initialize GloVe
glove = GloVe(name='6b', dim=300)

# Create custom vocab to match tokens
```

### Integration with PyTorch Model

```python
# Load pre-trained vectors
glove_vectors = glove.vectors

# Create embedding layer with pre-trained weights
embedding = nn.Embedding.from_pretrained(glove_vectors, freeze=False)
```

### Parameters

- **freeze**: 
  - `True` (default): Keep embeddings fixed
  - `False`: Allow fine-tuning on larger datasets

---

## Summary

- **Skip-gram**: Predicts context words from target word
- **Simplification**: Predict one context word at a time
- **PyTorch**: nn.Embedding + Linear layer
- **Pre-trained**: GloVe embeddings for improved NLP performance
- **Integration**: Load pre-trained vectors into PyTorch embedding layer

---

## Sequence-to-Sequence (Seq2Seq) Models

### Overview

- Revolutionized NLP tasks like machine translation, text summarization, chatbots
- Process variable length input sequences and generate variable length output sequences

### Applications

1. **Machine Translation**: English → French
2. **Chatbots**: Query → Conversational response
3. **Summarization**: Long text → Concise summary
4. **Code Generation**: Task description → Code

### Types of Sequence Tasks

1. **Sequence-to-Sequence**: Multiple inputs → Multiple outputs
   - Input: x₁, x₂, x₃, x₄, x₅
   - Output: y₁, y₂, y₃, y₄, y₅
   - Input and output can have different lengths

2. **Sequence-to-Label**: Multiple inputs → Single label
   - Example: Document classification

3. **Label-to-Sequence**: Single input → Full sequence
   - Example: Image generation from caption

### Importance of Context

- **Example**: "The man bites the dog" vs "The dog bites the man"
- Bag-of-words: Cannot differentiate (identical word frequencies)
- Sequence representation: Captures distinct meanings
- One-hot encoded words or embeddings capture word order

---

## Recurrent Neural Networks (RNNs)

### Overview

- Type of artificial neural network using sequential or time series data
- Designed to remember past information and use it to influence future decisions

### Key Components

1. **Input Layer (x_t)**: Receives data at each timestep
2. **Hidden State (h_t)**: Network's memory
   - Applies activation function (usually tanh)
   - Captures and retains information from previous inputs
3. **Output (z_t)**: Calculation at each timestep based on current input and hidden state
4. **Concatenated Layer**: Combines hidden state and current input

### How RNN Works

1. **Unroll over time**:
   - Start with initial hidden state (zeros)
   - At each timestep: Update hidden state h_t, produce output ŷ_t
   - Use previous hidden state to inform current prediction

2. **Process**:
   - x_t + h_{t-1} → h_t → ŷ_t
   - Repeat for each timestep

### Memory Limitation

- **Problem**: RNNs only remember short-term information
- **Challenge**: Difficult to train (vanishing gradients)

---

## Enhanced RNNs: GRU and LSTM

### Gated Recurrent Units (GRU)

- **Update gate (z)**: Determines proportion of previous hidden state to carry forward
- **Reset gate (r)**: Decides how much previous hidden state to disregard
- Together: Update hidden state and control information flow over time

### Long Short-Term Memory (LSTM)

- **Gates**:
  - **Input gate**: What to add to memory
  - **Forget gate**: What to discard from memory
  - **Output gate**: What to output

- **Key concept**:
  - **h**: Short-term memory (current relevant info)
  - **c**: Long-term memory (full scope of memory)
  - Selectively filters important information for current timestep

### Why LSTM/GRU?

- Extend network's memory beyond short-term
- Complement short-term with long-term recall
- Selectively retain and transport crucial data through time

---

## Seq2Seq Data Preparation

### Steps

1. **Add Special Tokens**:
   - BOS (Beginning of Sequence)
   - EOS (End of Sequence)
   - Helps model recognize start and stop points

2. **Sort by Length**:
   - Batch sentences of similar size together

3. **Padding**:
   - Append PAD symbols to shorter sentences
   - Equalize lengths for consistent batch sizes in PyTorch

---

## Decoding Methods

### Greedy Decoding

- Model picks highest score token at each step
- Returns prediction as output
- Simple but may not produce optimal results

### Top-k Sampling

- More flexible than greedy
- Samples from top k most likely tokens
- Produces more fluent text

---

## Summary

- **Seq2Seq**: Machine translation, chatbots, summarization, code generation
- **Sequence tasks**: Seq2seq, seq2label, label2seq
- **RNN**: Sequential data processing with memory
- **Enhancements**: GRU and LSTM for long-term dependencies
- **Data prep**: BOS/EOS tokens, sorting, padding
- **Decoding**: Greedy vs top-k sampling

---

## Encoder-Decoder RNN Models

### Overview

- Seq2Seq models use encoder-decoder architecture
- Encoder processes input sequence → produces context/state
- Decoder uses context to generate output sequence

### Dataset

- **Multi30K dataset**: English to German translations
  - Training set
  - Validation set
  - Test set

### Data Loading

- Use PyTorch DataLoader for batching
- **Collation**: Tokenization, numericalization, adding BOS/EOS/PAD tokens
- Output: Iterable batches of SRC (source) and TRG (target) tensors

---

## Training Seq2Seq Models

### Challenges

- More difficult to train than standard RNNs
- Goal: Minimize cross-entropy loss by comparing predictions with actual labels

### Training Procedure

```python
# 1. Initialize model in training mode
model.train()

# 2. Iterate through training data batches
for batch in train_data:
    # 3. Assign sequences to correct device
    src = batch.src  # Input sequence
    trg = batch.trg  # Target sequence
    
    # 4. Generate predictions
    output = model(src, trg)
    
    # 5. Reshape output for loss calculation
    # output: [target_len, batch_size, output_dim]
    output = output[1:].view(-1, output.shape[-1])
    trg = trg[1:].view(-1)
    
    # 6. Calculate loss
    loss = criterion(output, trg)
    
    # 7. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Key Points

- **Reshaping**: Align rows and columns for loss calculation
  - Exclude initial BOS token (target_len - 1)
  - Batch size = columns (separate sequences)
  - Output dimension = predicted tokens

### Evaluation

- Similar to training but:
  - Use validation/test data
  - Set model to evaluation mode: `model.eval()`
  - Disable dropout for faster computation

---

## Inference (Prediction)

### Prediction Function

```python
def predict(model, src, trg_vocab, max_len=50):
    # Convert source sentence to tensor
    src_tensor = tokenize_and_numericalize(src)
    
    # Feed to encoder
    hidden, cell = encoder(src_tensor)
    
    # Initialize target with BOS token
    trg = [trg_vocab['<bos>']]
    
    for _ in range(max_len):
        # Get last token and states
        decoder_input = torch.tensor([trg[-1]])
        
        # Decoder output
        output, hidden, cell = decoder(decoder_input, hidden, cell)
        
        # Get predicted token (highest probability)
        pred_token = output.argmax(1).item()
        
        # Add to translation
        trg.append(pred_token)
        
        # Stop if EOS token
        if pred_token == trg_vocab['<eos>']:
            break
    
    # Convert indices to words
    translated = [trg_vocab.get_token(idx) for idx in trg]
    
    # Remove special tokens and form sentence
    translated = [t for t in translated if t not in ['<bos>', '<eos>', '<pad>']]
    return ' '.join(translated)
```

### Steps

1. Convert source sentence to tensor format
2. Pass through encoder to get hidden and cell states
3. Start target with BOS token
4. Iterate up to max length:
   - Use last token and previous states in decoder
   - Get new outputs and states
   - Choose token with highest probability (argmax)
   - Add to translation
   - Stop if EOS token generated
5. Convert token indices to words
6. Remove special tokens, form sentence

---

## Summary

- **Encoder-Decoder**: Input sequence → Encoder → States → Decoder → Output sequence
- **Training**: Minimize cross-entropy loss, reshape outputs, backpropagation
- **Inference**: Complex function for translation, token-by-token generation
- **Data**: Multi30K dataset (English to German)
- **Key**: Exclude BOS from loss, use teacher forcing during training

---

# Module 2 Summary: Key Takeaways

## Word2Vec Models

### CBOW (Continuous Bag of Words)

- Uses context words to predict target word
- Input: Multiple context words → Output: Target word

### Skip-Gram

- Predicts context words from target word (reverse of CBOW)
- Simplifies by predicting one context word at a time

### Pre-trained Embeddings

- GloVe (Global Vectors): Stanford's pre-trained word embeddings
- Integration in PyTorch: `nn.Embedding.from_pretrained()`

## Sequence-to-Sequence Models

### Architecture

- Encoder-Decoder: Processes input, produces context, generates output
- Variable length input/output sequences

### RNN Basics

- Hidden state captures past information
- Processes sequential data timestep by timestep

### Enhanced RNNs

- **GRU**: Update gate + Reset gate
- **LSTM**: Input, Forget, Output gates for long-term memory

## Training & Inference

### Training

- Minimize cross-entropy loss
- Reshape outputs correctly for loss calculation
- Exclude BOS token from loss

### Inference

- Token-by-token generation
- Use encoder states as decoder context
- Generate until EOS token or max length

### Dataset

- Multi30K: English to German translations

---

## Encoder-Decoder Architecture Implementation

### Overview

- RNNs can create seq2seq models: input sequence X → output sequence Y
- X and Y can have different lengths
- Encoder-Decoder architectures solve this

### Architecture

- **Encoder**: Series of RNNs processing input sequence
  - Pass hidden states to next RNN
  - Last hidden state = context passed to decoder
  
- **Decoder**: Series of RNNs autoregressively generating output
  - Each generated token goes back as input for next RNN
  - Continue until EOS (end token) generated

### Key Points

- Encoder: Only use hidden state, discard output
- Decoder: Receives previously generated token, generates next token
- Representation: RNN cell (output from previous state recycled as input)

---

## Encoder Implementation in PyTorch

### Structure

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # Embedding transforms input token to vector
        embedded = self.dropout(self.embedding(src))
        
        # LSTM produces hidden and cell states
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Encoder only needs hidden and cell states
        return hidden, cell
```

### Key Points

- LSTM used (not GRU) - maintains both hidden and cell states
- **embedding_dim**: Size of embedding vectors
- **hidden_dim**: Size of hidden and cell states
- **n_layers**: Number of recurrent layers
- **dropout**: Regularization technique
- Output vector discarded (not needed for encoder)

---

## Decoder Implementation in PyTorch

### Structure

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        # Embed the input token
        embedded = self.dropout(self.embedding(input))
        
        # LSTM produces new hidden and cell states
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Linear layer maps to output dimension (vocabulary size)
        prediction = self.fc(output.squeeze(0))
        
        return prediction, hidden, cell
```

### Key Points

- **Autoregressive**: Receives previously generated token
- **embedding**: Maps tokens to dense vectors
- **LSTM**: Produces updated hidden state
- **Linear layer**: Maps to vocabulary size for token prediction
- **Softmax**: Applied later to get probability distribution

---

## Seq2Seq Model (Encoder + Decoder)

### Structure

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        output_dim = self.decoder.fc.out_features
        
        # Store predictions
        outputs = torch.zeros(max_len, batch_size, output_dim).to(self.device)
        
        # Encode source sequence
        hidden, cell = self.encoder(src)
        
        # First input is BOS token
        input = trg[0]
        
        for t in range(1, max_len):
            # Decode
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs
```

### Teacher Forcing

- Use true output from training data as decoder input
- Instead of using model's own predicted output
- Boosts training efficiency and convergence

### Forward Pass Steps

1. Initialize output tensor
2. Get encoder hidden and cell states
3. Start with BOS token as first input
4. For each timestep:
   - Pass input and states to decoder
   - Store prediction
   - Decide: use true token (teacher forcing) or predicted token
5. Return all predictions

---

## Summary

- **Encoder-Decoder**: Transform input sequence to output sequence
- **Encoder**: Process input, produce context (hidden + cell states)
- **Decoder**: Autoregressively generate output tokens
- **PyTorch**: LSTM layers for both encoder and decoder
- **Teacher Forcing**: Use true tokens during training for faster convergence

---

## Metrics for Evaluating Generated Text

### Overview

- Generative AI and LLMs used to generate text, images, etc.
- Measurement of success: ability to generate consistent and contextually relevant text
- **Perplexity**: Precious tool for evaluating efficiency of LLMs and GenAI models

### Perplexity

#### Definition

- Measure of how surprised/uncertain the model is when predicting next word
- Calculated as exponent of the loss obtained from the model

#### Cross-Entropy Loss

- Measures discrepancy between predicted and actual distribution
- As predicted distribution → true distribution, cross-entropy loss decreases
- When distributions match perfectly, cross-entropy = 0 (ideal)

#### Calculation

```
Perplexity = exp(average_cross_entropy_loss)
```

- For each sequence, calculate average loss of all tokens
- Apply exponential function to transform back to interpretable space
- **Lower perplexity = better performance**

#### Example

- Model 1: Loss → Perplexity = 21.7
- Model 2: Loss → Perplexity = 142.6
- Lower perplexity indicates better model

#### Limitations

- Provides overall measure of model performance
- Doesn't capture nuances of generated text quality
- Usually used only for training set evaluation

---

## Additional Evaluation Metrics

### N-gram Matching

- Measures similarity between generated text and reference texts
- Helpful when there's more than one valid generated text
- Example: Translation task - compare with multiple translation versions

### BLEU Score (Bilingual Evaluation Understudy)

- Counts matching n-grams between hypothesis and reference
- **Precision-based**: CountMatch / CountGenerated

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- Focuses on recall: coverage of important content from reference
- CountMatch / CountReference

### Precision vs Recall in Machine Translation

- **Precision**: Accuracy of generated translation
  - Formula: CountMatch / CountGenerated
  
- **Recall**: Completeness of generated translation
  - Formula: CountMatch / CountReference

- **F1 Score**: Harmonic mean of precision and recall

---

## Implementation Libraries

### NLTK Library

- Includes BLEU and METEOR implementations

```python
from nltk.translate.bleu_score import sentence_bleu
```

### PyTorch

- Perplexity and cross-entropy loss

```python
import torch.nn.functional as F

# Cross-entropy loss
loss = F.cross_entropy(logits, targets)

# Perplexity
perplexity = torch.exp(loss)
```

### BLEU Score Example

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, hypothesis):
    references = [ref.split()] for ref in reference]
    hypothesis = hypothesis.split()
    score = sentence_bleu(references, hypothesis)
    return score

# Example
references = ["The cat is on the mat", "A cat is on the mat"]
hypothesis = "The cat is on the"
print(calculate_bleu(references, hypothesis))
```

---

## Summary

- **Perplexity**: exp(cross_entropy_loss), lower = better
- **BLEU**: N-gram precision measure
- **ROUGE**: N-gram recall measure
- **F1**: Harmonic mean of precision and recall
- **Libraries**: NLTK (BLEU), PyTorch (perplexity, cross-entropy)

---

## Detailed Evaluation Metrics

### ROUGE-N (N-gram Overlap)

- Measures matching n-grams between hypothesis (H) and reference (R)
- **ROUGE-N Precision**: common_ngrams / total_ngrams_in_H
- **ROUGE-N Recall**: common_ngrams / total_ngrams_in_R
- **ROUGE-N F1**: Harmonic mean of precision and recall

### ROUGE-L (Longest Common Subsequence)

- Measures longest subsequence of words (not necessarily consecutive) shared
- **LCS**: Longest sequence appearing in same order in H and R
- **ROUGE-L Precision**: len(LCS) / len(H)
- **ROUGE-L Recall**: len(LCS) / len(R)

### ROUGE-S (Skip-gram)

- Allows unigram skipping (non-consecutive matching)
- Example: "the cat" matches "the big cat"
- **ROUGE-S Precision**: common_bigrams / total_bigrams_in_H
- **ROUGE-S Recall**: common_bigrams / total_bigrams_in_R

### BLEU (Clipped Precision)

- Compares generated translation with reference translations
- **Clipped precision**: Limits matching n-gram counts to prevent inflated scores
- **Brevity penalty**: Accounts for translation length
- **Formula**: BLEU = BP × (Π precision_n)^(1/n)

### METEOR (Multiple Matching Criteria)

- Considers: exact matches, synonymy, stemming, word reordering
- **Word alignment**: Map system translation to reference
- **Precision/Recall**: Based on matching unigrams
- **Harmonic mean**: Balance between precision and recall
- **Penalty**: Based on chunk count (adjacent matched words)

---

## Module 2 Summary: Key Takeaways

### Word2Vec

- **Word2Vec**: "word to vector" - produces numerical word representations
- **Neural network**: Input → Embedding → Output layers
- **Vocabulary size**: Neurons in input/output layers
- **Embedding dimension**: User-defined word vector size

### CBOW vs Skip-Gram

- **CBOW**: Context words → predict target word
- **Skip-Gram**: Target word → predict context words
- Both use embedding layer + output layer

### Pre-trained Embeddings

- **GloVe**: Stanford's Global Vectors
- **Integration**: `nn.Embedding.from_pretrained()`

### Sequence-to-Sequence Models

- **Applications**: Translation, chatbots, summarization
- **Types**: Seq2seq, seq2label, label2seq

### RNNs

- **RNN**: Sequential data processing with memory
- **Enhancements**: GRU (update/reset gates), LSTM (input/forget/output gates)

### Encoder-Decoder

- **Encoder**: Process input → hidden/cell states
- **Decoder**: Autoregressive token generation
- **Teacher forcing**: Use true tokens during training

### Evaluation Metrics

- **Perplexity**: exp(cross_entropy_loss), lower = better
- **BLEU**: Precision-based n-gram matching
- **ROUGE**: Recall-based n-gram matching
- **METEOR**: Precision + recall + word order

### Libraries

- **NLTK**: BLEU, METEOR
- **PyTorch**: Perplexity, cross-entropy loss