# Module 2: Advanced Concepts of Transformer Architecture

## Transformer Families Overview

### 1. Decoder Models (GPT-like)
- **Best for**: Text generation (chatbots, story writing, autocomplete)
- **Architecture**: Decoder-only, autoregressive
- **Examples**: GPT, LLaMA, Granite

### 2. Encoder Models (BERT-like)
- **Best for**: Understanding tasks (search engines, sentiment analysis, Q&A)
- **Architecture**: Encoder-only, bidirectional
- **Examples**: BERT, RoBERTa

### 3. Encoder-Decoder Models (Seq2Seq)
- **Best for**: Translation and input-output sequence mapping
- **Architecture**: Encoder + Decoder with cross-attention
- **Examples**: Transformer (original), T5, BART

---

## Decoder Models (GPT)

### Overview

- Transformers initially had encoder + decoder for translation
- Decoder evolved for text generation (GPT, LLaMA, Granite)
- Generative Pre-training (GPT): Self-supervised, predict next token

### Autoregressive Generation

- Predict next word based on previous words
- Process:
  1. Start with BOS token
  2. Predict next word (e.g., "IBM")
  3. Append to input: BOS IBM
  4. Predict next word (e.g., "taught")
  5. Continue until EOS or max tokens

### Masked Self-Attention

- **Key distinction**: Encoders use unmasked attention; Decoders use masked attention
- **Purpose**: Hide future tokens during training
- **Matrix multiplication core**: Ensures model only attends to previous tokens

### Training Methods

1. **Self-supervised**: Predict next token (causal language modeling)
2. **Fine-tuning**: Supervised optimization for specific tasks (QA, classification)
3. **RLHF**: Reinforcement Learning from Human Feedback
   - Effective for chatbot development

### Text Generation Process

1. Tokenize prompt
2. Convert to word embeddings
3. Add positional encodings
4. Pass through decoder layers (masked self-attention)
5. Generate contextual embeddings
6. Pass through linear layer → logits
7. Apply argmax to get next token
8. Append token to input
9. Repeat until EOS or max length

### Implementation in PyTorch

```python
from transformers import GPT2LMHeadModel, pipeline

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Text generation
generator = pipeline('text-generation', model=model)
output = generator("How are you", max_length=20)
```

---

## Encoder Models (BERT)

### Overview

- Bidirectional context understanding
- Does not generate text step-by-step
- Excels at understanding tasks

### Pre-training Tasks

#### 1. Masked Language Modeling (MLM)
- Randomly mask tokens (e.g., 15%)
- Train model to predict masked tokens
- Forces bidirectional context understanding
- Example: "The dog is chasing the ___" → predict "cat"

#### 2. Next Sentence Prediction (NSP)
- Given two sentences A and B
- Predict if B follows A logically
- Useful for Q&A, summarization tasks

### Data Preparation

1. **Tokenization**: Split text into subword tokens
2. **Special Tokens**: Add [CLS], [SEP]
3. **Segment IDs**: Mark sentence A vs sentence B
4. **Attention Masks**: Identify real tokens vs padding

### Implementation in PyTorch

```python
from transformers import BertModel, BertTokenizer

# Load pretrained model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare inputs
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

---

## Encoder-Decoder Models (Translation)

### Overview

- Combines encoder (read input) + decoder (generate output)
- Encoder: Full bidirectional reading of source
- Decoder: Autoregressive generation of target

### Architecture

1. **Encoder**: Processes full source sequence
2. **Cross-attention**: Decoder attends to encoder output
3. **Decoder**: Generates target word by word

### Training Process

1. Prepare source (e.g., German) and target (e.g., English) sequences
2. Encoder processes source → context embeddings
3. Decoder uses context + previous tokens to predict next token
4. Compare prediction with actual → compute loss → adjust model

### Implementation

```python
from torch.nn import TransformerEncoder, TransformerDecoder

# Encoder
encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
encoder = TransformerEncoder(encoder_layer, num_layers=6)

# Decoder
decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
decoder = TransformerDecoder(decoder_layer, num_layers=6)
```

---

## Hands-on Labs Overview

### Lab 1: GPT-like Decoder Model
- Text generation using decoder-only transformer
- Causal language modeling (predict next word)
- Causal masking: Prevent looking at future tokens
- Autoregressive text generation

### Lab 2: BERT-like Encoder Model
- Bidirectional understanding (not generation)
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Segment embeddings for sentence pairs

### Lab 3: Translation Model (Encoder-Decoder)
- Map German → English
- Encoder reads source, decoder generates target
- Cross-attention bridges input/output
- Applications: summarization, dialogue, code-to-text

---

## Key Differences Summary

| Aspect | Decoder (GPT) | Encoder (BERT) | Encoder-Decoder |
|--------|---------------|----------------|-----------------|
| Direction | Unidirectional | Bidirectional | Input bidirectional, output unidirectional |
| Use Case | Text generation | Understanding | Translation |
| Attention | Masked self-attention | Self-attention | Self-attention + cross-attention |
| Output | Next token | Pooled/sequence output | Target sequence |
| Examples | GPT, LLaMA | BERT, RoBERTa | Transformer, T5 |

---

## Summary

- **Decoder models**: Autoregressive text generation, masked attention
- **Encoder models**: Bidirectional understanding, MLM + NSP pre-training
- **Encoder-Decoder**: Translation and seq2seq tasks
- **Key concepts**: Q/K/V matrices, positional encoding, masking strategies

---

## Training Decoder Models

### Notations

- **ω̂_t**: Token at time t (predicted token index)
- **x̂_t**: Predicted word embedding vector

### Autoregressive Prediction Process

1. **t=0**: Input x₀ → Decoder → Contextual embedding → Predict ω̂₁ → x̂₁
2. **t=1**: Input x₀ + x̂₁ → Decoder → Contextual embeddings → Predict ω̂₂ → x̂₂
3. **t=2**: Input x₀ + x̂₁ + x̂₂ → Decoder → Contextual embeddings → Predict ω̂₃ → x̂₃
4. Continue recursively...

- **Note**: Positional encoding added at each step

### Training Data Format

- **Input tokens**: x₀, x₁, x₂, x₃ (original sequence)
- **Target tokens**: x₁, x₂, x₃, x₄ (shifted one step forward)
- Special tokens: BOS, EOS, PAD (for uniform length)

### Training Phase vs Inference

#### Inference (Prediction)
- Use predicted token (x̂) as input for next step
- Only final token used for next prediction

#### Training
- Use actual tokens throughout (not approximations)
- Predict next token at EVERY sequence position
- Process entire sequence (0-3) at once
- Use all output contextual embeddings for loss calculation

### Teacher Forcing

- Instead of feeding model's own predictions back
- Use actual previous token from sequence
- Example:
  - Input x₀ → Predict ω̂₁
  - Use actual x₁ (not x̂₁) for next step → Predict ω̂₂
  - Use actual x₂ → Predict ω̂₃
- Ensures model stays aligned with correct sequence

### Causal Attention Masking

#### Purpose
- Ensure each token only attends to preceding tokens (not future)
- Prevents looking at "future" tokens during prediction

#### Implementation
- Create upper triangular matrix with negative infinity
- Apply to attention scores before softmax
- After softmax, future token contributions become 0

```python
# Causal mask
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
```

#### Effect
- Token at position 3 can only attend to positions 0, 1, 2
- Token at position 2 can only attend to positions 0, 1
- Token at position 1 can only attend to position 0
- Token at position 0 can only attend to itself

### Loss Calculation

```python
# Compute loss across all sequence positions
loss = cross_entropy_loss(logits, target_tokens)
```

- Compare predicted tokens (ω̂) with actual tokens (ω)
- Calculate loss for entire sequence
- Backpropagate to train model

---

## Decoder Models PyTorch Implementation

### Dataset Preparation (IMDB Example)

```python
# Load IMDB dataset
from torchtext.datasets import IMDB

train_iter, test_iter = IMDB(split='train'), IMDB(split='test')

# Special tokens
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
EOS_TOKEN = '<eos>'
```

### Creating Training Samples

```python
def get_sample(text, block_size):
    # Random starting point
    start = random.randint(0, len(text) - block_size - 1)
    
    # Source: block_size tokens
    source = text[start:start + block_size]
    
    # Target: shifted by 1
    target = text[start + 1:start + block_size + 1]
    
    return source, target
```

### Causal Mask Generation

```python
def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

# Usage
causal_mask = generate_square_subsequent_mask(block_size)
```

### Custom GPT Architecture

```python
class CustomGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers):
        super(CustomGPT, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_size)
        
        # Transformer encoder (acts as decoder with mask)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=nhead
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x, src_mask=None):
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(embed_size)
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer(x, src_mask=src_mask)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        return logits
```

### Collate Function with Padding

```python
def collate_fn(batch):
    source, target = zip(*batch)
    # Pad sequences to same length
    source_padded = pad_sequence(source, padding_value=PAD_INDEX)
    target_padded = pad_sequence(target, padding_value=PAD_INDEX)
    return source_padded, target_padded
```

### Training Process

```python
# Initialize model
model = CustomGPT(vocab_size, embed_size, nhead, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        output = model(batch.source, src_mask=causal_mask)
        
        # Reshape for loss calculation
        output = output.view(-1, vocab_size)
        target = batch.target.view(-1)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Autoregressive Inference (Text Generation)

```python
def generate(model, prompt, max_new_tokens):
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer(prompt)['input_ids']
    
    for _ in range(max_new_tokens):
        # Truncate if too long
        input_ids = input_ids[-block_size:]
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids]).to(device)
        
        # Generate mask
        src_mask = generate_square_subsequent_mask(len(input_ids))
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_tensor, src_mask)
        
        # Get last token logits
        next_token_logits = logits[0, -1, :]
        
        # Greedy decoding (argmax)
        next_token = next_token_logits.argmax().item()
        
        # Append to input
        input_ids.append(next_token)
        
        # Stop if EOS
        if next_token == EOS_INDEX:
            break
    
    return tokenizer.decode(input_ids)
```

### Evaluation

```python
def evaluate(model, valid_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            output = model(batch.source, src_mask=causal_mask)
            output = output.view(-1, vocab_size)
            target = batch.target.view(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(valid_loader)
```

---

## Summary: Training Decoders

- **Autoregressive**: Predict next token based on previous
- **Teacher Forcing**: Use actual tokens (not predictions) during training
- **Causal Mask**: Prevent attending to future tokens
- **PyTorch**: Custom GPT with embedding, positional encoding, transformer, lm_head
- **Inference**: Greedy decoding (argmax) iteratively

---

## Encoder Models (BERT) - Detailed

### BERT Architecture

- **Full name**: Bidirectional Encoder Representations from Transformers
- **Developed by**: Google
- **Architecture**: Encoder-only (only the encoder part of the transformer)
- **Key feature**: Bidirectional context understanding

### BERT vs Autoregressive Models

- **Autoregressive models** (GPT): Only have access to previous tokens to predict masked words
- **BERT**: Can utilize full context from both sides of the masked token
- Example: "[CLS] IBM [MASK] me BERT [SEP]" → BERT can use "IBM" and "me BERT" to predict "taught"

### BERT Pre-training Tasks

#### 1. Masked Language Modeling (MLM)

- Randomly mask ~15% of input tokens
- Train BERT to predict the original masked tokens
- Encoder outputs contextual embeddings → pass through layer → convert to logits
- Masked word identified by highest logit value

**Masking Strategy** (80/10/10 rule):
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep unchanged

Example:
- Input: "The sun sets behind the distant mountains."
- 85% unchanged, from remaining 15%:
  - 80% replaced with [MASK]
  - 10% replaced with random token (e.g., "human")
  - 10% unchanged

#### 2. Next Sentence Prediction (NSP)

- Given two sentences A and B
- Predict if B logically follows A
- Binary classification: IsNext or NotNext

Example:
- Sentence A: "My dog is cute."
- Sentence B (IsNext): "He likes playing." → label = 1
- Sentence B (NotNext): "He likes studying medicine." → label = 0

### BERT Token Type Embeddings

1. **[CLS] token**: Added at start of sequence, used for classification
2. **[SEP] token**: Added to denote end of sentence
3. **Segment embeddings**: Distinguish first vs second sentence
   - First sentence tokens → embedding value 1
   - Second sentence tokens → embedding value 2
4. **Positional encodings**: Order of tokens in sequence

---

## Data Preparation for BERT

### Special Tokens and Indices

```python
PAD_IDX = 0
CLS_IDX = 1
SEP_IDX = 2
MASK_IDX = 3
UNK_IDX = 4
```

- **PAD**: Padding tokens for uniform length
- **CLS**: Start of sequence
- **SEP**: Separator between sequences
- **MASK**: Masked tokens for MLM
- **UNK**: Unknown tokens

### Masking Function

```python
def masking(token):
    if not mask:
        return token, pad_label
    
    # Random selection
    random_value = random.random()
    
    # Case 1: Replace with [MASK], label = original token
    if random_value < 0.8:
        return MASK_TOKEN, original_token
    
    # Case 2: Keep unchanged, label = original token
    elif random_value < 0.9:
        return token, original_token
    
    # Case 3: Replace with random token, label = original token
    else:
        return random_token, original_token
```

### Prepare for MLM Function

```python
def prepare_for_mlm(tokens):
    processed_sentences = []
    bert_labels = []
    raw_tokens = []
    
    for token in tokens:
        processed_token, label = masking(token)
        processed_sentences.append(processed_token)
        bert_labels.append(label)
        if include_raw:
            raw_tokens.append(token)
    
    return processed_sentences, bert_labels, raw_tokens
```

### Prepare for NSP Function

```python
def process_for_nsp(sentences, masked_labels):
    sentence_pairs = []
    pair_labels = []
    is_next_labels = []
    
    while enough_indices:
        # Randomly choose Next or NotNext
        is_next = random.choice([0, 1])
        
        if is_next == 1:
            # Select consecutive sentences
            sent_a = sentences[i]
            sent_b = sentences[i + 1]
            is_next_label = 1
        else:
            # Select random distinct sentences
            sent_a = sentences[i]
            sent_b = random.choice(sentences[:])
            is_next_label = 0
        
        # Add CLS and SEP tokens
        pair = [CLS] + sent_a + [SEP] + sent_b + [SEP]
        
        sentence_pairs.append(pair)
        pair_labels.append(pair_masked_labels)
        is_next_labels.append(is_next_label)
    
    return sentence_pairs, pair_labels, is_next_labels
```

### Prepare BERT Final Inputs

```python
def prepare_bert_final_inputs(bert_inputs, bert_labels, is_next_list):
    final_inputs = []
    segment_labels = []
    final_labels = []
    final_is_next = []
    
    for input_pair, label_pair, is_next in zip(bert_inputs, bert_labels, is_next_list):
        # Zero-pad to uniform length
        padded_input = zero_pad(input_pair, pad_idx)
        padded_label = zero_pad(label_pair, pad_idx)
        
        # Create segment labels
        segment_label = create_segment_labels(input_pair)
        
        final_inputs.append(padded_input)
        segment_labels.append(segment_label)
        final_labels.append(padded_label)
        final_is_next.append(is_next)
    
    return final_inputs, segment_labels, final_labels, final_is_next
```

### Sample Output

Input sentences: "he lives in new york" + "he likes studying"

Output structure:
- **BERT input**: [CLS] he lives in new york [SEP] he likes studying [SEP]
- **BERT labels**: [PAD] labels for each token
- **Segment labels**: 1 1 1 1 1 2 2 2 2 2
- **is_next**: 0 (second sentence does not follow first)

---

## Pre-training BERT with PyTorch

### Custom Dataset Class

```python
class BERTCSVDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'bert_input': torch.tensor(json.loads(row['bert_input'])),
            'bert_label': torch.tensor(json.loads(row['bert_label'])),
            'segment_label': torch.tensor(json.loads(row['segment_label'])),
            'is_next': torch.tensor(row['is_next'])
        }
    
    def __len__(self):
        return len(self.data)
```

### Create Data Loaders

```python
batch_size = 32

train_dataset = BERTCSVDataset('train.csv')
test_dataset = BERTCSVDataset('test.csv')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

### BERT Embedding Class

```python
class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, segment_vocab_size):
        super(BERTEmbedding, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(segment_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(512, embed_size)
    
    def forward(self, input_ids, segment_labels):
        position_ids = torch.arange(input_ids.size(1))
        
        token_embeds = self.token_embedding(input_ids)
        segment_embeds = self.segment_embedding(segment_labels)
        pos_embeds = self.positional_embedding(position_ids)
        
        return token_embeds + segment_embeds + pos_embeds
```

### Complete BERT Model

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super(BERT, self).__init__()
        
        self.embedding = BERTEmbedding(vocab_size, d_model, segment_vocab_size=2)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model*4, dropout=dropout),
            num_layers=n_layers
        )
        
        # NSP classification head
        self.nsp_head = nn.Linear(d_model, 2)
        
        # MLM prediction head
        self.mlm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, segment_labels):
        embeddings = self.embedding(input_ids, segment_labels)
        encoded = self.encoder(embeddings)
        
        # NSP: use CLS token output
        cls_output = encoded[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)
        
        # MLM: predict all tokens
        mlm_logits = self.mlm_head(encoded)
        
        return nsp_logits, mlm_logits
```

### Training BERT

```python
# Initialize model
model = BERT(
    vocab_size=len(vocab),
    d_model=256,
    n_layers=2,
    heads=2,
    dropout=0.1
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    for batch in train_loader:
        nsp_logits, mlm_logits = model(
            batch['bert_input'],
            batch['segment_label']
        )
        
        # Calculate NSP loss
        nsp_loss = criterion(nsp_logits, batch['is_next'])
        
        # Calculate MLM loss
        mlm_loss = criterion(
            mlm_logits.view(-1, vocab_size),
            batch['bert_label'].view(-1)
        )
        
        # Combined loss
        loss = nsp_loss + mlm_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate after each epoch
    evaluate(model, test_loader)
```

### Evaluation Function

```python
def evaluate(model, data_loader):
    model.eval()
    
    total_loss = 0
    total_nsp_loss = 0
    total_mlm_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            nsp_logits, mlm_logits = model(
                batch['bert_input'],
                batch['segment_label']
            )
            
            nsp_loss = criterion(nsp_logits, batch['is_next'])
            mlm_loss = criterion(mlm_logits.view(-1, vocab_size), batch['bert_label'].view(-1))
            
            total_nsp_loss += nsp_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_loss += (nsp_loss + mlm_loss).item()
    
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss
```

### Prediction Functions

#### Predict NSP

```python
def predict_nsp(sentence_a, sentence_b, model, tokenizer):
    # Tokenize
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
    
    # Get predictions
    with torch.no_grad():
        nsp_logits, _ = model(inputs['input_ids'], inputs['token_type_ids'])
    
    # Get prediction
    prediction = torch.argmax(nsp_logits, dim=1).item()
    
    return "IsNext" if prediction == 1 else "NotNext"
```

#### Predict MLM

```python
def predict_mlm(sentence, model, tokenizer):
    # Tokenize
    tokens = tokenizer.encode(sentence, return_tensors='pt')
    mask_pos = (tokens == MASK_IDX).nonzero()
    
    # Get predictions
    with torch.no_grad():
        _, mlm_logits = model(tokens, torch.zeros_like(tokens))
    
    # Get predicted token
    predicted_ids = torch.argmax(mlm_logits[0, mask_pos], dim=-1)
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_ids)
    
    return predicted_token
```

---

## Summary: BERT Encoder Models

- **Architecture**: Encoder-only, bidirectional
- **Pre-training tasks**: MLM (15% masked) + NSP
- **Special tokens**: [CLS], [SEP], [MASK]
- **Embeddings**: Token + Segment + Positional
- **MLM masking**: 80% [MASK], 10% random, 10% unchanged
- **NSP**: Binary classification (IsNext/NotNext)
- **Fine-tuning**: Train on task-specific datasets
- **Output**: CLS token for classification, full sequence for token-level tasks

---

## Summary and Highlights

- **BERT model's job**: Predict which sentence is the appropriate continuation
- **BERT's architecture**: Allows fine-tuning tasks like text summarization, question answering, and sentiment analysis
- **Data preparation**: Initialize tokenizer using `get_tokenizer` and define special symbols with corresponding indices

---

## Encoder-Decoder Transformer for Translation

### Overview

- **Purpose**: Language translation (e.g., German to English)
- **Key advantage**: Process entire text sequences simultaneously (faster than RNN/LSTM)
- **Architecture**: Encoder + Decoder with cross-attention

### Encoder Architecture

1. **Input**: Tokenized source sentence
2. **Embedding layer**: Each token → d-dimensional embedding vector
3. **Positional encoding**: Added to maintain word order
4. **Multi-head attention**: Model focuses on different parts of sentence
5. **Normalization layer**: Stabilizes training
6. **Feedforward network**: Processes vectors to same dimension D
7. **Output**: "Memory" - contextual embeddings (sequence length × embedding size D)

### Decoder Architecture

- Similar to decoder-only model with key differences:
  - **Cross-attention layer**: Attends to encoder's memory output
  - **Masking**: Ensures sequential prediction considering only preceding tokens
  - **Autoregressive generation**: One word at a time

### Cross-Attention Mechanism

- Computes attention scores between target positions (decoder) and source positions (encoder)
- Captures relevance of each source position to current target position
- Enables handling long-range dependencies
- Aligns input and output sequences effectively

### Translation Process

1. **Source encoding**: 
   - Tokenize source (German) → embeddings + positional encoding
   - Pass through encoder → memory

2. **Decoding** (autoregressive):
   - Start with BOS token
   - Generate next token using memory + previous tokens
   - Feed generated token back as input
   - Repeat until EOS or max length

### PyTorch Implementation

#### Create Masks

```python
def create_mask(src, tgt):
    # Causal mask for target
    tgt_mask = generate_square_subsequent_mask(tgt.size(1))
    
    # Source mask (no masking needed)
    src_mask = torch.zeros(src.size(1), src.size(1)).type(torch.bool)
    
    # Padding masks
    src_padding_mask = (src == PAD_INDEX)
    tgt_padding_mask = (tgt == PAD_INDEX)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
```

#### Transformer Model

```python
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(Seq2SeqTransformer, self).__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        # Source encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(d_model))
        memory = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        
        # Target encoding  
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(d_model))
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask, src_key_padding_mask=src_padding_mask)
        
        return self.generator(output)
    
    def encode(self, src, src_mask):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(d_model))
        return self.transformer.encoder(src_emb, src_mask)
    
    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)
```

#### Training

```python
def train(model, data_loader, optimizer, criterion, clip):
    model.train()
    
    for batch in data_loader:
        src = batch.src
        tgt = batch.tgt
        
        # Target input: all tokens except last
        tgt_input = tgt[:-1, :]
        # Target output: all tokens except first
        tgt_output = tgt[1:, :]
        
        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        # Forward pass
        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        
        # Compute loss
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
```

#### Inference (Translation)

```python
def translate(model, src_sentence, tokenizer, max_len):
    model.eval()
    
    # Encode source
    src_tokens = tokenizer(src_sentence)['input_ids']
    src_tensor = torch.tensor([src_tokens]).to(device)
    src_mask = torch.zeros(1, src_tensor.size(1)).to(device)
    
    memory = model.encode(src_tensor, src_mask)
    
    # Start with BOS token
    ys = torch.tensor([[BOS_INDEX]]).to(device)
    
    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        
        # Get next token
        next_token_logits = out[0, -1, :]
        next_token = next_token_logits.argmax(dim=-1).item()
        
        ys = torch.cat([ys, torch.tensor([[next_token]]).to(device)], dim=1)
        
        if next_token == EOS_INDEX:
            break
    
    return tokenizer.decode(ys[0])
```

### Summary: Encoder-Decoder Translation

- **Process**: Entire source sequence processed simultaneously
- **Memory**: Encoder output used by decoder
- **Cross-attention**: Decoder attends to encoder's memory
- **Autoregressive decoding**: One token at a time
- **Linear layer**: Maps embeddings to vocabulary for token prediction

---

## Summary and Highlights

- **Transformers vs RNNs**: Process entire text sequences simultaneously (faster and better context handling)
- **Encoder operations**: Embedding → positional encoding → multi-head attention → normalization → feedforward → normalization
- **Cross-attention**: Decoder references full encoder context
- **Masking**: Ensures sequential (autoregressive) prediction
- **Linear layer**: Maps embeddings to vocabulary logits
- **Translation process**: Autoregressive token-by-token generation until EOS
- **Decoder method**: Takes target sequence and memory as inputs; target receives token embedding and positional encoding similar to source
- **Transformer layer**: Handles both encoding and decoding processes
- **Cross-attention**: Computes attention scores between each target position and all source positions