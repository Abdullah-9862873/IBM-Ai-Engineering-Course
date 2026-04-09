# Module 1: Transformers and Fine-Tuning

## Course Overview

### About This Course

- **Topic**: Fine-tuning with transformers is becoming a cornerstone of AI strategies across industries
- **Focus**: Encoder models for simplicity (methods applicable to decoder models too)
- **Target Audience**: Existing and aspiring data scientists, ML engineers, deep learning engineers, AI engineers, and developers

### Prerequisites

- Basic knowledge of Python and PyTorch
- Awareness of transformers
- Understanding of how to load a model

### Learning Outcomes

After completing this course, you will be able to:
- Apply skills in working with transformer-based LLMs for generative AI engineering
- Use pretrained transformers for language tasks
- Fine-tune transformers for specific tasks
- Gain insights into Parameter Efficient Fine-Tuning (PEFT) using LoRA and QLoRA

---

## Course Content

### Major Topics Covered

1. **Transformers and Major Language Models**
   - Generative models and fine-tuning techniques
   - Advanced training methods
   - Frameworks: HuggingFace and PyTorch
   - Loading models for inference and training

2. **Fine-Tuning Models Using HuggingFace in PyTorch**
   - Pretraining and fine-tuning LLMs
   - Using HuggingFace transformers library

3. **PEFT Adapters**
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized Low-Rank Adaptation)
   - Soft prompts
   - Rank selection

4. **Model Quantization**
   - Definition using NLP
   - Unique quantization methods

---

## Framework Comparison: HuggingFace vs PyTorch

### HuggingFace

- Provides pretrained transformer models
- Easy-to-use API for loading models
- Extensive model hub
- Supports both inference and training
- Community-driven

### PyTorch

- Deep learning framework
- More control over model implementation
- Flexible for custom architectures
- Integrates well with HuggingFace

### Key Differences

| Feature | HuggingFace | PyTorch |
|---------|-------------|----------|
| Ease of use | Higher | Lower |
| Flexibility | Moderate | Higher |
| Pretrained models | Extensive library | Limited |
| Custom training | Supported | Full control |

---

## Fine-Tuning Concepts

### What is Fine-Tuning?

- Process of taking a pretrained model and training it further on a specific dataset
- Adapts the model to perform specific tasks
- Leverages knowledge from pretraining

### Why Fine-Tune?

- Achieve better performance on specific tasks
- Adapt to domain-specific vocabulary
- Reduce training time and cost
- Improve task accuracy

### Fine-Tuning Process

1. Load pretrained model
2. Prepare task-specific dataset
3. Configure training parameters
4. Train on new dataset
5. Evaluate and save model

---

## Parameter Efficient Fine-Tuning (PEFT)

### Overview

- Efficient method to fine-tune large language models
- Trains only a small number of parameters
- Reduces computational cost and memory usage

### LoRA (Low-Rank Adaptation)

- Adds low-rank matrices to model weights
- Trains only the added matrices
- Keeps pretrained weights frozen
- Achieves comparable performance to full fine-tuning

### QLoRA (Quantized LoRA)

- Combines quantization with LoRA
- Uses quantized pretrained weights
- Enables fine-tuning on consumer hardware

---

## Model Quantization

### Definition

- Process of reducing model precision
- Converts weights from float32 to lower precision (e.g., int8)
- Reduces model size and inference time

### Methods

1. **Post-Training Quantization (PTQ)**
   - Quantize after training
   - Simple to implement
   - May have accuracy loss

2. **Quantization-Aware Training (QAT)**
   - Simulates quantization during training
   - Better accuracy
   - More complex

3. **GPTQ**
   - Post-training quantization
   - Optimized for LLMs

4. **AWQ**
   - Activation-aware weight quantization
   - Preserves important weights

---

## Summary and Highlights

- **Fine-tuning**: Adapts pretrained transformers to specific tasks
- **HuggingFace**: Easy-to-use library for loading and fine-tuning models
- **PyTorch**: Deep learning framework with full control
- **LoRA**: Efficient fine-tuning via low-rank adaptation
- **QLoRA**: Combines quantization with LoRA for efficiency
- **Quantization**: Reduces model size and improves inference speed

---

## Labs Overview

The course includes hands-on labs:
- Pretraining and fine-tuning LLMs using HuggingFace
- Implementing LoRA and QLoRA
- Model quantization techniques
- Practice exercises in Jupyter notebooks

---

## Course Structure

- **Videos**: Short and focused on main topics
- **Readings**: Detailed content in text format
- **Labs**: Technical environment with code snippets
- **Quizzes**: Practice and assess knowledge

---

## Summary

- This course covers fine-tuning transformers for generative AI engineering
- Focuses on encoder models but methods apply to decoder models too
- Covers PEFT techniques (LoRA, QLoRA)
- Includes model quantization methods
- Uses both HuggingFace and PyTorch frameworks
- Prepares for working with LLMs in specialized contexts

---

## Hugging Face vs PyTorch

### Hugging Face Overview

- **Origin**: Started as a chatbot company
- **Evolution**: Became a platform and community dedicated to ML and data science
- **Purpose**: Assist users in developing, deploying, and training ML models
- **Nickname**: "GitHub of Machine Learning" (collaborative hub for AI developers)

### Key Features of Hugging Face

1. **Transformer Library**
   - Offers pre-trained models: BERT, GPT, T5
   - Ready to use for various NLP tasks
   - Extensive model hub

2. **Community**
   - Growing community contributes to model repositories and tools
   - Open sharing and testing among developers

3. **Infrastructure**
   - Tools for demonstrating, operating, and integrating AI
   - Platform for exploring models and datasets

### PyTorch Overview

- **Origin**: Developed by Facebook AI Research (now Meta)
- **Purpose**: Open source deep learning framework for building neural networks
- **Language**: Python-based high-level API
- **Popularity**: Leading ML framework in academic and research communities

### Key Features of PyTorch

1. **Dynamic Computation Graph**
   - Allows changes to network architecture on the fly during runtime
   - Excellent for rapid prototyping
   - Expedites debugging process

2. **Ease of Use**
   - Intuitive and straightforward Python-based syntax
   - Run and test portions of code in real time

3. **GPU Acceleration**
   - Strong GPU acceleration for large scale computations
   - Particularly beneficial for deep learning

4. **Flexibility**
   - Supports wide variety of neural network architectures
   - Full control over model implementation

### Comparison Summary

| Aspect | Hugging Face | PyTorch |
|--------|-------------|----------|
| Type | Platform/Library | Deep Learning Framework |
| Best For | Ready-to-use NLP models | Custom deep learning models |
| Primary Use | NLP tasks (classification, translation, QA) | Research and development |
| Integration | Works with PyTorch/TensorFlow | Base framework |

### Applications

1. **Sentiment Analysis**
   - Classifies sentiment of user reviews or social media posts

2. **Language Translation**
   - Models like T5 and M2M for translation

3. **Question Answering**
   - Build systems to provide answers based on questions and context

4. **Text Summarization**
   - Automatically generate concise summaries from large text

### Integration

- PyTorch can be integrated with Hugging Face transformers
- Allows tackling complex NLP tasks intuitively and efficiently
- Combines flexibility of PyTorch with ready-to-use models from Hugging Face

---

## Summary and Highlights

- **Hugging Face**: Platform and community for ML; known as "GitHub of ML"
- **Transformers Library**: Offers pre-trained models (BERT, GPT, T5)
- **PyTorch**: Open source deep learning framework by Meta
- **Dynamic Computation Graph**: Key feature allowing runtime changes
- **GPU Acceleration**: Efficient large scale computations
- **Integration**: PyTorch + Hugging Face for powerful NLP solutions
- **Applications**: Sentiment analysis, translation, QA, summarization

---

## Using Pre-Trained Transformers and Fine-Tuning

### Pre-Trained Transformer Models

- **Models**: BERT, Llama, GPT
- **Architecture**: Attention-based
- **Pre-training**: On large unlabeled text datasets
- **Benefit**: Learn rich language representations for downstream NLP tasks

### Why Fine-Tuning is Necessary

Training LLMs is:
- **Computationally expensive**: Requires powerful GPUs
- **Time consuming**: Weeks or months
- **Data intensive**: Requires substantial training data
- **Costly**: Infrastructure setup and maintenance

Fine-tuning:
- **Adapts** pre-trained models to specific tasks/domains
- **Uses domain-specific data**
- **Improves task performance** by adjusting model parameters
- **Leverages pre-existing language understanding**
- **Saves time and resources** compared to training from scratch

### Benefits of Fine-Tuning

1. **Transfer Learning**
   - Works well with limited labeled data
   - Provides time and resource efficiency

2. **Task-Specific Adaptation**
   - Tailors model responses to specific requirements
   - Ensures accurate and contextually relevant outputs
   - Example: Sentiment analysis, text generation in diverse domains

### Fine-Tuning Pitfalls to Avoid

1. **Overfitting**
   - Cause: Small dataset or too many training epochs
   - Solution: Use adequate data and reasonable epochs

2. **Underfitting**
   - Cause: Insufficient training or inappropriate learning rate
   - Solution: Ensure adequate training and appropriate learning rate

3. **Catastrophic Forgetting**
   - Cause: Model loses initial broad knowledge
   - Solution: Prevent performance degradation on various NLP tasks

4. **Data Leakage**
   - Cause: Training and validation datasets overlap
   - Solution: Keep datasets separate

### Fine-Tuning Approaches

#### 1. Self-Supervised Fine-Tuning
- Model predicts missing words (next words, masked words)
- Uses large unlabeled datasets

#### 2. Supervised Fine-Tuning
- Uses labeled data from target task
- Improves performance on specific tasks (e.g., sentiment classification)

#### 3. Reinforcement Learning from Human Feedback (RLHF)
- Adjusts model based on explicit human feedback
- Aligns outputs with human preferences

#### 4. Direct Preference Optimization (DPO)
- Emerging approach
- Optimizes based on human preferences
- **Features**:
  - Simpler than RL
  - Explicitly aligns with human preferences
  - No reward training needed
  - Faster convergence

### Supervised Fine-Tuning Methods

#### Full Fine-Tuning
- All model parameters are tuned for the specific task

#### Parameter-Efficient Fine-Tuning (PEFT)
- Large pre-trained models fine-tuned without modifying most original parameters
- More efficient than full fine-tuning

---

## Fine-Tuning with PyTorch

### Datasets Used

1. **IMDB Dataset**
   - 50,000 movie reviews
   - 2 classes: Positive, Negative
   - Small dataset for sentiment analysis

2. **AG News Dataset**
   - 120,000 training samples
   - 7,600 test samples
   - 4 classes: World, Sports, Business, Science/Technology

### Fine-Tuning Strategy

1. **Pre-train** on AG News dataset (robust language understanding)
2. **Fine-tune** on IMDB dataset (sentiment analysis)

### Model Implementation

```python
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, vocab_size, d_model, nhead, num_layers, dropout):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout),
            num_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average along first dimension
        return self.classifier(x)
```

### Data Preparation

```python
# Load dataset
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')

# Create vocabulary from GloVe embeddings
vocab = build_vocab_from_glove(embeddings, default_index=UNK_INDEX)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
```

### Fine-Tuning Complete Model

```python
# Create model with new output layer
model = TransformerClassifier(num_classes=2, vocab_size=len(vocab), ...)

# Load pretrained parameters
model.load_state_dict(torch.load('pretrained_model.pt'))

# Change final layer neurons to match new task (4 -> 2 for IMDB)
model.classifier = nn.Linear(d_model, 2)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    train_model(model, train_loader, optimizer, criterion)
```

### Fine-Tuning Only Final Layer

```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Train only classifier
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
```

### Results Comparison

| Method | Training Speed | Accuracy |
|--------|----------------|----------|
| Full Fine-Tuning | Slower | ~90% |
| Final Layer Only | Faster | Significantly worse |

### Summary: Fine-Tuning

- **Full Fine-Tuning**: ~90% accuracy, but slower
- **Final Layer Only**: Faster but worse performance
- **Trade-off**: Balance between training time and accuracy

---

## Summary and Highlights

- **Pre-trained models**: BERT, Llama, GPT use attention architecture
- **Fine-tuning**: Adapts models to specific tasks/domains
- **Pitfalls**: Overfitting, underfitting, catastrophic forgetting, data leakage
- **Approaches**: Self-supervised, Supervised, RLHF, DPO
- **Methods**: Full fine-tuning, PEFT
- **PyTorch**: Supports both full and final-layer fine-tuning
- **Trade-off**: Training speed vs accuracy

---

## Fine-Tuning with Hugging Face

### Overview

- **Platform**: Hugging Face - open-source ML platform with built-in transformers library
- **Purpose**: Share models, datasets, and showcase work
- **Advantages**: Simplifies fine-tuning process

### Loading Datasets

```python
from datasets import load_dataset

# Load Yelp reviews dataset
dataset = load_dataset('yelp_polarity')
```

### Dataset Structure

- **Yelp Reviews**: List-like object with user reviews and metadata
- **Each review**: Dictionary with:
  - `text`: Review text
  - `label`: Star rating (1-5)

### Tokenization

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Apply tokenization to dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove text column, rename label
tokenized_dataset = tokenized_dataset.remove_columns(['text'])
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')

# Convert to PyTorch tensors
tokenized_dataset.set_format('torch')
```

### DataLoader Creation

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=batch_size)
```

### Loading Pre-Trained Model

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5  # Number of output classes
)
```

### Training Loop

```python
from transformers import AdamW
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # Evaluate model performance
            pass
    return total_loss
```

### SFT Trainer (Supervised Fine-Tuning Trainer)

- **Purpose**: Simplifies and automates training tasks
- **Advantages**:
  - More efficient than manual PyTorch training
  - Less error-prone
  - Automated handling of training loop

#### Using SFT Trainer for MLM

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load IMDB dataset
imdb_dataset = load_dataset('imdb')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Load masked language model
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Create SFT Trainer
trainer = Trainer(
    model=mlm_model,
    args=training_args,
    train_dataset=imdb_dataset['train'],
    eval_dataset=imdb_dataset['test'],
)

# Train
trainer.train()
```

### Making Predictions

```python
from transformers import pipeline

# Create mask filler pipeline
mask_filler = pipeline(
    'fill-mask',
    model=mlm_model,
    tokenizer=tokenizer
)

# Make prediction
result = mask_filler('This is a [MASK] movie!')

# Output: List of predictions with token and score
# [{'token_str': 'great', 'score': 0.85, ...}]
```

### Summary: Fine-Tuning with Hugging Face

- **Load datasets**: Using `load_dataset` function
- **Tokenize**: Using BERT tokenizer with padding and truncation
- **Model**: Load pre-trained BERT classification model
- **Train**: Using custom training loop or SFT Trainer
- **SFT Trainer**: Automates and simplifies training process
- **Prediction**: Using pipeline for fill-mask tasks

---

## Summary and Highlights

### Fine-Tuning Overview

- **Fine-tuning**: Process of adapting a pretrained model for specific tasks or use cases
- **Process**:
  - Collate function tokenizes the dataset
  - Transformer-based model class defines classification in PyTorch
  - Forward method applies embeddings to the input
  - Train_model function trains a transformer model
- **Benefits**:
  - Enhances efficiency
  - Saves time and computational resources compared to training from scratch
  - Enables transfer learning
  - Provides time and resource efficiency
  - Allows tailored responses
  - Enables task-specific adaptation

### Hugging Face Overview

- **Hugging Face**: Open-source ML platform with built-in transformers library for NLP applications
- **Key Feature**: Built-in datasets can be loaded using `load_dataset` function
- **Also known as**: "GitHub of Machine Learning"

### Key Differences: Hugging Face vs PyTorch

| Feature | Hugging Face | PyTorch |
|---------|-------------|---------|
| Type | Platform and community dedicated to ML and data science | Software-based open-source deep learning framework |
| Most Popular Feature | Transformers library | Dynamic computation graph |
| Purpose | Share models, datasets, and tools for NLP | Build and train custom neural networks |
| Use Case | Ready-to-use NLP models | Research and development |

### Summary

- **Fine-tuning**: Adapts pretrained transformers for specific tasks
- **Hugging Face**: Platform for NLP with transformers library
- **PyTorch**: Deep learning framework for custom models
- **Difference**: Hugging Face provides ready-to-use models; PyTorch provides flexibility and control