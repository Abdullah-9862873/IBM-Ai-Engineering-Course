# Module 2: Parameter-Efficient Fine-Tuning (PEFT)

## Introduction to PEFT

### What is PEFT?

- **PEFT (Parameter-Efficient Fine-Tuning)**: Methods that reduce the number of trainable parameters to adapt large pretrained models to specific downstream applications
- **Purpose**: Significantly decrease computational resources and memory storage needed for fine-tuning

### Why PEFT?

Full fine-tuning has limitations:
- **High computational resources**: Requires significant GPU memory and processing power
- **Large memory storage**: Must store all model parameters
- **Risk of overfitting**: Especially with limited task-specific labeled data
- **Time-consuming**: Complex implementation requiring many epochs
- **Catastrophic forgetting**: Model forgets previously learned information when trained with new data

PEFT addresses these issues by:
- Reducing trainable parameters
- Preserving pretrained knowledge
- Lowering computational costs

---

## Types of PEFT Methods

### 1. Selective Fine-Tuning

- **Approach**: Updates only a subset of layers or parameters
- **Pros**: Less resource-intensive than full fine-tuning
- **Cons**: Less effective for transformer architectures due to high number of parameters
- **Use case**: Works for other network architectures, not ideal for transformers

### 2. Additive Fine-Tuning

- **Approach**: Adds new task-specific layers or components to pretrained model
- **Key Feature**: Keeps pretrained parameters fixed
- **Implementation**: Task-specific customization while preserving pretrained knowledge

#### Adapters

- **Location**: Added between attention blocks in transformers
- **Architecture**:
  - Down projection layer: Reduces input dimension
  - Non-linear transformation
  - Up projection layer: Restores dimension
- **Storage**: Only adapters need to be stored
- **Benefit**: Transformer maintains general language understanding; adapters store task-specific information

### 3. Reparameterization Fine-Tuning

- **Approach**: Uses low-rank transformations to reparameterize network weights
- **Benefit**: Reduces number of trainable parameters while maintaining performance

---

## Soft Prompts

### Overview

- **Soft Prompts**: Learnable tensors concatenated with input embeddings that can be optimized to a dataset
- **Purpose**: Improve training process for large pretrained models

### Types

1. **Prompt Tuning**: Learnable prompts added to input
2. **Prefix Tuning**: Append parameter embeddings to existing embeddings
3. **P-Tuning**: Learnable prompt embeddings
4. **Multitask Prompt Tuning**: Prompts for multiple tasks

### Example: Prefix Tuning

```
Original embeddings (purple) + New embeddings (red) → Freeze all parameters except new embeddings
```

---

## Understanding Rank

### Definition

- **Rank**: Minimum number of vectors needed to span a space
- **Concept**: Essentially a dimension

### Examples

1. **2D Space**: 2 vectors can reach any point → Rank = 2
2. **3D Space**: 2 vectors can only span a 2D plane → Rank = 2 (not full 3D)

### Application in Neural Networks

- **Low-Rank Operations**: Reduce number of parameters
- **Efficiency**: Only need 2D to span the space in higher dimensional context
- **Benefit**: More efficient model with fewer parameters

---

## LoRA (Low-Rank Adaptation)

### Overview

- **Full Name**: Low-Rank Adaptation
- **Method**: Adds low-rank layers to original layer, reducing parameters needed to represent weight matrices
- **Benefit**: Maintains model performance while significantly reducing computational costs

### How It Works

1. **Original Network**: Uses full network weights
2. **LoRA Addition**: Adds low-rank layers to original layer
3. **Reparameterization**: Captures most important directions in data
4. **Result**: Effective adaptation with fewer parameters

---

## Related Methods

### QLoRA (Quantized Low-Rank Adaptation)

- Combines low-rank adaptations with quantization
- Reduces memory footprint and computational requirements

### DoRA (Weight-Decomposed Low-Rank Adaptation)

- Adjusts rank in low-rank space based on magnitude of components
- Optimizes model performance and efficiency

---

## PEFT Methods Summary

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| Selective | Update subset of parameters | Less resource-intensive | Less effective for transformers |
| Additive | Add new task-specific layers | Preserves pretrained knowledge | Additional storage for adapters |
| Reparameterization | Use low-rank transformations | Most efficient | Complex implementation |

---

## Summary and Highlights

- **PEFT**: Reduces trainable parameters for efficient fine-tuning
- **Methods**: Selective, additive, reparameterization fine-tuning
- **Selective Fine-Tuning**: Updates subset of layers/parameters
- **Additive Fine-Tuning**: Adds task-specific layers (adapters)
- **Soft Prompts**: Learnable tensors concatenated with input embeddings
- **Rank**: Minimum vectors needed to span a space
- **Reparameterization**: Uses low-rank transformations to reduce parameters
- **LoRA**: Most popular PEFT method using low-rank adaptation
- **QLoRA**: Combines LoRA with quantization
- **DoRA**: Adjusts rank based on magnitude for efficiency

---

## LoRA (Low-Rank Adaptation) - Detailed

### Overview

- **Purpose**: Add lightweight plug-ins to original model for efficient functioning
- **Benefits**:
  - Reduced trainable parameters
  - Decreased training time
  - Lower resource usage and memory footprint
  - Produces smaller model weights for easy storage

### How LoRA Works

#### Example: Layer Reduction

- **Original Layer**: Input dimension 10, Output dimension 8 → 80 parameters
- **With LoRA**: 
  - Input 10 → 3 (30 parameters)
  - 3 → 8 (24 parameters)
  - Total: 54 parameters (reduced from 80)

#### Matrix Algebra

- **Original weight matrix**: W_0 (d × k)
- **Forward pass**: h(x) = W_0 × x
- **With LoRA**: 
  - Add δW to original matrix
  - h(x) = (W_0 + δW) × x
  - δW = B × A (where B is d×r, A is r×k)
  - r = rank (hyperparameter, smaller than d and k)

#### Fine-tuning with LoRA

- Original weight matrix W_0 remains **frozen**
- Only low-rank matrices B and A are updated
- **Parameters reduced**: d×k → d×r + r×k

### Optimization

- **Scaling factor**: (alpha / r) in forward pass
- **Hyperparameters**: alpha and r
- **Loss function**: Applied to updated parameters using gradient descent (e.g., Adam)
- **Applicable**: For transformers' attention layers (query, key, value parameters)
- **Can apply to**: Encoders and decoders

---

## LoRA with PyTorch

### Dataset Preparation (IMDb)

```python
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

# Create iterators
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')

# Convert to map-style datasets
train_dataset = list(train_iter)
test_dataset = list(test_iter)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Collate Function

```python
def collate_batch(batch):
    text, label = zip(*batch)
    return torch.tensor(text), torch.tensor(label)

train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_batch)
```

### LoRA Layer Class

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=1.0):
        super(LoRALayer, self).__init__()
        
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x):
        # LoRA: B × A × x, scaled by alpha
        lora_output = torch.matmul(torch.matmul(x, self.A), self.B)
        return self.alpha * lora_output
```

### Linear with LoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=1, alpha=1.0):
        super(LinearWithLoRA, self).__init__()
        
        self.original_linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        original_output = self.original_linear(x)
        lora_output = self.lora(x)
        return original_output + lora_output
```

### Model Modification

```python
# Replace hidden layer with LoRA layer
model.fc1 = LinearWithLoRA(model.fc1, rank=1, alpha=1.0)
```

### Training

```python
def train_model(model, dataloader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        for text, label in dataloader:
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        scheduler.step()
```

### Saving LoRA Parameters

```python
# Save only A, B, and alpha
lora_params = {
    'A': model.fc1.lora.A,
    'B': model.fc1.lora.B,
    'alpha': model.fc1.lora.alpha
}
# Storage: ~450 parameters vs 12,800 for full linear layer (~28x smaller)
```

---

## LoRA with HuggingFace

### Dataset Loading

```python
from datasets import load_dataset

imdb_dataset = load_dataset('imdb')
```

### Tokenization

```python
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

tokenized_dataset = imdb_dataset.map(tokenize_function, batched=True)
```

### Loading Model

```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=1,                          # Rank
    lora_alpha=1.0,                # Scaling factor
    lora_dropout=0.01,             # Dropout
    target_modules=['q_lin', 'v_lin'] # Query and Value attention layers
)
```

### Applying LoRA

```python
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=1e-4,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
)
```

### Training with Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

trainer.train()
```

### Advantages

- **Simple**: Easy integration with HuggingFace
- **Scalable**: Works with large pre-trained models
- **Efficient**: Only small adapter weights stored
- **Flexible**: Can target specific modules (q_lin, v_lin)

---

## Summary: LoRA

- **Definition**: Adds lightweight low-rank matrices to reduce parameters
- **Matrix algebra**: δW = B × A (where r = rank)
- **Original weights**: Remain frozen
- **Trainable parameters**: d×r + r×k (much smaller than d×k)
- **PyTorch implementation**: Custom LoRALayer class
- **HuggingFace**: Uses PEFT library
- **Storage reduction**: ~28x smaller than full weights

---

## QLoRA (Quantized Low-Rank Adaptation)

### Overview

- **Definition**: Combines quantization with LoRA for efficient fine-tuning
- **Purpose**: Optimize performance and efficiency of LLMs
- **Key Benefit**: Reduces memory footprint without sacrificing accuracy

### Quantization

- **Process**: Reduces numerical precision to finite discrete levels
- **Quantization Range**: -1 to 1
- **Effect**: Decreases memory usage, enables computation on limited precision hardware

### Quantization Levels

**3-bit quantization** (8 levels):
- Levels: -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1

**2-bit quantization** (4 levels):
- Levels: 1, -1/2, 1/2, 1

### QLoRA Methods

1. **NF4 (4-bit Normal Float)**: Unique quantization method
2. **Double Quantization**: Further reduces memory
3. **Paged Optimizers**: Memory management trick

### Memory Footprint Comparison (7B parameter model)

| Component | FP16 | 4-bit Quantized |
|-----------|------|----------------|
| Model | 14 GB | 3.5 GB |
| Gradients | 14 GB | 3.5 GB |
| Optimizer | 56 GB | 14 GB |
| Activations | 14 GB | 3.5 GB |
| **Total** | **98 GB** | **24.5 GB** |

**Reduction**: ~75% (98 GB → 24.5 GB)

---

## Soft Prompts

### Overview

- **Definition**: Learnable tensors concatenated with input embeddings
- **Purpose**: Modify input data to generate desired outputs
- **Alternative to**: Hard prompts (explicit textual instructions)

### Hard Prompts vs Soft Prompts

| Aspect | Hard Prompts | Soft Prompts |
|--------|-------------|------------|
| Creation | Manually crafted | Learnable tensors |
| Process | Discrete tokens | Continuous embeddings |
| Optimization | None | Gradient descent |
| Flexibility | Limited | High |

### Soft Prompt Methods

#### 1. Prompt Tuning
- Add learnable parameters to input embeddings
- Keep rest of model's parameters fixed
- Only prompt tokens are updated

#### 2. Prefix Tuning
- Add task-specific vectors to input
- Integrate across ALL model layers (not just input)
- Uses separate FFN for stability

#### 3. P-Tuning
- Trainable embedding tensor with prompt encoder
- Prompt encoder: bidirectional LSTM
- Flexible placement anywhere in sequence
- Uses anchor tokens for performance

### Implementation Steps

1. **Select Pre-trained Model**: Base model (usually frozen)
2. **Define Task**: e.g., sentiment analysis, summarization
3. **Initialize Soft Prompts**: Random or prior knowledge
4. **Integrate**: Modify input processing pipeline
5. **Freeze Parameters**: Keep model parameters frozen
6. **Optimize**: Train soft prompts via backpropagation
7. **Evaluate**: Test on validation data

### Best Practices

- Use robust pre-trained model as base
- Use diverse and representative datasets
- Regular evaluation and tuning
- Cross-validation to prevent overfitting

### Benefits

1. **Efficiency**: Fewer resources than full fine-tuning
2. **Flexibility**: Easy adaptation for various tasks
3. **Scalability**: Works across different models and datasets

---

## Ethical Considerations in Fine-Tuning LLMs

### Bias Amplification

- **Issue**: Models learn and amplify biases from training data
- **Sources**: Gender, race, ethnicity biases
- **Mitigation**: 
  - Debiasing techniques
  - Word embedding adjustments
  - Filtering biased data

### Data Privacy

- **Issue**: Models may memorize and reproduce sensitive data
- **Mitigation**:
  - Differential privacy (introduces noise)
  - Data anonymization

### Environmental Impact

- **Issue**: High energy consumption from training
- **Mitigation**:
  - PEFT methods (reduce computational needs)
  - Model distillation
  - Carbon offset initiatives

### Transparency and Accountability

- **Practice**:
  - Document fine-tuning process
  - Record data sources and modifications
  - Provide clear usage guidelines

### Ensuring Fair Representation

- **Practice**:
  - Use diverse datasets
  - Represent various demographics
  - Regular evaluations and updates

### Summary

- Address ethical concerns proactively
- Implement bias mitigation techniques
- Protect data privacy
- Ensure transparency
- Foster inclusive model development

---

## Summary and Highlights

### PEFT Overview

- **PEFT**: Parameter-efficient fine-tuning methods reduce trainable parameters
- **Methods include**: Selective fine-tuning, additive fine-tuning, reparameterization fine-tuning
- **Purpose**: Efficient adaptation with fewer resources

### LoRA Details

- **LoRA**: Low-rank adaptation reduces trainable parameters by leveraging pre-trained models
- **Explanation**: Uses matrix algebra
- **Key**: Original weight matrix remains frozen during fine-tuning
- **Implementation**: LinearWithLoRA class copies original linear model and creates LoRA layer object

### LoRA with PyTorch

- **Dataset**: IMDB (Internet Movie Database) for movie reviews
- **LoRALayer**: Implements LoRA module with two low-rank matrices (A and B)
- **Result**: Efficient parameter reduction

### LoRA with HuggingFace

- **Tokenizer**: Creates input IDs
- **Model**: Loads BERT-like models from HuggingFace library
- **Ease**: Helps train models easily

### QLoRA Details

- **QLoRA**: Quantized low-rank adaptation
- **Purpose**: Optimize performance and efficiency of LLMs
- **Quantization**: Reduces numerical precision to discrete levels
- **Range**: Between -1 and 1
- **Methods**: 4-bit NormalFloat (NF4), double quantization
- **Memory**: Uses model parameters, gradients, optimizer states, activations

### Quantization Benefits

- Reduces model size
- Improves inference speed
- Maintains model accuracy
- Enables running on less powerful hardware

### Model Quantization Tools

- **TensorFlow Lite**: For model quantization
- **PyTorch**: For quantization techniques

### Quantization Techniques

1. **Uniform quantization**: Equal spacing between levels
2. **Non-uniform quantization**: Unequal spacing
3. **Weight clustering**: Grouping similar weights
4. **Pruning**: Removing unnecessary parameters

### Summary

- PEFT methods reduce trainable parameters significantly
- LoRA is the most popular PEFT method
- QLoRA combines quantization with LoRA
- Quantization reduces memory footprint by ~75%
- Both enable efficient fine-tuning of large models