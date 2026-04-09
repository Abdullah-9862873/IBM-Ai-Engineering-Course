# Module 2: Data Preparation for LLMs

## Tokenization

### Overview

- Tokenization: Process of breaking a sentence into smaller pieces (tokens)
- Tokens help models understand text better
- Example: "IBM taught me tokenization" → tokens: IBM, taught, me, tokenization
- Tokenizer: Program that breaks down text into individual tokens

---

## Tokenization Methods

### 1. Word-Based Tokenization

- Text is divided into individual words
- Each word is considered a token
- **Advantages**: Preserves semantic meaning
- **Disadvantages**: Significantly increases model's vocabulary
- **Examples**: NLTK and spaCy tokenizers
- **Issue**: May treat similar words (unicorn, unicorns) as different

### 2. Character-Based Tokenization

- Text is split into individual characters
- **Advantages**: Smaller vocabularies
- **Disadvantages**: 
  - Single characters may not convey same information as entire words
  - Increases input dimensionality and computational needs
- **Example**: "this is a sentence" → T, H, I, S, , I, S, , A, , S, E, N, T, E, N, C, E

### 3. Subword-Based Tokenization

- Frequently used words remain unsplit
- Infrequent words are broken into meaningful subwords
- Combines advantages of word-based and character-based tokenization
- **Algorithms**:
  - **WordPiece**: Evaluates benefits/drawbacks of splitting and merging symbols
  - **Unigram**: Breaks text into smaller pieces, starts with large list, narrows down iteratively
  - **SentencePiece**: Segments text into manageable parts, assigns unique IDs

---

## Tokenization Examples

### WordPiece (BERT Tokenizer)

- Using transformers library
- `##` (double hash) before a word indicates it should be attached to previous word without space

### Unigram/XLNet Tokenizer

- Using transformers library
- Tokens prefixed with `_` indicate new words preceded by a space
- Tokenization appears without prefix if it directly follows preceding word without space

---

## Tokenization in PyTorch

### Using torchtext Library

1. **Tokenize text**: Use `get_tokenizer` function to tokenize sentences
2. **Build vocabulary**: Use `build_vocab_from_iterator` function
   - Assigns each token a unique integer index
   - Model uses these indices to map words

### Example Code

```python
# Create tokenizer
tokenizer = get_tokenizer("basic_english")

# Create vocabulary from tokens
vocab = build_vocab_from_iterator(yield_tokens(data_iter))

# Set unknown token
vocab.set_default_index(vocab["<unk>"])

# Get word to index mapping
vocab.get_stoi()  # dictionary mapping words to indices
```

### Special Tokens

- **UNK**: Unknown token for words not in vocabulary
- **BOS**: Beginning of sentence
- **EOS**: End of sentence
- **PAD**: Padding token to ensure all sentences have same length

### Process

1. Tokenize sentences
2. Add special tokens (BOS at beginning, EOS at end)
3. Pad tokenized lines to match longest sentence
4. Convert tokens to indices using vocabulary

---

## Data Loaders

### Overview

- Data loader: Helps prepare and load data for training generative AI models
- PyTorch has a dedicated `DataLoader` class for handling and preparing data
- NLP data loaders enable efficient loading and pre-processing of textual data

### Purpose

- **Efficient batching and shuffling**: Essential for training neural networks
- **On-the-fly pre-processing**: Optimizes memory by loading only required data during training
- **Seamless integration**: Works with PyTorch training pipeline
- **Data augmentation**: Apply various transformations to input data

### Dataset Structure

- **Dataset**: Collection of data samples and their labels
- **Typical split**:
  - Training set: Used to train the model
  - Validation set: Used to tweak and validate model parameters
  - Test set: Used to assess model's performance in real-world scenarios

---

## Creating Custom Dataset in PyTorch

### Dataset Class

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

### Key Methods

- **`__init__`**: Initialize dataset with list of data
- **`__len__`**: Returns number of samples
- **`__getitem__`**: Retrieves item at specific index

---

## DataLoader Class

### Overview

- Iterator object used for loading, shuffling, and batching data from a dataset
- **Iterator**: Object that can be looped over, contains elements that can be iterated
- **Methods**: `iter()` and `next()` for traversing large datasets

### Creating a DataLoader

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = CustomDataset(sentences)

# Create data loader
data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True
)
```

### Key Parameters

- **batch_size**: Number of samples grouped in each batch
- **shuffle**: Randomly shuffle sentences before dividing into batches
  - Prevents model from learning patterns based on order of data

### Iteration

- Each call to `next()` returns new batches of samples
- Iterate through data loader to see how data is loaded in batches

---

## Batch Functions (Collate)

### Overview

- Most data transformation is performed in the collate function
- Transformations include:
  - Tokenizing text
  - Numericalizing (converting tokens to indices)
  - Resizing to consistent size
  - Converting to tensors

### Padding Sequences

- Sequences may not be the same size after tokenization
- Each sample in a data loader must be the same length
- Use `pad_sequence` function in PyTorch

```python
padded = pad_sequence(
    tensor_batch, 
    batch_first=True, 
    padding_value=0
)
```

### pad_sequence Parameters

- **batch_first**: If True, batch dimension is first (batch_size, seq_len)
  - If False, sequence dimension is first (seq_len, batch_size)
- **padding_value**: Value to use for padding (default: 0)

### Custom Collate Function

```python
def collate_fn(batch):
    # Tokenize each sample
    tokenized = [tokenizer(sample) for sample in batch]
    
    # Map tokens to indices
    indices = [vocab(token) for token in tokenized]
    
    # Pad sequences
    padded = pad_sequence(indices, batch_first=True, padding_value=0)
    
    return padded
```

### Using Custom Collate

```python
data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    collate_fn=collate_fn
)
```

---

## Summary

- Data loaders prepare and load data for training generative AI models
- PyTorch has dedicated DataLoader class
- Enable efficient batching, shuffling, on-the-fly processing
- Seamlessly integrate with PyTorch training pipeline
- Custom dataset class inherits from `torch.utils.data.Dataset`
- Collate function handles transformations: tokenization, numericalization, padding, tensor conversion

---

## Data Quality and Diversity for Effective LLM Training

### Overview

- Quality and diversity of data are foundational to developing robust and inclusive LLMs
- Well-curated data is critical as models advance in sophistication

---

## Data Quality

### Definition

- Data quality refers to accuracy, consistency, and completeness of the dataset
- Poor-quality data introduces noise that lowers model accuracy and reliability

### Practices to Ensure High Data Quality

#### Noise Reduction
- Remove irrelevant or repetitive data
- Help model focus on significant patterns and linguistic structures
- **Example**: Clean datasets by removing typos and irrelevant information (forum tags)

#### Consistency Checks
- Regularly verify consistency to prevent conflicting or outdated information
- Essential for entities like public figure names or technical terms
- Ensures uniform usage throughout the dataset

#### Labeling Quality
- For labeled datasets, accurate labeling is crucial
- Clear guidelines for human annotators reduce subjective errors
- Improves labeling quality

---

## Diverse Representation

### Importance

- Diverse dataset enhances model inclusivity
- Ensures accurate responses to varied cultural, demographic, and regional inputs
- Without diversity, models may reflect narrow views and unintentional biases

### Achieving Meaningful Diversity

#### Inclusion of Varied Demographics
- Incorporate text from various demographic groups
- Avoid over-representing a single perspective
- Include sources in multiple languages or dialects
- Represent diverse cultural norms for global applicability

#### Balanced Data Sources
- Draw from balanced sources: news, social media, literature, technical documents
- Broadens model's knowledge base
- Reduces dependence on any single source

#### Regional and Linguistic Variety
- Datasets from diverse regions and languages
- Expands linguistic and cultural context
- Enhances accuracy in multilingual contexts
- Better supports translation tasks

---

## Regular Updates

### Why Updates Matter

- Language constantly evolves with new terminologies and usage patterns
- Regular updates keep model relevant and accurate

### Benefits

#### New Vocabulary and Trends
- Capture evolving language trends (e.g., "selfie", "cryptocurrency")
- Reflect current terminology

#### Cultural and Social Norms
- As societal perspectives shift, language models should adapt
- LLM trained on outdated data may reinforce outdated norms or stereotypes

#### Model Retraining
- Periodically update model with fresh data
- Maintains alignment with contemporary knowledge and societal standards

---

## Ethical Considerations in Data Collection

### Importance

- Ethics in data collection essential to protect user privacy and ensure fair representation
- Fundamental to building trust and reducing biases

### Key Aspects

#### Data Privacy
- Use anonymized data to protect personal information
- Especially important for datasets containing sensitive or identifiable information

#### Fair Representation
- Ensure inclusion of marginalized voices
- Avoid bias that reinforces societal inequalities

#### Transparency in Data Sources
- Disclose data sources used for model training
- Fosters user trust
- Allows understanding of model's knowledge foundation

---

## Conclusion

- Focusing on data quality, diversity, and ethical practices contributes to developing LLMs that are:
  - Accurate
  - Inclusive
  - Socially responsible
- With a well-prepared dataset, LLM will perform more effectively
- Helps bridge gaps in AI fairness and representation

---

## Module 2 Summary: Key Takeaways

### Tokenization

- Tokenization and data loading are part of data preparation activities for NLP
- Tokenization breaks a sentence into smaller pieces or tokens
- **Tokenizers**: Tools that break down text into tokens (words, characters, or subwords)
  - Examples: NLTK, spaCy
- **Word-based tokenization**: Preserves semantic meaning but increases vocabulary
- **Character-based tokenization**: Smaller vocabularies but may not convey same info as whole words
- **Subword-based tokenization**: Frequently used words unsplit, infrequent words broken into subwords
- **Algorithms**: WordPiece, Unigram, SentencePiece
- **Special tokens**: `<bos>` at beginning, `<eos>` at end of tokenized sentence

### Data Loaders

- **Dataset**: Object in PyTorch representing collection of data samples
  - Each sample consists of input features and corresponding target labels
- **Data loader**: Helps prepare and load data to train generative AI models
  - Output data in batches instead of one sample at a time
- **Key parameters**:
  - Dataset to load from
  - Batch size (samples per batch)
  - Shuffle (whether to shuffle data for each epoch)
- **Iterator interface**: Easy to iterate over batches during training
- **PyTorch DataLoader class**: Dedicated class for data loading
- **Integration**: Seamlessly integrates with PyTorch training pipeline
- **Benefits**: Simplifies data augmentation and preprocessing
- **Collate function**: Prepares and formats individual samples into batches
  - Essential for variable-length data (text, time series, sequences)