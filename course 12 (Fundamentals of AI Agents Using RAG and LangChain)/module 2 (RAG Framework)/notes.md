# Retrieval Augmented Generation (RAG)

## What is RAG?

- **RAG (Retrieval Augmented Generation)** is an AI framework that helps optimize LLM output
- Combines capabilities of LLMs with specific domain/internal organizational data
- **Without retraining** the model, RAG provides accurate responses to specialized queries

### Why RAG is Needed
- Pre-trained LLMs face challenges with **specific domain knowledge**
- They perform well on general tasks but may provide **inaccurate responses** to specialized queries
- External relevant knowledge sources ensure **more accurate responses**

### Example: Company Mobile Policy
- Chatbot can't access company's confidential mobile policy
- RAG helps generate **domain-specific responses** using internal knowledge base

---

## RAG Process Overview

### Two Main Components
1. **Retriever**: Core of RAG - encodes and retrieves relevant information
2. **Generator**: Functions as a chatbot to generate responses

### Steps in RAG Process

#### Step 1: Text Embedding
- **Question Encoder**: Converts inserted prompt/question into high-dimensional vectors
- **Context Encoder**: Converts knowledge base documents into high-dimensional vectors

#### Step 2: Retrieval
- System matches similar vectors from prompt with knowledge base vectors
- Retrieves most relevant information based on vector similarity

#### Step 3: Augmented Query Creation
- Combines text from retrieved vectors with original prompt
- Creates augmented query for the language model

#### Step 4: Model Generation
- Language model uses augmented query to generate response
- Uses content from knowledge base for accurate answers

---

## Prompt Encoding

### Token Embedding
- Each token (word or sub-word) in prompt is transformed into high-dimensional vector
- Uses pre-trained models like **BERT** or **GPT**

### Vector Averaging
- After embedding all tokens, take **average of all token vectors**
- Creates single vector representation for the entire prompt

---

## Knowledge Base Processing

### Why Chunk Text?
- Original policy documents are large
- Inserting entire documents into chatbot is challenging
- **Break into smaller, manageable chunks** for targeted retrieval

### Process
1. Break documents into text chunks
2. Embed each chunk into vectors using token embedding model
3. Average token vectors to create single vector per chunk
4. Store embeddings in **vector database** with chunk ID as key
5. Use **distance operations** to find relevant information

---

## Distance Metrics for Similarity

### Dot Product
- Considers vector's **direction and magnitude**
- Prioritizes overall alignment
- Prefers vector magnitude

### Cosine Distance
- Focuses on **direction** (angular difference)
- Measures angular similarity between vectors

### Selection
- For magnitude: use **dot product**
- For direction: use **cosine distance**

### Retrieval Process
1. Compare prompt vector with context vectors in knowledge base
2. Calculate distances using selected metric
3. Select **top K** context vectors closest to prompt vector
4. Use chunk IDs to retrieve relevant text chunks

---

## RAG Encoders: DPR

### Context Encoder (Dense Passage Retrieval)
- Focuses on encoding potential answer passages/documents
- Creates embeddings from extensive texts
- Allows comparison with question embeddings

### Implementation
```python
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

# Load tokenizer
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Load encoder
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Tokenize context
token_info = context_tokenizer(
    input_text,
    padding=True,
    max_length=256,
    truncation=True,
    return_tensors='pt'
)

# Generate embeddings
outputs = context_encoder(**token_info)
embeddings = outputs.pooler_output  # Shape: (batch, 768)
```

### Output
- **input_ids**: Token IDs for input text
- **token_type_ids**: Segment IDs
- **attention_mask**: Attention values

---

## Faiss (Facebook AI Similarity Search)

### What is Faiss?
- Library developed by Facebook AI Research
- Efficient algorithms for searching large collections of high-dimensional vectors
- Calculates distance between question embedding and context embeddings

### Implementation
```python
import faiss
import numpy as np

# Convert context embeddings to float32 numpy array
context_embeddings_np = np.array(context_embeddings).astype('float32')

# Initialize Faiss index for L2 (Euclidean) distance
index = faiss.IndexFlatL2(context_embeddings_np.shape[1])

# Add context embeddings to index
index.add(context_embeddings_np)
```

### Search
```python
# Search for top K closest embeddings
distances, indices = index.search(question_embedding, k=top_k)
```

---

## Question Encoder

### DPR Question Encoder
- Encodes input questions into fixed-dimensional vector representations
- Grasps meaning and context to facilitate answering

### Implementation
```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# Load tokenizer
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Load encoder
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Encode question
question_inputs = question_tokenizer(question_text, return_tensors='pt')
question_embedding = question_encoder(**question_inputs).pooler_output
```

---

## Retrieval and Generation Process

### Complete RAG Pipeline
1. **Encode question** using question encoder
2. **Search** Faiss index for top K similar contexts
3. **Retrieve** corresponding text chunks using indices
4. **Combine** retrieved context with original question
5. **Generate** response using language model (e.g., BART)

### Example: BART Generation
```python
from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Generate response
inputs = tokenizer(question, return_tensors='pt', max_length=1024, truncation=True)
outputs = model.generate(
    inputs['input_ids'],
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### With Context (RAG)
```python
# Get top K contexts
top_context_indices = indices[0]
top_contexts = [paragraphs[i] for i in top_context_indices]

# Combine question with context
augmented_input = f"Question: {question}\nContext: {top_contexts[0]}"

# Generate with context
inputs = tokenizer(augmented_input, return_tensors='pt', max_length=1024, truncation=True)
outputs = model.generate(inputs['input_ids'], max_length=150, min_length=40)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Summary

1. **RAG** combines LLM power with external knowledge retrieval
2. **Text embedding** converts prompts and documents into vectors
3. **DPR encoders** (context and question) create embeddings for retrieval
4. **Faiss** enables efficient similarity search in high-dimensional space
5. **Top K retrieval** selects most relevant context chunks
6. **Generator** (e.g., BART) produces final response using augmented input

---

# Module Summary and Cheat Sheet

## Key Takeaways

1. **RAG** is an AI framework that helps optimize LLM output
2. RAG combines retrieved information and generates natural language to create responses
3. **Two components**: Retriever (core of RAG) + Generator (chatbot)
4. **Retriever** encodes prompts and documents into vectors, stores in vector database, retrieves relevant context
5. **Generator** combines retrieved context with original prompt to produce response
6. **DPR Context Encoder** encodes potential answer passages/documents
7. **Faiss** calculates distance between question embedding and context embeddings
8. **DPR Question Encoder** encodes input questions into fixed-dimensional vectors

---

## Cheat Sheet: RAG with Hugging Face and PyTorch

### Text Processing
```python
def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    paragraphs = text.split('\n')
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs
```

### DPR Context Encoder
```python
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def encode_contexts(text_list):
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()
```

### Faiss Index
```python
import faiss

embedding_dim = 768
context_embeddings_np = np.array(context_embeddings).astype('float32')
index = faiss.IndexFlatL2(embedding_dim)
index.add(context_embeddings_np)
```

### Question Encoder
```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

def search_relevant_contexts(question, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    D, I = index.search(question_embedding, k)
    return D, I
```

### Generation (GPT2)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.generation_config.pad_token_id = tokenizer.pad_token_id

def generate_answer_without_context(question):
    inputs = tokenizer(question, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, 
                                  length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_answer(question, contexts):
    input_text = question + ' ' + ' '.join(contexts)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens=50, min_length=40,
                                  length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### BERT Mean Pooling
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def aggregate_embeddings(input_ids, attention_masks, bert_model=bert_model):
    mean_embeddings = []
    for input_id, mask in zip(input_ids, attention_masks):
        input_ids_tensor = torch.tensor([input_id]).to(DEVICE)
        mask_tensor = torch.tensor([mask]).to(DEVICE)
        with torch.no_grad():
            word_embeddings = bert_model(input_ids_tensor, attention_mask=mask_tensor)[0].squeeze(0)
            valid_embeddings_mask = mask_tensor[0] != 0
            valid_embeddings = word_embeddings[valid_embeddings_mask, :]
            mean_embedding = valid_embeddings.mean(dim=0)
            mean_embeddings.append(mean_embedding.unsqueeze(0))
    return torch.cat(mean_embeddings)
```

### RAG QA Function
```python
def RAG_QA(embeddings_questions, embeddings, n_responses=3):
    dot_product = embeddings_questions @ embeddings.T
    sorted_indices = torch.argsort(dot_product, descending=True).tolist()
    for index in sorted_indices[:n_responses]:
        print(responses[index])
```

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **RAG** | Architecture that enhances LLM responses by retrieving relevant external knowledge |
| **Embeddings** | Dense vector representations (BERT: 768-dim, MiniLM: 384-dim) |
| **BERT** | Pre-trained transformer generating contextual embeddings bidirectionally |
| **DPR** | BERT-based model with separate encoders for questions and contexts |
| **Context Encoder** | DPR component for encoding documents/passages |
| **Question Encoder** | DPR component for encoding queries |
| **Tokenization** | Converting text to subword tokens |
| **Attention Mask** | Binary tensor (1=content, 0=padding) for filtering |
| **Mean Pooling** | Averaging valid token embeddings for fixed-size vectors |
| **Faiss** | Library for efficient similarity search in high-dimensional space |
| **IndexFlatL2** | Faiss index computing Euclidean distance |
| **Vector Store** | Database for storing/searching high-dimensional vectors |
| **Dot Product** | Similarity metric for ranking retrieved documents |