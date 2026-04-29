# Introduction to Document Embedding

## Overview
Document embedding is the process of converting textual documents into numerical vectors. These vectors capture the semantic meaning of the documents, enabling machines to understand and process human language. Embedding models serve as the backbone for numerous NLP tasks such as text classification, sentiment analysis, and information retrieval.

--- 

## Understanding watsonx's Embedding Model 

IBM's watsonx.ai offers powerful embedding models tailored for modern NLP applications. These models excel at creating high-quality embeddings that capture the nuances of language across various contexts.

---

## Steps to Embed Documents Using watsonx.ai
 
### 1. Preparation of Data
- Ensure documents are clean and preprocessed
- Remove special characters, normalize text, and perform tokenization
- Organize documents into a format compatible with the model's input requirements (typically a list of strings or dataset)

### 2. Loading the watsonx Embedding Model
- Access embedding models through watsonx.ai's API or platform interface
- Load the pretrained embedding model, optimized for generating document embeddings

### 3. Embedding Process
- Pass prepared documents through the embedding model
- The model converts each document into a fixed-size numerical vector
- These vectors are dense and capture the semantic meaning of the documents

### 4. Postprocessing
- Normalize the vectors if necessary
- Store embeddings in a suitable format (such as a database) for downstream tasks

---

## Applications of Document Embeddings

### Document Clustering
- Use embeddings to group similar documents together
- Useful for organizing large document collections or creating topic-based clusters

### Semantic Search
- Implement a semantic search engine where queries are matched with documents based on semantic similarity (not just keyword matching)

### Text Classification
- Utilize embeddings as input features for classification models to categorize documents into predefined labels

---

## Benefits of Using watsonx's Embedding Model

| Benefit | Description |
|---------|-------------|
| **High Accuracy** | Produces embeddings that accurately reflect semantic content, leading to better NLP performance |
| **Scalability** | Handles large datasets efficiently, suitable for enterprise-level applications |
| **Versatility** | Applied to various use cases: search engines, recommendation systems, and more |

---

## Challenges and Considerations

### Computational Resources
- Embedding large volumes of documents requires substantial computational power
- Particularly challenging for real-time applications

### Model Interpretability
- Embeddings are powerful but difficult to interpret directly
- Vector representation is abstract and not human-readable

---

## Conclusion
Embedding documents is a powerful NLP technique that enables advanced applications such as document clustering, semantic search, and text classification. By converting text into meaningful numerical representations, watsonx enables machines to better understand and process human language, driving innovation in AI-powered applications.

---

## Introduction to Vector Databases for Storing Embeddings

After loading, splitting, and embedding data from various sources, the next crucial step is storing the embeddings. This is achieved using a vector store specifically designed to store embeddings.

### What is a Vector Database?

A vector database does more than just store data. It also retrieves required information based on queries using similarity search.

**How it works:**
1. The query is first converted into embeddings
2. Input into the vector database
3. Database performs similarity calculations to search for and retrieve the most relevant content that matches the query

### Why Vector Databases?

- Embeddings convert unstructured data (like text) into numerical vector formats within a high-dimensional space
- Traditional databases (like SQL) are not optimized for storing and querying extensive vector data
- Vector stores can index and quickly search for similar vectors using sophisticated similarity algorithms
- This enables applications to find related vectors based on a target vector query

### Supported Similarity Metrics
- Euclidean distance
- Cosine similarity
- Manhattan distance

---

## Chroma DB - A Vector Database Supported by LangChain

Chroma DB is an open-source vector store for storing and retrieving vector embeddings. Its primary use is to save embeddings and metadata for later use by large language models. It is also a powerful tool for semantic search engines over text data.

### Implementing Chroma DB

**Prerequisites:**
- Load and split target data into chunks
- Have an embedding model object ready (e.g., using watsonx)

**Creating the Vector Database:**

```python
from langchain.vectorstores import Chroma

# Create vector database using chunks and embedding model
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)
```

Chroma DB handles the rest automatically, making the process seamless and efficient.

---

## Similarity Search in Vector Database

### How Similarity Search Works

1. **Query Input**: Start with a query (any question text you're interested in)
2. **Embedding**: The embedding model converts the query into a numerical vector format (high-dimensional)
3. **Comparison**: The vector database compares the query vector to all stored vectors
4. **Retrieval**: Find the most similar vectors to the query

**Performing Similarity Search:**

```python
# Perform similarity search
query = "What is the email policy?"
results = vector_db.similarity_search(query)

# Returns top 4 most similar content by default
for doc in results:
    print(doc.page_content)
```

### Key Points
- The system performs similarity calculations based on various metrics
- Retrieves the most relevant content that matches the query
- Enables efficient and effective information retrieval

---

## Summary

| Concept | Description |
|---------|-------------|
| **Vector Store** | Database specifically designed to store and retrieve embeddings |
| **Chroma DB** | Open-source vector store supported by LangChain for saving embeddings and metadata |
| **Similarity Search** | Process of comparing query vector to stored vectors to find most similar matches |
| **Similarity Metrics** | Euclidean distance, Cosine similarity, Manhattan distance |

**Process Flow:**
1. Load documents → 2. Split into chunks → 3. Create embeddings → 4. Store in vector database → 5. Perform similarity search to retrieve relevant content

---

## Advanced Retrievers in LangChain

### What is a LangChain Retriever?

A LangChain retriever is an interface that returns documents based on an unstructured query. It is more general than a vector store - it doesn't necessarily store documents as its purpose is to retrieve them or their chunks.

**Input**: A string query
**Output**: A list of documents or chunks

---

## Vector Store-Based Retriever

The simplest type of retriever that retrieves documents from a vector database.

**How it works:**
1. Vector database is created by loading source documents, splitting them into chunks, and embedding them
2. Retriever plugs into this existing vector store
3. Accepts a query and retrieves the most similar data (chunks)
4. Uses similarity search or Maximum Marginal Relevance (MMR)

### Similarity Search
- Retriever accepts a query and retrieves the most similar data
- Embeds the query and compares with embedded chunks

### Maximum Marginal Relevance (MMR)
- Balances relevance and diversity of retrieved results
- Selects documents that are both highly relevant to the query AND minimally similar to previously selected documents
- Helps avoid redundancy and ensures comprehensive coverage of different aspects of the query

**Creating a Vector Store Retriever:**
```python
retriever = vector_db.as_retriever()
```

---

## Multi-Query Retriever

Similar to vector-based retriever, except it uses an LLM to create different versions of the query, generating a richer set of retrieved documents.

**Purpose:** Overcome issues from:
- Different results due to subtle changes in query wording
- Embeddings not capturing semantics of data well

**How it works:**
1. LLM generates alternative versions of the original query
2. For each query version, retrieves relevant documents
3. Takes the unique union across all queries
4. Results in a larger set of potential relevant documents

**Creating a Multi-Query Retriever:**
```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

---

## Self-Query Retriever

Used when documents contain both text AND metadata. Converts the query into two components:
1. A string to look up semantically
2. A metadata filter to go along with it

**How it works:**
1. LLM determines which metadata filters to apply
2. Separates semantic search from metadata filtering
3. Can filter by year, rating, director, etc.

**Example:**
Query: "I want to watch a movie rated higher than 8.5"
- Semantic lookup: "movie"
- Metadata filter: year > 8.5

**Creating a Self-Query Retriever:**
```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains import RetrievalQA

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents="Descriptions of movies",
    metadata_field_info=[
        {"name": "year", "description": "Year movie was released", "type": "int"},
        {"name": "rating", "description": "IMDB rating of the movie", "type": "float"},
    ]
)
```

---

## Parent Document Retriever

Addresses conflicting requirements in document retrieval:
- **Problem**: Need small chunks for accurate embeddings, but long enough chunks for context
- **Solution**: Parent document retriever fetches small chunks, then returns the larger parent documents

**How it works:**
1. **Parent Splitter**: Splits text into large chunks to be retrieved
2. **Child Splitter**: Splits documents into small chunks to generate meaningful embeddings
3. During retrieval:
   - First fetches the smaller chunks
   - Looks up their parent IDs
   - Returns the larger documents in which the small chunks live

**Creating a Parent Document Retriever:**
```python
from langchain.retrievers import ParentDocumentRetriever

parent_splitter = TextSplitter(chunk_size=2000)
child_splitter = TextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

retriever.add_documents(documents)
```

---

## Summary of Retrievers

| Retriever Type | Use Case | Key Feature |
|----------------|----------|-------------|
| **Vector Store Retriever** | Simple retrieval from vector database | Uses similarity search or MMR |
| **Multi-Query Retriever** | Varying query formulations | Uses LLM to generate multiple query versions |
| **Self-Query Retriever** | Documents with metadata | Separates semantic lookup from metadata filtering |
| **Parent Document Retriever** | Need both small embeddings and full context | Returns parent documents based on child chunk matches |

Each retriever addresses different retrieval challenges and can be chosen based on specific requirements of the RAG application.
