# Quiz: RAG Framework

## Question 1
What is the primary function of RAG in AI systems?

- Only store large amounts of data without performing actions.
- **Optimize LLM output by combining external knowledge with language generation.** ✓
- Replace human intelligence completely in all tasks.
- Randomly perform actions without any input or goals.

---

## Question 2
Which component of RAG is responsible for encoding documents and queries into vectors?

- Generator
- **Retriever** ✓
- Database
- Chatbot

---

## Question 3
What is the purpose of text chunking in the RAG process?

- To increase the size of the knowledge base
- **To break large documents into smaller, manageable pieces for targeted retrieval** ✓
- To train the language model
- To speed up the generation process

---

## Question 4
Which distance metric focuses on the direction (angular difference) between vectors?

- Dot product
- **Cosine distance** ✓
- Euclidean distance only
- Manhattan distance

---

## Question 5
What is Faiss primarily used for in RAG?

- Storing raw text documents
- **Searching through large collections of high-dimensional vectors** ✓
- Training language models
- Generating responses

---

## Question 6
Which encoder is used to encode potential answer passages or documents in RAG?

- Question encoder
- **Context encoder (DPR)** ✓
- Tokenizer
- Generator

---

## Question 7
What does the DPR question encoder do?

- Encodes documents into vectors for storage
- **Encodes input questions into fixed-dimensional vector representations** ✓
- Generates final responses
- Stores embeddings in database

---

## Question 8
In the RAG process, what happens in the augmented query creation step?

- The model generates a response directly
- **The system combines retrieved context with the original prompt** ✓
- The knowledge base is updated
- The embeddings are stored

---

## Question 9
What is the output shape of DPR context embeddings typically?

- (batch, 256)
- **(batch, 768)** ✓
- (batch, 1024)
- (batch, 512)

---

## Question 10
How does RAG help with specialized domain knowledge that LLMs aren't trained on?

- By retraining the LLM
- **By retrieving relevant external information and using it to generate responses** ✓
- By using a smaller model
- By limiting the questions users can ask

---

# Answers

1. **Optimize LLM output by combining external knowledge with language generation.** - RAG combines retrieval with generation.

2. **Retriever** - The retriever encodes documents and queries, finds relevant content.

3. **To break large documents into smaller, manageable pieces for targeted retrieval** - Chunking enables efficient retrieval.

4. **Cosine distance** - Focuses on direction/angle between vectors.

5. **Searching through large collections of high-dimensional vectors** - Faiss provides efficient similarity search.

6. **Context encoder (DPR)** - Encodes potential answer passages/documents.

7. **Encodes input questions into fixed-dimensional vector representations** - Question encoder creates query embeddings.

8. **The system combines retrieved context with the original prompt** - Creates augmented query for generator.

9. **(batch, 768)** - DPR produces 768-dimensional embeddings.

10. **By retrieving relevant external information and using it to generate responses** - RAG provides accurate domain-specific answers.

---

## Additional Questions

## Question 11
Facebook AI Similarity Search, also known as Faiss, is essentially a tool to:

- Design training and inference for machine learning models
- Address high-level data retrieval and operations like SQL querying
- **Calculate the distance between the question embedding and the vector database of context vector embeddings** ✓
- Take care of tasks related to data management and preprocessing

---

## Question 12
What is the primary function of a contextual tokenizer with respect to tokenizers in natural language processing (NLP)?

- **To break text into numeric representations or contextual embeddings** ✓
- To summarize long paragraphs into concise sentences
- To translate text from one language to another
- To split text into individual characters

---

## Question 13
How do the inserted prompts typically get encoded before being used to retrieve relevant documents using the retrieval-augmented generation (RAG) process?

- **You can apply token embedding and vector averaging to convert the prompts into a vector representation.** ✓
- You can use bag-of-words to create a frequency matrix of the words in the prompts.
- You can use one-hot encoding to represent each word as a unique vector.
- You can convert the entire prompt into a single unique token.

---

# Answers (Additional)

11. **Calculate the distance between the question embedding and the vector database of context vector embeddings** - Faiss calculates similarity between question and context embeddings.

12. **To break text into numeric representations or contextual embeddings** - Tokenizers convert text into numerical format for model processing.

13. **You can apply token embedding and vector averaging to convert the prompts into a vector representation.** - RAG uses token embedding and averaging to create prompt vectors.

---

## Process Questions

## Question 14
Which of the following best describes the correct order of the retrieval-augmented generation (RAG) process for generating accurate responses?

- **Encoding → Retrieval → Generation** ✓
- Retrieval → Generation → Encoding
- Retrieval → Encoding → Generation
- Generation → Retrieval → Encoding

---

## Question 15
Which of the following is the typical pipeline for converting a knowledge document into a vectorized form in retrieval-augmented generation (RAG)?

- Search → Tokenize → Embed → Generate
- Tokenize → Translate → Index → Generate
- Generate → Tokenize → Embed → Translate
- **Chunk → Embed → Index → Retrieve → Generate** ✓

---

## Question 16
Rommy has developed an AI assistant that answers employee queries using recent HR documents. What task does the retrieval step perform in this scenario?

- **Identifies the most relevant documents in the knowledge base to match the user's question.** ✓
- Saves and summarizes all input for future queries for the relevant document vectors to provide the generator with useful context.
- Translates user queries to match policy documents with relevant vectors to generate useful context.
- Generates a standard response from previously stored outputs using relevant document vectors.

---

## Question 17
If you want to add a query to a chatbot powered by RAG, which part of the system first processes this input for retrieval?

- Context encoder
- Generator model
- **Question encoder** ✓
- FAISS indexer

---

## Question 18
How does RAG support the development of a chatbot that answers questions based on private medical guidelines in a healthcare organization?

- Train the model on the open medical data.
- Summarize internet results for medical advice.
- Replace the chatbot with the manual's frequently asked questions (FAQs).
- **Retrieves internal guidelines to supplement the chatbot's replies.** ✓

---

# Answers (Process)

14. **Encoding → Retrieval → Generation** - Correct RAG pipeline order.

15. **Chunk → Embed → Index → Retrieve → Generate** - Typical document to vector pipeline.

16. **Identifies the most relevant documents in the knowledge base to match the user's question.** - Retrieval finds relevant documents.

17. **Question encoder** - First processes query input for retrieval.

18. **Retrieves internal guidelines to supplement the chatbot's replies.** - RAG provides domain-specific answers.