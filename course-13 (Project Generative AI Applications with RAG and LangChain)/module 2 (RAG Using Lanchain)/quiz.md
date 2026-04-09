# Quiz: Embeddings and Vector Database

## Question 1
When splitting a document into small, semantically meaningful chunks to fit into the LLM's context window, a text splitter ensures that:

- Each chunk is complete.
- The source formatting is preserved.
- **It creates a new chunk with some overlap from the previous one.**
- It splits below the character level.

## Question 2
Which of the following text splitting methods aims to keep text with a common context together and honor the document's structure?

- Split Code
- Recursively Split by Character
- **Markdown Header Text Splitter**
- Split by Character

## Question 3
Which of the following text splitting methods aims to keep all paragraphs (then sentences, then words) together as much as possible when trying to split text?

- Split Code
- Split by Character
- Markdown Header Text Splitter
- **Recursively Split by Character**

---

## Question 4
When inputting a document into the embedding model, you cannot input the entire document because:

- The embedding model cannot be used for downstream tasks.
- **The embedding model has a maximum input token limit.**
- The embedding model does not capture the semantic meaning of the text.
- The embedding model cannot capture the context of the document.

## Question 5
Which of the following options best describes the primary advantage of vector databases over traditional databases like SQL?

- Vector databases can store and manage data in a structured format.
- Vector databases can be scaled to handle growing data.
- **Vector databases can index and quickly search for similar vectors using sophisticated similarity algorithms.**
- Vector databases have the capacity to maintain data integrity and consistency.

## Question 6
A similarity search in a vector database involves:

- Complex relational queries that include joins and transactions.
- Finding keywords that are an exact match to the words in the given query.
- Using manual feature engineering to identify items similar to the query.
- **Finding items that are most similar to a given query item based on their vector representations.**