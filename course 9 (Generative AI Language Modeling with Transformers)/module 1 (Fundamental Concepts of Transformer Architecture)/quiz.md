# Module 1 Quiz: Transformers Fundamentals

## Question 1

In a transformer model's self-attention mechanism, consider a token at position 't' is 'apple' in a sentence 'I love to eat apples every day.' You've applied a self-attention mechanism to predict the token 'apple' representation based on the context provided in the sentence. Which of the following is the most influential factor for the token 'apple' prediction in the self-attention mechanism?

- A) The presence of the word 'eat'
- B) The word 'love'
- C) Position of the token 'apple' in the sequence
- D) The word 'apple'

**Answer: A**

---

## Question 2

In the following embedding, what is the dimensionality of the embedding for the word Transformers?

Given: Transformers → [0.1, 0.3, 0.2, 0.9]

- A) 0.2
- B) 1
- C) 3
- D) 4

**Answer: D**

---

## Question 3

What purpose does the following formula serve in the context of using the attention mechanism in language translation?

w = argmax_i {hV^T}

- A) Provides the word vector for the translated word
- B) Applies attention mechanism to word embeddings
- C) Retrieves the translated word from the translated vector
- D) Finds the index of the embedding that is more similar to H

**Answer: C**

---

## Question 4

Which statement is true about the scaled dot-product attention mechanism with multiple heads?

- A) It follows the setting batch_first = True for PyTorch implementations.
- B) It is implemented using the nn.Attention module of PyTorch.
- C) There is a constraint that the input dimension must be a prime number.
- D) Each head can attend to distinct segments of the input sequence in parallel.

**Answer: D**

---

## Question 5

When using transformer-based models for text classification, creating the text pipeline is a key activity. Identify the missing step (step number 5) from the following list of steps for creating the text pipeline.

Steps for creating the text pipeline:

1. Create iterators and allocate a training set
2. Generate tokens and construct a vocabulary
3. Design a custom collate function
4. Apply padding
5. ?

- A) Add positional encoding
- B) Record cumulative losses and epoch accuracies
- C) Create a data loader
- D) Apply the transformer encoder layers

**Answer: C**

---

## Answer Summary

1. Self-attention: "eat" is most influential for predicting "apple"
2. Embedding dimensionality: 4 (vector has 4 values)
3. Formula w = argmax_i {hV^T}: Retrieves translated word from vector
4. Multi-head attention: Each head attends to different segments in parallel
5. Text pipeline missing step: Create a data loader

---

## Additional Quiz Questions

## Question 1

You are working as a software developer in an MNC and have been assigned a project on natural language processing (NLP) that involves implementing a self-attention mechanism. What is the primary purpose of the self-attention mechanism that you will explain when you kick off a meeting with your new team members?

- A) Encode contextual information from surrounding words
- B) Perform part-of-speech tagging on the individual words
- C) Generate alternative text based on the input sequence
- D) Remove irrelevant words from the input sentence

**Answer: A**

---

## Question 2

You are analyzing positional encoding across embedding dimensions in a transformer model and have identified a parameter that influences the values that are computed for each dimension. What is the main role of this parameter?

- A) Determines the frequency of sine and cosine waves
- B) Indicates the phase offset for sinusoidal values
- C) Tracks where each word appears in the sentence
- D) Counts the total number of input tokens

**Answer: A**

---

## Question 3

Anika is teaching a class on how AI translates words using an attention mechanism. For the word "chat," the system should return "cat" by using the matrix structures provided. What does the system do to ensure that it picks "cat" as the translation for "chat"?

- A) By using the key vector to directly replace the query vector with the dot product of the words
- B) By randomly selecting a value vector from the value matrix and matching it with the key metrics
- C) By multiplying the value and key matrices without involving the query vector in translation
- D) By matching the query vector with the transposed key matrix and separating the matching value

**Answer: D**

---

## Question 4

You want to implement a text summarization model using PyTorch. To allow the model to focus on different parts of a sentence simultaneously, you decide to use multi-head attention. What best describes the role of multi-head attention in this model?

- A) Mask future tokens in both the encoder and decoder by default
- B) Apply multiple scaled dot-product attention operations in parallel on different segments
- C) Apply a single attention mechanism using query, keys, and values
- D) Multiply all vectors without dividing them for parallel processing

**Answer: B**

---

## Question 5

You're building a text classification model using transformers and have just instantiated the embedding layer. What is the next step before applying the encoder layers?

- A) Generate tokens
- B) Add positional encoding
- C) Construct a vocabulary
- D) Record cumulative losses

**Answer: B**

---

## Answer Summary 2

1. Self-attention encodes contextual information from surrounding words
2. Parameter determines frequency of sine and cosine waves
3. Matching query with transposed key matrix retrieves matching value
4. Multi-head attention applies multiple attention operations in parallel
5. After embedding, add positional encoding before encoder layers