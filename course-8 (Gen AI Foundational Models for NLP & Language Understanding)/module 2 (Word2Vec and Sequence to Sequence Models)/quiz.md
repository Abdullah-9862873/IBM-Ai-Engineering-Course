# Module 2 Quiz: Word2Vec and Sequence to Sequence Models

## Question 1

In word2vec models, the number of neurons in input and output layers corresponds to which of the following?

- A) Context words
- B) Numerical representations
- C) Vocabulary size
- D) Word vector dimensions

**Answer: C**

---

## Question 2

While creating the skip-gram model in PyTorch, what does the following code do?

```python
self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
```

- A) It passes the input text through the embedding layer.
- B) It performs the forward pass.
- C) It defines the fully connected layer.
- D) It defines the embeddings layer.

**Answer: D**

---

## Question 3

Which of the following statements is correct?

- A) Sequence-to-label tasks generate a part of a sequence from a descriptive input, as seen in generative models for image creation.
- B) Label-to-label tasks take a single input to produce multiple labels, which is useful in document classification.
- C) Sequence-to-sequence models help with code generation, where you can describe your task, and the AI generates the appropriate code.
- D) Neural networks can only operate under the assumption that each sample is dependent on others and distinctly distributed.

**Answer: C**

---

## Question 4

Consider the following code, which is used to reshape the output tensor while training a sequence-to-sequence model.

```python
output_dim = output.shape[-1]
output = output[1:].view(-1, output_dim)
trg = trg[1:].contiguous().view(-1)
```

For sequence models like RNNs, where the input shape often differs from other model types, why is it crucial to reshape the output tensor as well?

- A) To align the rows and columns correctly for loss calculation
- B) To calculate the average loss per batch
- C) To initialize the model in training mode to activate essential layers
- D) To generate predictions by output

**Answer: A**

---

## Question 5

"For the encoder, the interest lies only in the hidden state." What is the reason for this?

- A) Because the encoder generates the output text.
- B) Because the encoder seamlessly integrates with the PyTorch training pipeline.
- C) Because the encoder is only responsible for encoding the input sequence.
- D) Because the encoder autoregressively generates the translation as one token at a time.

**Answer: C**

---

## Question 6

Consider the following reference and hypothesis:

Reference: The dog runs on the ground

Hypothesis: The big dog runs in the park

Find the count of matching n-grams by comparing a hypothesis sequence with a reference sequence.

- A) Unigrams: 4; Bigrams: 2
- B) Unigrams: 5; Bigrams: 2
- C) Unigrams: 4; Bigrams: 1
- D) Unigrams: 3; Bigrams: 1

**Answer: C**

---

## Answer Summary

1. Word2Vec input/output neurons → Vocabulary size
2. nn.Embedding → Defines embeddings layer
3. Seq2Seq → Code generation example
4. Reshape output → Align rows/columns for loss calculation
5. Encoder hidden state → Encoder encodes input sequence only
6. Matching n-grams: Unigrams=4, Bigrams=1 (The, dog, runs, on match; "the dog" matches)

---

## Additional Quiz Questions

## Question 1

In the sentence "she loves watching football", using the Word2Vec CBOW model with a window size of 1, what are the context words and the target word at position t = 2?

- A) Context: loves, football; Target word: watching
- B) Context: she, watching; Target word: loves
- C) Context: she, football; Target word: loves, watching
- D) Context: loves, watching; Target word: she, football

**Answer: B**

---

## Question 2

Anika wants to train a skip-gram model on fitness-related sentences. She selected the word 'morning' in the sentence 'she exercises every morning'. What is the model's main task?

- A) Use the word 'morning' based on all other words in the sentence.
- B) Use the word 'morning' to predict the surrounding words 'every' and 'exercises'.
- C) Predict the words 'she' and 'exercises' from 'every morning'.
- D) Use the word 'she' to predict 'morning' and 'exercises'.

**Answer: B**

---

## Question 3

Which generative AI model matches the description below?

"It is a type of simulated neural network that uses time series data. It is designed to remember past information."

- A) Feedforward neural network
- B) Word2vec's neural network
- C) Generative adversarial networks (GANs)
- D) Recurrent neural networks (RNNs)

**Answer: D**

---

## Question 4

Which statement best describes using the beginning of sequence (BOS) token in the decoder during training in a sequence-to-sequence model?

- A) Helps the encoder to terminate the sequence early.
- B) Serves as a replacement for unknown words during tokenization.
- C) Signals the decoder to generate the output sequence from the beginning.
- D) Informs the decoder to generate complete output at once.

**Answer: C**

---

## Question 5

In an encoder-decoder RNN architecture, how is the output sequence typically generated during translation?

- A) The encoder predicts the complete output
- B) The model merges input and output vectors
- C) The decoder repeats the same tokens
- D) The decoder modules with RNN cells

**Answer: D**

---

## Question 6

How is perplexity typically computed for evaluating a language model?

- A) It divides the vocabulary size by the number of predicted tokens.
- B) It applies the exponential function to the average cross-entropy loss.
- C) It multiplies all predicted probabilities for each sequence.
- D) It averages the squared error of predicted and actual words.

**Answer: B**

---

## Answer Summary 2

1. CBOW window=1: Context=[she, watching], Target=loves (t=2 = "loves")
2. Skip-gram: Target="morning" predicts context words "every" and "exercises"
3. RNN: Uses time series data, remembers past information
4. BOS token: Signals decoder to start generating from beginning
5. Decoder with RNN cells generates output sequence step by step
6. Perplexity = exp(average cross-entropy loss)