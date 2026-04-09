# Module 1 Quiz: Fundamentals of Language Understanding

## Question 1

Which statement is true regarding one-hot encoding?

- A) It sums up all the vectors from bag-of-words and multiplies the result with the embedding matrix.
- B) It combines embedding weights to form a matrix.
- C) It is a representation of an entire document as the aggregate or average of vectors.
- D) It converts categorical data into feature vectors.

**Answer: D**

---

## Question 2

To determine the class of an article, logits are input from the output layer into the argmax function. What does the Argmax function do?

- A) It corresponds to the vocabulary size.
- B) It processes the text through the pipeline.
- C) It decides whether a neuron should be activated or not.
- D) It identifies the index of the highest logit value.

**Answer: D**

---

## Question 3

Which of the following expressions represents the gradient descent equation?

- A) Σ_y P(y) ⋅ (ln(P(y)) − ln(P(y|x,θ)))
- B) θ_(k+1) = θ_k − η ∇_θ L(θ_k)
- C) P(y=i|x) = e(z^[2])_i / Σ_{j=1}^K e(z^[2])_j
- D) μ_f = (1/N) Σ_{n=1}^N f(z_n)

**Answer: B**

---

## Question 4

What happens during neural network training?

- A) Learnable parameters are decoded to generate text.
- B) Learnable parameters are encoded to generate text.
- C) Learnable parameters are fine-tuned to enhance model performance.
- D) Learnable parameters are optimized for text-to-text transfer.

**Answer: C**

---

## Question 5

Match the following and select the correct option.

| | |
|---|---|
| 1. Training data | a. Evaluating real-world performance |
| 2. Validation data | b. Hyperparameter tuning |
| 3. Test data | c. Learning |

- A) 1, a; 2, b; 3, c
- B) 1, a; 2, c; 3, b
- C) 1, b; 2, c; 3, a
- D) 1, c; 2, b; 3, a

**Answer: D**

---

## Answer Summary

1. One-hot encoding → Converts categorical data into feature vectors
2. Argmax → Identifies index of highest logit value
3. Gradient descent → θ_(k+1) = θ_k − η ∇_θ L(θ_k)
4. Neural network training → Fine-tune learnable parameters
5. Data split → Training: Learning, Validation: Hyperparameter tuning, Test: Real-world performance

---

## Additional Quiz Questions

## Question 1

Using the following data, calculate the input dimension (context vector) of the neural network in a given n-gram-based language model:

Vocabulary = {I, hate, like, surgeons, surgery, football, vacations, broccoli}

Context size = 2

- A) 16
- B) 14
- C) 4
- D) 10

**Answer: A**

---

## Question 2

Consider the phrase, "I like watching movies and sports on…" Using the tri-gram model, what will be the context and predicted word(s) at 't=5'?

- A) Context: ["I", "like"]; Predicted word: "watching"
- B) Context: ["like", "watching"]; Predicted word: "and"
- C) Context: ["watching", "movies"]; Predicted word: "and"
- D) Context: ["watching", "movies"]; Predicted word: "on"

**Answer: C**

---

## Question 3

Which of the following set of codes converts the list of token indices into a PyTorch tensor?

- A) x_c = torch.tensor(context)
- B) out = model(x_c)
- C) predicted_index = torch.argmax(out, 1)
- D) index_to_token[predicted_index]
- E) context = text_pipeline("Never gonna")

**Answer: A**

---

## Answer Summary 2

1. Input dimension = vocabulary_size × context_size = 8 × 2 = 16
2. Tri-gram at t=5: Context = ["watching", "movies"], Predicted = "and"
3. Convert to tensor: `x_c = torch.tensor(context)`

---

## Additional Quiz Questions (Set 3)

## Question 1

Which of the following best defines how a bag-of-words model encodes a document?

- A) Aggregates one-hot encoded vectors of all tokens.
- B) Increases the vocabulary size by token length.
- C) Assigns fixed embeddings to unknown tokens
- D) Uses a unique vector for pronunciation.

**Answer: A**

---

## Question 2

How does the argmax function help during document classification using a neural network?

- A) Determines the number of hidden layers
- B) Converts logits into a probability distribution
- C) Selects the index of the output neuron with the highest logit
- D) Transforms raw text into tokenized output

**Answer: C**

---

## Question 3

While training a document classification model in PyTorch, you want to ensure the model's parameters are adjusted effectively during training. Which step directly contributes to reducing the model's loss by changing its parameters?

- A) Convert labels into one-hot encoded format
- B) Apply backpropagation using the computed loss
- C) Pass inputs through embedding layers
- D) Initialize cross-entropy as the loss function

**Answer: B**

---

## Question 4

A developer wants to set up a data loader but forgets to shuffle the training dataset while defining the loop. Which of the following will be the potential issue arising in this scenario?

- A) The model will overfit the training data due to early stopping.
- B) The gradient descent may converge to a suboptimal local minimum.
- C) The batch size will dynamically increase over time.
- D) The model may skip validation in the training dataset.

**Answer: B**

---

## Question 5

Seema wants to implement a feedforward neural network for an N-gram model with a vocabulary size of five and a context size of three. Your colleague recommends computing a single one-hot vector for the context. Which would properly define the context vector in this neural network setup?

- A) Replace vocabulary with embeddings of output words
- B) Sum the one-hot vectors of the context words into one
- C) Focus on the embedding vectors of each context word
- D) Multiply each word by a context-based attention weight

**Answer: C**

---

## Question 6

You're training a neural n-gram model. Which performance metric should you monitor during each epoch to evaluate model learning?

- A) Context
- B) Accuracy
- C) Prediction
- D) Loss

**Answer: D**

---

## Answer Summary 3

1. Bag-of-words → Aggregates one-hot encoded vectors of all tokens
2. Argmax → Selects index of output neuron with highest logit
3. Backpropagation → Reduces loss by changing parameters
4. No shuffle → Gradient descent may converge to suboptimal local minimum
5. Context vector → Focus on embedding vectors of each context word
6. Metric to monitor → Loss