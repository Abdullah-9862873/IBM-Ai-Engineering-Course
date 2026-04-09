# Module 2 Quiz: Data Preparation for LLMs

## Question 1

Which statement is true about the Unigram algorithm for tokenization?

- A) It evaluates the benefits and drawbacks of splitting and merging two symbols to ensure its decisions are valuable.
- B) It segments text into manageable parts and assigns unique IDs.
- C) It begins with a large list of possibilities and gradually narrows down based on how frequently they appear in the text.
- D) It involves splitting text into individual characters.

**Answer: C**

---

## Question 2

Identify the advantages of using data loaders in natural language processing (NLP). Select all that apply.

- A) Splits text into characters to ensure vocabulary is small
- B) Enables batching of data
- C) Seamlessly integrates with the PyTorch training pipeline
- D) Enables shuffling of data

**Answers: B, C, D**

---

## Question 3

Fill in the blank.

You can use the _____ to ensure that all sentences have the same length after tokenization, matching the length of the longest sentence among the input sentences.

- A) `<pad>` token
- B) `<eos>` special token
- C) `##` symbol
- D) Underscore symbol

**Answer: A**

---

## Question 4

You are designing a text classification model for a language with complex morphology. You want to use a tokenization method that minimizes the size of the vocabulary but increases the computational requirements due to higher input dimensionality. Which tokenization method will you choose?

- A) Subword-based tokenization
- B) Word-based tokenization
- C) WordPiece tokenization
- D) Character-based tokenization

**Answer: D**

---

## Question 5

You want to loop batches of a large natural language processing (NLP) dataset using a PyTorch data loader for training. Which of the following concepts will allow you to access one batch at a time?

- A) Iteration
- B) Batching
- C) Padding
- D) Shuffling

**Answer: A**

---

## Question 6

You are implementing subword-based tokenization for your NLP model and need to indicate that a word should be attached to the previous word without adding a space between them. Which symbol is used for this purpose?

- A) `##` symbol
- B) Underscore symbol
- C) `<eos>` special token
- D) `<pad>` token

**Answer: A**

---

## Question 7

Sonia is developing a large language model (LLM). She noticed that the model's performance has decreased due to inconsistent text formatting, repeated forum tags, and user typos. What should they prioritize?

- A) Increase batch size while training the model.
- B) Add more data from the same source.
- C) Clean data to remove inconsistencies and noise.
- D) Apply token-level augmentation in the model.

**Answer: C**

---

## Question 8

You are configuring a data loader for training your deep learning model and want to prevent it from learning patterns based on sequential order in your dataset during training. Which input parameter will you use?

- A) The padding value
- B) The shuffle argument
- C) The dataset
- D) The batch size

**Answer: B**