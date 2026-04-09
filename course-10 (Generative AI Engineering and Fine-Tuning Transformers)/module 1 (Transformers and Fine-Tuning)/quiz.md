# Module 1 Quiz: Transformers and Fine-Tuning

## Course Overview Quiz

## Question 1

What is the primary focus of this course on fine-tuning with transformers?

- A) Decoder models exclusively
- B) Encoder models for simplicity (applicable to decoder models too)
- C) Only text classification tasks
- D) Image generation models

**Answer: B**

---

## Question 2

Which frameworks are covered in this course for working with transformers?

- A) TensorFlow only
- B) HuggingFace and PyTorch
- C) Keras and TensorFlow
- D) Scikit-learn and NumPy

**Answer: B**

---

## Question 3

What does PEFT stand for in the context of fine-tuning?

- A) Pre-trained Encoder Fine-Tuning
- B) Parameter Efficient Fine-Tuning
- C) Probabilistic Embedding Fine-Tuning
- D) Progressive Evaluation Fine-Tuning

**Answer: B**

---

## Question 4

Which technique combines quantization with LoRA for more efficient fine-tuning?

- A) QLoRA
- B)GPTQ
- C) AWQ
- D) PTQ

**Answer: A**

---

## Answer Summary

1. Focus on encoder models (applicable to decoders too)
2. HuggingFace and PyTorch frameworks
3. PEFT = Parameter Efficient Fine-Tuning
4. QLoRA combines quantization with LoRA

---

## Hugging Face vs PyTorch Quiz

## Question 1

Hugging Face is often called the __________ of machine learning.

- A) Google
- B) GitHub
- C) SourceForge
- D) GitLab

**Answer: B**

---

## Question 2

Which feature of PyTorch allows changes to network architecture during runtime?

- A) Static computation graph
- B) Dynamic computation graph
- C) Lazy evaluation
- D) Automatic differentiation

**Answer: B**

---

## Question 3

What is the most popular feature of Hugging Face?

- A) Datasets library
- B) Transformers library
- C) Tokenizers library
- D) Evaluate library

**Answer: B**

---

## Question 4

PyTorch was originally developed by which organization?

- A) Google
- B) Microsoft
- C) Facebook AI Research (Meta)
- D) Amazon

**Answer: C**

---

## Question 5

Which of the following is NOT an application of Hugging Face and PyTorch integration?

- A) Sentiment analysis
- B) Image generation
- C) Language translation
- D) Text summarization

**Answer: B**

---

## Question 6

What type of models does the Hugging Face transformers library provide?

- A) Only computer vision models
- B) Pre-trained models like BERT, GPT, and T5
- C) Only reinforcement learning models
- D) Only audio processing models

**Answer: B**

---

## Question 7

Which PyTorch feature makes it excellent for rapid prototyping?

- A) Static computation graphs
- B) Dynamic computation graphs
- C) Pre-defined model architectures
- D) Limited GPU support

**Answer: B**

---

## Question 8

What programming language is PyTorch built on?

- A) C++
- B) Java
- C) Python
- D) Ruby

**Answer: C**

---

## Question 9

Which company/institution developed the original PyTorch framework?

- A) Google Brain
- B) Microsoft Research
- C) Facebook AI Research (Meta)
- D) OpenAI

**Answer: C**

---

## Question 10

How can you integrate PyTorch with Hugging Face for NLP tasks?

- A) Only through API calls
- B) Using Hugging Face as transformers with PyTorch backend
- C) Cannot be integrated
- D) Only through TensorFlow

**Answer: B**

---

## Answer Summary

1. Hugging Face = "GitHub of Machine Learning"
2. Dynamic computation graph allows runtime changes
3. Transformers library is most popular feature of Hugging Face
4. Developed by Facebook AI Research (Meta)
5. Image generation is not an NLP application
6. Provides pre-trained models (BERT, GPT, T5)
7. Dynamic computation graphs enable rapid prototyping
8. Built on Python
9. Developed by Facebook AI Research (Meta)
10. Using Hugging Face transformers with PyTorch backend

---

## Additional Quiz Questions: Fine-Tuning

## Question 1

Why is fine-tuning necessary instead of training from scratch?

- A) Fine-tuning is always more accurate
- B) Training LLMs from scratch is computationally expensive
- C) Pre-trained models are always perfect
- D) Fine-tuning requires less data

**Answer: B**

---

## Question 2

What is the process of adapting a pre-trained model for specific tasks called?

- A) Pre-training
- B) Fine-tuning
- C) Transfer learning
- D) Both B and C

**Answer: D**

---

## Question 3

Which of the following is a pitfall of fine-tuning?

- A) Overfitting
- B) Underfitting
- C) Catastrophic forgetting
- D) All of the above

**Answer: D**

---

## Question 4

What is self-supervised fine-tuning?

- A) Learning with labeled data
- B) Learning by predicting missing words
- C) Learning with human feedback
- D) Learning with reinforcement learning

**Answer: B**

---

## Question 5

Which fine-tuning approach uses explicit human feedback?

- A) Self-supervised
- B) Supervised
- C) RLHF
- D) DPO

**Answer: C**

---

## Question 6

What does DPO stand for?

- A) Direct Preference Optimization
- B) Distributed Parameter Optimization
- C) Deep Pre-training Optimization
- D) Dynamic Parameter Opposition

**Answer: A**

---

## Question 7

What is the main advantage of full fine-tuning?

- A) Faster training
- B) Higher accuracy
- C) Less memory
- D) Simpler implementation

**Answer: B**

---

## Question 8

When fine-tuning only the final layer, what happens to other layers?

- A) They are deleted
- B) They are frozen
- C) They are retrained
- D) They are compressed

**Answer: B**

---

## Question 9

What is the accuracy when fine-tuning only the final layer compared to full fine-tuning?

- A) Better
- B) Worse
- C) Same
- D) Cannot be measured

**Answer: B**

---

## Question 10

What is the accuracy when fully fine-tuning the model on IMDB dataset?

- A) ~50%
- B) ~70%
- C) ~90%
- D) ~100%

**Answer: C**

---

## Answer Summary

1. Fine-tuning needed because training from scratch is computationally expensive
2. Fine-tuning and transfer learning are related concepts
3. Pitfalls include overfitting, underfitting, catastrophic forgetting
4. Self-supervised: predicts missing words
5. RLHF: uses human feedback
6. DPO: Direct Preference Optimization
7. Full fine-tuning: higher accuracy
8. Final layer only: other layers frozen
9. Final layer only: worse performance
10. Full fine-tuning: ~90% accuracy

---

## Additional Quiz Questions: Hugging Face Fine-Tuning

## Question 1

How can you load a built-in dataset in Hugging Face?

- A) load_model function
- B) load_dataset function
- C) load_data function
- D) load_tokenizer function

**Answer: B**

---

## Question 2

What does the tokenizer function do in Hugging Face?

- A) Removes text from dataset
- B) Converts text to token indices with attention masks
- C) Creates new datasets
- D) Trains the model

**Answer: B**

---

## Question 3

What parameter specifies the number of neurons in the final classification layer?

- A) num_layers
- B) num_heads
- C) num_labels
- D) num_params

**Answer: C**

---

## Question 4

What is the purpose of SFT Trainer?

- A) To load datasets faster
- B) To simplify and automate training tasks
- C) To tokenize data
- D) To create models

**Answer: B**

---

## Question 5

What is the tokenization parameter that handles variable length sequences?

- A) padding only
- B) truncation only
- C) Both padding and truncation
- D) Neither

**Answer: C**

---

## Question 6

Which optimizer is commonly used with Hugging Face for fine-tuning?

- A) SGD
- B) RMSprop
- C) AdamW
- D) Adagrad

**Answer: C**

---

## Question 7

What is the output from the fill-mask pipeline?

- A) Single token prediction
- B) List of predictions with token and score
- C) Only the score
- D) Only the token

**Answer: B**

---

## Question 8

When using the fill-mask pipeline, what token is used as placeholder?

- A) [CLS]
- B) [SEP]
- C) [MASK]
- D) [UNK]

**Answer: C**

---

## Question 9

For the Yelp dataset, what does the label represent?

- A) Review text
- B) Star rating (1-5)
- C) User ID
- D) Date

**Answer: B**

---

## Question 10

What format can Hugging Face datasets be converted to?

- A) Only NumPy
- B) PyTorch tensors
- C) Only Pandas
- D) Only lists

**Answer: B**

---

## Answer Summary

1. Use load_dataset function to load datasets
2. Tokenizer converts text to token indices with attention masks
3. num_labels specifies neurons in final layer
4. SFT Trainer automates training tasks
5. Both padding and truncation handle variable lengths
6. AdamW optimizer common for fine-tuning
7. Fill-mask returns list of predictions with token and score
8. [MASK] token is placeholder
9. Label = star rating (1-5)
10. Can convert to PyTorch tensors

---

## Additional Quiz Questions: Model Training

## Question 1

Which of the following is the most popular feature of Hugging Face?

- A) Tokenizers library
- B) Transformers library
- C) Datasets library
- D) Building neural networks

**Answer: B**

---

## Question 2

Which of the following fine-tuning approaches do you apply if the model wants to learn to predict missing words in a large, unlabeled dataset, such as next words or masked words?

- A) Direct preference optimization (DPO)
- B) Self-supervised fine-tuning
- C) Supervised fine-tuning
- D) Reinforcement learning from human feedback (RLHF)

**Answer: B**

---

## Question 3

Consider the following code snippet:

Select the correct statement regarding the given code snippet.

- A) This code snippet takes in a text and a text pipeline, which preprocesses the text for machine learning.
- B) This code snippet trains a transformer model using the provided optimizer and loss criterion.
- C) This code snippet converts the dataset into map-style datasets and performs a random split.
- D) This code snippet indicates the constructor that initializes the text classifier with configurations such as the number of classes, vocabulary size, and transformer settings.

**Answer: D**

---

## Question 4

Which of the following statements is correct regarding an SFT Trainer?

- A) It evaluates the model's performance.
- B) It extracts the text from the dataset.
- C) It determines the number of neurons in the final layer.
- D) It simplifies and automates training tasks.

**Answer: D**

---

## Answer Summary

1. Transformers library is the most popular feature of Hugging Face
2. Self-supervised fine-tuning predicts missing words in unlabeled data
3. Code snippet is the constructor that initializes the text classifier
4. SFT Trainer simplifies and automates training tasks

---

## Additional Quiz Questions: Advanced Fine-Tuning

## Question 1

You are iterating on a deep learning model and want to test sections of your code without waiting for the full implementation to finish. Which of the following reasons makes PyTorch an excellent platform for rapid prototyping and expedites debugging?

- A) It allows real-time testing of code segments without waiting for full implementation.
- B) It offers essential tools for creating a variety of machine learning models.
- C) It has an intuitive and straightforward syntax, which is Python-based.
- D) It supports diverse neural architectures with extensive libraries of pre-configured models.

**Answer: A**

---

## Question 2

A research team has access to a large pre-trained transformer but needs to adapt it for a new domain with limited labeled examples. They want to avoid excessive computational costs and prevent the model from forgetting its general language knowledge. Which fine-tuning model should a company leverage in this scenario?

- A) Parameter-efficient fine-tuning (PEFT) on domain-specific labeled data
- B) Train the model for various epochs on a very small dataset for fine-tuning
- C) Use the pre-trained model as-is without any specific training
- D) Full fine-tuning of model parameters on a large irrelevant dataset

**Answer: A**

---

## Question 3

You are fine-tuning a transformer pretrained on a multi-class text dataset for a binary classification task using PyTorch. Which practice ensures that the fine-tuned model matches the new task's requirements?

- A) Replace the last classification layer with two neurons for binary classes
- B) Eliminate the training validation split to maximize data use
- C) Freeze the entire model to retain pretrained models
- D) Use the original output of the pretrained model without modifications

**Answer: A**

---

## Question 4

While fine-tuning a transformer model using Hugging Face's transformers library, you need to adjust the final classification layer. How does the num_labels parameter affect the architecture of a pre-trained BERT model for classification?

- A) Set the learning rate to optimize during fine-tuning models
- B) Define the batch size for the data loader during training
- C) Specify the number of output classes and determine the number of neurons in the final layer
- D) Load the tokenizer needed for preprocessing the output text data

**Answer: C**

---

## Question 5

Why should Thomas leverage the supervised fine-tuning (SFT) trainer instead of writing his own training loop in PyTorch?

- A) Selects the best learning rate for the model automatically
- B) Augments datasets automatically with new examples
- C) Automates training steps, reducing complexity and errors
- D) Generates tokenizers from the dataset to train the model

**Answer: C**

---

## Answer Summary

1. PyTorch allows real-time testing without full implementation
2. PEFT avoids computational costs and prevents catastrophic forgetting
3. Replace final layer to match new binary classification task
4. num_labels specifies number of output classes/neurons
5. SFT Trainer automates training steps