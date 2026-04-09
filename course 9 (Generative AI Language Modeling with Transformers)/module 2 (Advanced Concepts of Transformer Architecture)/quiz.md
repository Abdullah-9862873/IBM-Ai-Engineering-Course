# Module 2 Quiz: Advanced Transformer Architecture

## Question 1

In transformer architectures, excluding the translation task, what is the principal distinction between encoders and decoders?

- A) Encoders generate text by predicting the previous tokens in a sequence.
- B) Encoders apply a SoftMax function before generating the output.
- C) Decoders include a multi-head attention mechanism, but encoders do not.
- D) Decoders use masked self-attention to prevent attending to future tokens, while encoders use unmasked self-attention.

**Answer: D**

---

## Question 2

Which of the following statements is true for the characteristics of a decoder model implemented using PyTorch with a causal language model (LM) architecture?

- A) The decoder models in PyTorch using causal LM initially focus on understanding the context from bidirectional dependencies within a sequence.
- B) The decoder models in PyTorch use causal LM relays on input embeddings without considering the order of tokens in the sequence.
- C) The decoder models in PyTorch using causal LM restrict attending sequential data due to its limitations in capturing temporary tokens.
- D) The decoder models in PyTorch using causal LM help to generate sequence-to-sequence tokens while attending to the previous tokens and ensuring coherence in sequence generation.

**Answer: D**

---

## Question 3

Which of the following best describes the purpose of using decoder model neural network architecture?

- A) Generate outputs from the learned representations of the input data.
- B) Preprocess the input data before feeding it to the neural network.
- C) Initialize the weights of the neural network parameters.
- D) Reduce the dimensionality of the data for faster computation.

**Answer: A**

---

## Question 4

Which of the following is one of the advantages of implementing causal attention masking in natural language processing (NLP)?

- A) Speech recognition
- B) Language translation
- C) Image classification
- D) Sentiment analysis

**Answer: B**

---

## Answer Summary

1. Decoders use masked self-attention; encoders use unmasked self-attention
2. Decoder with causal LM generates sequence-to-sequence tokens while attending to previous tokens
3. Decoder generates outputs from learned representations
4. Language translation benefits from causal attention masking

---

## Additional Quiz Questions

## Question 1

What is the primary purpose of Masked Language Modeling (MLM) in BERT pre-training?

- A) To predict the next token in a sequence autoregressively
- B) To train the model to predict masked tokens by understanding bidirectional context
- C) To generate text step-by-step
- D) To perform language translation between two languages

**Answer: B**

---

## Question 2

In the autoregressive text generation process, what happens after the decoder generates a prediction for the next token?

- A) The generation process stops immediately
- B) The predicted token is appended to the input sequence for the next prediction
- C) The model resets to the beginning
- D) The positional encodings are removed

**Answer: B**

---

## Question 3

Which pre-training task is specifically used by BERT for understanding whether one sentence logically follows another?

- A) Masked Language Modeling (MLM)
- B) Next Sentence Prediction (NSP)
- C) Causal Language Modeling (CLM)
- D) Sentence Piece Encoding (SPE)

**Answer: B**

---

## Question 4

In the Encoder-Decoder architecture for translation, what is the role of the cross-attention mechanism?

- A) Allows the decoder to attend to the encoder's output representations
- B) Prevents the encoder from attending to future tokens
- C) Masks tokens randomly during training
- D) Generates the initial input embeddings

**Answer: A**

---

## Question 5

What is the purpose of teacher forcing in training decoder models?

- A) To use the model's own predictions as input for the next step
- B) To use actual previous tokens from the sequence instead of model's predictions
- C) To compute loss only at the final position
- D) To remove positional encodings

**Answer: B**

---

## Answer Summary 2

1. MLM trains model to predict masked tokens using bidirectional context
2. Predicted token is appended to input for next prediction
3. NSP predicts if one sentence follows another logically
4. Cross-attention allows decoder to attend to encoder output
5. Teacher forcing uses actual tokens during training

---

## Additional Quiz Questions: BERT Encoder Models

## Question 1

What is the key advantage of BERT's bidirectional encoder architecture compared to autoregressive models like GPT?

- A) BERT can generate text more quickly than autoregressive models
- B) BERT can utilize context from both sides of a masked token for prediction
- C) BERT requires less computational resources
- D) BERT uses fewer parameters

**Answer: B**

---

## Question 2

In BERT's Masked Language Modeling (MLM), what percentage of tokens are typically masked during pre-training?

- A) 5%
- B) 10%
- C) 15%
- D) 25%

**Answer: C**

---

## Question 3

According to BERT's masking strategy (80/10/10 rule), when a token is selected for masking, what percentage of the time is it replaced with a random token?

- A) 5%
- B) 10%
- C) 15%
- D) 20%

**Answer: B**

---

## Question 4

What is the purpose of segment embeddings in BERT?

- A) To distinguish between first and second sentences in paired sentence tasks
- B) To encode the position of tokens in the sequence
- C) To represent the meaning of individual tokens
- D) To mask sensitive information

**Answer: A**

---

## Question 5

In the BERT model, which special token is added at the start of the sequence and used for classification tasks?

- A) [SEP]
- B) [MASK]
- C) [CLS]
- D) [UNK]

**Answer: C**

---

## Question 6

What does the Next Sentence Prediction (NSP) task train BERT to determine?

- A) The sentiment of the input text
- B) Whether the second sentence logically follows the first sentence
- C) The topic of the input text
- D) The grammatical correctness of the sentence

**Answer: B**

---

## Question 7

In PyTorch implementation of BERT, what does the BERTEmbedding class combine?

- A) Token embeddings only
- B) Token embeddings and segment embeddings
- C) Token embeddings, segment embeddings, and positional encodings
- D) Only positional encodings

**Answer: C**

---

## Question 8

When pre-training BERT, what is the combined loss function calculated from?

- A) Only NSP loss
- B) Only MLM loss
- C) Sum of NSP loss and MLM loss
- D) Sum of reconstruction loss and prediction loss

**Answer: C**

---

## Question 9

What is the purpose of the [SEP] token in BERT?

- A) To mark the beginning of a sequence
- B) To denote the end of a sentence or sequence
- C) To replace masked tokens
- D) To represent unknown tokens

**Answer: B**

---

## Question 10

In the BERT training process with PyTorch, what is the purpose of zero-padding input sequences?

- A) To reduce memory usage
- B) To maintain consistent input shapes during training
- C) To improve accuracy
- D) To increase training speed

**Answer: B**

---

## Answer Summary 3

1. Bidirectional context allows BERT to use both sides of masked token
2. 15% of tokens are masked during MLM pre-training
3. 10% of masked tokens are replaced with random tokens
4. Segment embeddings distinguish sentence pairs
5. [CLS] token is used for classification
6. NSP determines if second sentence follows first
7. BERTEmbedding combines token, segment, and positional embeddings
8. Combined loss = NSP loss + MLM loss
9. [SEP] token marks end of sentence
10. Zero-padding maintains consistent input shapes

---

## Additional Quiz Questions: BERT Architecture

## Question 1

Why does BERT use an encoder-only architecture, that is, only the encoder part of the transformer model?

- A) It allows BERT to be used for text-generation tasks.
- B) Because in encoder models, causal attention is visually represented by an 'X' for the masked attention and 'O's' for the active attention units.
- C) It allows BERT to process entire sequences of text simultaneously.
- D) Because encoder models possess a unidirectional training method.

**Answer: C**

---

## Question 2

In the next sentence prediction (NSP) task, which of the following determines whether the second sentence logically follows the first for a given pair of sentences?

- A) CLS token's contextual embedding
- B) Segment embeddings
- C) Positional encoding
- D) Separate token

**Answer: A**

---

## Question 3

How many classes are there in the output layer of the neural network for mask language modeling (MLM)?

- A) The number of classes will be equal to the number of special tokens.
- B) The number of classes will be equal to the size of the vocabulary.
- C) The number of classes will be equal to the size of NSP.
- D) There will be two classes.

**Answer: B**

---

## Question 4

Identify one of the most common pretraining objectives useful for training the BERT model in PyTorch.

- A) You can supervise the learning with labeled data.
- B) You can use unsupervised learning with masked language modeling (MLM).
- C) You can use semi-supervised learning with limited labeled data.
- D) You can initialize the model's parameters randomly.

**Answer: B**

---

## Answer Summary 4

1. Encoder-only architecture allows BERT to process entire sequences simultaneously
2. CLS token's contextual embedding determines NSP output
3. MLM output classes equal to vocabulary size
4. MLM is a common unsupervised pretraining objective for BERT

---

## Additional Quiz Questions: Encoder-Decoder Translation

## Question 1

What is the primary advantage of transformers over RNNs for language translation?

- A) Transformers use fewer parameters
- B) Transformers process entire text sequences simultaneously
- C) Transformers always produce more accurate translations
- D) Transformers require less memory

**Answer: B**

---

## Question 2

In the encoder-decoder transformer architecture for translation, what is the output of the encoder called?

- A) Logits
- B) Memory
- C) Embeddings
- D) Hidden state

**Answer: B**

---

## Question 3

What is the purpose of cross-attention in the decoder?

- A) To predict the next token in the sequence
- B) To attend to the encoder's memory output
- C) To generate the final translation
- D) To mask future tokens

**Answer: B**

---

## Question 4

In the translation process, how does the decoder generate tokens during inference?

- A) All at once
- B) Sequentially, using previously generated tokens
- C) Randomly
- D) Using only the source sequence

**Answer: B**

---

## Question 5

What does the linear layer in the decoder do?

- A) Applies dropout to the embeddings
- B) Transforms contextual embeddings into logits for vocabulary prediction
- C) Applies normalization
- D) Creates positional encodings

**Answer: B**

---

## Question 6

When training the encoder-decoder model, what is the target input?

- A) The complete target sequence
- B) The target sequence with the last token removed
- C) The target sequence with the first token removed
- D) A random sequence

**Answer: B**

---

## Question 7

What is the purpose of the causal mask in decoder training?

- A) To hide future tokens from the model
- B) To speed up computation
- C) To reduce memory usage
- D) To improve accuracy

**Answer: A**

---

## Question 8

In the translate function, when do you stop generating tokens?

- A) After a fixed number of iterations
- B) When EOS token is generated or max length reached
- C) When the source is exhausted
- D) After one iteration

**Answer: B**

---

## Answer Summary 5

1. Transformers process entire sequences simultaneously (faster than RNN sequential processing)
2. Encoder output is called "memory"
3. Cross-attention allows decoder to attend to encoder's memory
4. Decoder generates tokens sequentially using previously generated tokens
5. Linear layer maps embeddings to vocabulary for prediction
6. Target input = target sequence with last token removed
7. Causal mask hides future tokens from the model
8. Stop when EOS token or max length reached

---

## Additional Quiz Questions: Transformer Translation Implementation

## Question 1

Why does the decoder use a cross-attention layer for translation?

- A) To maintain the words' order
- B) To convert a token into a D-dimensional embedding vector
- C) To complete the decoder's operation
- D) To attend to the encoder's hidden representations

**Answer: D**

---

## Question 2

Which of the following functions is designed to construct masks for the source and target sequences?

- A) "src_padding_mask"
- B) "Masking(token)"
- C) "generate_square_subsequent_mask"
- D) "create_mask"

**Answer: D**

---

## Question 3

What is the following code used for?

```python
def generate_square_subsequent_mask(sz, device=DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

- A) To create a mask
- B) To initiate the token embeddings and positional encodings
- C) To construct the memory tensor
- D) To generate a causal mask

**Answer: D**

---

## Answer Summary 6

1. Cross-attention allows decoder to attend to encoder's hidden representations
2. "create_mask" function constructs masks for source and target sequences
3. This code generates a causal mask to hide future tokens

---

## Additional Quiz Questions: Transformer and BERT

## Question 1

You are analyzing text generated from a GPT-like model for the step-by-step output "Good. Thank you!" for the prompt "How are you?" What best describes how the GPT-like model generates this?

- A) It reads the complete target phrase before choosing any words.
- B) It matches the input to the memory responses automatically.
- C) It uses past tokens to sequentially predict the next token.
- D) Convert the input through an encoder for final decoding.

**Answer: C**

---

## Question 2

Which attention technique limits each token to only focus on earlier tokens in the sequence during generation?

- A) Global attention masking
- B) Local attention masking
- C) Causal attention masking
- D) Multi-head attention masking

**Answer: C**

---

## Question 3

The hospitality industry wants to implement causal language modeling (LM) for text generation sequences where each token is generated based only on the tokens next to it. Which of the following PyTorch implementation techniques is commonly used for text generation in this case?

- A) Convolutional neural networks (CNNs)
- B) Gated recurrent units (GRUs)
- C) Transformer architecture
- D) Bi-directional LSTM

**Answer: C**

---

## Question 4

Identify the primary function of the "torch.nn.TransformerDecoder" module in the PyTorch implementation while implementing decoder models for sequence-to-sequence tasks.

- A) It provides positional encoding for the input sequence.
- B) It encodes the input sequence into a fixed-size representation.
- C) It adds the attention scores for the decoder input and encoder output.
- D) It decodes the encoded input sequence to generate the output.

**Answer: D**

---

## Question 5

A financial services company is using BERT to analyze customer feedback and wants the model to understand the meaning of each word based on its surrounding context. Which training approach allows BERT to consider both the left and right context of a word during encoding?

- A) Contextual representation generation
- B) Generating causal masks
- C) Predicting the masked words
- D) Bidirectional training method

**Answer: D**

---

## Question 6

While implementing BERT from scratch, you include CLS and SEP tokens to train the model with sentence pairs labeled with 0 or 1 for the next sentence prediction (NSP). What does the NSP task do in BERT training?

- A) To change a sentence from one language to another
- B) To evaluate whether a word expresses a positive or negative feeling
- C) To validate if one sentence sensibly follows the additional
- D) To build text from a given sentence prompt

**Answer: C**

---

## Question 7

Mei is preparing input for a BERT model using PyTorch. To represent the input text accurately, she must include which set of embeddings?

- A) Token embeddings, position embeddings, and segment embeddings
- B) Token embeddings and position embeddings
- C) Token embeddings
- D) Position embeddings and segment embeddings

**Answer: A**

---

## Question 8

Which optimizer is widely recommended for fine-tuning BERT models due to its adaptive learning rate and momentum features?

- A) Adaptive Moment Estimation (Adam)
- B) Root Mean Square Propagation (RMSprop)
- C) Adaptive Gradient Algorithm (Adagrad)
- D) Stochastic Gradient Descent (SGD)

**Answer: A**

---

## Question 9

Which of the following decoder components ensures the model sequentially predicts each word, considering only the preceding tokens in the target sequence?

- A) Linear layer
- B) Masking layer
- C) Normalization layer
- D) Cross-attention layer

**Answer: B**

---

## Question 10

A developer is working with a transformer decoder in PyTorch. Which component generates the logits used to predict the output?

- A) Feedforward layer
- B) Linear layer
- C) Multi-head attention layer
- D) Normalization layer

**Answer: B**

---

## Answer Summary 7

1. GPT uses past tokens to sequentially predict next token
2. Causal attention masking limits focus to earlier tokens
3. Transformer architecture used for causal language modeling
4. TransformerDecoder decodes encoded input to generate output
5. Bidirectional training allows BERT to use both left and right context
6. NSP validates if one sentence follows another
7. BERT uses token, position, and segment embeddings
8. Adam optimizer recommended for fine-tuning BERT
9. Masking layer ensures sequential prediction
10. Linear layer generates logits for output prediction