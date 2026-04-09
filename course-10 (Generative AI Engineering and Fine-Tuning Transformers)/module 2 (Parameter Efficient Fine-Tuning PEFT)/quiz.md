# Module 2 Quiz: Parameter-Efficient Fine-Tuning (PEFT)

## Question 1

What is the primary advantage of Parameter-Efficient Fine-Tuning (PEFT)?

- A) It always produces better accuracy than full fine-tuning
- B) It reduces the number of trainable parameters significantly
- C) It eliminates the need for labeled data
- D) It requires no computational resources

**Answer: B**

---

## Question 2

Which PEFT method involves adding new task-specific layers to the pretrained model without modifying existing parameters?

- A) Selective fine-tuning
- B) Additive fine-tuning
- C) Reparameterization fine-tuning
- D) Full fine-tuning

**Answer: B**

---

## Question 3

In transformer architectures, where are adapter layers typically added for additive fine-tuning?

- A) Before the embedding layer
- B) After the output layer
- C) Between the attention blocks
- D) At the very beginning

**Answer: C**

---

## Question 4

What are soft prompts?

- A) Fixed text inputs for models
- B) Learnable tensors concatenated with input embeddings
- C) Pre-defined token sequences
- D) Model checkpoints

**Answer: B**

---

## Question 5

What does the rank represent in the context of low-rank adaptations?

- A) The number of neurons in a layer
- B) The minimum number of vectors needed to span a space
- C) The batch size during training
- D) The learning rate

**Answer: B**

---

## Question 6

Which PEFT method is considered the most popular based on the video?

- A) Selective fine-tuning
- B) Additive fine-tuning (Adapters)
- C) LoRA
- D) QLoRA

**Answer: C**

---

## Question 7

What is a key advantage of using adapters in additive fine-tuning?

- A) They require retraining the entire model
- B) Only adapters need to be stored
- C) They replace the original model
- D) They increase model size

**Answer: B**

---

## Question 8

What does LoRA stand for?

- A) Linear Rank Adaptation
- B) Low-Rank Adaptation
- C) Localized Representation Adaptation
- D) Large Parameter Adaptation

**Answer: B**

---

## Question 9

QLoRA combines LoRA with which technique?

- A) Dropout
- B) Quantization
- C) Gradient descent
- D) Attention mechanisms

**Answer: B**

---

## Question 10

What is one issue with full fine-tuning that PEFT aims to solve?

- A) Underfitting
- B) Catastrophic forgetting
- C) Poor accuracy
- D) Slow inference

**Answer: B**

---

## Answer Summary

1. PEFT reduces trainable parameters significantly
2. Additive fine-tuning adds new task-specific layers
3. Adapters are added between attention blocks
4. Soft prompts are learnable tensors
5. Rank is minimum vectors to span a space
6. LoRA is the most popular PEFT method
7. Only adapters need to be stored
8. LoRA = Low-Rank Adaptation
9. QLoRA combines LoRA with quantization
10. PEFT aims to solve catastrophic forgetting

---

## Additional Quiz Questions: LoRA Implementation

## Question 1

In LoRA, what happens to the original weight matrix during fine-tuning?

- A) It is completely replaced
- B) It remains frozen
- C) It is deleted
- D) It is doubled

**Answer: B**

---

## Question 2

In LoRA, how is δW (delta W) decomposed?

- A) δW = B + A
- B) δW = B - A
- C) δW = B × A
- D) δW = A / B

**Answer: C**

---

## Question 3

What is the reduced parameter count with LoRA compared to full fine-tuning for a layer with d×k parameters?

- A) d + k
- B) d × r + r × k
- C) r only
- D) No reduction

**Answer: B**

---

## Question 4

Which scaling factor is applied in LoRA forward pass?

- A) rank / alpha
- B) alpha / rank
- C) rank × alpha
- D) 1 / (alpha × rank)

**Answer: B**

---

## Question 5

In LoRA with HuggingFace, which modules are commonly targeted for LoRA adaptation in transformers?

- A) Only embedding layers
- B) Query and Value attention layers (q_lin, v_lin)
- C) Only output layers
- D) All linear layers

**Answer: B**

---

## Question 6

What is the storage advantage of LoRA according to the video?

- A) Saves only 2x parameters
- B) Saves approximately 28x parameters compared to full linear layer
- C) No storage advantage
- D) Saves memory only during training

**Answer: B**

---

## Question 7

In PyTorch implementation, what is the purpose of the LoRA layer?

- A) Replace the original layer completely
- B) Add low-rank matrices to the original layer
- C) Delete the original layer
- D) Increase model capacity

**Answer: B**

---

## Question 8

When applying LoRA to the TextClassifier, which layer is typically modified?

- A) Embedding layer
- B) Output layer
- C) Hidden layer
- D) All layers

**Answer: C**

---

## Question 9

In LoRA, what is 'r' (rank) compared to d and k?

- A) r is much larger than d and k
- B) r is smaller than d and k
- C) r is equal to d and k
- D) r is unrelated

**Answer: B**

---

## Question 10

What is the function of the TrainingArguments class in HuggingFace?

- A) Create the model
- B) Load the tokenizer
- C) Encapsulate all hyperparameters for training
- D) Define the loss function

**Answer: C**

---

## Answer Summary

1. Original weight matrix remains frozen
2. δW = B × A (matrix multiplication)
3. Parameters reduced to d×r + r×k
4. Scaling factor: alpha / rank
5. Target modules: q_lin, v_lin
6. Saves approximately 28x parameters
7. LoRA layer adds low-rank matrices
8. Modify hidden layer for LoRA
9. r is smaller than d and k
10. TrainingArguments encapsulates hyperparameters

---

## Additional Quiz Questions: QLoRA and Soft Prompts

## Question 1

What does QLoRA stand for?

- A) Quantitative Low-Rank Adaptation
- B) Quantized Low-Rank Adaptation
- C) Quick Low-Rank Adaptation
- D) Quality Low-Rank Adaptation

**Answer: B**

---

## Question 2

What is the quantization range in QLoRA?

- A) 0 to 1
- B) -1 to 1
- C) 0 to 255
- D) -10 to 10

**Answer: B**

---

## Question 3

How many discrete levels does 3-bit quantization represent?

- A) 2
- B) 4
- C) 8
- D) 16

**Answer: C**

---

## Question 4

What is the approximate memory reduction when using 4-bit quantization compared to FP16 for a 7B parameter model?

- A) 25%
- B) 50%
- C) 75%
- D) 90%

**Answer: C**

---

## Question 5

What is a key difference between hard prompts and soft prompts?

- A) Hard prompts are learnable
- B) Soft prompts are manually crafted text
- C) Soft prompts use learnable tensors
- D) Hard prompts use embeddings

**Answer: C**

---

## Question 6

Which soft prompt method uses a bidirectional LSTM as a prompt encoder?

- A) Prompt tuning
- B) Prefix tuning
- C) P-Tuning
- D) All of the above

**Answer: C**

---

## Question 7

In prefix tuning, where are the soft prompts integrated?

- A) Only at input
- B) Only at output
- C) Across all model layers
- D) Only in attention layers

**Answer: C**

---

## Question 8

What is the main benefit of soft prompts over hard prompts?

- A) They are always more accurate
- B) They can be optimized via gradient descent
- C) They require no training
- D) They are easier to write

**Answer: B**

---

## Question 9

Which technique helps protect data privacy during fine-tuning?

- A) Model amplification
- B) Differential privacy
- C) Data duplication
- D) Model compression

**Answer: B**

---

## Question 10

What environmental impact is associated with training large language models?

- A) Lower water usage
- B) Reduced energy consumption
- C) High energy consumption and carbon emissions
- D) No environmental impact

**Answer: C**

---

## Answer Summary

1. QLoRA = Quantized Low-Rank Adaptation
2. Quantization range: -1 to 1
3. 3-bit quantization: 8 levels
4. Memory reduction: ~75%
5. Soft prompts use learnable tensors
6. P-Tuning uses bidirectional LSTM
7. Prefix tuning integrates across all layers
8. Soft prompts can be optimized via gradient descent
9. Differential privacy protects data
10. LLMs have high energy consumption

---

## Additional Quiz Questions: PEFT Advanced

## Question 1

Which of the following reasons makes selective fine-tuning less effective for transformer architectures?

- A) Because it allows for task-specific customization in transformers.
- B) Because of the higher number of parameters in transformer architectures.
- C) Because it involves updating the neural parameters, layers, and neurons.
- D) Because it allows adding layers to a pre-trained transformer model between the attention blocks.

**Answer: B**

---

## Question 2

Which of the following is the key aspect of the low-rank adaptation (LoRA) for enhancing the efficiency of fine-tuning large language models (LLMs)?

- A) LoRA parallelizes the model across the GPUs to increase the training speed.
- B) LoRA introduces a new architecture for replacing transformers in LLMs.
- C) LoRA uses low-rank decomposition for trainable parameters for fine-tuning LLMs using weight matrices.
- D) During the training process, LoRA eliminates large datasets.

**Answer: C**

---

## Question 3

Which of the following statements is correct with respect to quantization to quantized low-rank adaptation (QLoRA)?

- A) Quantization to QLoRA reduces the model's accuracy by increasing inference speed
- B) Quantization to QLoRA streamlines the model architecture, making it simpler to interpret.
- C) Quantization to QLoRA helps the model be fine-tuned using less memory; however maintains the same performance.
- D) Quantization to QLoRA helps the model handle more complex tasks and increases the calculation's precision.

**Answer: C**

---

## Question 4

In the context of training LoRA with PyTorch, identify the correct statement for the initialization of low-rank matrices A and B.

- A) Initialize the matrices A and B using any distribution method by ensuring stable training and proper scaling of the values.
- B) Initialize the A and B matrices as identity matrices, ensuring they start as neutral elements in matrix multiplication.
- C) Initialize the matrices A and B using random values and leveraging standard normal distribution.
- D) Initialize the matrices A and B with zeros to prevent initial influence on the model's predictions.

**Answer: C**

---

## Answer Summary

1. Selective fine-tuning less effective due to higher number of parameters in transformers
2. LoRA uses low-rank decomposition for trainable parameters
3. QLoRA maintains same performance while using less memory
4. LoRA matrices initialized with random values using standard normal distribution

---

## Additional Quiz Questions: PEFT Applications

## Question 1

Nick is tasked with fine-tuning a large language model (LLM) on limited hardware resources. Which technique would be helpful in quantized low-rank adaptation (QLoRA) to reduce memory usage while maintaining high precision?

- A) Few-shot inference
- B) Zero-shot inference
- C) LoRA adaptation
- D) 4-bit quantization

**Answer: D**

---

## Question 2

A data scientist enhances a model's performance by optimizing input embeddings using learnable tensors without modifying model weights. Which PEFT techniques should data scientists use in this scenario?

- A) Additive fine-tuning
- B) LoRA
- C) Soft prompts
- D) Full fine-tuning

**Answer: C**

---

## Question 3

What happens when low-rank matrices are introduced to the weight layers of a pre-trained model using LoRA?

- A) The low-rank matrices track all trainable parameters during backpropagation.
- B) The low-rank matrices replace the original weights to reduce dimensionality.
- C) The low-rank matrices add a small number of trainable parameters to the existing structure.
- D) The low-rank matrices increase trainable parameters by attaching full-rank matrices.

**Answer: C**

---

## Question 4

You are fine-tuning a sentiment classification model using LoRA with the HuggingFace on the IMDB dataset. Which of the following best explains how LoRA integrates with the HuggingFace model during fine-tuning?

- A) LoRA modifies transformer layers and retrains the model from scratch.
- B) LoRA uses low-rank matrices A and B, updates them, and integrates with PEFT and training arguments.
- C) LoRA disables dropout and uses standard tokenizers from HuggingFace.
- D) LoRA adds more weight metrics and freezes all original model parameters.

**Answer: B**

---

## Question 5

Ricky is deploying a transformer-based model and needs to fine-tune it quickly on a domain-specific dataset. Which mechanism does LoRA employ to enable faster and memory-efficient model fine-tuning?

- A) Maintains the full network from scratch using low learning rates
- B) Freezes main weights and updates only added low-rank matrices
- C) Applies quantization techniques to reduce model size
- D) Tailors the system architecture to support small batch training

**Answer: B**

---

## Answer Summary

1. 4-bit quantization reduces memory while maintaining precision
2. Soft prompts optimize input embeddings without modifying weights
3. LoRA adds small number of trainable parameters
4. LoRA uses low-rank matrices with PEFT and training arguments
5. LoRA freezes main weights, updates only low-rank matrices