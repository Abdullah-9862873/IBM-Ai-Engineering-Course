# Basics of Instruction Tuning

## What is Instruction Tuning?

- Also known as **Supervised Fine-Tuning (SFT)**
- Trains models with **expert-curated datasets**
- Enhances model performance by providing specific instructions and contexts
- Typically performed **before DPO (Direct Preference Optimization)** and **RLHF (Reinforcement Learning from Human Feedback)**
- Establishes a strong foundational understanding for precise and reliable outputs

## Training a GPT-like Model

1. **Pre-training**: Predict next word in sequence (standard GPT training)
2. **Instruction Tuning**: Fine-tune with expert-labeled examples
3. **RLHF or DPO**: Apply preference optimization techniques

## Key Components of Instruction Tuning

### 1. Instructions
- Directions or commands specifying what the user wants
- Defines tasks or actions the model should perform
- Examples: generate text, translate languages, summarize content, answer questions, perform computations

### 2. Input
- Data or context to process to fulfill the instruction
- Can be text passages, lists of items, questions, or other data
- **Note**: Not all datasets include input (instruction-only datasets exist)

### 3. Output
- Expected result or response from the model

## Special Symbols and Prompt Format

- Use special tokens to format prompts for tokenizer compatibility
- Examples:
  - `### Instruction` and `### Response` (triple hashtag format)
  - `### Human` and `### Assistant`
- These tokens help the model correctly interpret structured data
- Different models/datasets may use different tokens

## Instruction Masking

### Concept
- Focus loss calculation on **critical tokens** rather than all output tokens
- Similar to generative pre-training (predict next token)
- Loss is modified to target specific tokens

### How It Works
1. Given sequence with special tokens:
   - `[INST]` instruction prompt `[INST]`
   - `[OUTPUT]` response `[OUTPUT]`
2. Model learns to predict the shifted sequence
3. Loss calculated only on response tokens (not instruction tokens)

### Considerations
- Some studies show unmasked instructions perform better on smaller datasets
- Hugging Face uses `DataCollatorForCompletionOnlyLM` to specify instruction masking
- Special tokens are typically masked by default

## Key Takeaways

1. Instruction tuning trains models with expert-curated datasets
2. Model requires both instructions and answers
3. Uses three components: instructions, input, and output
4. Adjust prompt format using special tokens for tokenizer compatibility
5. Instruction masking focuses loss on important response tokens
6. Enables models to interpret and execute instructions more effectively

---

# Instruction-Tuning with Hugging Face

## Loading and Formatting Datasets

### CodeAlpaca-20k Dataset
- Programming code dataset
- Components:
  - **Instruction**: Task description for the model
  - **Input**: Optional context (present in ~40% of examples)
  - **Output**: Answer to the instruction

### Dataset Preprocessing
- Split into 80% training, 20% validation
- Drop samples containing input values

### Formatting Functions

**formatting_prompts_func:**
- Formats instruction and output into template
- Format: `### Instruction` ... `### Response` ... EOS token
- EOS token indicates model should stop generating

**formatting_prompts_func_no_response:**
- Similar to above but without responses
- Used for validation samples

## Model Creation

### Base Model
- Load from Hugging Face (e.g., facebook/opt-350m)

### PEFT with LoRA
- Use `peft` library to convert base model for LoRA
- Define LoRA config with:
  - LoRA rank
  - Target modules
  - Task type: CAUSAL_LM

### Training Configuration (SFT Config)
- Output directory for model files
- Number of training epochs
- Batch sizes for training and evaluation
- Evaluation strategy (e.g., after each epoch)
- Max sequence length
- FP16 enabled for efficient training

## Collator and Trainer

### DataCollatorForCompletionOnlyLM
- From TRL library
- Prepares data batches for text completion tasks
- Masks instruction part when calculating loss
- Adds padding tokens for uniform batch length
- Creates attention masks
- Handles special tokens including padding

### SFT Trainer
- Pass formatting function for data pre-processing
- Set packing to false
- Control max prompt length
- Pass collator to trainer

## Data Pipeline

1. Load training dataset
2. Apply formatting prompt function
3. Tokenization and masking handled by data collator

## Text Generation Pipeline

- Automatically handles model loading, tokenization, padding
- Define with:
  - Task: text generation
  - Model and tokenizer
  - Max length
  - Return full text (set to false for response only)
- Generation parameters:
  - num_beams: for higher quality text
  - early_stopping: for beam-based methods

## Evaluation

### BLEU Score
- Bilingual Evaluation Understudy (BLEU)
- SacreBLEU: standardized variant of BLEU
- Measures similarity between generated and reference text

### Results Example
- Base model: 0.4 (baseline)
- Fine-tuned model: 14.7 (significant improvement)

---

# Best Practices for Instruction-Tuning

## Data Selection

### High-Quality Data
- Diverse instructions and responses
- Helps model generalize across scenarios

### Dataset Collection
- Wide range of topics, contexts, instructions
- Different prompt types and response styles
- Balance specialized and general data

## Prompt Engineering

### Effective Prompt Design
- Reflect real-world use cases
- Vary formality, complexity, specificity

### Testing Variability
- Experiment with different prompts
- Ensure model generalizes to unseen instructions

## Response Consistency

### Measure Accuracy
- Test with similar instructions
- Monitor task-specific performance
- Consistent responses indicate well-tuned model

## Avoid Overfitting

### Style Variety
- Include variety in tones and structures
- Balance precision and flexibility

## Regular Evaluation

### Metrics for Instruction Adherence
- Evaluate alignment with provided instructions
- Human review for quality checks

---

# Reward Modeling and Response Evaluation

## What is Reward Modeling?

- Quantifies response quality by assigning numerical values/scores
- Evaluates **degree of alignment** with human preferences
- Guides model optimization by maximizing assigned scores
- Incorporates user preferences into scoring function
- Ensures **consistency and reliability** in evaluation

## Response Evaluation Process

### Scoring Function
- Takes query + response as input
- Assigns numerical score (reward) to each response
- Higher scores = better alignment with preferences

### Example 1: Preference Ranking
- Query: "Do you prefer cats or dogs?"
- LLM1 response: "I like cats" → high score (preferred)
- LLM2 response: "Cats are just ok" → lower score (0.5)

### Example 2: Factual Accuracy
- Query: "Which country owns Antarctica?"
- Response A: "Governed by Antarctic Treaty System" → 0.89 (factual)
- Response B: "Penguin overlords run the show" → 0.03 (incorrect)

### Tokenization
- Query tokens: ω (omega with subscript for index)
- Response tokens: ω̂ (omega hat for estimated/response)
- Sequences can have different lengths
- Scoring function: r(query, response)

---

# Reward Model Training

## Why Pairwise Ranking?

- Humans find it **easier to rank** responses than assign precise numerical scores
- Pairwise comparison is more reliable than absolute scoring

## Reward Model Loss

### Bradley-Terry Model
- Goal: Ensure reward for better response > reward for worse response
- r(X, Y) = reward function where X = query, Y = response
- φ (phi) = learnable parameters of transformer model

### Loss Function
- Convert comparison into probability using **sigmoid function**
- Probability that response A is better than B:
  - P(A > B) = σ(r_A - r_B) where σ is sigmoid
- Convert to minimization: Loss = -log(σ(r_A - r_B))

### Optimization
- Maximum likelihood estimation
- Use **negative log-likelihood** for simplification
- Log transforms products into sums for easier differentiation

### Key Relationship
- As reward difference (delta) increases:
  - Loss decreases
  - Model learns to give higher rewards to better responses

---

# Reward Modeling with Hugging Face

## Dataset: Dahoas/synthetic-instruction-gptj-pairwise

- Designed for training instruction-following models
- Each data point contains:
  - **prompt**: Text prompt for model
  - **chosen**: Preferred response
  - **rejected**: Disliked response

## Preprocessing

### get_response_function
- Structures data as query-response pairs

### add_columns Function
- Creates `prompt_chosen`: prompt + chosen response (with human/assistant labels)
- Creates `prompt_rejected`: prompt + rejected response

### Pre-process Function
- Tokenizes both `prompt_chosen` and `prompt_rejected`
- Returns keys: `input_ids_chosen`, `input_ids_rejected`, `attention_mask_chosen`, `attention_mask_rejected`
- Apply using `map` method with `batched=True`

### Filtering
- Filter samples shorter than max length

## Model Setup

### Base Model
- Use GPT-2 for sequence classification as score function
- Output: single scalar value (reward score)

### LoRA Configuration
- Use PEFT library for parameter-efficient fine-tuning
- Configure for sequence classification task

### Training Arguments
```python
per_device_train_batch_size = 3
num_train_epochs = 3
gradient_accumulation_steps = 8
learning_rate = 1.41e-5
```

## Reward Trainer

- Specialized trainer from TRL library
- Orchestrates: batching, optimization, evaluation, checkpointing
- Parameters: model, args, tokenizer, train_dataset, eval_dataset, peft_config

### Training
- Use `trainer.train()` to initiate training

## Evaluation

### Score Generation
- Tokenize text and get model output scores

### Pairwise Comparison
- Compare scores for chosen vs rejected responses
- Correct if chosen score > rejected score

### Win Rate
- Calculate accuracy of correct selections
- Example: 100% on synthetic data
- Real models: 60-70% win rate

---

# Reward Modeling Deep Dive (Lab)

## Key Aspects of Reward Modeling

### 1. Alignment with Human Preferences
- Evaluates how well model responses align with human preferences
- Example: "Who was first US president?" → "George Washington" gets high reward

### 2. Quantifying Response Quality
- Assigns numerical values for performance assessment
- Example: Chatbot A more accurate → higher score than Chatbot B

### 3. Guiding Model Optimization
- Optimize parameters to maximize assigned scores
- Example: If reward model values concise answers, LLM generates concise responses

### 4. Incorporating User Preferences
- Customizes scoring function for specific user needs
- Example: Score creative responses higher for creative tasks

### 5. Ensuring Consistency and Reliability
- Same query should get consistent high scores for correct answers
- Example: "What is capital of France?" → "Paris" always gets high score

## Mathematical Formulation

### Embedding Generation
- Tokenized input Ω and response Ω̂ converted to contextual embeddings
- E(Ω) = [CLS], e_{ωb1}, e_{ωb2}, ..., e_{ωbn}

### Reward Score Calculation
- R(Ω, Ω̂) = W^T · E(Ω ⊕ Ω̂) + b
- Where:
  - Ω ⊕ Ω̂: Concatenated embeddings
  - W, b: Learnable weights and bias

### Example Scores
- Response A (factual): R = 0.89
- Response B (incorrect): R = 0.03

## Bradley-Terry Loss Function

### Single Pair Loss
- L(φ) = -log(σ(R(Ω, Ω̂^A) - R(Ω, Ω̂^B)))
- σ(x) = 1 / (1 + e^(-x)) (sigmoid function)

### Multiple Pairs Loss
- φ̂ = arg min_φ Σ[-log(σ(r(X_n, Y_n,a|φ) - r(X_n, Y_n,b|φ)))]
- Sum over all N training examples

## Gradient Descent Update

### Parameters
- φ: Model parameters (weights W, bias b)
- η: Learning rate
- ∇_φ L(φ): Gradient of loss

### Update Rule
- φ ← φ - η∇_φ L(φ)

### Example Calculation
- Given: ∇_W R_A = [2.1, -0.3], ∇_W R_B = [1.5, 0.4], σ(Δ) = 0.88, η = 0.01
- ∇_φ L = (0.88 - 1) · ([2.1, -0.3] - [1.5, 0.4])
- = (-0.12) · [0.6, -0.7] = [-0.072, 0.084]
- W_new = W + [0.00072, -0.00084]

## Loss vs Reward Difference Table

| Δ (Reward Difference) | Loss (-log σ(Δ)) | Effect |
|------------------------|------------------|--------|
| 0.0 | 0.693 | Δ = 0, high loss |
| 1.0 | 0.313 | Δ = 1, moderate loss |
| 2.0 | 0.126 | Δ = 2, low loss |
| 3.0 | 0.048 | Δ = 3, very low loss |

## Key Takeaways

1. **Human-centric learning**: Models align with preferences, not just accuracy
2. **Measurable quality**: Clear assessment of response quality
3. **Continuous optimization**: Gradient descent refines based on feedback
4. **Preference differentiation**: Distinguishes preferred from non-preferred
5. **Scalable human input**: Efficient at scale

---

# Module Summary

## Instruction Tuning

1. **Instruction-tuning** involves training models with expert-curated datasets
2. Model requires **instructions and answers**
3. Uses three components: **Instructions, input, and output**
4. **Adjust prompt format** using special tokens for tokenizer compatibility
5. **Instruction masking** focuses loss calculation on specific tokens

## Hugging Face Implementation

1. Load dataset using **CodeAlpaca-20k** dataset
2. Format dataset using **formatting_prompts_func** function
3. Use two code blocks for generating instructions with and without responses
4. Fine-tune **facebook/opt-350m** model
5. Define collator using **DataCollatorForCompletionOnlyLM**
6. Create **SFTTrainer** object
7. Generate text pipeline from Transformers library
8. Evaluate using **BLEU score**

## Reward Modeling

1. **Reward model** takes prompt as input, response as output, returns reward/score
2. Quantifies quality responses, guides optimization, incorporates preferences
3. Scoring function takes query and appends chatbot's responses
4. Dataset: **synthetic-instruction-gptj-pairwise** from Hugging Face
5. **preprocess_function()** formats keys and tokenizes data for reward trainer
6. **TrainingArguments** defines training parameters
7. **Reward trainer** orchestrates training, saves, and evaluates using trainer.train()
8. **Tokenizing process** generates scores for pairwise comparison to achieve win rate

## Reward Model Training

1. Trains model to identify desired outputs and assign scores based on relevance/accuracy
2. **Scoring function** generates rewards effectively
3. Encoder model generates responses as **contextual embeddings**
4. **Bradley-Terry reward loss model** generates cost/loss function