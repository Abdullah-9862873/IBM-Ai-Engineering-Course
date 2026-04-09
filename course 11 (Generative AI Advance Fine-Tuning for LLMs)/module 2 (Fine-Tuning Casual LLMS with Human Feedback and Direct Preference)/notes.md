# LLMs as Distributions

## LLMs as Distribution and Sampling

- Language model takes query X and generates response Y where Y is a **random variable**
- Example query: "Which is the largest ocean?"
- Possible responses with different probabilities:
  - "Pacific Ocean"
  - "Pacific Ocean is the largest ocean on Earth"
  - "Pacific Ocean is 155 million square kilometers"
  - "Atlantic Ocean"
- Represented as: Y ~ π(Y|X) where π is the distribution/policy

## Transformer Model Generating Probabilities

- Transformer generates probabilities for different words using **softmax function** applied to final layer
- At each timestamp t, the model outputs probability distribution over vocabulary
- Visualized as bar graph with:
  - X-axis: possible words at time ω_t
  - Y-axis: probabilities for each word

### Token Generation Process

- At time t, various realizations for ω_t exist (Pacific, Atlantic, Indian, etc.)
- Words selected based on their probabilities
- Number of realizations for each word is proportional to the distribution from softmax

### Sequential Dependency

- Distribution at time t+1 **depends on previous values** at time t
- As t increases, the distribution is influenced by all earlier timestamps
- This creates various possible sequences (rollouts)

## Generation Parameters

### Temperature (τ)

- Hyperparameter affecting the softmax function
- Controls randomness in probability distribution
- Lower τ = less random (peakier distribution)
- Higher τ = more uniform/random

**Effect on Distribution:**
- τ = 1: Original distribution
- τ = 2: More uniform
- τ = 5: Higher randomness
- τ = 10: More flat/uniform
- τ = 100: Almost uniform (maximum randomness)

### Top-K Sampling

- Restricts selection to top K highest probability tokens
- Steps:
  1. Compute softmax values
  2. Select top K highest probability tokens
  3. Filter out less preferred options
  4. Normalize top K values (sum to 1)

### Other Parameters

- **Beam Search**: Tracks and expands various top sequences at each step
- **Top-P (Nucleus) Sampling**: Limits sampling pool to smallest tokens where cumulative probability exceeds threshold p
- **Repetition Penalty**: Penalizes repeated sequences to encourage diverse output
- **Max/Min Tokens**: Sets maximum or minimum number of tokens to generate

---

# From Distributions to Policies

## Policy in Reinforcement Learning

- Policy is a strategy or mapping that an agent uses to decide actions based on current state
- In RL, policies determine distributions for generating sequences of actions
- For LLMs, policies guide generation process, allowing exploration of various text generation paths

## Importance of Policy in LLMs

- **Enhances decision-making**: Helps models learn optimal responses
- **Improves accuracy**: Contextually appropriate outputs
- **Enables exploration**: Uses randomness to discover unseen possibilities
- **Makes model robust**: Adapts to new contexts

## Policy as Language Model Distribution

- Language model generates responses following policy distribution
- Notation: Y ~ π(Y|X)
- Where:
  - X = input sequence of length m
  - Y = output sequence of length n

### Example
- Query: "Which is the largest ocean?"
- Possible responses (rollouts):
  1. "Pacific Ocean"
  2. "Pacific Ocean is the largest ocean on Earth"
  3. "Atlantic Ocean is 155 million..."
  4. "Indian Ocean if you..."
  5. "What is your timeline?"

## Rollouts

- **Rollouts**: How the model generates different responses for each query
- For each query, multiple responses are generated
- Each response is a sample from the policy distribution
- Note: In Hugging Face libraries, rollout definition differs from traditional RL (no reward included)

---

# Reinforcement Learning from Human Feedback (RLHF)

## RLHF Concept

- Imagine monkeys typing on typewriters with rewards (bananas)
- With rewards, they need fewer attempts to produce meaningful text
- Same principle applies to LLMs with human feedback

### Process
1. Model generates response
2. Human evaluates and provides feedback/reward
3. Model updates parameters based on reward
4. Repeat until satisfactory outputs

## Reward Calculation

- Use reward function r(X, Y) where X = query, Y = response

### Examples
| Query | Response | Reward |
|-------|----------|--------|
| Which country owns Antarctica? | ?9dfsa | 0 (irrelevant) |
| Which country owns Antarctica? | No country owns Antarctica | 0.9 |
| Which country owns Antarctica? | Antarctica is governed by an international treaty | 1.0 |

## Expected Reward

### Empirical Formula
- Approximates expected reward by averaging over multiple queries and responses
- N = total number of queries
- K = number of responses per query
- E[reward] ≈ (1/NK) Σ r(x_n, y_{n,k})

### Actual Expected Value
- E[reward] = Σ_x Σ_y p(x) · π(y|x) · r(x, y)
- Where p(x) is data distribution and π(y|x) is model response distribution

## Incorporating Human Feedback

1. Start with pretrained LLM (policy π_θ)
2. Input query X to the agent (LLM)
3. Generate response Y (rollout)
4. Input (X, Y) to reward model
5. Get reward value from reward model
6. Use reward to update policy parameters θ
7. Repeat for multiple queries

### Example
- Query: "Who made this course?"
- Response: "He looks like Brad Pitt"
- Reward: -10,000 (poor response)
- This feedback helps fine-tune the model to generate better responses

---

# Proximal Policy Optimization (PPO)

## Policy Gradient Methods

- Objective: Maximize expected reward
- Policy gradient methods form foundation of various RL algorithms

## PPO Overview

- PPO is a method to maximize the objective function
- Key aspects:
  - **Clipped surrogate objectives**: Ensures updates are not too drastic
  - **KL penalty coefficient**: Regulates divergence between old and new policies
  - **Advantage function**: Estimates the reward improvement

## Policy Gradient Objective Function

1. **Step 1**: For given query x, sample response y from policy
2. **Step 2**: Estimate reward using reward function r(x, y)
3. **Step 3**: Extend to entire dataset
4. **Goal**: Find optimal policy π_θ that maximizes expected reward

### Objective with Regularization
- J(θ) = E[r(x, y)] - β · KL(π_θ || π_ref)
- Where:
  - π_ref = reference model (regularizing term)
  - β = hyperparameter controlling regularization strength

## Log Derivative Trick

- Used to convert sampling expression into gradient calculable form

### Process
1. Start with: ∇_θ E[π_θ(y|x) · r(x,y)]
2. Use identity: ∇_θ π_θ(y|x) = π_θ(y|x) · ∇_θ log π_θ(y|x)
3. Convert to: E[π_θ(y|x) · ∇_θ log π_θ(y|x) · r(x,y)]
4. This allows gradient calculation using samples

## Training Tips

- **Regular evaluation**: Use human feedback to evaluate model
- **Moderate beta**: Start with moderate β value
- **Explore more**: Increase temperature to explore options
- **Avoid drastic updates**: PPO clipping prevents large policy changes

## Key Takeaways

1. Policy gradient methods maximize objective function
2. PPO stabilizes training through clipping and KL penalties
3. Log derivative trick enables gradient estimation from samples
4. KL divergence ensures model doesn't deviate too far from original
5. Reward model evaluates (query, response) pairs
6. Expected reward considers both data distribution and model distribution

---

# PPO with Hugging Face

## Scoring Function for Sentiment Analysis

- **Sentiment analysis** serves as a scoring/ reward function
- Rewards positive responses over negative ones
- Initialize using pre-trained model fine-tuned on IMDB reviews

### Sentiment Pipeline Setup
```python
sent_kwargs = {'return_all_scores': True, 'function_to_apply': None, 'batch_size': 2}
```

### Output
- Returns positive/negative sentiment scores for each text
- Scores evaluate quality/relevance of generated responses
- Extract scores and convert to tensors for use as rewards

## Dataset and Tokenization

### IMDB Dataset
- 50,000 movie reviews
- Filter reviews with length ≤ 200 characters

### Length Sampler
- Varies text lengths for data processing
- Enhances model robustness
- Simulates realistic training conditions
- Range: input_min_text_length to input_max_text_length

### Tokenization
- Load pre-trained tokenizer for causal LLM
- Set padding token as EOS token
- Tokenize review text → input IDs
- Truncate to desired length
- Create input IDs and queries for model input

### Dataset Processing Function
- Combines all preprocessing steps
- Adds keys to dataset
- Removes texts shorter than 200 characters

---

# PPO Trainer

## PPO Configuration

- **PPOConfig**: Specifies model and learning rate
- Model name and learning rate are key parameters

## Model and Reference Model

- **Reference Model**: Stabilizes training using KL divergence
- **KL divergence**: Between current policy (model) and reference policy
- **AutoModelForCausalLMWithValueHead**: Extends auto model for causal LM for RL

## Collator Function

- Prepares data batches for PPO trainer
- Groups each feature from data samples together

## PPO Trainer

- Processes query samples and optimizes chatbot policy
- Handles complex tasks for high-quality responses

### Initialization
1. Configure PPO settings (learning rate, model name)
2. Fine-tune primary model using PPO
3. Input reference model
4. Input tokenizer
5. Insert dataset for training
6. Data collator handles batching

### Training Statistics
- `list_stats_all` stores training statistics for each batch

### Sentiment Score Setting
- **score_change = 1**: Higher reward for positive sentiment → positive responses
- **score_change = 0**: Higher reward for negative sentiment → negative responses

## Training Loop

1. Extract input IDs, queries from each batch
2. Select random sample output length
3. Generate response using PPO trainer
4. Decode responses to text
5. Concatenate query + response
6. Apply sentiment analysis pipeline
7. Extract sentiment scores → tensors
8. Perform PPO step with queries, responses, rewards
9. Log statistics

## Results

- **PPO Loss**: Decreases over time
- **PPO Mean Reward**: Increases over time

## Text Generation Comparison

- Use same input text across models
- **Positive model**: Generates positive responses (score ~1)
- **Negative model**: Generates negative responses (score ~0)
- **Reference model**: Provides neutral response

---

# Log-Derivative Trick

## Definition

- Technique for deriving functions with respect to parameters
- Deals with likelihood functions (probability of random variables)
- Simplifies deriving non-logarithmic functions

## Mathematical Form

### Basic Identity
∇_θ log p(x;θ) = ∇_θ p(x;θ) / p(x;θ)

Rearranging:
∇_θ p(x;θ) = p(x;θ) · ∇_θ log p(x;θ)

## Score Function Estimator

- Used to estimate gradient of expectation
- Many applications: RL (REINFORCE), variational inference, finance

### Problem
∇_θ E_p(x;θ)[f(x)] = ∇_θ ∫ p(x;θ) f(x) dx

### Solution using Log-Derivative Trick
∇_θ E_p(x;θ)[f(x)] = E_p(x;θ)[∇_θ log p(x;θ) · f(x)]

### Monte Carlo Approximation
≈ (1/n) Σ ∇_θ log p(x_i;θ) · f(x_i) for samples x_i from p(x;θ)

## Applications

- **Reinforcement Learning**: REINFORCE method (policy gradient)
- **Variational Inference**: Optimizing ELBO
- **Computational Finance**: Sensitivity analysis
- **Discrete Latent Variable Models**: Gumbel-softmax, concrete distribution

---

# Module Summary

1. **Reward function** provides human feedback for queries
2. **Rollouts** help review sampling process (differs in Hugging Face vs RL)
3. **Expected rewards** use empirical formula for agent performance
4. **RLHF** uses response distribution to fine-tune pre-trained LLMs
5. **Pre-trained reward model** evaluates query + response pairs
6. **PPO** provides feedback on policy action quality

## Generation Parameters
- Temperature, top-k, beam search, top-p, repetition penalty, max/min tokens

## Key Concepts
- **Objective function**: Measures difference between predicted and target values
- **KL divergence**: Measures difference between two probability distributions
- **Policy gradient**: Maximizes objective function
- **PPO**: Achieves maximization through clipping and KL penalties
- **Log derivative trick**: Enables gradient estimation from samples

## Training Tips
- Regular evaluation with human feedback
- Use moderate beta value
- Increase temperature for exploration
- Positive reward → positive update
- Negative reward → negative update

---

# Direct Preference Optimization (DPO)

## DPO Concept

- **Direct Preference Optimization (DPO)** is a RL technique
- Fine-tunes models based on human preferences more directly and efficiently
- Collects preference data by showing users different outputs and asking them to choose better one
- Directly optimizes model parameters to produce outputs that align with human choices

### Comparison with RLHF
- RLHF: Uses reward model + PPO (complex, reward-based)
- DPO: Reward-free method, directly optimizes against preferences
- State-of-the art results on academic benchmarks often use DPO

## DPO Models

### Three Models Involved
1. **Reward Function**: Uses encoder model (e.g., BERT)
2. **Target Decoder**: Model with parameters θ to optimize
3. **Reference Model**: Baseline for regularization

### Example
- Input: "this is a"
- Response: "cat" → reward = 0.1 (irrelevant)
- Response: "reward function" → reward = 0.99 (relevant)

### Objective
- Maximize: E[log π(y|x)] - β · KL(π || π_ref)
- Where β = regularization term measuring divergence from reference model

## Partition Function

### Logistic Function σ(x)
- Maps any real number to value between 0 and 1
- Basis for logistic probability function

### Partition Function Z(x)
- Z(x) = P(y=0|x) + P(y=1|x) = 1 (ensures valid probability)
- Used to normalize probabilities
- For sequence length T: Z(x) sums over V^T possible sequences (exponential growth)

### Creating Custom Distributions
1. Start with base probability (e.g., exponential, gaussian)
2. Apply partition function for normalization
3. Result: valid probability distribution summing to 1

---

# DPO Optimal Solution

## KL Divergence
- KL(π_star || π_ref) measures difference between two probability distributions
- Minimized to zero if and only if π_star = π_ref

### Mathematical Trick
1. **Max to Min**: Multiply by -1 to convert max to min
2. **Scaling**: Multiply by 1/β doesn't change optimum location
3. **Expectation**: Express as expected value for simplification

## Deriving DPO Objective

### Steps
1. Start with RL objective: maximize E[r(x,y)] - β·KL(π||π_ref)
2. Use optimal policy: π_r(y|x) = (1/Z(x)) · π_ref(y|x) · exp(β·r(x,y))
3. Solve for reward: r(x,y) = log(π_r(y|x)) - log(π_ref(y|x)) + log(Z(x))
4. Substitute into Bradley-Terry loss

### Final DPO Loss Function
- Loss = -log(σ(β · (log π(y_w|x) - log π(y_l|x))))
- Where y_w = winning response, y_l = losing response
- This eliminates the need for partition function Z(x)

### Loss as Function of U
- U = π(y_w|x) / π(y_l|x) (ratio of probabilities)
- Loss decreases as U increases (winning response becomes more likely)
- Convert to cost function for minimization

---

# From Optimal Policy to DPO

## Bradley-Terry Model for DPO
- WIN (y_w): preferred response
- LOSS (y_l): rejected response
- Loss = log(σ(r(y_w) - r(y_l)))

### Dataset Notation
- D: dataset
- ~(tilde): sampling
- (X, Y_w, Y_l) ~ D: query, winning response, losing response drawn from dataset

### DPO Loss Derivation
1. Express reward in terms of optimal policy
2. Substitute into Bradley-Terry loss
3. Partition function Z(x) cancels out
4. Result: Loss depends only on LLM and reference model (no separate reward model needed)

### Simplified Form (β=1)
- Loss = -log(σ(log(π(y_w|x)/π(y_l|x))))
- Can convert to cost: -log likelihood for minimization

---

# DPO with Hugging Face

## Two Main Steps

### 1. Data Collection
- Gather preference dataset with positive/negative pairs
- Each record: prompt, chosen (preferred), rejected

### 2. Optimization
- Maximize log likelihood of DPO loss directly
- Simpler than PPO

## Dataset: Barra Home Preference Data

- Available on Hugging Face
- 6 splits, 7 features per record
- Key features: chosen, rejected, prompt

### Preprocessing
- Extract prompt, rejected, chosen responses
- Reformat to match DPO trainer input

## Model Creation

### Load Models
- Load GPT2 model (target decoder)
- Load reference model (another GPT2 instance)
- Load tokenizer (set pad token = EOS)

### PEFT Configuration
- Apply LoRA for parameter-efficient fine-tuning
- Configure attention parameters

## Training Arguments

- Similar to other methods
- **Key parameter**: Beta (temperature for DPO loss)
- Typical range: 0.1-0.5

## DPO Trainer

- reference_model = None (using PEFT config)
- Call trainer.train() to start training

## Evaluation

- Plot training loss (should decrease)
- Retrieve from trainer.state.log_history

## Inference

- Load trained DPO model
- Compare with original GPT2
- Use pipeline or generate() method
- DPO model generates more relevant responses

---

# InstructLab: Fine-Tune LLMs Locally

## Introduction

- Library for easy fine-tuning of LLMs on local machine (including laptops)
- Uses synthetic data generation using a teacher model
- Fine-tunes a student model on synthetic data

### Key Features
- Alleviates problem of insufficient training examples
- Provides few initial Q&A pairs → teacher model generates synthetic pairs
- Capable of fine-tuning for new knowledge or skills
- Uses taxonomy for structured knowledge/skill separation
- Uses QLoRA for parameter-efficient fine-tuning
- Quantized by default → runs on consumer-grade hardware

## Installation

```bash
# Create directory
mkdir instructlab
cd instructlab

# Create virtual environment
python3 -m venv --upgrade-deps venv
source venv/bin/activate

# Install InstructLab
pip cache remove llama_cpp_python
pip install instructlab

# Test installation
ilab
```

## Initialization

```bash
ilab config init
```

## Model Download

### Teacher Model (Default - Merlinite 7B)
```bash
ilab model download
```

### Student Model (Granite 7B)
```bash
ilab download --repository instructlab/granite-7b-lab-GGUF --filename granite-7b-lab-Q4_K_M.gguf --hf-token <Access Token>
```

## Chat with Base Models

### Serve Model
```bash
# Serve default Merlinite model
ilab model serve

# Serve Granite model
ilab model serve --model-path models/granite-7b-lab-Q4_K_M.gguf
```

### Chat
```bash
# Chat with default model
ilab model chat

# Chat with specific model
ilab model chat --model models/granite-7b-lab-Q4_K_M.gguf
```

## Fine-tuning with InstructLab

### Step 1: Create Seed Examples (YAML)

Create `qna.yaml` in taxonomy directory:
- Minimum 5 seed examples (more = better synthetic data)
- Format: question/answer pairs
- Save to: `taxonomy/compositional_skills/linguistics/tokenizer/qna.yaml`

### Step 2: Validate Taxonomy
```bash
ilab taxonomy diff
```

### Step 3: Generate Synthetic Data
```bash
ilab data generate --num-instructions 100
```

### Step 4: Fine-tune Student Model
```bash
ilab model train --model-dir instructlab/granite-7b-lab
```

### Step 5: Test Fine-tuned Model
```bash
ilab model test --model-dir instructlab-granite-7b-lab-mlx-q
```

### Step 6: Convert and Use Model
```bash
# Convert to GGUF
ilab model convert --model-dir instructlab-granite-7b-lab-mlx-q

# Serve
ilab model serve --model-path instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf

# Chat
ilab model chat --model instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf
```

## Module Summary

1. **DPO** is a reinforcement learning technique for fine-tuning based on human preferences
2. Collects preference data by showing users different outputs
3. Uses three models: reward function, target decoder, reference model
4. Converts complex RL problem to simpler objective function

### Two Main Steps for DPO
1. Data collection (preference dataset)
2. Optimization (maximize log likelihood of DPO loss)

### DPO with Hugging Face Steps
1. Preprocess dataset
2. Create and configure model/tokenizer
3. Define training arguments and DPO trainer
4. Train and evaluate
5. Inference

### Key DPO Formulas
- Reward policy: π_r(y|x) = (1/Z(x)) · π_ref(y|x) · exp(β·r(x,y))
- DPO Loss: -log(σ(β · (log π(y_w|x) - log π(y_l|x))))

### InstructLab
- Local fine-tuning using synthetic data generation
- Teacher model generates synthetic examples from seed Q&A
- Student model fine-tuned using QLoRA
- Can run on consumer hardware (laptops)