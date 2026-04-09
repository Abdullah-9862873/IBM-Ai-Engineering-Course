# IBM AI Engineering with LLMs - Course 1: Generative AI and LLMs

## Course Overview

- First course in a 6-course specialization program
- Focus on building NLP-based applications using LLMs
- Hands-on labs and exercises for skill development

## Impact of Generative AI

- Automatic code completion
- Music composition
- Game design
- Drug discovery
- Contextually relevant conversations
- Document summarization
- Language translation

## Career Opportunities

- Lucrative careers in AI engineering with focus on language modeling
- Build innovative AI applications for human-machine natural language interaction

## Target Audience

- Existing and aspiring data scientists
- Machine learning engineers
- Deep learning engineers
- AI engineers

## Prerequisites

- Basic Python knowledge (advantage)
- Basic PyTorch knowledge (advantage)
- Awareness of machine learning and neural networks (advantage)

## Learning Outcomes

- Describe how to use LLMs to develop generative AI applications
- Explain how text is preprocessed and loaded for LLM training/analysis
- Use generative AI libraries and tools for LLM applications

---

# Module 1: Generative AI Architecture

## Significance of Generative AI

### Definition

- Deep learning models that generate high-quality text, images, and other content based on training data
- Models understand patterns and structures in existing data and apply them to produce new, relevant data
- Analogy: Like an artist who studies paintings, understands patterns, and creates original art

### Types of Generative AI Models

#### 1. Text Generation Models
- Understand context within text data and relationships between words/phrases
- Generate contextually relevant text (e.g., story continuation)
- Language translation while preserving tone
- **Example**: GPT (Generative Pre-trained Transformer)

#### 2. Image Generation Models
- **Text-to-image**: Generate images from text prompts (e.g., "a robot playing a piano")
  - **Example**: DALL-E (Data Analysis Learning with Language Model for Generation and Exploration)
- **Image-to-image**: Generate images from seed images or sketches
  - Applications: Realistic images from sketches, deep fakes (bringing actors back to life)
  - **Examples**: GAN (Generative Adversarial Network), Diffusion models

#### 3. Audio Generation Models
- Generate natural-sounding speech
- Text-to-speech synthesis
- **Example**: WaveNet

### Applications of Generative AI

- **Content Creation**: Automated articles, blog posts, marketing materials
- **Visual/Video Creation**: Entertainment and advertising
- **Summarization**: Condense long documents for quick absorption
- **Language Translation**: More natural-sounding translations
- **Chatbots/Virtual Assistants**: Human-like conversations, customer support
- **Data Analysis**: NLP tasks, uncover insights, suggest creative solutions

### Industry-Specific Applications

- **Healthcare**: Analyze medical images, create patient reports
- **Finance**: Predictions and forecasts from financial data
- **Gaming**: Interactive elements, dynamic storylines
- **IT**: Create artificial data to train models for ML

### Market Growth

- Bloomberg Intelligence: Generative AI market expected to reach $1.3 trillion by 2032
- Future applications: Personalized recommendations, drug discovery, smart homes, autonomous vehicles

---

## Generative AI Architectures and Models

### Architectures Overview

- Recurrent Neural Networks (RNNs)
- Transformers
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Diffusion Models

### 1. Recurrent Neural Networks (RNNs)

- Artificial neural networks using sequential or time series data
- Solve data problems with natural order or time-based dependencies
- **Key Feature**: Loop-based design enables remembering previous inputs
- This design is crucial for tasks dealing with sequences (language modeling)
- **Fine-tuning**: Adjust weights and structure to align with specific tasks/datasets
- **Applications**: NLP, language translation, speech recognition, image captioning

### 2. Transformers

- Deep learning models for translating text and speech in near real-time
- Pass data (words/numbers) through different layers (information flows in one direction)
- Employ feedback mechanisms to improve accuracy
- **Key Feature**: Self-attention mechanism - focus on most important parts of information
- Enables parallelization for efficient training
- **Fine-tuning**: Keep pretrained model intact, train only final output layers
- **Example**: GPT (Generative Pre-trained Transformer)
  - Trained to predict and generate text sequences based on patterns
  - Produces text that mirrors training data distribution

### 3. Generative Adversarial Networks (GANs)

- Consists of two submodels: Generator and Discriminator
- **Generator**: Creates fake samples
- **Discriminator**: Checks authenticity by comparing with real samples
- Assigns probability score indicating likelihood of being authentic
- **Adversarial Process**: Like a friendly competition
  - Generator strives to make things look real
  - Discriminator learns to distinguish real from fake
  - Both improve their outputs
- **Applications**: Image and video generation

### 4. Variational Autoencoders (VAEs)

- Operate on encoder-decoder framework
- **Encoder Network**: Compresses input data into simplified abstract space (latent space)
- **Decoder Network**: Uses condensed information to recreate original data
- **Key Feature**: Learn underlying patterns, create new samples with similar characteristics
- Represent data using probability distributions in latent space
- Produce range of possible outputs reflecting uncertainty in real-world data
- **Applications**: Art and creative design

### 5. Diffusion Models

- Probabilistic generative model
- Generate images by learning to remove noise or reconstruct distorted examples
- **Key Feature**: Learn statistical properties of training data
- Generate highly creative images based on prompts
- **Applications**: Generate high-quality images from noisy/low-quality inputs (e.g., restoring old photos)

### Training Approach Differences

| Model | Approach |
|-------|----------|
| RNNs | Loop-based design |
| Transformers | Self-attention mechanism |
| GANs | Competitive training (generator vs discriminator) |
| VAEs | Characteristics-based (encoder-decoder) |
| Diffusion Models | Statistical properties (denoising/reconstruction) |

### Generative AI and Reinforcement Learning

- Traditional RL: How agents interact with environment to maximize rewards
- Generative AI uses RL techniques during training to fine-tune and optimize performance for specific tasks

---

## Large Language Models (LLMs)

### What is an LLM?

- A Large Language Model (LLM) is a type of artificial intelligence trained on vast amounts of text data
- Can understand, generate, and manipulate human language
- Based on deep learning architectures, particularly Transformers
- Trained on massive datasets to learn patterns, grammar, facts, and reasoning abilities
- Capable of performing various natural language tasks without specific task training

### Key Characteristics of LLMs

1. **Scale**: Millions to billions of parameters
2. **Training Data**: Massive text corpora from diverse sources
3. **Emergent Abilities**: Show unexpected capabilities as scale increases
4. **Few-shot Learning**: Can learn from few examples
5. **Zero-shot Capabilities**: Can perform tasks without explicit training

### How LLMs Work

1. **Training Phase**:
   - Pre-training on large text corpora
   - Learn to predict next word (causal language modeling)
   - Or learn to fill in masked words (masked language modeling)

2. **Inference Phase**:
   - Take input text (prompt)
   - Process through neural network
   - Generate probability distribution over vocabulary
   - Sample next token
   - Repeat to generate complete response

### Architecture

- **Transformer-based**: Uses self-attention mechanisms
- **Encoder-only**: BERT-style (bidirectional)
- **Decoder-only**: GPT-style (autoregressive)
- **Encoder-Decoder**: T5, BART (sequence-to-sequence)

### Popular LLMs

1. **GPT Series** (OpenAI)
2. **BERT** (Google)
3. **PaLM** (Google)
4. **LLaMA** (Meta)
5. **Claude** (Anthropic)
6. **Mistral** (Mistral AI)

### Applications

- Text generation and completion
- Translation
- Summarization
- Question answering
- Code generation
- Chatbots and virtual assistants
- Content creation
- Sentiment analysis

### Training Process

1. **Pre-training**: Learn general language patterns from large datasets
2. **Fine-tuning**: Adapt to specific tasks or domains
   - Instruction fine-tuning
   - RLHF (Reinforcement Learning from Human Feedback)
   - Domain-specific adaptation

### Challenges

- Computational resources required
- Bias in training data
- Hallucinations (generating incorrect information)
- Context window limitations
- Energy consumption and environmental impact
- Ethical considerations

### Evaluation Metrics

- Perplexity
- BLEU score (translation)
- ROUGE score (summarization)
- Human evaluations
- Benchmark datasets (MMLU, BIG-Bench, etc.)

---

## Basics of AI Hallucinations

### Definition

- AI hallucinations: Model generates output presented as accurate but seen as unrealistic, inaccurate, irrelevant, or nonsensical by humans
- Similar to human hallucinations
- Strongly associated with LLMs

### Example

- ChatGPT falsely claimed an Australian mayor was found guilty and imprisoned in a bribery case
- In reality, the mayor had notified authorities about the bribery issue
- Such incidents are rare but can have significant implications

### Causes of AI Hallucinations

- Biases in training data
- Limited training
- Complexity of the model
- Lack of human oversight
- Outputs may not be based on patterns learned from training data

### Problems Caused by AI Hallucinations

- Generation of inaccurate information
- Creation of biased views or misleading information
- Wrong input provided to sensitive applications (autonomous vehicles, medical domain)
- Legal disputes from incorrect summarization of legal documents

### Methods for Mitigating Hallucinations

1. Eliminate bias in training data
2. Perform extensive training on high-quality data
3. Avoid manipulation of inputs fed into models
4. Ongoing evaluation and improvement of models
5. Fine-tune pre-trained LLM on domain-specific data

### Best Practices to Prevent Problems

1. **Be vigilant**: Understand models don't understand actual meaning of words
   - Models focus on predicting next word based on patterns
   - Trained on vast amounts of data, learn statistical patterns
   - Lack semantic understanding or comprehension like humans

2. **Ensure human oversight**: Regular fact-checking and continuous testing

3. **Provide additional context**: In prompts/inputs
   - Enables LLMs to understand desired output better
   - Generates more accurate and contextually relevant responses

---

## Generative AI for NLP

### Role of Generative AI in NLP

- Enable machines to comprehend human language and generate human-like responses
- Improve language processing by incorporating context awareness
- Ensure coherent interactions
- Enable meaningful conversations through predictive analytics and advanced modeling
- NLP systems based on generative AI sense feelings and grasp intentions behind words

### Evolution of Generative AI for NLP

1. **Rule-Based Systems**
   - Strictly follow predefined linguistic rules (grammar)
   - Precise but lacked flexibility

2. **Machine Learning Approaches**
   - Employ statistical methods to learn from vast language datasets
   - More adaptable than rule-based systems
   - Still limited in understanding complex language nuances

3. **Deep Learning**
   - Train artificial neural networks with extensive datasets
   - Many computational units working together for nuanced language interpretations

4. **Transformers (Latest)**
   - Specifically designed to handle sequential data
   - Greater ability to understand context and dependencies in language
   - Self-attention mechanism for parallel processing

### Applications in Language Tasks

1. **Machine Translation**
   - More precise and context-aware conversions between languages
   - Better preservation of meaning and tone

2. **Chatbots/Virtual Assistants**
   - More natural and human-like conversations
   - Degree of empathy and personalization
   - Enhanced user experience

3. **Sentiment Analysis**
   - Grasp subtle language expressions
   - Deeper insights into user sentiments

4. **Text Summarization**
   - Recognize core meaning and significance of text
   - More precise and accurate summaries

### Large Language Models (LLMs) for NLP

- **Definition**: Foundation models using AI and deep learning with vast datasets (websites, books)
- **Purpose**: Generate text, translate languages, create various content types

### LLM Characteristics

- **Training Data Size**: May reach petabytes
- **Parameters**: Billions of variables defining model behavior
- **Fine-tuning**: Parameters optimized during training for specific tasks
- **Example**: Parameter represents weights assigned to words (e.g., "happy" or "sad")

### LLM Capabilities

- Understand language structures and contexts comprehensively
- Capture nuances of human language
- Predict next word in sequence
- Produce creative content with minimal task-specific training

### Popular LLMs

| Model | Architecture | Best For |
|-------|--------------|----------|
| GPT (Generative Pre-trained Transformer) | Decoder-only | Text generation, chatbots |
| BERT (Bidirectional Encoder Representations) | Encoder-only | Sentiment analysis, question answering |
| BART (Bidirectional and AutoRegressive Transformers) | Encoder-Decoder | Sequence-to-sequence tasks |
| T5 (Text-to-Text Transfer Transformer) | Encoder-Decoder | Various NLP tasks |

### GPT vs ChatGPT

- **GPT**: Focuses on diverse text generation tasks
- **ChatGPT**: Focuses on generating conversations

| Feature | GPT | ChatGPT |
|---------|-----|---------|
| Primary Use | Text generation | Conversations |
| Training | Supervised learning | Supervised + RLHF |
| Human Feedback | Not incorporated | Uses RLHF |

- **RLHF**: Reinforcement Learning from Human Feedback - uses human feedback to create a reward model

### LLM Usage in Industry

- **Pre-training**: Train for generic purposes
- **Fine-tuning**: Adapt with smaller dataset for specific domains
- **Example**: Generic text classifier → Fine-tuned for retail product categorization (electronics, clothing)

### Considerations

- May generate authoritative-sounding but inaccurate information (hallucinations)
- Need to address biases in training data
- Consider potential impact of generated content on society

---

## Libraries and Tools for Generative AI NLP

### Overview

- AI/NLP engineers design, develop, and deploy generative AI applications
- Rapid advancement has increased accessibility to various tools and libraries
- Key libraries: PyTorch, TensorFlow, Hugging Face, LangChain, Pydantic

---

## PyTorch

### Overview

- Open source deep learning framework developed by Facebook's AI Research (Meta)
- Python-based library known for ease of use, flexibility, dynamic computation graphs
- Dynamic computation graph: Network structure can change during execution
- Highly customizable, used by OpenAI, Tesla

### Key Features

#### Dynamic Computation Graphs (Autograd)
- Autograd system allows dynamic changes to network during training
- Enhances flexibility and eases development process
- Particularly beneficial for research and experimentation

#### Rich Ecosystem and Community
- Comprehensive ecosystem for computer vision, NLP, and other ML domains
- Vibrant community with tutorials and third-party extensions
- **torchtext**: Library within PyTorch ecosystem for text data (datasets, pretrained models, preprocessing)

### Application in NLP

- Develop and train neural network models for language understanding/generation
- Particularly favored for research and development
- Provides adaptable environment for model experimentation and prototyping

---

## TensorFlow

### Overview

- Open-source framework developed by Google
- Provides tools and libraries for ML and deep learning development/deployment
- Robust architecture suitable for research and production

### Key Features

#### Scalability
- Scalable architecture for seamless transition from research to production
- Streamlines training and large-scale deployment of ML models

#### TensorFlow Extended (TFX)
- Platform for deploying production-ready ML pipelines
- Built on TensorFlow foundation
- Integrates phases of ML system deployment: defining, launching, monitoring

#### Keras Integration
- Tightly integrated with Keras
- Keras provides user-friendly high-level neural networks API
- Users can use tf.keras module to define and train neural networks

### Application in NLP

- NLP tasks: sentiment analysis, text classification, machine translation
- Contains latest AI models and libraries for raw text
- Preferred for enterprise-level NLP applications due to large-scale deployment capacity

---

## Hugging Face

### Overview

- Platform with open-source library for pretrained models and tools
- Streamlines training and fine-tuning of generative AI models
- Made state-of-the-art NLP technologies more accessible

### Key Features

#### Extensive Model Repository
- **Model Hub**: Online platform with vast collection of pretrained NLP models
- Allows sharing, discovering, and using models for text classification, translation, QA
- Supports PyTorch and TensorFlow frameworks
- Provides detailed model info: architecture, training data, usage examples
- Community-driven platform for researchers and developers

#### Simplicity
- Simplifies deployment of complex models
- Makes cutting-edge NLP techniques accessible to beginners and experts
- **Transformers library**: User-friendly interface for pretrained models

#### Community-Driven Development
- Vibrant community enabling collaborative approach
- Developers contribute and share NLP models and tools

### Application in NLP

- Tasks: named entity recognition, sentiment analysis, text summarization
- Readily available resources streamline building and deploying NLP applications

### Useful Hugging Face Libraries

1. **Transformers**
   - Most famous library from Hugging Face
   - Pretrained models for text tasks: generation, summarization, translation, classification, QA
   - Designed for PyTorch and TensorFlow

2. **Datasets**
   - Access and share large-scale datasets and evaluation metrics
   - Wide array of datasets in different languages and tasks
   - Easier benchmarking and evaluation

3. **Tokenizers**
   - Optimized for performance and versatility in tokenization
   - Handles pre-tokenization for models like BERT and GPT

---

## LangChain

### Overview

- Open-source framework for AI application development using LLMs
- Improves LLM accessibility and functionality for diverse applications

### Key Features

#### Advanced Prompt Engineering
- Sophisticated tools for designing effective prompts
- Prompts guide model behavior
- Crucial for tailoring responses and guiding outputs toward desired outcomes

#### Seamless Integration with Leading Models
- Compatibility with major models (GPT)
- Simplifies application development on advanced models
- Smoother transition from concept to deployment

### Application in NLP

- Essential for developers leveraging LLMs
- Ideal for creating: interactive chatbots, analytical tools
- Harmonizes model integration, prompt engineering, and application-specific customization

---

## Pydantic

### Overview

- Python library for streamlined data handling
- Parses and validates data using Python-type annotations

### Key Features

#### Robust Data Validation
- Ensures accuracy of data types and formats before processing
- Enhances reliability of applications
- **BaseModel class**: Define data models and validation rules

#### Efficient Settings Management
- Manages application settings and environment variables
- Vital for scalability of larger projects

### Application in NLP

- Important role in NLP pipelines for validating and managing data
- Ensures data integrity and consistency with diverse, large datasets

---

## Summary: Libraries and Tools

| Tool | Purpose | Key Feature |
|------|---------|-------------|
| PyTorch | Deep learning framework | Dynamic computation graphs (Autograd) |
| TensorFlow | ML/DL framework | Scalability, TFX, Keras integration |
| Hugging Face | Pretrained models & tools | Model Hub, Transformers library |
| LangChain | LLM application development | Prompt engineering, model integration |
| Pydantic | Data validation | BaseModel class, settings management |

---

# Module 2: Data Preparation for LLMs

- Tokenization
- Data loaders
- Implement tokenization (lab exercise)
- Create NLP data loader (lab exercise)
- Course glossary and cheat sheet

---

# Course Structure

- **Videos**: Short and focused on main topics
- **Readings**: Detailed text content
- **Labs**: Technical environment with instructions and code snippets
- **Practice Quizzes**: Ungraded self-assessment
- **Graded Quizzes**: Apply and assess knowledge
- **Glossary & Cheat Sheet**: Quick reference content

## Tips for Success

- Watch all videos
- Complete all labs
- Attempt all quizzes

---

# IBM AI Engineering with LLMs - Program Overview

## Introduction to AI & LLMs in the Workplace

- **WEF Prediction**: ~75% of organizations will adopt AI/ML technologies
- **Career Opportunity**: Organizations need experts to apply AI in their context
- **Value Addition**: Knowledge of generative AI architectures and models is crucial for AI/ML/Data Science careers

## Target Audience

- AI Engineers
- Machine Learning Engineers
- Deep Learning Engineers
- Data Scientists

## Prerequisites

- Basic Python knowledge (required)
- Knowledge of PyTorch, machine learning, and neural networks (beneficial)

## Program Features

- Hands-on labs and exercises
- Skills demonstration for potential employers
- Partial coverage for Professional Certificate in AI Engineering

## Program Structure

### Course 1: Generative AI and LLMs - Architecture and Data Preparation
- Generative AI models and LLM applications for NLP
- Libraries and tools for developing LLM applications
- Data preparation for training LLMs
- Tokenization and NLP data loaders

### Course 2: AI Foundational Models for NLP
- N-gram models
- Word2vec
- Sequence-to-sequence models
- Building, training, and integrating models for NLP tasks

### Course 3: Language Modeling with Transformers
- Positional encoding
- Word embedding
- Attention mechanisms
- Multi-head attention
- Decoder-based models (GPT)
- Encoder-based models (BERT)
- Language translation applications

### Course 4: Fine Tuning with Transformers
- General framework for optimizing transformer-based LLMs
- Fine-tuning generative AI models
- Hugging Face and PyTorch platforms
- Parameter Efficient Fine Tuning (PEFT)
- Low Rank Adaptation (LoRA)
- Quantized Low Rank Adaptation (QLoRA)
- Loading models, inference, and training with adapters

### Course 5: Advanced Fine Tuning for LLMs
- Instruction tuning
- Reward modeling
- Proximal Policy Optimization (PPO)
- LLMs as policies
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Performance Optimization (DPO)

### Course 6: AI Agents with RAG and LangChain
- RAG process (context and question encoders)
- FAISS (Facebook AI Similarity Search)
- In-context learning with LangChain
- Advanced prompt engineering
- LangChain tools, components, chat models, and agents

### Course 7: Capstone Project on AI Applications with RAG and LangChain
- Apply all skills learned
- Build real-world applications using LangChain
- Integration and deployment of sophisticated language models
- Portfolio project for completion certificate

## Assessment & Certification

- Quizzes with weightage toward course completion
- AI/peer-graded projects with weightage
- Completion certificate awarded after successfully finishing all content