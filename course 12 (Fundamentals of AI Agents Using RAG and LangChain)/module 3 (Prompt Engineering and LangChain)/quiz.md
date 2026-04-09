# Quiz: Prompt Engineering and LangChain

## Question 1
What is LangChain primarily used for?

- Building websites and web applications
- **Developing applications using large language models** ✓
- Creating database management systems
- Designing user interfaces

---

## Question 2
Which component of LangChain includes chains, agents, and retrieval strategies?

- LangChain Community
- LangChain Core
- **LangChain (main library)** ✓
- LangChain Expression Language

---

## Question 3
What is in-context learning?

- A training method that requires fine-tuning on specific datasets
- **A prompt engineering method where task demonstrations are provided in the prompt** ✓
- A technique for compressing model size
- A method for evaluating model performance

---

## Question 4
What are the two main components of a prompt?

- Questions and answers
- **Instructions and context** ✓
- Input and output
- Tokens and embeddings

---

## Question 5
Which advanced prompt engineering method provides a single example to help the LLM perform a similar task?

- Zero-shot prompting
- **One-shot prompting** ✓
- Chain-of-thought prompting
- Self-consistency

---

## Question 6
What is Chain-of-thought (CoT) prompting used for?

- Generating creative stories
- **Guiding LLM through complex reasoning step-by-step** ✓
- Translating languages
- Classifying sentiments

---

## Question 7
Which technique generates multiple independent answers to verify consistency?

- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
- **Self-consistency** ✓

---

## Question 8
What is the key benefit of using prompt templates in LangChain?

- They automatically generate code
- They improve model training speed
- **They simplify prompt creation and make them consistent** ✓
- They reduce the need for tokenization

---

## Question 9
In a well-structured prompt, what does the "input data" component contain?

- Instructions for what to do
- **The actual data the LLM will process** ✓
- The expected output format
- Background information

---

## Question 10
What type of agents does LangChain support for Q&A systems?

- Image generation agents
- **Q&A agents with sources** ✓
- Video processing agents
- Audio transcription agents

---

# Answers

1. **Developing applications using large language models** - LangChain is an open-source framework for LLM applications.

2. **LangChain (main library)** - Contains chains, agents, and retrieval strategies for cognitive architecture.

3. **A prompt engineering method where task demonstrations are provided in the prompt** - In-context learning provides examples without fine-tuning.

4. **Instructions and context** - Prompts consist of instructions (what to do) and context (background information).

5. **One-shot prompting** - Provides a single example to help the LLM perform similar tasks.

6. **Guiding LLM through complex reasoning step-by-step** - CoT is effective for multi-step reasoning problems.

7. **Self-consistency** - Generates multiple answers and selects most consistent one.

8. **They simplify prompt creation and make them consistent** - Prompt templates provide predefined recipes for prompts.

9. **The actual data the LLM will process** - Input data is the content the LLM operates on.

10. **Q&A agents with sources** - LangChain supports various agent types including Q&A with sources.

---

## Additional Questions

## Question 11
What are the two key attributes of a Document object in LangChain?

- Text and images
- **page_content and metadata** ✓
- Embeddings and vectors
- Source and destination

## Question 12
Which component in LangChain is responsible for splitting large documents into manageable chunks?

- Document loader
- **Text splitter** ✓
- Vector database
- Retriever

## Question 13
In LangChain chains, what happens to the output of each step?

- It is discarded
- **It becomes the input for the next step** ✓
- It is stored in a database
- It is sent to the user immediately

## Question 14
What is the purpose of memory in LangChain?

- To store model weights
- **To read and write historical data for continuity** ✓
- To cache embeddings
- To manage API keys

## Question 15
Which type of retriever in LangChain is based on similarity search?

- Parent document retriever
- **Vector store retriever** ✓
- Self-query retriever
- Multi-query retriever

---

# Answers (Additional)

11. **page_content and metadata** - Document object has page_content (content) and metadata (attributes like doc_id).

12. **Text splitter** - Splits large documents into manageable chunks for processing.

13. **It becomes the input for the next step** - Sequential chain creates seamless flow where each output feeds into next input.

14. **To read and write historical data for continuity** - Memory enables continuity across interactions.

15. **Vector store retriever** - Uses similarity search to retrieve relevant documents from vector database.

---

## Final Assessment Questions

## Question 16
One of the key advantages of in-context learning in prompt engineering is that:

- **It doesn't require the model to be fine-tuned on specific data sets.** ✓
- It can perform required tasks without the need for specific examples.
- It is not constrained by what can realistically be provided in a context.
- It can perform complex tasks with a single simplified step.

---

## Question 17
How do you describe the prompt template in LangChain?

- Prompt template optimizes the computational efficiency of language model inference.
- Prompt template converts natural language queries into structured database queries.
- **Prompt template defines the structure and format of prompts used to interact with language models.** ✓
- Prompt template stores large data sets for training language models.

---

## Question 18
In the context of LangChain components, which of the following components manages the information retrieving process from various sources based on the query?

- Prompt template
- **Retrieval module** ✓
- Document loader
- Memory module

---

## Question 19
The LangChain component _____ is responsible for managing interactions between various parts of LangChain applications?

- **Agents** ✓
- Chains
- Document loaders
- Prompt templates

---

## Question 20
Which of the following is the role of LangChain documents in building RAG applications?

- LangChain documents provide a framework for training large-scale language models.
- LangChain documents summarize the textual data for improving model performance.
- **In LangChain documents, the document object is a data information container using attributes including metadata and page_content.** ✓
- LangChain documents fine-tune the pre-trained models specifically for the task generation.

---

## Question 21
In LangChain, the LangChain component ____ is usually responsible for managing interactions between the retriever and generation steps.

- **Chains** ✓
- Prompt template
- Document retriever
- Embeddings

---

# Answers (Final)

16. **It doesn't require the model to be fine-tuned on specific data sets.** - Key advantage of in-context learning.

17. **Prompt template defines the structure and format of prompts used to interact with language models.** - Templates provide structure for prompts.

18. **Retrieval module** - Manages information retrieval based on queries.

19. **Agents** - Manage interactions between various parts of LangChain applications.

20. **In LangChain documents, the document object is a data information container using attributes including metadata and page_content.** - Document object stores content and metadata.

21. **Chains** - Manage interactions between retriever and generation steps.

---

## Module 3 Assessment Questions

## Question 1
A logistics company is building a chatbot for answering internal queries using existing data without retraining the model. Why do they leverage LangChain?

- Fine-tunes models based on warehouse data
- **Connects LLMs to data sources and custom workflows** ✓
- Helps in tracking and routing GPS
- Provides inventory management APIs

## Question 2
What is the role of prompt engineering in working with large language models (LLMs), such as GPT-3.5?

- Optimize the AI's architecture and improve its training process.
- **Update and modify the AI's outputs by carefully crafting input prompts.** ✓
- Create new algorithms to train AI models to generate responses.
- Manage and allocate computational resources for AI training.

## Question 3
A team is testing various prompt variations to see GPT-4's responds. Which tool should they use?

- **OpenAI's Playground** ✓
- Google Classroom
- Microsoft's Copilot
- Google's Gemini

## Question 4
A developer wants to design a chatbot by structuring the conversation to provide coherent replies. How would a LangChain component, a prompt template, ChatMessagePrompt, be useful in this scenario?

- **To format user queries and manage conversation history for generating contextually relevant responses** ✓
- To directly retrieve data from a database or knowledge base without any formatting or processing
- To create and train machine learning models for natural language processing tasks
- To handle user authentication and manage access control to data sources

## Question 5
In LangChain, the MarkdownHeaderTextSplitter is used for breaking down documents into smaller, manageable pieces before generating embeddings in LangChain. Which component in LangChain's data processing pipeline should you use in this scenario?

- Document source
- Document embedding
- **Document splitter** ✓
- Document loader

## Question 6
A data analyst is interacting with a DataFrame using natural language to extract row counts and summaries via a LangChain-based interface. How can LangChain help in this scenario?

- Saving previous queries in a standard chain format
- Using documents to structure prompts manually
- **Deploying an agent that interprets queries and uses tools – Pandas** ✓
- Using chains to manually input and output code

## Question 7
A software developer is researching frameworks that support fast integration of GPT-4 into new natural language processing (NLP) tools. What is the primary reason a developer may choose LangChain for building NLP applications?

- **Draft AI-powered content tools that use LLMs efficiently** ✓
- Store and handle large image datasets
- Generate mobile applications for video editing
- Restore traditional SQL databases entirely

---

# Answers (Module 3 Assessment)

1. **Connects LLMs to data sources and custom workflows** - LangChain provides environment for building and integrating LLM applications into external data and workflows.

2. **Update and modify the AI's outputs by carefully crafting input prompts.** - Prompt engineering designs and refines prompts to get relevant and accurate responses from AI.

3. **OpenAI's Playground** - Tool for developing, experimenting, and evaluating prompts with various models.

4. **To format user queries and manage conversation history for generating contextually relevant responses** - Prompt templates translate questions into clear instructions and manage conversation flow.

5. **Document splitter** - Text splitters like MarkdownHeaderTextSplitter break documents into manageable chunks before embedding.

6. **Deploying an agent that interprets queries and uses tools – Pandas** - LangChain agents can transform natural language queries into code and execute them.

7. **Draft AI-powered content tools that use LLMs efficiently** - LangChain simplifies integration of language models like GPT-4 for building NLP applications.