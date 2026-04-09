# Introduction to LangChain

## What is LangChain?

- **Open-source framework** for developing applications using LLMs
- Generic interface for LLMs providing environment to build and integrate applications
- Integrates with external data sets and workflows
- Customizes generated models, improves accuracy, provides relevant information

### Example
- Developers can use LangChain components to create prompt chains
- Use existing prompt templates to allow LLMs to generate responses based on different variables

## Why LangChain is Important?

1. **Simplifies Integration**: Makes GPT-4 and other LLMs accessible for building NLP applications
2. **Unlocks AI Capabilities**: Enables developers to implement AI capabilities seamlessly
3. **Flexibility**: Developers can customize, dig into codebase, and develop commercial products

---

## LangChain Components

### 1. LangChain (Main Library)
- Chains, agents, and retrieval strategies
- Builds application's cognitive architecture

### 2. LangChain Core
- LangChain Expression Language (LCEL)
- Base for abstractions

### 3. LangChain Community
- Third-party integrations
- Partner packages: LangChain IBM, LangChain OpenAI, LangChain Anthropic
- Lightweight packages depending on LangChain Core

---

## LangChain Use Cases

1. **Chatbots and AI Agents**: Enhance user interactions
2. **API Integration**: Connect seamlessly with various services
3. **Code & Data Understanding**: Extract insights from code and tabular data
4. **Q&A Systems**: Generate document-based questions and answers
5. **Text Summarization**: Summarize texts and enhance information retrieval
6. **Content Management**: Manage and process large content libraries

---

# Introduction to In-Context Learning

## What is In-Context Learning?

- **Specific method of prompt engineering**
- Demonstrations of task provided to model as part of prompt in natural language
- **No additional training required**
- New task learned from small set of examples at inference time

### Advantages
- Doesn't require fine-tuning on specific datasets
- Drastically reduces resources and time needed to adapt LLMs
- Improves performance on specific tasks

### Disadvantages
- Constrained by what can be provided in context
- Complex tasks may require gradient steps or traditional ML training

---

## Prompt Engineering Fundamentals

### What are Prompts?
- Instructions or inputs given to LLM to guide it toward specific task or output
- Two main components:

1. **Instructions**: Clear, direct commands telling AI what to do
2. **Context**: Background information helping AI make sense of instruction

### Why Prompt Engineering is Important?

1. **Boosts Effectiveness**: Directly influences LLM accuracy
2. **Ensures Relevance**: Enables precise, contextually appropriate responses
3. **Meets Expectations**: Clear prompts reduce misunderstandings
4. **Eliminates Fine-tuning**: Model adapts within its context

### Example
```
Prompt: "The wind is"
Response: "Blowing gently through the trees, whispering secrets and stories..."
```

---

## Prompt Structure

A well-structured prompt consists of:

1. **Instructions**: What the LLM needs to do (e.g., "Classify the following customer review")
2. **Context**: Background scenario (e.g., "Review is part of feedback for recently launched product")
3. **Input Data**: Actual data for LLM to process (e.g., "The product arrived late but quality exceeded expectations")
4. **Output Indicator**: Where to deliver analysis (e.g., "Sentiment:")

---

# Advanced Methods of Prompt Engineering

## 1. Zero-Shot Prompting
- LLM performs task **without any prior specific training or examples**
- Example: "Classify as true or false: The Eiffel Tower is located in Berlin"

## 2. One-Shot Prompting
- Gives LLM **single example** to help perform similar task
- Example: Shows translation template, then translates new sentence

## 3. Few-Shot Prompting
- Provides **small set of examples** before tackling similar task
- Helps AI generalize from few instances to new data

## 4. Chain-of-Thought (CoT) Prompting
- Guides LLM through **complex reasoning step-by-step**
- Effective for problems requiring multiple intermediate steps
- Example: Arithmetic problem with clear sequential steps

## 5. Self-Consistency
- Generates **multiple independent answers** to same question
- Evaluates to determine **most consistent result**
- Cross-verifies reliability of responses

---

## Tools for Prompt Engineering

### Popular Tools
1. **OpenAI's Playground**: Develop, experiment, evaluate prompts
2. **LangChain**: Build and test prompt templates
3. **Hugging Face's Model Hub**: Access various pre-trained models
4. **IBM's AI Classroom**: Collaborative editing and sharing

### Capabilities
- Real-time tweaking and testing
- Access to various models for different tasks
- Share and collaborate on prompts
- Track changes and analyze results

---

## LangChain Prompt Templates

LangChain provides **predefined recipes** for generating effective prompts:

```python
from langchain.prompts import PromptTemplate

joke_template = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}"
)

# Use template
prompt = joke_template.format(adjective="funny", content="chickens")
# Output: "Tell me a funny joke about chickens"
```

### Benefits
- Simplifies prompt creation
- Makes prompts consistent and adaptable

---

## Agents in LangChain

Agents are powered by LLMs and integrated tools to perform complex tasks:

1. **Q&A Agents with Sources**: Answer questions using specific sources
2. **Content Agents**: Create and summarize content
3. **Analytic Agents**: Data analysis and business intelligence
4. **Multilingual Agents**: Translation and communication

---

## Summary

1. **LangChain** is an open-source framework for building LLM applications
2. **Components**: LangChain, LangChain Core, LangChain Community
3. **In-context learning** provides examples in prompt without fine-tuning
4. **Prompt engineering** improves LLM responses through carefully crafted prompts
5. **Advanced methods**: Zero-shot, few-shot, Chain-of-thought, Self-consistency
6. **Prompt templates** in LangChain simplify prompt creation
7. **Agents** perform complex tasks using integrated tools

---

# LangChain Core Concepts

## What is LangChain?

- **Open-source interface** that simplifies application development using LLMs
- Integrates language models into NLP and data retrieval use cases
- Key components: Documents, Chains, Agents, Language Model, Chat Model, Chat Message, Prompt Templates, Output Parsers

## Language Model in LangChain

- Uses **text input → text output** for task completion
- Example: Using IBM WatsonX.AI with Mixtral 8x7B Instruct model

```python
from ibm_watson_machine_learning.foundation_models import Model

model = Model(
    model_id='mistralai/mixtral-8x7b-instruct-v01',
    params=GenParams(
        max_tokens=200,
        temperature=0.5
    )
)
response = model.generate(prompt)
```

## Chat Model

- Designed for **efficient conversations**
- Understands questions/prompts and responds like a human

### Chat Messages Types
- **HumanMessage**: User inputs
- **AIMessage**: Model-generated responses
- **SystemMessage**: Instructions for the model
- **FunctionMessage**: Function call outcomes
- **ToolMessage**: Tool interaction results

### Properties
- **Role**: Who is speaking
- **Content**: What is being said

## Prompt Templates

### Types
1. **String Prompt Template**: Single-string formatting
2. **Chat Prompt Template**: Message lists
   - ChatMessagePromptTemplate (flexible role)
   - HumanMessagePromptTemplate
   - AIMessagePromptTemplate
   - SystemMessagePromptTemplate
3. **MessagesPlaceholder**: Full control over rendering
4. **FewShotPromptTemplate**: Specific examples/shots

### Example
```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} bot"),
    ("human", "{question}")
])
```

## Example Selectors

Used to select most relevant examples:
- **Semantic Similarity**: Meaning-based matching
- **Max Marginal Relevance**: Diversity in selection
- **N-Gram Overlap**: Textual similarity

## Output Parsers

Transform LLM output into structured formats:
- JSON, XML, CSV, Pandas DataFrame

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
```

---

# LangChain Documents for RAG Applications

## Document Object

```python
from langchain.schema import Document

doc = Document(
    page_content="Text content here",
    metadata={"doc_id": "123", "filename": "example.pdf"}
)
```

### Attributes
- **page_content**: String containing document content
- **metadata**: Arbitrary data (doc_id, filename, etc.)

## Document Loaders

Load documents from **100+ sources**:
- HTML, PDF, code files
- S3 buckets, public websites
- Examples: Airbyte, Unstructured

```python
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

## Text Splitters

Split large documents into manageable chunks:
- **RecursiveCharacterTextSplitter**: Recursive text splitting
- **MarkdownHeaderTextSplitter**: Split by markdown headers

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=100
)
chunks = splitter.split_text(text)
```

## Embeddings

- Capture semantic meaning of text
- Example: Using watsonx.ai embeddings

## Vector Database

Store embeddings for similarity search:
- Example: ChromaDB

```python
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(docs, embedding_model)
```

## Retrievers

Algorithms for efficient data retrieval:
- **VectorStore Retriever**: Similarity search
- **Parent Document Retriever**: Search within document chunks
- **Self-Query Retriever**: Semantic separation

```python
retriever = vectorstore.as_retriever()
results = retriever.get_relevant_documents(query)
```

---

# LangChain Chains and Agents

## Chains in LangChain

- **Sequence of calls** where output from one step becomes input for next
- Sequential chain creates seamless flow of information

### Example: Recipe Chain
```python
# Chain 1: Get famous dish
template1 = "What is a famous dish in {location}?"
prompt1 = PromptTemplate(template=template1, input_variables=["location"])
location_chain = LLMChain(llm=llm, prompt=prompt1, output_key="meal")

# Chain 2: Get recipe
template2 = "Provide a simple recipe for {meal}"
prompt2 = PromptTemplate(template=template2, input_variables=["meal"])
dish_chain = LLMChain(llm=llm, prompt=prompt2, output_key="recipe")

# Chain 3: Get cooking time
template3 = "Estimate cooking time for {recipe}"
prompt3 = PromptTemplate(template=template3, input_variables=["recipe"])
time_chain = LLMChain(llm=llm, prompt=prompt3, output_key="time")

# Combine into sequential chain
overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, time_chain],
    input_variables=["location"],
    output_variables=["meal", "recipe", "time"],
    verbose=True
)
```

## Memory in LangChain

- Reads and writes historical data
- Chain reads from memory to enhance user inputs
- Writes current inputs/outputs back to memory

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_ai_message("Hi")
history.add_user_message("What is the capital of France?")
```

## Agents in LangChain

- **Dynamic systems** where LLM determines and sequences actions
- Integrates with tools (search engines, databases, websites)

### Example: Pandas DataFrame Agent

```python
from langchain.agents import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm=chat_model,
    df=dataframe,
    verbose=True
)

agent.invoke("How many rows are in the dataframe?")
```

### Capabilities
- Transforms natural language queries into code
- Executes code in background
- Returns precise answers

---

# Summary and Highlights

Congratulations! You have completed this lesson. At this point in the course, you know that:

- LangChain provides an environment for building and integrating large language model (LLM) applications into external data sets and workflow.

- LangChain simplifies the integration of language models like GPT-4 and makes it accessible for developers to build natural language processing or NLP applications.

- The components of LangChain are: Chains, agents, and retriever; LangChain-Core; LangChain-Community.

- Generative models understand and capture the underlying patterns and data distribution to resemble the given data sets. Generative models are applicable in generating images, text, and music, augmenting data, discovering drugs, and detecting anomalies.

- Types of generative models are: Gaussian mixture models (GMMs); Hidden Markov models (HMMs); Restricted Boltzmann machines (RBMs); Variational autoencoders (VAEs); Generative adversarial networks (GANs); Diffusion models.

- In-context learning is a method of prompt engineering where task demonstrations are provided to the model as part of the prompt.

- Prompts are inputs given to an LLM to guide it toward performing a specific task. They consist of instructions and context.

- Prompt engineering is a process where you design and refine the prompts to get relevant and accurate responses from AI.

- Prompt engineering has several advantages: It boosts the effectiveness and accuracy of LLMs; It ensures relevant responses; It facilitates meeting user expectations; It eliminates the need for continual fine-tuning.

- A prompt consists of four key elements: instructions, context, input data, and output indicator.

- Advanced methods for prompt engineering are: zero-shot prompting, few-shot prompting, chain-of-thought prompting, and self-consistency.

- Prompt engineering tools facilitate interactions with LLMs.

- LangChain uses "prompt templates," which are predefined recipes for generating effective prompts for LLMs.

- An agent is a key component in prompt applications that can perform complex tasks across various domains using different prompts.

- The language models in LangChain use text input to generate text output.

- The chat model understands the questions or prompts and responds like a human.

- The chat model handles various chat messages, such as: HumanMessage; AIMessage; SystemMessage; FunctionMessage; ToolMessage.

- The prompt templates in LangChain translate the questions or messages into clear instructions.

- An example selector instructs the model for the inserted context and guides the LLM to generate the desired output.

- Output parsers transform the output from an LLM into a suitable format.

- LangChain facilitates comprehensive tools for retrieval-augmented generation (RAG) applications, focusing on the retrieval step to ensure sufficient data fetching.

- The "Document object" in LangChain serves as a container for data information, including two key attributes, such as page_content and metadata.

- The LangChain document loader handles various document types, such as HTML, PDF, and code, from various locations.

- LangChain in document retrieves relevant isolated sections from the documents by splitting them into manageable pieces.

- LangChain embeds documents and facilitates various retrievers.

- LangChain is a platform that embeds APIs for developing applications.

- Chains in the LangChain is a sequence of calls. In chains, the output from one step becomes the input for the next step.

- In LangChain, chains first define the template string for the prompt, then create a PromptTemplate using the defined template and create an LLMChain object name.

- In LangChain, memory storage is important for reading and writing historical data.

- Agents in LangChain are dynamic systems where a language model determines and sequences actions, such as predefined chains.

- Agents integrate with tools such as search engines, databases, and websites to fulfill user requests.

---

# Cheat Sheet: Fundamentals of Building AI Agents using RAG and LangChain

## Package/Method Description and Code Examples

### Generate Text

Generates text sequences without computing gradients:

```python
output_ids = model.generate(
    inputs.input_ids, 
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50, 
    num_return_sequences=1
)
```

Or with torch.no_grad():

```python
with torch.no_grad():
    outputs = model(**inputs) 
```

### formatting_prompts_func_no_response Function

Creates formatted prompts from a dataset:

```python
def formatting_prompts_func(mydataset):
    output_texts = []
    for i in range(len(mydataset['instruction'])):
        text = (
            f"### Instruction:\n{mydataset['instruction'][i]}"
            f"\n\n### Response:\n{mydataset['output'][i]}"
        )
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_no_response(mydataset):
    output_texts = []
    for i in range(len(mydataset['instruction'])):
        text = (
            f"### Instruction:\n{mydataset['instruction'][i]}"
            f"\n\n### Response:\n"
        )
        output_texts.append(text)
    return output_texts
```

### torch.no_grad()

Generates text with gradient computation disabled:

```python
with torch.no_grad():
    pipeline_iterator = gen_pipeline(instructions_torch[:3],
                                    max_length=50,
                                    num_beams=5,
                                    early_stopping=True)
    generated_outputs_lora = []
    for text in pipeline_iterator:
        generated_outputs_lora.append(text[0]["generated_text"])
```

### mixtral-8x7b-instruct-v01 WatsonX.AI Inference Model

Creates model with customizable parameters:

```python
model_id = 'mistralai/mixtral-8x7b-instruct-v01'
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
project_id = "skills-network"
model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
```

### String Prompt Templates

Formats single string inputs:

```python
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}
prompt.invoke(input_)
```

### Chat Prompt Templates

Formats lists of messages:

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])
input_ = {"topic": "cats"}
prompt.invoke(input_)
```

### MessagesPlaceholder

Adds list of messages at specific place:

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])
input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}
prompt.invoke(input_)
```

### Example Selector

Selects relevant examples for prompts:

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
```

### JSON Parser

Returns JSON with specified schema:

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

output_parser = JsonOutputParser(pydantic_object=Joke)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | mixtral_llm | output_parser
```

### Comma Separated List Parser

Returns comma-separated items:

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)
```

### Document Object

Contains content and metadata:

```python
from langchain_core.documents import Document
Document(page_content="Python is an interpreted high-level programming language.",
         metadata={
             'my_document_id': 234234,
             'my_document_source': "About Python",
             'my_document_create_time': 1680013019
         })
```

### Text Splitter

Splits text into chunks:

```python
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)
```

### Embedding Models

Generates vector representations:

```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
from langchain_ibm import WatsonxEmbeddings
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
```

### Vector Store-Backed Retriever

Retrieves documents using similarity search:

```python
retriever = docsearch.as_retriever()
docs = retriever.invoke("Langchain")
```

### ChatMessageHistory Class

Stores conversation history:

```python
from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of France?")
```

### LLMChain

Creates a basic language model chain:

```python
from langchain.chains import LLMChain
template = """Your job is to come up with a classic dish from the area that the users suggests.
                {location}
                YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template, input_variables=['location'])
location_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='meal')
location_chain.invoke(input={'location': 'China'})
```

### Simple Sequential Chain

Chains multiple LLM calls together:

```python
from langchain.chains import SequentialChain
# chain 1: location -> meal
location_chain = LLMChain(llm=mixtral_llm, prompt=prompt1, output_key='meal')
# chain 2: meal -> recipe  
dish_chain = LLMChain(llm=mixtral_llm, prompt=prompt2, output_key='recipe')
# chain 3: recipe -> time
recipe_chain = LLMChain(llm=mixtral_llm, prompt=prompt3, output_key='time')
# overall chain
overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=['location'],
    output_variables=['meal', 'recipe', 'time'],
    verbose=True
)
```

### load_summarize_chain

Creates summarization chain:

```python
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm=mixtral_llm, chain_type="stuff", verbose=False)
response = chain.invoke(web_data)
```

### llm_model Function

Wrapper for LLM inference:

```python
def llm_model(prompt_txt, params=None):
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }
    if params:
        default_params.update(params)
    # ... create model and return response
```

### Zero-shot Prompt

Tests model without examples:

```python
prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'
            Answer:
"""
response = llm_model(prompt, params)
```

### One-shot Prompt

Gives single example:

```python
prompt = """Here is an example of translating English to French:
            English: "How is the weather today?"
            French: "Comment est le temps aujourd'hui?"
            Now translate: English: "Where is the nearest supermarket?"
"""
```

### Few-shot Prompt

Provides multiple examples:

```python
prompt = """Here are few examples of classifying emotions:
            Statement: 'I just won my first marathon!'
            Emotion: Joy
            ...
            Now classify: Statement: 'That movie was so scary...'
"""
```

### Chain-of-thought (CoT) Prompting

Guides through reasoning steps:

```python
prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples...'
            Break down each step of your calculation
"""
```

### Self-consistency

Generates multiple验证ations:

```python
prompt = """When I was 6, my sister was half of my age. Now I am 70...
            Provide three independent calculations, then determine the most consistent result.
"""
```

### Text Summarization

Summarizes provided content:

```python
template = """Summarize the {content} in one sentence."""
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input={"content": content})
```

### Question Answering

Answers based on provided content:

```python
template = """Answer the {question} based on the {content}.
            Respond 'Unsure about answer' if not sure.
            Answer:"""
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key="answer")
response = llm_chain.invoke(input={"question": question, "content": content})
```

### Code Generation

Generates SQL queries:

```python
template = """Generate an SQL query based on the {description}
            SQL Query:"""
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key="query")
```

### Role Playing

Configures LLM to assume roles:

```python
template = """You are an expert {role}. I have this question {question}.
            I would like our conversation to be {tone}.
            Answer:"""
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key="answer")
```

### read_and_split_text

Reads and splits text files:

```python
def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    paragraphs = text.split('\n')
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs
```

### encode_contexts

Encodes texts to embeddings:

```python
def encode_contexts(text_list):
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()
```

### FAISS Index

Creates similarity search index:

```python
import faiss
embedding_dim = 768
context_embeddings_np = np.array(context_embeddings).astype('float32')
index = faiss.IndexFlatL2(embedding_dim)
index.add(context_embeddings_np)
```

### search_relevant_contexts

Searches similar contexts:

```python
def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    D, I = index.search(question_embedding, k)
    return D, I
```

### generate_answer_without_context

Generates answers without RAG:

```python
def generate_answer_without_context(question):
    inputs = tokenizer(question, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer
```

### generate_answer with DPR contexts

Generates answers with retrieved context:

```python
def generate_answer(contexts):
    input_text = ' '.join(contexts)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### aggregate_embeddings

Computes mean embeddings:

```python
def aggregate_embeddings(input_ids, attention_masks, bert_model=bert_model):
    mean_embeddings = []
    for input_id, mask in tqdm(zip(input_ids, attention_masks)):
        input_ids_tensor = torch.tensor([input_id]).to(DEVICE)
        mask_tensor = torch.tensor([mask]).to(DEVICE)
        with torch.no_grad():
            word_embeddings = bert_model(input_ids_tensor, attention_mask=mask_tensor)[0].squeeze(0)
            valid_embeddings_mask = mask_tensor[0] != 0
            valid_embeddings = word_embeddings[valid_embeddings_mask, :]
            mean_embedding = valid_embeddings.mean(dim=0)
            mean_embeddings.append(mean_embedding.unsqueeze(0))
    return torch.cat(mean_embeddings)
```

### text_to_emb

Converts text to embeddings:

```python
def text_to_emb(list_of_text, max_input=512):
    data_token_index = tokenizer.batch_encode_plus(list_of_text, add_special_tokens=True, padding=True, truncation=True, max_length=max_input)
    question_embeddings = aggregate_embeddings(data_token_index['input_ids'], data_token_index['attention_mask'])
    return question_embeddings
```

### RAG_QA

Question answering with RAG embeddings:

```python
def RAG_QA(embeddings_questions, embeddings, n_responses=3):
    dot_product = embeddings_questions @ embeddings.T
    dot_product = dot_product.reshape(-1)
    sorted_indices = torch.argsort(dot_product, descending=True)
    for index in sorted_indices[:n_responses]:
        print(yes_responses[index])
```

### model_name_or_path

Initializes GPT-2 model:

```python
model_name_or_path = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = GPT2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
max_length = 1024
```

### add_combined_columns

Combines prompt with responses:

```python
def add_combined_columns(example):
    example['prompt_chosen'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["chosen"]
    example['prompt_rejected'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["rejected"]
    return example
dataset['train'] = dataset['train'].map(add_combined_columns)
```

### RetrievalQA

Retrieval-augmented question answering:

```python
qa = RetrievalQA.from_chain_type(
    llm=flan_ul2_llm, 
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=False
)
query = "what is mobile policy?"
qa.invoke(query)
```