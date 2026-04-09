# Fundamentals of AI Agents

## Introduction to AI Agents

### Course Overview
- Focus on AI agents, RAG (Retrieval Augmented Generation), in-context learning, and LangChain
- Build job-ready skills using RAG, PyTorch, Hugging Face, LLMs, and LangChain

### Learning Objectives
- Explain fundamentals of AI agents
- Understand how tool calling enables LLMs to perform real-world tasks
- Apply fundamentals of in-context learning and advanced prompt engineering
- Use RAG with Hugging Face and PyTorch for different applications
- Describe LangChain's core concepts, components, and chat models

---

## From Monolithic Models to Compound AI Systems

### Limitations of Single Models
- Models are limited by their training data
- Hard to adapt - requires tuning with data and resources
- Cannot access personal/sensitive information (e.g., vacation days)

### What are Compound AI Systems?
- Multiple components working together to solve problems
- More modular and easier to adapt than tuning a single model
- Combine models with programmatic components (verifiers, search, tools)

### Example: Vacation Planning System
1. User Query: "How many vacation days do I have?"
2. LLM creates a search query
3. Query accesses vacation database
4. Result returns to LLM
5. LLM generates final answer: "You have 10 days left"

### Control Logic Types
- **Programmatic Control Logic**: Human-defined paths (most RAG systems)
- **LLM-driven Control Logic**: Agentic approach where LLM decides the path

---

## What are AI Agents?

### The Shift to Agentic Systems
- Put LLM in charge of control logic
- Enabled by improved reasoning capabilities of LLMs

### Two Modes of Problem Solving
1. **Think Fast**: Give first answer that pops into head (programmatic)
2. **Think Slow**: Create a plan, break down problem, iterate (agentic)

### Core Capabilities of LLM Agents

#### 1. Reasoning (Think)
- Model at core of problem-solving
- Prompted to create plans and reason through each step

#### 2. Acting (Tool Calling)
- External programs called "tools"
- Model decides WHEN and HOW to call tools
- Examples of tools:
  - Web search
  - Database queries
  - Calculator
  - Code execution
  - Translation models
  - APIs

#### 3. Memory
- **Inner logs**: Think out loud reasoning stored
- **Conversation history**: Previous human-agent interactions

---

## ReAct Agent Framework

### What is ReAct?
- Combines Reasoning and Acting components
- Most popular framework for configuring AI agents

### How ReAct Works
1. **User Query** → Fed into LLM
2. **Prompt**: "Think slow, plan your work, then execute"
3. **Action**: Model decides to use external tools
4. **Observation**: Model reviews tool output
5. **Iteration**: Adjust plan if needed
6. **Final Answer**: Once satisfied with result

### Example: Vacation Sunscreen Calculation
1. Retrieve vacation days from memory
2. Check Florida weather for sun hours
3. Look up sunscreen dosage recommendations
4. Calculate number of 2-oz bottles needed
5. Multiple complex paths can be explored

---

## AI System Design Paradigms

### 1. Single LLM Features
- Simple, one-shot tasks
- Stateless processing (no memory)
- Direct input-output flow

**Best for:**
- Text summarization
- Sentiment classification
- Translation

**Advantages:** Fast, simple, low cost
**Limitations:** No adaptability, no memory

### 2. Structured Workflows
- Multi-step, predictable processes
- Deterministic execution with predefined paths
- Explicit control flow

**Best for:**
- Document pipelines (OCR → extraction → validation → storage)
- Batch report generation
- Financial/healthcare processing

**Advantages:** Predictable, audit-ready, compliance-friendly
**Limitations:** Rigid, difficulty handling new scenarios

### 3. Autonomous Agents
- Flexible, context-aware reasoning
- Dynamic planning and tool orchestration
- Real-time adaptation

**Best for:**
- Complex, open-ended tasks
- Research agents
- Adaptive customer support

**Advantages:** Highly adaptable, reduces human intervention
**Limitations:** Unpredictable outcomes, higher complexity

---

## Choosing the Right Approach

### Sliding Scale of LLM Autonomy
- **Narrow problems** → Programmatic approach (more efficient)
- **Complex, open-ended problems** → Agentic approach (too complex to predefine)

### Real-World Trends
- **Hybrid architectures**: Combine workflow reliability with agent flexibility
- **MCP (Model Context Protocol)**: Anthropic's standard for integration
- **ACP (Agent Communication Protocol)**: IBM's standard for agent communication

### Key Takeaways
1. Start simple: Use most straightforward solution
2. Leverage workflows: When predictability, compliance matter
3. Deploy agents selectively: Only when complex reasoning required
4. Human in the loop: Required in most cases as accuracy improves

---

## Summary

1. **Compound AI systems** combine models with tools for better problem-solving
2. **Agents** put LLMs in charge of control logic with reasoning, acting, and memory
3. **ReAct** is the popular framework combining reasoning and acting
4. **Three paradigms**: Single LLM (simple), Workflows (predictable), Agents (flexible)
5. **Choose based on task complexity**: Narrow = programmatic, Complex = agentic

---

# When (and When Not) to Use AI Agents

## Why Evaluating Agent Use Matters

- Agents offer powerful capabilities but aren't always the best solution
- Need framework to decide when agents make sense vs. simpler tools

## AI System Spectrum

| Type | Description | Best Use Cases |
|------|-------------|----------------|
| Simple AI Features | Classification, text summarization | Fast, repeatable tasks |
| Orchestrated Workflows | Predefined multi-step logic | Structured processes |
| Autonomous Agents | Independent decisions, adapt to new info | Complex reasoning, exploration |

## Four-Criteria Framework for Using Agents

### 1. Is the task ambiguous or predictable?

**Use Agents when ambiguous:**
- Decision path unclear or cannot be mapped in advance
- Tasks involve exploration, troubleshooting, creativity

**Use Workflows when predictable:**
- Can define all rules and outcomes
- Process follows clear, repeatable structure

### 2. Is the value of the task worth the cost?

- Agents are 10-100× more expensive than workflows (more tokens)
- Use for high ROI tasks (strategic planning)
- Avoid for low-margin tasks (basic support)

### 3. Does the agent meet minimum capabilities?

Test agent on 3-5 key skills before launch:
- Research agent: identify, filter, summarize credible sources
- Coding agent: write, fix, validate code snippets
- Support agent: classify issues, resolve queries, escalate appropriately
- Data analysis agent: clean datasets, detect anomalies, summarize trends

### 4. What happens if the agent makes a mistake?

- Can errors be caught and corrected quickly?
- What's the risk if something is missed?
- Does agent include built-in correction/validation tools?
- Use agents when risk is manageable or reversible

## Current AI Agent Challenges

| Challenge | Why It Matters |
|-----------|----------------|
| Reasoning inconsistency | May succeed once but fail on similar tasks |
| Unpredictable costs | Resource use can spike depending on complexity |
| Tool integration issues | Need well-integrated tools and stable APIs |

## When NOT to Use Agents

Avoid agents for:
- **High-volume, low-margin tasks** (basic chat support)
- **Real-time applications** (instant fraud detection)
- **Zero-error systems** (medical, security decisions)
- **Heavily regulated industries** needing deterministic outcomes

## Guidelines for Managing Risk

### Key Elements of Effective Agent Architecture
1. **Environment**: Digital space where agent operates
2. **Tools**: Interfaces for agent to act or observe
3. **System Prompts**: Rules, goals, behaviors guiding agent

### Deployment Plan Assessment

| Risk Level | Response Strategy |
|------------|-------------------|
| High-stakes, difficult to notice | Human review, multiple validation layers |
| High-stakes, visible | Automated checks, oversight mechanisms |
| Low-stakes | Monitor, user feedback, lightweight validation |

### Best Practices
- Start with read-only access to tools
- Add human approvals for critical steps
- Use staged deployments with monitoring
- Enable comprehensive logging

## Implementing Agents Responsibly

### Phased Deployment
1. **Validate POC**: Low-risk, reversible tasks
2. **Pilot Program**: Moderate-risk tasks under supervision
3. **Production Scaling**: Expand only after safety/performance proven

### Expected Improvements
- More consistent reasoning
- Smarter, leaner architectures
- Advanced monitoring and error detection tools

## Key Takeaways

1. Agents excel at autonomous decision-making but need careful evaluation
2. Use four-step framework: task ambiguity, cost/value, capabilities, failure impact
3. Avoid agents for simple, predictable, or high-risk tasks
4. Current challenges: reliability, cost, tool integration
5. Manage risk with boundaries, logging, human-in-the-loop
6. Start simple, add complexity only after reliable performance confirmed

---

# Tool Calling in AI Agents

## What is Tool Calling?

- Technique to make LLM context-aware of real-time data (databases, APIs)
- Enables LLMs to interact with real-world systems

### Traditional Tool Calling

1. Client application sends message + tool definitions to LLM
2. LLM recommends a tool to call
3. Client application executes the tool
4. Tool response sent back to LLM
5. LLM provides final answer or calls another tool

### Tool Definition Components
- **Name**: Identifier for each tool
- **Description**: How to use the tool, when to use it
- **Input Parameters**: Required parameters for tool call

### Example: Weather in Miami
1. User: "What's the temperature in Miami?"
2. LLM recommends calling weather API
3. Client calls weather API → returns "71 degrees"
4. LLM: "The weather in Miami is 71 degrees"

### Downsides of Traditional Tool Calling
- LLM can hallucinate
- LLM can make up incorrect tool calls

### Embedded Tool Calling
- Library/framework between application and LLM
- Library contains tool definitions AND executes tool calls
- Prevents hallucination by handling execution
- Library retries tool calls if needed

---

# Why AI Needs Tools

## Limitations of LLMs Without Tools

- LLMs are pattern recognition machines
- Don't know real-time facts
- Can't access APIs
- Can't interact with the world
- Like a super-smart person blindfolded with no calculator

## Problems Without Tools

- **Hallucinations**: Model confidently makes things up
- **Math errors**: Guessing from patterns (e.g., 371 × 492 = 158,213 wrong answer)

## Capabilities Tools Provide

1. **Retrieval**: Access private data, company documents, RAG
2. **Multimodal**: Analyze images, audio, other non-text inputs
3. **Extended Memory**: Maintain conversation across sessions
4. **External Systems**: Interact with APIs, software, digital services

## Tools Transform LLMs to Agents

### Agentic Process
1. User asks question
2. LLM selects right tool
3. Tool performs action
4. LLM provides response

### Examples of Tools
- Calculator → precise math
- Web tool → real-time weather, news
- Code tool → write and execute Python
- SQL tool → query business databases

---

# Methods for Creating AI Agents

## Core Components of AI Agents

1. **Perception**: Gather information from environment (user input, web data)
2. **Reasoning/Planning**: Determine best action based on goal
3. **Memory**: Store info to maintain context
4. **Action**: Execute chosen action (response, API call)

### Continuous Cycle
Perceive → Reason → Act → (repeat)

## Two Approaches

### 1. From Scratch

- Manually implement every part of agent lifecycle
- Complete control over behavior
- Coding logic for:
  - Understanding user input
  - Deciding what to do
  - Producing response

#### Example: Weather Agent (From Scratch)
```python
class WeatherAgent:
    def __init__(self, api_key, memory_agent):
        self.api_key = api_key
        self.memory = memory_agent
    def answer(self, city):
        # Call OpenWeather API
        # Store in memory
        # Construct response
        return f"The weather in {city} is..."
```

#### Example: Daily Dish Agent
- Uses TF-IDF vectorization
- Cosine similarity for semantic matching
- Manual threshold for matching (0.08)

#### Router Agent
- Decides which agent handles request
- Simple keyword-based routing:
```python
def router(user_query):
    if "weather" in user_query:
        return weather_agent
    elif "food" in user_query:
        return daily_dish
    else:
        return default_agent
```

### 2. Framework Approach

- Use pre-existing library/framework (e.g., LangChain)
- Pre-built modules for:
  - Orchestrating workflows
  - Managing memory
  - Connecting to tools/APIs

#### Framework Example
```python
agent = Agent(
    tools=[weather_tool],
    memory=ConversationMemory()
)
agent.run("What's the weather in London?")
```

## Comparison: From Scratch vs Framework

| Aspect | From Scratch | Framework |
|--------|-------------|-----------|
| Design | Standalone classes with explicit logic | High-level abstractions |
| Control | Full control over behavior | Control via configuration |
| Routing | Manual keyword checks | Built-in router modules |
| Memory | Explicit storage/retrieval | Framework abstractions |
| Complexity | More boilerplate code | Less boilerplate |
| Learning Value | Excellent for understanding internals | Better for rapid development |
| Flexibility | Highly flexible, custom logic | Bounded by framework features |
| Debugging | Easier to trace issues | May need framework knowledge |
| Scalability | Requires custom code | Designed to scale naturally |

## Key Takeaways

1. Tool calling enables LLMs to access real-time data and APIs
2. Tools transform LLMs from "guessers" to intelligent agents
3. Two approaches: build from scratch (learning) or use framework (speed)
4. Router agents coordinate multiple specialized agents
5. Framework approach reduces complexity but offers less control

---

# Pros and Cons of AI Agent Frameworks vs. From Scratch

## Benefits of Using a Framework

### 1. Rapid Development
- Ready-made structure and pre-built components
- Connect to different LLMs, manage memory, call external tools
- Significantly reduces boilerplate code
- Build functional agent much faster

### 2. Simplified Complexity
- Abstracts difficult challenges:
  - Orchestrating multi-step workflows
  - Handling asynchronous processes
  - Managing conversational memory
- Easier to create complex, multi-agent systems
- Agents can communicate and collaborate

### 3. Community and Support
- Active communities and extensive documentation
- Wealth of examples and tutorials
- Easy to find solutions and stay updated

### 4. Reduced Overhead
- Built-in tools for debugging, monitoring, tracing
- Valuable visibility into agent decision-making
- Major challenge when starting from scratch

## Challenges of Building from Scratch

### 1. High Complexity and Effort
- Massive undertaking for production-ready agent
- Responsible for everything:
  - Robust architecture design
  - Managing context windows
  - Implementing error handling
  - Integrating APIs and tools
- Requires deep expertise in software engineering and AI

### 2. Time and Resource Intensive
- Very long development time
- Reinventing wheel on common features
- Drain on resources and project delays

### 3. Difficulty in Debugging
- Without framework's observability → "black box" problem
- Hard to pinpoint unexpected behavior
- Can't easily trace thought process, tool calls, data flow

### 4. Security and Maintenance
- Handle all security aspects:
  - Data encryption
  - Access control
- Ongoing maintenance of:
  - Agent logic
  - Dependencies
  - Performance as LLMs/APIs evolve

## Summary: Choosing the Right Approach

| Use Framework When... | Build From Scratch When... |
|----------------------|---------------------------|
| Rapid prototyping | Highly specialized application |
| Complex tasks need simplification | Complete control required |
| Benefiting from community | Custom architecture needed |
| Standard features suffice | Unique requirements |

**Key Point**: Framework is excellent for speed and community support. From scratch is better for complete control but requires significant time and expertise investment.

---

# Module Summary and Cheat Sheet

## Key Takeaways from Module

1. **AI Agents** are autonomous systems that perceive, reason, and act to achieve goals
2. **Not every problem needs agents** - simple LLM calls or workflows work for predictable tasks
3. **Agents excel** at complex tasks requiring flexibility, decision-making, and tool interaction
4. **Agents differ** from traditional AI by combining reasoning, memory, and tool use in iterative cycle
5. **Tool calling** enables LLMs to access real-time data, APIs, databases, code execution
6. **Traditional vs Embedded** tool calling - embedded reduces hallucinations via library
7. **Limitations**: reliability concerns, higher costs, need for human oversight
8. **Agent design** requires clear boundaries, thoughtful tools, monitoring
9. **Two approaches**: from scratch (learning) vs frameworks (speed)

---

## Cheat Sheet: AI Agents - Weather and Daily Dish

### WeatherAgent Components

```python
class WeatherAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.openweathermap.org/data/2.5/weather"
        self.memory = Memory()
    
    def answer(self, query):
        city = self.extract_city(query)
        if not city:
            return "Please specify a city for weather information."
        return self.get_weather(city)
    
    def get_weather(self, city):
        params = {"q": city, "appid": self.api_key, "units": "metric"}
        response = requests.get(self.url, params=params)
        data = response.json()
        return self.format_weather_response(city, data)
    
    def extract_city(self, query):
        # Pattern matching for city extraction
        patterns = ['weather in (\w+)', 'temperature in (\w+)', ...]
        # Returns city name or None
    
    def format_weather_response(self, city, data):
        previous = self.memory.recall(city)
        current_temp = data["main"]["temp"]
        # Store and compare with previous
        return f"The weather in {city} is {description} with {current_temp}°C"
```

### Memory Class

```python
class Memory:
    def __init__(self):
        self.storage = {}
    
    def store(self, key, value):
        self.storage[key] = value
    
    def recall(self, key):
        return self.storage.get(key, None)
    
    def clear(self, key=None):
        # Clear specific key or all
```

### DailyDishAgent Components

```python
class DailyDishAgent:
    def __init__(self, questions, answers):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000
        )
        self.doc_vectors = self.vectorizer.fit_transform(questions)
    
    def answer(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        best_idx = similarities.argmax()
        if similarities[best_idx] >= 0.08:
            return self.answers[best_idx]
        return "I don't have information about that."
```

### AgentRouter

```python
class AgentRouter:
    def __init__(self, weather_agent, daily_dish_agent):
        self.weather_agent = weather_agent
        self.daily_dish_agent = daily_dish_agent
        self.weather_keywords = ["weather", "temperature", "forecast", ...]
    
    def route(self, query):
        query_lower = query.lower()
        for keyword in self.weather_keywords:
            if keyword in query_lower:
                return "weather"
        return "daily_dish"
    
    def answer(self, query):
        route = self.route(query)
        if route == "weather":
            return self.weather_agent.answer(query)
        else:
            return self.daily_dish_agent.answer(query)
```

### Advanced Features

**route_with_confidence**: Calculate confidence scores for each agent
**answer_with_fallback**: Comprehensive error handling
**MonitoredRouter**: Track interactions and performance

### Key Concepts

| Concept | Description |
|---------|-------------|
| Agent | Autonomous system that perceives environment and acts to achieve goals |
| Routing | Directing queries to appropriate specialized agent |
| Memory | Storage for maintaining context across interactions |
| TF-IDF | Term Frequency-Inverse Document Frequency - text to numerical vectors |
| Cosine Similarity | Metric for measuring text similarity |
| Semantic Matching | Meaning-based similarity vs exact keywords |