# Quiz: Fundamentals of AI Agents

## Question 1
Which of the following is a limitation of single monolithic language models?

- They can easily adapt to new tasks without additional data
- They have access to personal information about users
- **They are limited by the data they've been trained on** ✓
- They can automatically connect to external databases

---

## Question 2
What is an example of a compound AI system?

- A single LLM generating text summaries
- A model translating text without any external tools
- **A system where an LLM creates a search query to access a vacation database** ✓
- A model classifying sentiment using only its training data

---

## Question 3
In the context of AI agents, what does "reasoning" capability refer to?

- The ability to call external tools
- Storing conversation history for future sessions
- **The model being prompted to create plans and reason through each step** ✓
- Processing user queries in real-time

---

## Question 4
What are tools in the context of LLM agents?

- Internal model parameters that control behavior
- **External programs that the model can call to help execute solutions** ✓
- Training datasets used to fine-tune the model
- Output formats that the model generates

---

## Question 5
Which of the following is an example of memory in AI agents?

- The model's parameters learned during training
- **The history of conversations between human and agent** ✓
- The prompt provided to the model
- The training data used to create the model

---

## Question 6
What is the ReAct framework?

- A method for training language models
- **A combination of reasoning and acting for AI agents** ✓
- A type of neural network architecture
- A database for storing agent interactions

---

## Question 7
What is the key difference between "think fast" and "think slow" approaches?

- Think fast uses more computational resources
- **Think fast gives the first answer; think slow creates a plan and iterates** ✓
- Think slow is only used for simple tasks
- Think fast requires more memory than think slow

---

## Question 8
Which type of AI system is best for repetitive, multi-step tasks with clear logic?

- Single LLM features
- **Structured workflows** ✓
- Autonomous agents
- Random combination of models

---

## Question 9
What is the primary advantage of autonomous agents over structured workflows?

- Lower cost and simpler implementation
- **High adaptability to handle unforeseen situations** ✓
- Deterministic outputs that are easy to audit
- Faster execution time

---

## Question 10
When should you choose a programmatic (non-agentic) approach over an agentic approach?

- When the task is complex and requires reasoning
- When you need the system to adapt to new scenarios
- **When the problem is narrow and well-defined** ✓
- When you need the system to learn from feedback

---

# Answers

1. **They are limited by the data they've been trained on** - Single models cannot access information they weren't trained on.

2. **A system where an LLM creates a search query to access a vacation database** - This shows a compound AI system with multiple components.

3. **The model being prompted to create plans and reason through each step** - Reasoning is the ability to plan and think through problems.

4. **External programs that the model can call to help execute solutions** - Tools are external APIs or programs the agent can call.

5. **The history of conversations between human and agent** - Memory includes conversation history for personalized interactions.

6. **A combination of reasoning and acting for AI agents** - ReAct combines reasoning (thinking) and acting (tool use).

7. **Think fast gives the first answer; think slow creates a plan and iterates** - Think slow involves planning and iterative problem-solving.

8. **Structured workflows** - Best for predictable, multi-step tasks with clear logic.

9. **High adaptability to handle unforeseen situations** - Agents can adapt to new scenarios dynamically.

10. **When the problem is narrow and well-defined** - Programmatic approach is more efficient for narrow, predictable problems.

---

## Additional Questions

## Question 11
Analyze the scenario: Jamie is considering implementing an AI solution to automate customer service tasks. The tasks require understanding complex queries and providing personalized responses. Which AI system design should Jamie consider for this task?

- **Compound AI system** ✓
- Monolithic AI model
- Single LLM feature
- Programmatic approach

---

## Question 12
Apply the four-step decision framework to determine when to use AI agents. Which step involves evaluating the adaptability requirements of the task?

- Assess task complexity
- Evaluate risk management guidelines
- Determine operational requirements
- **Identify adaptability needs** ✓

---

## Question 13
Select the answer that correctly contrasts monolithic AI models and compound AI systems regarding their structural design.

- **Monolithic AI models are designed as a single, unified system, while compound AI systems comprise interconnected modules.** ✓
- Both monolithic AI models and compound AI systems are designed as single, unified systems.
- Both monolithic AI models and compound AI systems are interconnected modules.
- Compound AI systems are designed as a single, unified system, while monolithic AI models are made of interconnected modules.

---

# Answers (Additional)

11. **Compound AI system** - Complex, personalized customer service requires multiple components (LLM + tools + memory).

12. **Identify adaptability needs** - This step evaluates whether task requires dynamic adaptation vs. predictable processes.

13. **Monolithic AI models are designed as a single, unified system, while compound AI systems comprise interconnected modules** - Contrast between single model vs. multi-component system.

---

## Tool Calling and Methods Questions

## Question 14
What are the main components of a tool definition in tool calling?

- API endpoint and response format
- **Name, description, and input parameters** ✓
- Function code and return type
- Server location and authentication

## Question 15
What is a key advantage of embedded tool calling over traditional tool calling?

- Faster response time from the LLM
- **Prevents hallucination by handling tool execution** ✓
- Requires less setup time
- Works with any LLM without configuration

## Question 16
Which of the following is NOT a core component of an AI Agent?

- Perception
- Reasoning/Planning
- **Code generation** ✓
- Action

## Question 17
In the "from scratch" approach to building AI agents, who is responsible for implementing the logic for each part of the agent's cycle?

- The LLM itself
- **The developer** ✓
- The framework
- The user

## Question 18
What is the primary purpose of a Router Agent in a multi-agent system?

- To store conversation history
- **To decide which agent should handle a given request** ✓
- To execute API calls
- To generate final responses

## Question 19
Which approach gives developers full control over agent behavior but requires more boilerplate code?

- Framework-based approach
- **From scratch approach** ✓
- API-based approach
- Template-based approach

## Question 20
What is a key limitation of LLMs without tools?

- They cannot understand natural language
- **They can hallucinate and make things up** ✓
- They cannot process text
- They lack creativity

---

# Answers (Tool Calling and Methods)

14. **Name, description, and input parameters** - Tool definition includes how to call the tool and when to use it.

15. **Prevents hallucination by handling tool execution** - Library takes care of execution and retries.

16. **Code generation** - Core components are perception, reasoning, memory, and action.

17. **The developer** - From scratch means manually implementing all agent logic.

18. **To decide which agent should handle a given request** - Router analyzes input and routes to appropriate agent.

19. **From scratch approach** - Explicit control but more code to maintain.

20. **They can hallucinate and make things up** - Without tools, LLMs guess from patterns leading to errors.

---

## Framework vs. From Scratch Questions

## Question 21
What is a key benefit of using an AI Agent framework like LangChain or CrewAI?

- Complete control over every aspect of agent behavior
- **Rapid development with pre-built components** ✓
- Direct access to all LLM internals
- No need to understand AI concepts

## Question 22
Which of the following is a challenge of building an AI Agent from scratch?

- Limited community support
- **Debugging becomes a "black box" problem** ✓
- Less flexibility in agent design
- Requires less technical expertise

## Question 23
What overhead do frameworks often include that helps with agent development?

- Network infrastructure
- **Built-in tools for debugging, monitoring, and tracing** ✓
- Physical hardware management
- Data storage solutions

## Question 24
When is building from scratch the better approach?

- For rapid prototyping
- **For highly specialized applications requiring complete control** ✓
- When you have limited time
- When working with standard features only

## Question 25
What aspect of agent maintenance is a challenge when building from scratch?

- Using pre-built templates
- Handling all security aspects and ongoing updates as LLMs evolve ✓
- Choosing which framework to use
- Writing documentation

---

## Additional Tool Calling Questions

## Question 26
What is one key advantage of using embedded tool calling over traditional tool calling in LLMs?

- Embedded tool calling eliminates the need for tool definitions.
- Embedded tool calling requires less initial setup than traditional tool calling.
- **Embedded tool calling reduces hallucinations by managing tool execution within a dedicated library.** ✓
- Embedded tool calling allows LLMs to operate offline.

---

## Question 27
How do tools transform LLMs into intelligent agents capable of interacting with real-world data?

- Tools allow LLMs to function without any human intervention.
- **Tools enable LLMs to perform actions and access real-time data, similar to a traditional computer program.** ✓
- Tools increase the LLM's ability to generate text more quickly.
- Tools simplify the LLM's internal processing mechanisms.

---

## Question 28
What is the role of a library in embedded tool calling?

- The library eliminates the need for external data sources.
- **The library manages tool definitions and execution, reducing LLM hallucinations.** ✓
- The library simplifies the tool definition process.
- The library increases the LLM's processing speed.

---

## Question 29
Alex is developing an AI assistant that needs to provide real-time weather updates. Which tool calling method should Alex consider to reduce hallucinations and improve reliability?

- Traditional tool calling
- Using the no-tool calling method
- **Embedded tool calling** ✓
- A hybrid approach combining both methods

---

## Question 30
Jamie is tasked with developing a chatbot that can perform complex calculations reliably. What tool should Jamie consider integrating to improve the chatbot's accuracy?

- A natural language processing tool to enhance text generation.
- An image recognition tool to process visual inputs.
- A data visualization tool to present results graphically.
- **A calculator tool to perform accurate mathematical computations.** ✓

---

# Answers (Tool Calling Scenarios)

26. **Embedded tool calling reduces hallucinations by managing tool execution within a dedicated library.** - Library handles execution and retries.

27. **Tools enable LLMs to perform actions and access real-time data, similar to a traditional computer program.** - Tools transform LLMs into agents that can interact with real world.

28. **The library manages tool definitions and execution, reducing LLM hallucinations.** - Key role of library in embedded tool calling.

29. **Embedded tool calling** - Best for reducing hallucinations and improving reliability.

30. **A calculator tool to perform accurate mathematical computations.** - Tools like calculator improve accuracy beyond LLM guesses.

---

# Answers (Framework vs. From Scratch)

21. **Rapid development with pre-built components** - Frameworks provide ready-made structure for common tasks.

22. **Debugging becomes a "black box" problem** - Without framework's observability, hard to trace issues.

23. **Built-in tools for debugging, monitoring, and tracing** - Provides visibility into agent decisions.

24. **For highly specialized applications requiring complete control** - When custom architecture is needed.

25. **Handling all security aspects and ongoing updates as LLMs evolve** - From scratch requires handling all security and maintenance.

---

## Core Concepts Questions

## Question 31
What is the primary function of an AI agent?

- Only store large amounts of data without performing actions.
- **Perceive its environment and act to achieve goals.** ✓
- Replace human intelligence completely in all tasks.
- Randomly perform actions without any input or goals.

---

## Question 32
Which feature differentiates an AI agent from a simple software program?

- It can only run on specialized hardware.
- It passively waits for instructions without analyzing inputs.
- **Perceive, reason, and act autonomously.** ✓
- It requires constant human supervision for every task.

---

## Question 33
A customer asks an AI assistant for the latest stock price of a company. How should the LLM respond using tool calling?

- Refuse to answer since it doesn't know.
- Guess the stock price based on its training data.
- Use a pre-stored static database of prices.
- **Call a stock market API to get the current price.** ✓

---

## Question 34
A user wants to translate a paragraph into French using an AI model. What is the best way to handle this with tool calling?

- **Use a translation API to get an accurate result.** ✓
- Ignore the request because the model cannot do it natively.
- Invoke a language translation tool integrated with the LLM.
- Translate it manually within the model's memory.

---

## Question 35
An AI assistant is asked to book a flight for a user. Without tool access, it can only guess flight details. How does using tools improve its response?

- It refuses to respond to the request.
- It continues to guess based on past data.
- **It accesses a booking system to provide real flight options.** ✓
- It makes up flight options based on similar past requests.

---

# Answers (Core Concepts)

31. **Perceive its environment and act to achieve goals** - AI agents perceive environment and take actions to achieve specific goals.

32. **Perceive, reason, and act autonomously** - Key feature differentiating agents from simple programs is autonomous perception, reasoning, and action.

33. **Call a stock market API to get the current price** - Tool calling allows LLM to access real-time data via APIs.

34. **Use a translation API to get an accurate result** - Translation tool provides accurate results beyond LLM's capabilities.

35. **It accesses a booking system to provide real flight options** - Tools enable agents to interact with real-world systems for accurate information.