# LearnLangChain

A comprehensive 20-week structured learning guide for mastering **LangChain**, a framework for building intelligent applications with Large Language Models (LLMs).

## Overview

This repository provides a complete curriculum for learning LangChain from fundamentals to production-ready applications. Whether you're new to LLM development or looking to deepen your expertise, this guide will take you through a progressive learning journey.

### What is LangChain?

LangChain is a framework that simplifies building applications powered by large language models. It provides abstractions for working with language models, chains of operations, retrieval-augmented generation (RAG), agents, and graph-based orchestration.

## 📚 Repository Contents

The repository is organized around three complementary documentation files:

1. **LANGCHAIN_KNOWLEDGE_MAP.md** - The "What & Why"
   - Comprehensive knowledge architecture
   - Detailed explanations of all LangChain concepts
   - Best practices and design patterns
   - Use this as your reference guide

2. **PRACTICE_GUIDE.md** - The "How To"
   - Practical code templates and examples
   - Implementation patterns for common tasks
   - Runnable code snippets
   - Use this for hands-on learning

3. **STUDY_SCHEDULE.md** - The "When & What Order"
   - Week-by-week learning progression
   - Specific tasks and milestones
   - Code exercises and checkpoints
   - Use this to track your learning journey

4. **CLAUDE.md** - Development guidance
   - Development standards and conventions
   - Code style expectations
   - Common patterns and pitfalls

## 🎯 Learning Path (7 Stages)

```
Week 1-2   → Foundation
Week 3-4   → Chains & Composition
Week 5-7   → Data & RAG
Week 8-10  → Tools & Agents
Week 11-13 → LangGraph
Week 14-16 → Advanced Applications
Week 17-20 → Capstone Projects
```

### Stage 1: Foundation (Weeks 1-2)
- Core LangChain concepts
- LLM integration and configuration
- Prompts and messages
- Output parsing
- **Skills gained**: Can create basic LLM queries and parse responses

### Stage 2: Chains & Composition (Weeks 3-4)
- LCEL (LangChain Expression Language)
- Runnable interface and pipe operators
- Composing complex chains
- Error handling and retries
- **Skills gained**: Can build sequential chains and handle errors gracefully

### Stage 3: Data & RAG (Weeks 5-7)
- Document loading and processing
- Text splitting and chunking
- Embeddings and vector stores
- Retrieval-augmented generation
- **Skills gained**: Can build RAG systems that retrieve and generate based on documents

### Stage 4: Tools & Agents (Weeks 8-10)
- Tool definition and calling
- Agent frameworks and loops
- ReAct pattern implementation
- Tool-calling agents
- **Skills gained**: Can build agents that use tools autonomously

### Stage 5: LangGraph (Weeks 11-13)
- Graph-based orchestration
- State management and MessagesState
- Node and edge definitions
- Conditional routing
- **Skills gained**: Can design complex multi-step workflows with state management

### Stage 6: Advanced Applications (Weeks 14-16)
- Multi-agent systems
- Streaming and async operations
- Caching and performance optimization
- Evaluation and monitoring
- **Skills gained**: Can build production-scale applications

### Stage 7: Capstone Projects (Weeks 17-20)
- End-to-end application development
- Integration of all concepts
- Deployment considerations
- Real-world scenarios
- **Skills gained**: Portfolio-ready projects demonstrating full mastery

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Basic understanding of Python
- API keys for LLM providers (OpenAI recommended for learning)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LearnLangChain.git
   cd LearnLangChain
   ```

2. **Install dependencies**
   ```bash
   pip install langchain langchain-openai langchain-community langgraph python-dotenv
   ```

3. **Configure environment**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

4. **Start learning**
   - Begin with Week 1 in **STUDY_SCHEDULE.md**
   - Reference concepts in **LANGCHAIN_KNOWLEDGE_MAP.md**
   - Use code examples from **PRACTICE_GUIDE.md**

## 📖 How to Use This Repository

### For Self-Directed Learning
1. Open **STUDY_SCHEDULE.md** and follow the week-by-week progression
2. For each week, read the corresponding sections in **LANGCHAIN_KNOWLEDGE_MAP.md**
3. Implement the code examples from **PRACTICE_GUIDE.md**
4. Complete the checkpoint exercises

### For Classroom Use
- Use **LANGCHAIN_KNOWLEDGE_MAP.md** as lecture material
- Have students follow **STUDY_SCHEDULE.md** for assignments
- Use **PRACTICE_GUIDE.md** for live coding demonstrations

### As a Reference
- Use **LANGCHAIN_KNOWLEDGE_MAP.md** to understand concepts
- Use **PRACTICE_GUIDE.md** to find implementation patterns
- Use **CLAUDE.md** for development standards

## 🔑 Key Concepts at a Glance

### LCEL (LangChain Expression Language)
Modern, declarative way to compose components using the pipe operator (`|`)

```python
chain = prompt_template | llm_model | output_parser
result = chain.invoke({"variable": value})
```

### Runnable Interface
Core abstraction with methods: `invoke()`, `batch()`, `stream()`, `ainvoke()`, `astream()`, `abatch()`

### RAG (Retrieval-Augmented Generation)
Combines document retrieval with generation for more accurate, context-aware responses

```
Documents → Embeddings → Vector Store → Retrieval → Generation → Response
```

### Agents
Autonomous systems that can plan, act, and observe to complete tasks using tools

```
Plan → Use Tools → Observe Results → Repeat until Done
```

### LangGraph
Graph-based orchestration for complex workflows with state management

## 📚 Recommended Learning Approach

### Option 1: Linear (Recommended for beginners)
- Follow weeks 1-20 in order
- Build on foundational concepts
- Estimated time: 20 weeks at 7-10 hours/week

### Option 2: Modular (For experienced developers)
- Skip foundation stages based on your level
- Focus on specific areas of interest
- Combine multiple stages in parallel

### Option 3: Project-Based
- Start with a capstone project you want to build
- Jump to relevant stages as needed
- Use guides as reference materials

## 🎓 Learning Objectives

By completing this curriculum, you will be able to:

- ✅ Build LLM applications from scratch
- ✅ Implement retrieval-augmented generation (RAG) systems
- ✅ Create autonomous agents that use tools
- ✅ Design complex workflows with LangGraph
- ✅ Optimize performance and handle errors
- ✅ Deploy production-ready LLM applications

## ⚠️ Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| Context window overflow | Use text splitters with appropriate chunk_size |
| Poor retrieval quality | Implement query transformation and fusion retrieval |
| High token costs | Cache responses, use cheaper models for preprocessing |
| State management issues | Use reducer functions (operator.add) |
| Unreliable agents | Implement fallback chains and error recovery |

## 📦 Dependencies

Core packages used throughout the curriculum:

```
langchain >= 0.1.x          # Core framework
langchain-openai            # OpenAI integration
langchain-community         # Third-party integrations
langgraph >= 0.1.x          # Graph orchestration
python-dotenv              # Environment management
pydantic >= 2.0             # Data validation
```

## 🔄 Modern vs Legacy Approaches

This curriculum uses **modern patterns**:

| Concept | Modern Approach | Legacy Approach |
|---------|-----------------|-----------------|
| Chain composition | LCEL with `\|` operator | LLMChain class |
| Retrieval | RAG Chain | RetrievalQA |
| Agents | Tool-calling + StateGraph | AgentExecutor |

## 🌐 Additional Resources

- [Official LangChain Documentation](https://python.langchain.com)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs)

## 📋 Progress Tracking

Use **STUDY_SCHEDULE.md** to track your progress through the 20-week curriculum. Each week includes:
- Learning objectives
- Key concepts to study
- Code exercises to complete
- Checkpoint assessments

## 💡 Tips for Success

1. **Code along**: Don't just read the examples, run them yourself
2. **Experiment**: Modify examples and try different approaches
3. **Debug systematically**: Use print statements and error messages to understand issues
4. **Build projects**: Apply concepts to real problems you care about
5. **Review regularly**: Revisit earlier concepts as you progress
6. **Join communities**: Connect with other LangChain learners

## 🤝 Contributing

We welcome contributions to improve this learning guide:
- Report errors or unclear explanations
- Suggest additional examples or use cases
- Share your own learning experiences
- Propose new capstone projects

## 📄 License

This learning guide is open source and available for educational use.

## ❓ FAQ

**Q: How long will this take?**
A: Approximately 20 weeks at 7-10 hours per week. You can adjust based on your pace and experience level.

**Q: Do I need LLM/AI experience?**
A: No! This guide is designed for beginners. We start from the fundamentals.

**Q: Can I skip sections?**
A: If you already have experience, you can skip foundational stages. However, we recommend going through them for completeness.

**Q: What if I get stuck?**
A: Review the relevant section in LANGCHAIN_KNOWLEDGE_MAP.md and try the PRACTICE_GUIDE.md examples.

**Q: Are there solutions to the exercises?**
A: Check PRACTICE_GUIDE.md for complete examples you can reference and learn from.

---

**Ready to start?** Begin with [Week 1 in STUDY_SCHEDULE.md](./STUDY_SCHEDULE.md)!
