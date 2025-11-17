# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LearnLangChain** is a comprehensive 20-week structured learning guide for mastering LangChain, a framework for building LLM-powered applications. The repository contains three main documentation files:

- **LANGCHAIN_KNOWLEDGE_MAP.md**: Detailed knowledge architecture covering 7 learning stages
- **PRACTICE_GUIDE.md**: Practical templates and code examples for implementation
- **STUDY_SCHEDULE.md**: Granular weekly breakdown with specific tasks and code exercises

## Core Architecture & Learning Structure

### Knowledge Organization (7 Stages)

The project is organized into progressive learning stages that build upon each other:

1. **Foundation (Weeks 1-2)**: Core concepts, LLM integration, prompts, messages, output parsing
2. **Chains & Composition (Weeks 3-4)**: Runnable interface, LCEL language, error handling
3. **Data & RAG (Weeks 5-7)**: Document loading, embeddings, vector stores, retrieval-augmented generation
4. **Tools & Agents (Weeks 8-10)**: Tool definition, tool calling, agent frameworks, ReAct pattern
5. **LangGraph (Weeks 11-13)**: Graph-based orchestration, state management, conditional routing
6. **Advanced Applications (Weeks 14-16)**: Multi-agent systems, streaming, async, evaluation
7. **Capstone Projects (Weeks 17-20)**: End-to-end implementation of production-ready systems

### Key Concepts & Patterns

**LCEL (LangChain Expression Language)**
- Pipe operator `|` for composing Runnables
- Declarative syntax for building chains
- Built-in support for batch, stream, and async operations
- Foundation for all chain composition in modern LangChain

**Runnable Interface**
- Core abstraction that all components implement
- Methods: `invoke()`, `batch()`, `stream()`, `ainvoke()`, `astream()`, `abatch()`
- Consistent interface across prompts, models, parsers, and custom functions

**RAG Architecture**
- Document loading → Text splitting → Embeddings → Vector storage → Retrieval → Generation
- Performance heavily depends on retrieval quality and context ranking
- Query transformation and multi-hop retrieval improve effectiveness

**State Management in LangGraph**
- TypedDict-based state definitions
- MessagesState for conversation history patterns
- Reducer functions (typically operator.add) for state merging
- Node functions that take and return state

**Agent Patterns**
- Tool-calling agents: Modern approach using model tool_calls
- ReAct agents: Think-action-observe cycle
- Multi-agent supervisor patterns for complex tasks
- Memory integration for context awareness

## Development Standards

### Code Style & Organization

```
Expected structure for any implementation:
- Type hints throughout (Python 3.10+ syntax)
- Docstrings for all functions/classes
- Environment variables for API keys (never hardcode)
- Async/await patterns for I/O-bound operations
- Structured error handling with specific exception types
```

### Common Code Patterns

**Basic Chain** (LCEL composition):
```python
chain = prompt_template | llm_model | output_parser
result = chain.invoke({"variable": value})
```

**RAG System** (retrieval + generation):
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm_model
    | output_parser
)
```

**Tool-Calling Agent**:
```python
model_with_tools = model.bind_tools(tools)
# Use StateGraph with agent node and tool node
# Conditional routing based on tool_calls
```

**LangGraph State Pattern**:
```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator

class MyState(TypedDict):
    messages: Annotated[list, operator.add]  # Reducer pattern

graph = StateGraph(MyState)
# Add nodes and edges, compile with .compile()
```

### Common Implementation Tasks

When adding new learning materials or code examples:

1. **Document Loading Examples**: Always show both simple (TXT/Markdown) and complex (PDF) cases
2. **Vector Store Setup**: Demonstrate both local (FAISS) and cloud (Pinecone/Weaviate) options
3. **Model Selection**: Compare cost vs. quality trade-offs (gpt-3.5-turbo vs gpt-4)
4. **Error Handling**: Include retry logic and graceful degradation for API calls
5. **Async Patterns**: Show both sync and async versions for I/O operations

## Documentation-First Approach

This repository prioritizes comprehensive documentation over code. The three main guides serve different purposes:

- **LANGCHAIN_KNOWLEDGE_MAP.md**: Use this as the "what and why" reference
- **PRACTICE_GUIDE.md**: Use this for "how to implement" code templates
- **STUDY_SCHEDULE.md**: Use this for "when and in what order" learning progression

When updating guides, maintain consistency across all three files and ensure code examples are tested and executable.

## Key Dependencies & Versions

Core packages (implied from documentation):
- `langchain >= 0.1.x` - Core framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Third-party integrations
- `langgraph >= 0.1.x` - Graph orchestration
- `python-dotenv` - Environment configuration
- `pydantic >= 2.0` - Data validation

Note: The repository doesn't include a requirements.txt. When creating new code examples, specify version constraints for stable reproduction.

## Important Distinctions from Documentation

- **Chain vs LCEL**: Newer code uses LCEL (pipe operator) rather than LLMChain class
- **RetrievalQA vs RAG Chain**: Documentation shows both legacy and modern LCEL approaches
- **Agents**: Modern approach uses tool_calls + StateGraph, not legacy AgentExecutor

## Common Pitfalls & Solutions

1. **Context Window Overflow**: Use text splitters with appropriate chunk_size and overlap
2. **Retrieval Quality**: Implement query transformation and fusion retrieval for complex questions
3. **Token Cost**: Cache responses, use cheaper models for preprocessing, optimize prompts
4. **State Management**: Use reducer functions (operator.add) for proper message accumulation
5. **Tool Reliability**: Implement fallback chains and error recovery in agent workflows

## Testing & Validation Strategy

While the repository is documentation-focused, any code examples should be:
- **Self-contained**: Can run independently with minimal setup
- **Reproducible**: Use deterministic seeds for testing
- **Evaluable**: Include assertions or simple quality checks
- **Error-handling**: Show graceful failure modes

## Progress Tracking

The STUDY_SCHEDULE.md provides week-by-week milestones. When adding new content:
- Map it to a specific week/stage in the 20-week curriculum
- Ensure prerequisites from earlier stages are satisfied
- Include checkpoint items in each section for self-assessment

## Integration with SuperClaude Framework

This repository benefits from:
- **Deep research capabilities** for investigating new LangChain features
- **Backend architect** perspective for RAG and system design
- **Task management** for organizing multi-week learning progression
- **Documentation** mode for maintaining clarity in guides

When working on improvements, consider:
- Use `/sc:research` for validating documentation against latest LangChain releases
- Use `/sc:design` for architecting complex systems (multi-agent, RAG optimization)
- Use `/sc:document` for ensuring technical clarity and completeness
