# LangChain 实践指南

## 🔧 快速开始模板

### 项目初始化

```bash
# 创建项目目录
mkdir my_langchain_project
cd my_langchain_project

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install langchain langchain-openai langchain-community python-dotenv
pip install ipython jupyter  # 可选，用于交互式开发

# 配置环境变量
cat > .env << EOF
OPENAI_API_KEY=your-api-key-here
EOF
```

### 项目结构
```
my_project/
├── .env                    # 环境变量
├── .gitignore             # Git 忽略
├── requirements.txt       # 依赖列表
├── README.md             # 项目说明
├── src/
│   ├── __init__.py
│   ├── chains.py         # 链定义
│   ├── prompts.py        # 提示模板
│   ├── tools.py          # 工具定义
│   ├── agents.py         # 代理实现
│   └── utils.py          # 工具函数
├── tests/
│   ├── test_chains.py
│   └── test_agents.py
└── examples/
    ├── basic_rag.py
    ├── agent_example.py
    └── langgraph_example.py
```

---

## 📝 常用代码模板

### 1. 基础 LLM 调用

```python
# src/chains.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

def create_basic_chain():
    """创建最基础的 LLM 链"""
    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template(
        "你是一个友好的助手。\n问题: {question}"
    )

    output_parser = StrOutputParser()

    # 使用 LCEL 组合
    chain = prompt | model | output_parser

    return chain

# 使用
if __name__ == "__main__":
    chain = create_basic_chain()
    result = chain.invoke({"question": "什么是 Python?"})
    print(result)
```

### 2. 多轮对话

```python
# src/chains.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def create_conversation_chain():
    """创建支持多轮对话的链"""
    model = ChatOpenAI()

    # 构建对话模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的编程助手"),
        ("user", "{input}")
    ])

    return prompt | model

def chat_with_history():
    """演示多轮对话"""
    chain = create_conversation_chain()
    messages = []

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # 添加消息
        result = chain.invoke({"input": user_input})
        print(f"Assistant: {result.content}")

# 使用
if __name__ == "__main__":
    chat_with_history()
```

### 3. 工具定义与使用

```python
# src/tools.py
from langchain_core.tools import tool
from typing import Any
import requests
import json

@tool
def search_web(query: str) -> str:
    """
    搜索网络信息

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    # 这是一个模拟实现
    return f"关于 '{query}' 的搜索结果..."

@tool
def calculate(expression: str) -> float:
    """
    计算数学表达式

    Args:
        expression: 数学表达式，如 "2+3*4"

    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_current_weather(location: str) -> dict:
    """
    获取当前天气（模拟实现）

    Args:
        location: 地点名称

    Returns:
        天气信息
    """
    weather_data = {
        "北京": {"温度": 20, "天气": "晴朗"},
        "上海": {"温度": 18, "天气": "多云"},
        "深圳": {"温度": 25, "天气": "晴朗"}
    }
    return weather_data.get(location, {"温度": "未知", "天气": "无数据"})

def get_tools():
    """获取所有工具列表"""
    return [search_web, calculate, get_current_weather]

# 测试工具
if __name__ == "__main__":
    print("搜索:", search_web.invoke("Python"))
    print("计算:", calculate.invoke("2+3*4"))
    print("天气:", get_current_weather.invoke("北京"))
```

### 4. RAG 系统

```python
# src/rag.py
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, doc_dir: str):
        """加载文档"""
        loader = DirectoryLoader(doc_dir, glob="*.md")
        documents = loader.load()
        return documents

    def process_documents(self, documents):
        """处理文档"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "，", ""]
        )
        chunks = splitter.split_documents(documents)
        return chunks

    def build_vectorstore(self, doc_dir: str):
        """构建向量存储"""
        documents = self.load_documents(doc_dir)
        chunks = self.process_documents(documents)

        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )

    def setup_qa_chain(self):
        """设置 QA 链"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""使用以下文件回答问题。

文件内容:
{context}

问题: {question}

答案:"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def query(self, question: str) -> str:
        """查询"""
        if not self.qa_chain:
            raise ValueError("Please setup QA chain first")
        return self.qa_chain.run(question)

# 使用
if __name__ == "__main__":
    rag = RAGSystem()
    rag.build_vectorstore("./documents")
    rag.setup_qa_chain()

    answer = rag.query("文档中讲了什么?")
    print(answer)
```

### 5. 代理实现

```python
# src/agents.py
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
import operator

# 定义工具
@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索 '{query}' 的结果..."

@tool
def calculator(expression: str) -> str:
    """计算表达式"""
    return str(eval(expression))

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def create_agent():
    """创建工具调用代理"""
    model = ChatOpenAI(model="gpt-4")
    tools = [search, calculator]

    # 绑定工具
    model_with_tools = model.bind_tools(tools)

    # 定义节点函数
    def agent(state: AgentState):
        """代理节点"""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        """决定是否继续"""
        messages = state["messages"]
        last_message = messages[-1]

        # 如果有工具调用，继续
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # 构建图
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# 使用
if __name__ == "__main__":
    graph = create_agent()

    result = graph.invoke({
        "messages": [HumanMessage(content="计算 2+3*4")]
    })

    print(result["messages"][-1].content)
```

### 6. LangGraph 工作流

```python
# src/langgraph_example.py
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def create_simple_graph():
    """创建简单的 LangGraph 工作流"""
    model = ChatOpenAI()

    def node_1(state: State):
        """第一个节点"""
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    def node_2(state: State):
        """第二个节点"""
        messages = state["messages"]
        # 进行某些处理
        return state

    def route(state: State) -> str:
        """路由函数"""
        messages = state["messages"]
        last_message = messages[-1]

        if "问题" in last_message.content:
            return "node_1"
        return "node_2"

    # 构建图
    graph = StateGraph(State)
    graph.add_node("node_1", node_1)
    graph.add_node("node_2", node_2)

    graph.add_edge(START, "node_1")
    graph.add_conditional_edges(
        "node_1",
        route,
        {"node_1": "node_1", "node_2": "node_2"}
    )
    graph.add_edge("node_2", END)

    return graph.compile()

# 使用
if __name__ == "__main__":
    graph = create_simple_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="这是一个问题")]
    })
    print(result)
```

---

## 🧪 测试与评估

### 单元测试模板

```python
# tests/test_chains.py
import unittest
from src.chains import create_basic_chain

class TestChains(unittest.TestCase):
    def setUp(self):
        self.chain = create_basic_chain()

    def test_basic_chain_returns_string(self):
        """测试基础链返回字符串"""
        result = self.chain.invoke({"question": "什么是 Python?"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_chain_handles_edge_cases(self):
        """测试边界情况"""
        result = self.chain.invoke({"question": ""})
        self.assertIsInstance(result, str)

if __name__ == "__main__":
    unittest.main()
```

### 评估工具

```python
# src/evaluation.py
from typing import List, Dict
import json

class RAGEvaluator:
    """RAG 系统评估工具"""

    @staticmethod
    def evaluate_retrieval(retrieved_docs: List[str],
                          expected_docs: List[str]) -> Dict:
        """评估检索质量"""
        if not retrieved_docs:
            return {"precision": 0, "recall": 0}

        retrieved_set = set(retrieved_docs)
        expected_set = set(expected_docs)

        intersection = retrieved_set & expected_set

        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
        recall = len(intersection) / len(expected_set) if expected_set else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * (precision * recall) / (precision + recall)
                  if (precision + recall) > 0 else 0
        }

    @staticmethod
    def evaluate_generation(generated: str,
                           reference: str) -> Dict:
        """评估生成质量"""
        # 简单的相似性评估
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())

        overlap = gen_words & ref_words
        similarity = len(overlap) / len(gen_words) if gen_words else 0

        return {
            "similarity": similarity,
            "length_ratio": len(generated) / len(reference)
                           if reference else 0
        }

# 使用
if __name__ == "__main__":
    evaluator = RAGEvaluator()

    metrics = evaluator.evaluate_retrieval(
        ["doc1", "doc2"],
        ["doc1", "doc3"]
    )
    print(metrics)
```

---

## 🐛 常见问题与解决方案

### 问题 1: API 密钥错误

```python
# 解决方案 1: 使用环境变量
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

# 解决方案 2: 直接传递
model = ChatOpenAI(api_key="your-key")
```

### 问题 2: Token 限制

```python
# 解决方案：使用更小的模型或分割文档
from langchain_openai import ChatOpenAI

# 选择较小的模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 或限制输出长度
model = ChatOpenAI(max_tokens=500)
```

### 问题 3: 向量存储性能

```python
# 解决方案：使用有效的搜索参数
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 只返回前 3 个结果
)

# 或使用 MMR 搜索
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

### 问题 4: 处理超长输入

```python
# 解决方案：使用文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(long_text)
```

---

## 📊 调试技巧

### 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain")
logger.setLevel(logging.DEBUG)

# 现在所有 LangChain 操作都会输出详细日志
model = ChatOpenAI()
result = model.invoke("test")
```

### 追踪执行步骤

```python
from langchain.callbacks import StdOutCallbackHandler

# 方法 1: 全局回调
callbacks = [StdOutCallbackHandler()]

# 方法 2: 链级回调
chain.invoke(input, config={"callbacks": callbacks})
```

### 使用 LangSmith

```python
import os

# 启用 LangSmith 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"

# 现在所有操作都会被追踪到 LangSmith dashboard
```

---

## 🚀 性能优化

### 缓存优化

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 启用缓存
set_llm_cache(InMemoryCache())

# 重复调用会使用缓存结果
model = ChatOpenAI()
result1 = model.invoke("same input")
result2 = model.invoke("same input")  # 使用缓存
```

### 批处理优化

```python
# 使用 batch 方法而不是循环
inputs = ["input1", "input2", "input3"]

# 低效：循环
results = []
for inp in inputs:
    results.append(model.invoke(inp))

# 高效：批处理
results = model.batch(inputs)
```

### 异步优化

```python
import asyncio

async def process_multiple():
    # 异步并发处理
    tasks = [
        model.ainvoke("input1"),
        model.ainvoke("input2"),
        model.ainvoke("input3")
    ]
    results = await asyncio.gather(*tasks)
    return results

# 运行
results = asyncio.run(process_multiple())
```

---

## 🔑 最佳实践

### 1. 错误处理

```python
try:
    result = chain.invoke(input)
except Exception as e:
    print(f"Error: {e}")
    # 提供降级方案
    result = get_default_response()
```

### 2. 提示优化

```python
# 不好：模糊的提示
"翻译这个文本"

# 好：清晰的提示
"""将以下英文文本翻译成中文。
确保：
1. 准确表达原意
2. 自然流畅
3. 保留技术术语

文本：{text}"""
```

### 3. 模型选择

```python
# 成本优化：选择合适的模型
model = ChatOpenAI(model="gpt-3.5-turbo")  # 便宜

# 质量优化：选择更强的模型
model = ChatOpenAI(model="gpt-4")  # 贵但更好

# 平衡：
model = ChatOpenAI(model="gpt-4-turbo")  # 折中方案
```

### 4. 参数调整

```python
# 创意任务：高温度
model = ChatOpenAI(temperature=0.8)

# 分析任务：低温度
model = ChatOpenAI(temperature=0.1)

# 平衡：
model = ChatOpenAI(temperature=0.5)
```

---

## 📚 推荐项目案例

### 项目 1: 企业知识库 QA

```python
# 完整实现骨架
class EnterpriseKnowledgeBase:
    def __init__(self):
        self.rag = RAGSystem()

    def load_knowledge(self, path):
        self.rag.build_vectorstore(path)

    def answer_question(self, question):
        return self.rag.query(question)
```

### 项目 2: AI 代码审查工具

```python
class CodeReviewer:
    def __init__(self):
        self.model = ChatOpenAI()
        self.tools = [analyze_complexity, check_bugs]

    def review_code(self, code):
        # 使用代理分析代码
        pass
```

### 项目 3: 多语言文档翻译

```python
class MultilingualTranslator:
    def __init__(self, target_languages):
        self.languages = target_languages

    def translate_document(self, doc_path):
        # 并行翻译到多个语言
        pass
```

---

开始实践吧！🚀
