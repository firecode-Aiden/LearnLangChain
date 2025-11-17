# LangChain 详细学习计划（20 周）

## 📅 整体计划概览

| 阶段 | 周数 | 主题 | 目标 |
|------|------|------|------|
| 第一阶段 | 1-2 | 基础概念 | 理解 LangChain 架构和核心概念 |
| 第二阶段 | 3-4 | 链与组合 | 掌握 LCEL 和 Runnable |
| 第三阶段 | 5-7 | 数据管理与 RAG | 构建完整 RAG 系统 |
| 第四阶段 | 8-10 | 工具与代理 | 实现自主代理 |
| 第五阶段 | 11-13 | LangGraph | 掌握图形编排 |
| 第六阶段 | 14-16 | 高级应用 | 实现多代理、流式、异步 |
| 第七阶段 | 17-20 | 综合项目 | 完成完整项目 |

---

## 📚 第一阶段：基础概念（第 1-2 周）

### 第 1 周：核心概念与模型集成

#### 周一 - 环境准备与架构理解 (3 小时)
```
任务清单:
□ 了解 LangChain 发展历程
  - 阅读官方介绍文档 (30 分钟)
  - 观看项目结构讲解视频 (30 分钟)

□ 理解三个核心模块
  - langchain-core: 基础抽象 (30 分钟)
  - langchain-community: 第三方集成 (20 分钟)
  - langchain: 高级链 (20 分钟)

□ 设置开发环境
  - 创建虚拟环境 (10 分钟)
  - 安装依赖包 (10 分钟)
  - 验证安装 (10 分钟)

学习资源:
- https://python.langchain.com/docs/get_started/introduction
- https://github.com/langchain-ai/langchain
```

#### 周二 - LLM 集成基础 (3 小时)
```
任务清单:
□ LLM 与 Chat Model 的区别
  - 接口对比 (30 分钟)
  - 使用场景分析 (30 分钟)

□ OpenAI 集成
  - 获取 API 密钥 (10 分钟)
  - 实现第一个 LLM 调用 (30 分钟)
  - 测试不同参数 (30 分钟)

□ 其他模型探索
  - Anthropic Claude (20 分钟)
  - 本地模型 (Ollama) (20 分钟)
  - 模型选择标准 (20 分钟)

代码任务:
```python
# 创建文件: 01_llm_basics.py
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key="your-key",
    model="gpt-4",
    temperature=0.7
)

response = model.invoke("Hello!")
print(response.content)

# 实验不同参数
for temp in [0.0, 0.5, 1.0]:
    model = ChatOpenAI(temperature=temp)
    print(f"Temperature {temp}: {model.invoke('Write a poem').content}")
```

学习资源:
- https://python.langchain.com/docs/integrations/llms/
- https://python.langchain.com/docs/integrations/chat/
```

#### 周三 - Prompt 工程基础 (3 小时)
```
任务清单:
□ Prompt 设计原理
  - 清晰指令 (30 分钟)
  - 上下文提供 (30 分钟)
  - 输出格式指定 (20 分钟)

□ PromptTemplate 使用
  - 基础模板 (30 分钟)
  - 变量替换 (30 分钟)
  - 动态提示生成 (20 分钟)

□ ChatPromptTemplate
  - 系统消息 (20 分钟)
  - 用户消息 (20 分钟)
  - 多轮对话 (20 分钟)

代码任务:
```python
# 创建文件: 02_prompt_templates.py
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 基础模板
template = PromptTemplate(
    input_variables=["topic"],
    template="写一篇关于 {topic} 的文章"
)
print(template.format(topic="Python"))

# Chat 模板
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手"),
    ("user", "翻译: {text}")
])
print(chat_template.format_messages(text="Hello"))

# 动态提示
few_shot_template = ChatPromptTemplate.from_messages([
    ("system", "根据示例进行翻译"),
    ("user", "示例: {examples}"),
    ("user", "翻译: {text}")
])
```

学习资源:
- https://python.langchain.com/docs/concepts/prompt_templates
- Few-shot 学习: https://python.langchain.com/docs/concepts/prompting
```

#### 周四 - 消息与输出解析 (3 小时)
```
任务清单:
□ Message 系统
  - HumanMessage (20 分钟)
  - AIMessage (20 分钟)
  - SystemMessage (20 分钟)
  - 消息角色与作用 (20 分钟)

□ 输出解析器
  - StrOutputParser (30 分钟)
  - JsonOutputParser (30 分钟)
  - PydanticOutputParser (30 分钟)

□ 错误处理
  - 解析失败处理 (20 分钟)
  - 重试机制 (20 分钟)

代码任务:
```python
# 创建文件: 03_messages_parsing.py
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel

# 消息使用
messages = [
    SystemMessage(content="你是一个代码审查师"),
    HumanMessage(content="审查这个代码"),
    AIMessage(content="代码看起来不错")
]

# 输出解析
class CodeReview(BaseModel):
    rating: int
    comments: str

parser = JsonOutputParser(pydantic_object=CodeReview)
model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("评分: {code}")
chain = prompt | model | parser

result = chain.invoke({"code": "print('hello')"})
print(result)
```

学习资源:
- https://python.langchain.com/docs/concepts/messages
- https://python.langchain.com/docs/concepts/output_parsers
```

#### 周五 - 本周回顾与实践 (3 小时)
```
任务清单:
□ 复习核心概念
  - 整理笔记 (30 分钟)
  - 回答关键问题 (30 分钟)
  - 概念对比 (20 分钟)

□ 综合练习
  - 创建多模型调用程序 (30 分钟)
  - 实现动态提示生成 (30 分钟)
  - 组合输出解析 (20 分钟)

综合项目:
```python
# 创建文件: 04_week1_project.py
"""
任务: 构建一个"智能翻译机"
功能:
- 支持多种目标语言
- 返回结构化输出 (原文、译文、难度评分)
- 处理长文本
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class TranslationResult(BaseModel):
    original: str
    translated: str
    difficulty: int  # 1-5

# 实现翻译链
```

学习检查:
- □ 能解释 LangChain 的核心优势
- □ 能使用 ChatOpenAI 进行基础调用
- □ 能创建 PromptTemplate 和 ChatPromptTemplate
- □ 能使用 OutputParser 解析结构化数据
- □ 能处理基础错误情况
```

### 第 2 周：进阶概念与实践

#### 周一 - 消息历史与对话 (3 小时)
```
任务清单:
□ 维护对话历史
  - 消息列表管理 (30 分钟)
  - 上下文窗口 (30 分钟)
  - 消息清理策略 (20 分钟)

□ 实现多轮对话
  - 状态管理 (30 分钟)
  - 角色切换 (20 分钟)
  - 上下文连贯性 (20 分钟)

代码任务:
```python
# 创建文件: 05_conversation.py
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

# 构建对话历史
messages = [
    SystemMessage(content="你是一个友好的助手"),
    HumanMessage(content="你好"),
    AIMessage(content="你好!"),
    HumanMessage(content="你叫什么名字?"),
    AIMessage(content="我是 Claude 的助手")
]

# 继续对话
messages.append(HumanMessage(content="你能帮我做什么?"))
response = model.invoke(messages)
messages.append(AIMessage(content=response.content))

print(response.content)
```
```

#### 周二 - 模型参数与成本计算 (3 小时)
```
任务清单:
□ 模型参数详解
  - temperature (30 分钟)
  - max_tokens (20 分钟)
  - top_p (20 分钟)
  - frequency_penalty (20 分钟)

□ Token 与成本
  - Token 计数 (30 分钟)
  - 成本估算 (30 分钟)
  - 成本优化 (20 分钟)

代码任务:
```python
# 创建文件: 06_parameters_cost.py
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

model = ChatOpenAI(temperature=0.7)

# 成本跟踪
with get_openai_callback() as cb:
    response = model.invoke("写一个 Python 函数")
    print(f"Token 用量: {cb.total_tokens}")
    print(f"成本: ${cb.total_cost}")

# 不同参数对比
for temp in [0, 0.5, 1.0]:
    model = ChatOpenAI(temperature=temp)
    # 测试输出差异
```
```

#### 周三 - 链式提示优化 (3 小时)
```
任务清单:
□ 提示工程最佳实践
  - 清晰指令 (30 分钟)
  - Few-shot 学习 (30 分钟)
  - 角色扮演 (20 分钟)

□ 提示优化
  - A/B 测试 (30 分钟)
  - 效果评估 (20 分钟)
  - 迭代改进 (20 分钟)

代码任务:
```python
# 创建文件: 07_prompt_optimization.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI()

# 不同风格的提示
prompts = {
    "simple": "翻译: {text}",
    "detailed": """请将以下文本翻译成中文。
确保翻译：
1. 准确表达原意
2. 自然流畅
3. 适合中文阅读习惯

文本: {text}""",
    "few_shot": """翻译示例：
"Hello" -> "你好"
"Goodbye" -> "再见"

现在翻译: {text}"""
}

# 比较不同提示效果
```
```

#### 周四 - Streaming 与批处理 (3 小时)
```
任务清单:
□ 流式输出
  - stream() 方法 (30 分钟)
  - Token 级流式 (30 分钟)
  - 实时显示 (20 分钟)

□ 批处理
  - batch() 方法 (30 分钟)
  - 并行处理 (20 分钟)
  - 性能对比 (20 分钟)

代码任务:
```python
# 创建文件: 08_streaming_batch.py
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

# 流式输出
print("流式输出:")
for chunk in model.stream("讲一个笑话"):
    print(chunk.content, end="", flush=True)

print("\n批处理:")
inputs = ["笑话 1", "笑话 2", "笑话 3"]
results = model.batch(inputs)
for result in results:
    print(result.content)
```
```

#### 周五 - 第一阶段总结与测试 (3 小时)
```
任务清单:
□ 知识总结
  - 创建概念导图 (30 分钟)
  - 整理常用代码 (30 分钟)
  - 总结最佳实践 (20 分钟)

□ 综合测试项目

项目: 构建一个"智能客服"系统
要求:
- 支持多轮对话
- 维护对话历史
- 处理常见问题
- 返回结构化信息
- 优化成本

检查清单:
- □ 环境配置正确
- □ 所有代码能够运行
- □ 理解核心概念
- □ 完成两个测试项目
- □ 准备进入第二阶段
```

---

## 🔗 第二阶段：链与组合（第 3-4 周）

### 第 3 周：LCEL 与 Runnable

#### 周一 - Runnable 接口深入 (3 小时)
```
任务清单:
□ 理解 Runnable
  - 什么是 Runnable (30 分钟)
  - 核心方法详解 (40 分钟)
  - Runnable 设计模式 (20 分钟)

□ 核心方法实践
  - invoke() 同步调用 (30 分钟)
  - batch() 批量处理 (30 分钟)
  - stream() 流式处理 (20 分钟)

代码任务:
```python
# 创建文件: 09_runnable_interface.py
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")

# 所有这些都是 Runnable
print("Model invoke:", model.invoke("Hello"))
print("Prompt invoke:", prompt.invoke({"topic": "Python"}))

# 尝试 batch
results = model.batch(["Hello", "Hi", "Hey"])
print("Batch results:", results)

# 尝试 stream
for chunk in model.stream("Tell a joke"):
    print(chunk, end="")
```
```

#### 周二 - LCEL 管道操作 (3 小时)
```
任务清单:
□ LCEL 语法
  - 管道操作符 (|) (30 分钟)
  - 组合多个 Runnable (30 分钟)
  - 字典式输入输出 (20 分钟)

□ 实际应用
  - 构建简单链 (30 分钟)
  - 构建复杂链 (30 分钟)

代码任务:
```python
# 创建文件: 10_lcel_basics.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 最简单的链
prompt = ChatPromptTemplate.from_template("翻译: {text}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"text": "Hello"})
print(result)

# 更复杂的链
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"input": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

result = chain.invoke("Test")
```
```

#### 周三 - RunnableParallel 与分支 (3 小时)
```
任务清单:
□ 并行执行
  - RunnableParallel (40 分钟)
  - 独立任务并行 (40 分钟)

□ 条件分支
  - RunnableBranch (30 分钟)
  - 动态路由 (30 分钟)

代码任务:
```python
# 创建文件: 11_parallel_branch.py
from langchain_core.runnables import RunnableParallel, RunnableBranch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI()

# 并行执行
parallel = RunnableParallel(
    grammar_check=ChatPromptTemplate.from_template(
        "检查语法: {text}"
    ) | model,
    tone_analysis=ChatPromptTemplate.from_template(
        "分析语气: {text}"
    ) | model
)

result = parallel.invoke({"text": "This is a test."})

# 分支
branch = RunnableBranch(
    (lambda x: "Chinese" in x["lang"], ChatPromptTemplate.from_template(
        "翻译成中文: {text}"
    ) | model),
    (lambda x: "English" in x["lang"], ChatPromptTemplate.from_template(
        "Translate to English: {text}"
    ) | model),
    ChatPromptTemplate.from_template("Unknown language") | model
)
```
```

#### 周四 - 错误处理与调试 (3 小时)
```
任务清单:
□ 异常处理
  - try-except 模式 (30 分钟)
  - 重试机制 (30 分钟)
  - 降级方案 (20 分钟)

□ 调试工具
  - 日志记录 (30 分钟)
  - 中间步骤追踪 (20 分钟)

代码任务:
```python
# 创建文件: 12_error_handling.py
from langchain_core.runnables import RunnableRetry
from langchain_openai import ChatOpenAI
import logging

logging.basicConfig(level=logging.DEBUG)

model = ChatOpenAI()

# 重试机制
retry_chain = model.with_retry(max_attempts=3)

try:
    result = retry_chain.invoke("Test", timeout=10)
except Exception as e:
    print(f"Error: {e}")
    # 降级方案
    result = "Default response"
```
```

#### 周五 - 第二周回顾 (3 小时)
```
综合项目: 构建一个文本处理管道

功能:
- 语法检查
- 语气分析
- 要点提取
- 长度优化

检查清单:
- □ 理解 Runnable 接口的所有方法
- □ 能使用 LCEL 语法编写链
- □ 能处理并行和分支
- □ 能进行错误处理
```

### 第 4 周：高级 LCEL 与最佳实践

#### 周一 - RunnablePassthrough 与字典操作 (3 小时)
```
任务清单:
□ 数据流处理
  - RunnablePassthrough (30 分钟)
  - 字典输入输出 (30 分钟)
  - 数据转换 (20 分钟)

代码任务:
```python
# 创建文件: 13_passthrough.py
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI()

# 保留原始输入
chain = (
    {"text": RunnablePassthrough(), "language": lambda x: "English"}
    | ChatPromptTemplate.from_template(
        "Translate to {language}: {text}"
    )
    | model
)

result = chain.invoke("Hello")
```
```

#### 周二 - Lambda 与自定义函数 (3 小时)
```
任务清单:
□ 函数转换
  - lambda 函数 (30 分钟)
  - 自定义函数转为 Runnable (30 分钟)
  - 数据预处理 (20 分钟)

代码任务:
```python
# 创建文件: 14_custom_functions.py
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

# 自定义预处理
def preprocess(text):
    return text.strip().lower()

# 转换为 Runnable
preprocess_runnable = RunnableLambda(preprocess)

chain = (
    preprocess_runnable
    | ChatPromptTemplate.from_template("翻译: {text}")
    | model
)
```
```

#### 周三 - 链的可观测性与追踪 (3 小时)
```
任务清单:
□ 调试工具
  - 中间步骤打印 (30 分钟)
  - 回调系统 (30 分钟)
  - LangSmith 集成 (20 分钟)

代码任务:
```python
# 创建文件: 15_observability.py
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI

# 启用详细日志
callbacks = [StdOutCallbackHandler()]
model = ChatOpenAI(callbacks=callbacks)

result = model.invoke("Test", callbacks=callbacks)
```
```

#### 周四 - 异步编程基础 (3 小时)
```
任务清单:
□ 异步方法
  - ainvoke() (30 分钟)
  - astream() (30 分钟)
  - abatch() (20 分钟)

代码任务:
```python
# 创建文件: 16_async_basics.py
import asyncio
from langchain_openai import ChatOpenAI

async def main():
    model = ChatOpenAI()

    # 异步调用
    result = await model.ainvoke("Hello")
    print(result)

    # 异步流式
    async for chunk in model.astream("Tell a joke"):
        print(chunk.content, end="")

asyncio.run(main())
```
```

#### 周五 - 第二阶段项目 (3 小时)
```
综合项目: 构建一个"内容分析管道"

功能:
1. 接收文本输入
2. 并行处理:
   - 关键词提取
   - 情感分析
   - 长度计算
3. 聚合结果
4. 生成报告

要求:
- 使用 LCEL 语法
- 实现并行处理
- 处理错误情况
- 支持流式输出

检查清单:
- □ 理解 LCEL 所有操作
- □ 能处理复杂的数据流
- □ 能进行错误处理
- □ 能使用异步方法
- □ 准备进入第三阶段
```

---

## 📖 简化版周计划表（第 5-20 周）

由于篇幅限制，以下是简化的周计划表。详细版本请参考完整计划文档。

### 第三阶段：数据管理与 RAG（第 5-7 周）

| 周次 | 周一 | 周二 | 周三 | 周四 | 周五 |
|------|------|------|------|------|------|
| 第 5 周 | 文档加载 | PDF 处理 | 文本分割 | 元数据管理 | 项目小结 |
| 第 6 周 | 嵌入模型 | 向量化 | 向量存储 | 相似性搜索 | 项目小结 |
| 第 7 周 | 检索器 | 多层检索 | RAG 实现 | 性能优化 | **RAG 项目** |

### 第四阶段：工具与代理（第 8-10 周）

| 周次 | 周一 | 周二 | 周三 | 周四 | 周五 |
|------|------|------|------|------|------|
| 第 8 周 | 工具定义 | 工具调用 | 模型绑定 | 工具验证 | 项目小结 |
| 第 9 周 | 代理框架 | 工具调用代理 | 多工具代理 | 代理调试 | 项目小结 |
| 第 10 周 | ReAct 代理 | 自定义代理 | 代理记忆 | 高级模式 | **代理项目** |

### 第五阶段：LangGraph（第 11-13 周）

| 周次 | 周一 | 周二 | 周三 | 周四 | 周五 |
|------|------|------|------|------|------|
| 第 11 周 | 图基础 | 状态管理 | 节点与边 | 条件路由 | 项目小结 |
| 第 12 周 | 消息状态 | 子图 | 持久化 | 可视化 | 项目小结 |
| 第 13 周 | RAG 图 | 图调试 | 性能优化 | 扩展性 | **LangGraph 项目** |

### 第六阶段：高级应用（第 14-16 周）

| 周次 | 周一 | 周二 | 周三 | 周四 | 周五 |
|------|------|------|------|------|------|
| 第 14 周 | 多代理 | 代理通信 | 主管模式 | 协作策略 | 项目小结 |
| 第 15 周 | 流式处理 | 异步编程 | 实时交互 | 并发优化 | 项目小结 |
| 第 16 周 | 评估与监控 | 自动评估 | LangSmith | 性能监控 | **高级应用项目** |

### 第七阶段：综合项目（第 17-20 周）

| 周次 | 内容 |
|------|------|
| 第 17 周 | 项目选择与需求分析 |
| 第 18 周 | 核心功能开发 |
| 第 19 周 | 集成、优化与测试 |
| 第 20 周 | 部署、评估与总结 |

---

## 📋 每周时间分配建议

### 推荐每周 20 小时安排

```
周一-周四: 各 3 小时 (共 12 小时)
├─ 学习新概念: 1.5 小时
├─ 代码实践: 1 小时
└─ 练习与复习: 0.5 小时

周五: 3 小时
├─ 本周回顾: 1 小时
├─ 综合项目: 1.5 小时
└─ 知识整理: 0.5 小时

周末: 2 小时
├─ 阅读补充资料: 1 小时
└─ 社区交流: 1 小时
```

### 灵活调整建议

- **基础差的同学**: 每个阶段延长 1 周
- **基础好的同学**: 可以压缩到 15 周
- **有工作的同学**: 平均每天 1.5 小时，延长到 6-7 个月

---

## 🎯 每周学习目标检查

### 第一阶段检查清单
```
第 1 周结束:
□ 能创建 OpenAI 模型实例
□ 理解 PromptTemplate 和 ChatPromptTemplate
□ 能使用 OutputParser 解析结果
□ 能处理基础错误

第 2 周结束:
□ 能维护多轮对话历史
□ 理解 Token 和成本计算
□ 能优化提示词
□ 能使用 stream 和 batch 方法
```

### 第二阶段检查清单
```
第 3 周结束:
□ 理解 Runnable 接口的所有核心方法
□ 能使用 LCEL 语法构建简单链
□ 能使用 RunnableParallel 并行处理
□ 能使用 RunnableBranch 条件分支

第 4 周结束:
□ 能处理复杂的数据流
□ 能实现自定义转换函数
□ 能调试和追踪链的执行
□ 能使用异步方法
```

（以此类推，每个阶段都有详细的检查清单）

---

## 🚀 加速学习技巧

### 1. 代码库组织
```
LearnLangChain/
├── week01/
│   ├── 01_llm_basics.py
│   ├── 02_prompt_templates.py
│   └── 03_messages_parsing.py
├── week02/
├── ...
├── projects/
│   ├── project_rag.py
│   ├── project_agent.py
│   └── project_langgraph.py
├── notes/
│   ├── week01_summary.md
│   └── concepts.md
└── LANGCHAIN_KNOWLEDGE_MAP.md
```

### 2. 学习记录模板

每周创建一个学习记录：
```markdown
# 第 X 周学习总结

## 本周学习内容
- 概念 1: ...
- 概念 2: ...

## 关键收获
- ...

## 遇到的问题
- 问题 1: 解决方案
- 问题 2: 解决方案

## 代码示例
```python
# 关键代码片段
```

## 下周准备
- ...
```

### 3. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 打印中间结果
from langchain.callbacks import StdOutCallbackHandler
chain.invoke(input, config={"callbacks": [StdOutCallbackHandler()]})

# 使用 LangSmith 追踪
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"
```

### 4. 快速参考

建立一个 `quick_reference.md` 文件：

```markdown
# LangChain 快速参考

## 最常用代码片段
```python
# 基础链
chain = prompt | model | parser

# 并行处理
parallel = RunnableParallel(a=chain_a, b=chain_b)

# 条件分支
branch = RunnableBranch(
    (condition, chain_if_true),
    default_chain
)
```

## 常见错误与解决
- 错误 1: ...
- 错误 2: ...
```

---

## 📚 推荐学习资源

### 官方资源
- 📖 LangChain Python 文档: https://python.langchain.com/
- 🎓 LangGraph 文档: https://langchain-ai.github.io/langgraph/
- 💻 GitHub 示例: https://github.com/langchain-ai/langchain/tree/master/examples

### 视频教程
- YouTube: LangChain Crash Course
- YouTube: Building AI Apps with LangChain
- 官方演讲与教程

### 交互式学习
- LangChain 官方笔记本示例
- DeepLearning.AI LangChain 课程

### 社区支持
- GitHub Discussions
- Discord 社群
- Stack Overflow (langchain 标签)

---

## ✅ 最终检查清单

学习完成后，你应该能够：

**基础能力**
- ☐ 创建和配置 LLM
- ☐ 设计有效的提示词
- ☐ 处理模型输出
- ☐ 管理对话历史

**进阶能力**
- ☐ 使用 LCEL 构建复杂链
- ☐ 实现 RAG 系统
- ☐ 创建工具和代理
- ☐ 使用 LangGraph 设计工作流

**专业能力**
- ☐ 构建多代理系统
- ☐ 实现流式和异步处理
- ☐ 评估和优化系统
- ☐ 部署到生产环境

**项目经验**
- ☐ 完成 3+ 个实际项目
- ☐ 处理真实世界的数据和需求
- ☐ 优化性能和成本
- ☐ 参与社区贡献

---

祝学习顺利！有任何问题，欢迎查看知识图谱或联系社区。🚀
