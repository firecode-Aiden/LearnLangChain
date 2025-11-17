# LearnLangChain

一份全面的20周结构化学习指南，用于掌握**LangChain**，这是一个用于构建由大型语言模型(LLMs)驱动的智能应用的框架。

## 概述

本仓库提供了一套完整的课程，从基础到生产就绪的应用程序，全面学习LangChain。无论你是LLM开发的新手还是想深化你的专业知识，本指南都将带你走过一个渐进式的学习之旅。

### 什么是LangChain？

LangChain是一个简化使用大型语言模型构建应用程序的框架。它为处理语言模型、操作链、检索增强生成(RAG)、代理和基于图的编排提供了抽象。

## 📚 仓库内容

本仓库围绕三个互补的文档文件进行组织：

1. **LANGCHAIN_KNOWLEDGE_MAP.md** - "是什么"和"为什么"
   - 综合知识架构
   - 所有LangChain概念的详细解释
   - 最佳实践和设计模式
   - 用作你的参考指南

2. **PRACTICE_GUIDE.md** - "如何做"
   - 实用的代码模板和示例
   - 常见任务的实现模式
   - 可运行的代码片段
   - 用于动手学习

3. **STUDY_SCHEDULE.md** - "何时"和"顺序"
   - 按周进行的学习进度
   - 具体的任务和里程碑
   - 代码练习和检查点
   - 用于追踪学习进度

4. **CLAUDE.md** - 开发指导
   - 开发标准和规范
   - 代码风格期望
   - 常见模式和陷阱

## 🎯 学习路径（7个阶段）

```
第1-2周   → 基础
第3-4周   → 链与组合
第5-7周   → 数据与RAG
第8-10周  → 工具与代理
第11-13周 → LangGraph
第14-16周 → 高级应用
第17-20周 → 毕业项目
```

### 阶段1：基础（第1-2周）
- LangChain核心概念
- LLM集成和配置
- 提示词和消息
- 输出解析
- **获得技能**：能够创建基本的LLM查询并解析响应

### 阶段2：链与组合（第3-4周）
- LCEL（LangChain表达式语言）
- Runnable接口和管道操作符
- 组合复杂链
- 错误处理和重试
- **获得技能**：能够构建序列链并优雅地处理错误

### 阶段3：数据与RAG（第5-7周）
- 文档加载和处理
- 文本分割和分块
- 嵌入和向量存储
- 检索增强生成
- **获得技能**：能够构建基于文档的RAG系统

### 阶段4：工具与代理（第8-10周）
- 工具定义和调用
- 代理框架和循环
- ReAct模式实现
- 工具调用代理
- **获得技能**：能够构建自主使用工具的代理

### 阶段5：LangGraph（第11-13周）
- 基于图的编排
- 状态管理和MessagesState
- 节点和边的定义
- 条件路由
- **获得技能**：能够设计具有状态管理的复杂多步工作流

### 阶段6：高级应用（第14-16周）
- 多代理系统
- 流式和异步操作
- 缓存和性能优化
- 评估和监控
- **获得技能**：能够构建生产规模的应用程序

### 阶段7：毕业项目（第17-20周）
- 端到端应用程序开发
- 所有概念的集成
- 部署考虑
- 真实场景
- **获得技能**：展示完全掌握的投资组合项目

## 🚀 快速开始

### 前置要求
- Python 3.10+
- 基本的Python理解
- LLM提供商的API密钥（推荐用OpenAI学习）

### 设置步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/LearnLangChain.git
   cd LearnLangChain
   ```

2. **安装依赖**
   ```bash
   pip install langchain langchain-openai langchain-community langgraph python-dotenv
   ```

3. **配置环境**
   ```bash
   # 创建 .env 文件
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

4. **开始学习**
   - 在 **STUDY_SCHEDULE.md** 中从第1周开始
   - 在 **LANGCHAIN_KNOWLEDGE_MAP.md** 中参考概念
   - 使用 **PRACTICE_GUIDE.md** 中的代码示例

## 📖 如何使用本仓库

### 自主学习
1. 打开 **STUDY_SCHEDULE.md** 并按周进行学习
2. 每周阅读 **LANGCHAIN_KNOWLEDGE_MAP.md** 中的对应部分
3. 从 **PRACTICE_GUIDE.md** 实现代码示例
4. 完成检查点练习

### 课堂教学
- 使用 **LANGCHAIN_KNOWLEDGE_MAP.md** 作为讲座材料
- 让学生按照 **STUDY_SCHEDULE.md** 完成作业
- 使用 **PRACTICE_GUIDE.md** 进行现场编码演示

### 作为参考
- 使用 **LANGCHAIN_KNOWLEDGE_MAP.md** 理解概念
- 使用 **PRACTICE_GUIDE.md** 查找实现模式
- 使用 **CLAUDE.md** 了解开发标准

## 🔑 核心概念速览

### LCEL（LangChain表达式语言）
使用管道操作符（`|`）组合组件的现代声明式方法

```python
chain = prompt_template | llm_model | output_parser
result = chain.invoke({"variable": value})
```

### Runnable接口
核心抽象，具有以下方法：`invoke()`、`batch()`、`stream()`、`ainvoke()`、`astream()`、`abatch()`

### RAG（检索增强生成）
结合文档检索和生成，提供更准确、上下文感知的响应

```
文档 → 嵌入 → 向量存储 → 检索 → 生成 → 响应
```

### 代理
能够规划、执行和观察以使用工具完成任务的自主系统

```
规划 → 使用工具 → 观察结果 → 重复直到完成
```

### LangGraph
用于复杂工作流的基于图的编排，支持状态管理

## 📚 推荐学习方法

### 选项1：线性学习（推荐初学者）
- 按顺序学习第1-20周
- 在基础概念上逐步构建
- 预计时间：20周，每周7-10小时

### 选项2：模块化学习（适合有经验的开发者）
- 根据你的水平跳过基础阶段
- 专注于感兴趣的特定领域
- 并行组合多个阶段

### 选项3：项目驱动学习
- 从你想构建的毕业项目开始
- 根据需要跳到相关阶段
- 将指南用作参考资料

## 🎓 学习目标

完成本课程后，你将能够：

- ✅ 从头开始构建LLM应用程序
- ✅ 实现检索增强生成（RAG）系统
- ✅ 创建使用工具的自主代理
- ✅ 用LangGraph设计复杂工作流
- ✅ 优化性能和处理错误
- ✅ 部署生产就绪的LLM应用程序

## ⚠️ 常见陷阱与解决方案

| 问题 | 解决方案 |
|------|--------|
| 上下文窗口溢出 | 使用适当chunk_size的文本分割器 |
| 检索质量差 | 实现查询转换和融合检索 |
| 高代币成本 | 缓存响应，使用更便宜的模型进行预处理 |
| 状态管理问题 | 使用归约函数（operator.add） |
| 代理不可靠 | 实现回退链和错误恢复 |

## 📦 依赖项

整个课程使用的核心包：

```
langchain >= 0.1.x          # 核心框架
langchain-openai            # OpenAI集成
langchain-community         # 第三方集成
langgraph >= 0.1.x          # 图编排
python-dotenv              # 环境管理
pydantic >= 2.0             # 数据验证
```

## 🔄 现代与遗留方法对比

本课程使用**现代模式**：

| 概念 | 现代方法 | 遗留方法 |
|------|---------|---------|
| 链组合 | LCEL与`\|`操作符 | LLMChain类 |
| 检索 | RAG链 | RetrievalQA |
| 代理 | 工具调用+StateGraph | AgentExecutor |

## 🌐 额外资源

- [LangChain官方文档](https://python.langchain.com)
- [LangChain GitHub仓库](https://github.com/langchain-ai/langchain)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API参考](https://platform.openai.com/docs)

## 📋 进度追踪

使用 **STUDY_SCHEDULE.md** 来追踪你在20周课程中的进度。每周包括：
- 学习目标
- 关键概念
- 代码练习
- 检查点评估

## 💡 成功建议

1. **动手编码**：不仅要阅读示例，要自己运行它们
2. **实验创新**：修改示例并尝试不同的方法
3. **系统调试**：使用打印语句和错误消息来理解问题
4. **构建项目**：将概念应用于你关心的真实问题
5. **定期复习**：随着进度推进重新访问早期概念
6. **加入社区**：与其他LangChain学习者建立联系

## 🤝 贡献

欢迎为改进本学习指南做出贡献：
- 报告错误或不清楚的解释
- 建议其他示例或用例
- 分享你自己的学习经验
- 提议新的毕业项目

## 📄 许可证

本学习指南是开源的，可用于教育用途。

## ❓ 常见问题

**问：这需要多长时间？**
答：大约20周，每周7-10小时。你可以根据自己的进度和经验水平进行调整。

**问：我需要有LLM/AI经验吗？**
答：不需要！本指南是为初学者设计的。我们从基础开始。

**问：我可以跳过某些部分吗？**
答：如果你已有经验，可以跳过基础阶段。但我们建议你完整学习以确保全面理解。

**问：如果我卡住了怎么办？**
答：查看LANGCHAIN_KNOWLEDGE_MAP.md中的相关部分并尝试PRACTICE_GUIDE.md中的示例。

**问：练习题有解决方案吗？**
答：查看PRACTICE_GUIDE.md中的完整示例，你可以参考和学习。

---

**准备好开始了吗？** 从[STUDY_SCHEDULE.md中的第1周](./STUDY_SCHEDULE.md)开始！
