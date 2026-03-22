# AI Agent Learning

这是一个面向初学者的 Python Agent 学习仓库，按 `day1` 到 `day4` 逐步练习，从最基础的 LLM 调用，一路走到 function calling 和带长期记忆的 Agent。

项目特点：
- 以小步练习为主，每一天都尽量是独立示例，方便单独运行和理解
- 代码偏教学风格，适合边看边学
- 重点练习 Agent 的核心能力：对话、参数控制、工具调用、长期记忆

## 学习路线

### Day 1: 基础 LLM Agent
Day 1 主要学习：
- Python 基础组织方式
- 如何通过 API 调用大模型
- 如何维护最简单的多轮对话历史
- 如何把请求和响应记录到日志中

相关文件：
- `day1/llm_agent.py`
- `day1/agent_runner.py`
- `day1/学习总结.md`

### Day 2: 参数控制、重试与历史压缩
Day 2 在 Day 1 的基础上增加了更实用的能力：
- `temperature`、`top_p`、`max_tokens` 等采样参数控制
- 请求失败后的自动重试
- 历史消息过长时的压缩总结
- 更适合真实项目的请求封装

相关文件：
- `day2/day2_llm_agent.py`
- `day2/day2_runner.py`
- `day2/day2_llm_lab.py`
- `day2/学习总结.md`

### Day 3: Function Calling
Day 3 重点学习大模型如何调用工具：
- 定义工具描述
- 注册本地函数
- 让模型根据问题自动选择工具
- 把工具结果再交回模型生成最终回答

当前示例内置了两个简单工具：
- 查询天气
- 查询当前时间

相关文件：
- `day3/function_calling_demo.py`
- `day3/学习总结.md`

### Day 4: Agent Memory
Day 4 进入更接近真实 Agent 的长期记忆系统：
- 使用 embedding 将文本转成向量
- 使用 Qdrant 存储长期记忆
- 支持简单检索、过滤检索、混合检索
- 将检索到的记忆注入 prompt
- 自动从用户输入中提取值得保存的信息

默认技术组合：
- LLM: DashScope OpenAI 兼容接口
- Embedding: 本地 Ollama
- Vector DB: 本地 Qdrant

相关文件：
- `day4/agent_with_memory.py`
- `day4/vector_retriever.py`
- `day4/学习总结.md`
- `day4/学习任务.md`

## 目录结构

```text
ai-agent-learning/
├─ day1/                  # 基础 Agent
├─ day2/                  # 参数控制、重试、历史压缩
├─ day3/                  # Function Calling
├─ day4/                  # 长期记忆 Agent
├─ memory/                # 学习记录和笔记
├─ utils/                 # 通用工具，例如 LLM Markdown 日志
├─ requirements.txt       # Python 依赖
├─ AGENTS.md              # 仓库协作规范
└─ readme.md              # 项目说明
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 中的基础依赖有：
- `requests`
- `python-dotenv`

注意：
- Day 4 代码还使用了 `httpx`、`qdrant-client`
- 如果你要运行 Day 4，需要额外安装这些依赖

可以手动执行：

```bash
pip install httpx qdrant-client
```

### 2. 配置 API Key

请不要把 API Key 直接写进代码里。

推荐方式：
- 放到 `.env`
- 或放到 `.local/openai_api_key.txt`
- 或通过环境变量设置 `OPENAI_API_KEY`

示例 `.env`：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
OPENAI_MODEL=qwen3.5-plus
```

## 运行方式

### 运行 Day 1

```bash
python day1/agent_runner.py
```

### 运行 Day 2

```bash
python day2/day2_runner.py
```

如果你想做 Day 2 的实验脚本，也可以查看：

```bash
python day2/day2_llm_lab.py
```

### 运行 Day 3

```bash
python day3/function_calling_demo.py
```

### 运行 Day 4

运行前请确保：
- 本地 Ollama 已启动
- 本地 Qdrant 已启动
- `OPENAI_API_KEY` 已正确配置

启动命令：

```bash
python day4/agent_with_memory.py
```

如果你想先注入一些演示记忆，可以这样运行：

Windows PowerShell:

```powershell
$env:DAY4_SEED_DEMO="true"
python day4/agent_with_memory.py
```

Windows CMD:

```cmd
set DAY4_SEED_DEMO=true
python day4/agent_with_memory.py
```

## 常用校验命令

快速做语法检查：

```bash
python -m py_compile day4/agent_with_memory.py day4/vector_retriever.py
```

你也可以按需检查其他脚本，例如：

```bash
python -m py_compile day1/llm_agent.py day2/day2_llm_agent.py day3/function_calling_demo.py
```

## Day 4 环境变量

Day 4 支持以下环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OLLAMA_BASE_URL`
- `OLLAMA_EMBED_MODEL`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `MEMORY_COLLECTION`
- `MEMORY_USER_ID`
- `DAY4_SEED_DEMO`

示例：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
OPENAI_MODEL=qwen3.5-plus
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
QDRANT_HOST=localhost
QDRANT_PORT=6333
MEMORY_COLLECTION=day4_agent_memory
MEMORY_USER_ID=default-user
```

## 编码约定

本仓库遵循这些原则：
- 使用 4 个空格缩进
- 函数、变量、模块名使用 `snake_case`
- 类名使用 `PascalCase`
- 不要在代码中硬编码密钥、Token、主机地址
- 复杂逻辑尽量增加注释，方便 Python 初学者理解

详细规范请查看：
- `AGENTS.md`

## 安全提醒

请特别注意：
- 不要提交真实 API Key
- 不要提交 `.env` 中的真实敏感信息
- 不要提交 `.local/` 下的密钥文件
- `__pycache__/` 等临时文件应忽略

## 适合谁

这个项目比较适合：
- 正在入门 Python 的同学
- 想从零理解 Agent 基础能力的人
- 想自己动手实现对话、工具调用、记忆系统的人

## 后续可以继续做什么

如果你继续往下练习，可以考虑：
- 给 Day 1 到 Day 3 补充更完整的中文注释
- 给 Day 2 和 Day 4 增加测试
- 给 Day 3 增加更多工具
- 给 Day 4 增加记忆更新、去重、遗忘机制
- 补充每个 day 独立的 README
