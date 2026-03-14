# Day 2 模板（参照 Day 1 风格）

## 文件结构

- `day2_llm_agent.py`：Agent 类（参数控制 + 历史管理 + 重试）
- `day2_runner.py`：交互式命令行 Runner

## 依赖安装

```bash
pip install -U requests
```

## 配置密钥

```bash
export OPENAI_API_KEY="你的key"
```

## 运行

```bash
python3 day2_runner.py
```

## 交互命令

- `chat <text>`：多轮对话
- `experiment <text>`：同一 prompt 下做参数实验
- `temp <float>`：设置 temperature
- `top_p <float>`：设置 top_p
- `max_tokens <int>`：设置 max_tokens
- `retries <int>`：设置最大重试次数
- `history`：查看历史条数
- `clear`：清空历史
- `prompt <text>`：更新系统提示词
- `info`：查看当前配置
- `quit` / `exit`：退出

## Day 2 学习点落地

1. 参数理解：通过 `experiment` 对比 `temperature/top_p/max_tokens`
2. 多轮上下文：自动保存历史，并在过长时做摘要压缩
3. 错误处理：对 `429/5xx/网络超时` 自动重试（指数退避 + jitter）
