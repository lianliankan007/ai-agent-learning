# Day17 Function Calling 案例拆解

## 案例名称

请填写：
## Response
```json
{
  "created": 1774273684,
  "usage": {
    "completion_tokens": 747,
    "prompt_tokens": 285,
    "completion_tokens_details": {
      "text_tokens": 747,
      "reasoning_tokens": 710
    },
    "prompt_tokens_details": {
      "text_tokens": 285
    },
    "total_tokens": 1032
  },
  "model": "qwen3.5-plus",
  "id": "chatcmpl-307289bb-767b-901e-ab62-fa55464ead4f",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": ,
        "reasoning_content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: \"1+ 5*38 等于多少\" (What is 1 + 5 * 38 equal to?)\n    *   Task: Decide whether to answer directly or call a tool.\n    *   Constraint: Must output structured JSON. No Markdown code blocks. No extra explanation.\n    *   Available Tools: `calculator`, `doc_search`, `memory_lookup`.\n\n2.  **Evaluate the Need for Tools:**\n    *   The user is asking for a mathematical calculation: `1 + 5 * 38`.\n    *   According to the tool descriptions, `calculator` is used for \"executing precise mathematical calculations\" and \"expression evaluation\".\n    *   While I (the AI) can calculate this mentally (1 + 190 = 191), the system instruction defines me as a \"Teaching Function Calling Decision Maker\". The goal is to demonstrate the decision-making process for function calling.\n    *   Usually, for math problems, especially those involving order of operations, using the `calculator` tool ensures precision and aligns with the tool's purpose.\n    *   Therefore, calling the `calculator` tool is the appropriate action.\n\n3.  **Construct the Tool Call:**\n    *   Tool Name: `calculator`\n    *   Arguments: `expression` should be \"1+ 5*38\" or \"1 + 5 * 38\". I'll normalize it slightly to \"1 + 5 * 38\".\n    *   Reason: The user is asking for a mathematical calculation result.\n\n4.  **Format the Output:**\n    *   JSON format required: `{\"mode\":\"function_call\",\"tool_name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 5 * 38\"},\"reason\":\"用户需要进行数学计算\"}`\n    *   Constraint Check: No Markdown code blocks (```json ... ```), just raw JSON.\n\n5.  **Final Review:**\n    *   Does it match the schema? Yes.\n    *   Is it valid JSON? Yes.\n    *   Did I choose the right tool? Yes (`calculator`).\n    *   Did I avoid Markdown? Yes.\n\n6.  **Drafting the JSON:**\n    {\"mode\":\"function_call\",\"tool_name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 5 * 38\"},\"reason\":\"用户请求计算数学表达式的结果\"}\n\n    *Self-Correction on Reason:* Keep it concise and clear. \"用户需要进行精确数学计算\" (User needs precise mathematical calculation).\n\n    *Final JSON:*\n    {\"mode\":\"function_call\",\"tool_name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 5 * 38\"},\"reason\":\"用户需要进行精确数学计算\"}\n\n    *Wait, one more check:* The prompt says \"If no tool needed, output text_answer\". Since math calculation is explicitly listed under `calculator` tool usage (\"适用于表达式求值\"), I should call the tool.\n\n    *Final Decision:* Call `calculator`.\n\n    *JSON Content:*\n    {\"mode\":\"function_call\",\"tool_name\":\"calculator\",\"arguments\":{\"expression\":\"1 + 5 * 38\"},\"reason\":\"用户需要进行精确数学计算\"}\n\n    *Remove Markdown:* Ensure no ```json wrappers.\n\n    *Final Output Generation.*"
      },
      "logprobs": null
    }
  ],
  "system_fingerprint": null,
  "object": "chat.completion"
}
```

---
## 2026-03-23 21:48:27

- time: 2026-03-23 21:48:27
- provider: dashscope-compatible
- model: qwen3.5-plus
- endpoint: https://coding.dashscope.aliyuncs.com/v1/chat/completions
- status_code: 200
- duration_ms: 24392.21
- agent_name: Day17FunctionCallingAgent
- stage: final_answer

### Request
```json
{
  "model": "qwen3.5-plus",
  "messages": [
    {
      "role": "system",
      "content": "你是一个教学型 AI 助手。\n你已经拿到了工具调用结果 observation。\n请基于 observation 回答用户问题，并清楚说明：\n1. 系统调用了什么工具\n2. observation 提供了什么信息\n3. 最终结论是什么\n请保持中文、结构清晰、适合初学者阅读。"
    },
    {
      "role": "user",
      "content": "用户问题: 1+ 5*38 等于多少\n工具名: calculator\n工具参数: {\"expression\": \"1 + 5 * 38\"}\nobservation: 计算结果: 191"
    }
  ],
  "temperature": 0.1,
  "max_tokens": 800
}
```

## 1. 用户问题

请填写：ask 1+ 5*38 等于多少

## 2. 模型返回的结构化 tool call

请填写：
{
  "mode": "function_call",
  "tool_name": "calculator",
  "arguments": {
    "expression": "1 + 5 * 38"
  },
  "reason": "用户需要进行精确数学计算"
}
## 3. 程序如何执行工具

请填写：调用calculator "1 + 5 * 38"

## 4. observation 是什么

请填写：计算结果: 191

## 5. 最终回答如何形成

请填写：LLm根据上下文和 observation生成

## 我的结论

请总结这个案例里最关键的 2 到 3 个点。
