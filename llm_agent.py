#!/usr/bin/env python3
"""
简单的 LLM Agent 调用程序
支持 OpenAI 格式的 API 调用
"""

import os
from typing import Optional, List, Dict, Any
import requests


class LLMAgent:
    """简单的 LLM Agent 类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        初始化 LLM Agent
        
        Args:
            api_key: API 密钥，默认从环境变量 OPENAI_API_KEY 读取
            base_url: API 基础 URL
            model: 模型名称
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请提供 api_key 或设置 OPENAI_API_KEY 环境变量")
        
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 对话历史
        self.messages: List[Dict[str, str]] = []
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        发送对话消息
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            stream: 是否流式输出
            
        Returns:
            AI 的回复内容
        """
        # 构建消息列表
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史对话
        messages.extend(self.messages)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": message})
        
        # 调用 API
        response = self._call_api(messages, stream)
        
        # 保存对话历史
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": response})
        
        return response
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        """
        调用 LLM API
        
        Args:
            messages: 消息列表
            stream: 是否流式输出
            
        Returns:
            AI 回复内容
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 调用失败: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"解析响应失败: {e}")
    
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.messages.copy()


def main():
    """示例用法"""
    print("=" * 50)
    print("🤖 简单 LLM Agent 演示")
    print("=" * 50)
    
    # 初始化 Agent
    # 方式1: 直接传入 API Key
    # agent = LLMAgent(api_key="your-api-key")
    
    # 方式2: 从环境变量读取 (推荐)
    # 先设置环境变量: export OPENAI_API_KEY="your-api-key"
    try:
        agent = LLMAgent()
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请设置环境变量后再运行:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\n或使用自定义 API:")
        print("  agent = LLMAgent(")
        print("      api_key='your-api-key',")
        print("      base_url='https://coding.dashscope.aliyuncs.com/v1',")
        print("      model='qwen3.5-plus'")
        print("  )")
        return
    
    # 设置系统提示词
    system_prompt = "你是一个 helpful 的 AI 助手，回答简洁明了。"
    
    print("\n💬 开始对话 (输入 'quit' 退出, 'clear' 清空历史)\n")
    
    while True:
        # 获取用户输入
        user_input = input("你: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == "quit":
            print("\n👋 再见!")
            break
            
        if user_input.lower() == "clear":
            agent.clear_history()
            print("\n🗑️ 对话历史已清空\n")
            continue
        
        try:
            # 调用 LLM
            print("\n🤖 思考中...")
            response = agent.chat(user_input, system_prompt=system_prompt)
            print(f"AI: {response}\n")
            
        except Exception as e:
            print(f"\n❌ 出错: {e}\n")


if __name__ == "__main__":
    main()
