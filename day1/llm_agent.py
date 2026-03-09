#!/usr/bin/env python3
"""
LLM Agent 模块
负责与 LLM API 通信
"""

import os
from typing import Optional, List, Dict, Any
import requests


class LLMAgent:
    """
    LLM Agent 类
    负责调用大语言模型 API，维护对话历史
    """
    
    def __init__(
        self,
        name: str = "默认助手",
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 1.9,
        max_tokens: int = 1000
    ):
        """
        初始化 LLM Agent
        
        Args:
            name: Agent 的名称，用于显示
            api_key: API 密钥，默认从环境变量 OPENAI_API_KEY 读取
            base_url: API 基础 URL
            model: 模型名称
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
        """
        self.name = name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(f"[{name}] 请提供 api_key 或设置 OPENAI_API_KEY 环境变量")
        
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
    
    def get_info(self) -> str:
        """获取 Agent 信息"""
        return f"[{self.name}] 模型: {self.model}"
