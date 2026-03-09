#!/usr/bin/env python3
"""
Agent 运行器模块
负责管理多个 Agent，处理用户交互
"""

from typing import Dict, Optional
from llm_agent import LLMAgent


class AgentRunner:
    """
    Agent 运行器类
    管理多个 Agent，支持切换，处理用户交互
    """
    
    def __init__(self):
        """初始化 Agent 运行器"""
        # 存储所有可用的 Agent，用字典管理
        self.agents: Dict[str, LLMAgent] = {}
        # 当前正在使用的 Agent
        self.current_agent: Optional[LLMAgent] = None
        # 当前 Agent 的名称
        self.current_agent_name: str = ""
        # 系统提示词
        self.system_prompt: str = "你是一个 helpful 的 AI 助手，回答简洁明了。"
    
    def register_agent(self, name: str, agent: LLMAgent) -> None:
        """
        注册一个 Agent
        
        Args:
            name: Agent 的名称标识
            agent: LLMAgent 实例
        """
        self.agents[name] = agent
        print(f"✅ 已注册 Agent: {agent.get_info()}")
    
    def switch_agent(self, name: str) -> bool:
        """
        切换到指定的 Agent
        
        Args:
            name: Agent 的名称
            
        Returns:
            切换是否成功
        """
        if name not in self.agents:
            print(f"❌ Agent '{name}' 不存在，可用 Agent: {list(self.agents.keys())}")
            return False
        
        self.current_agent = self.agents[name]
        self.current_agent_name = name
        print(f"🔄 已切换到: {self.current_agent.get_info()}")
        return True
    
    def list_agents(self) -> None:
        """列出所有可用的 Agent"""
        print("\n📋 可用 Agent 列表:")
        for name, agent in self.agents.items():
            marker = " 👈 当前" if name == self.current_agent_name else ""
            print(f"  - {name}: {agent.get_info()}{marker}")
        print()
    
    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt
        print(f"📝 系统提示词已更新")
    
    def chat(self, message: str) -> Optional[str]:
        """
        与当前 Agent 对话
        
        Args:
            message: 用户消息
            
        Returns:
            AI 回复内容，如果没有当前 Agent 返回 None
        """
        if self.current_agent is None:
            print("❌ 请先切换到一个 Agent (使用 'switch <name>')")
            return None
        
        try:
            print(f"\n🤖 [{self.current_agent_name}] 思考中...")
            response = self.current_agent.chat(message, system_prompt=self.system_prompt)
            return response
        except Exception as e:
            print(f"\n❌ 出错: {e}")
            return None
    
    def clear_history(self) -> None:
        """清空当前 Agent 的对话历史"""
        if self.current_agent:
            self.current_agent.clear_history()
            print(f"🗑️ [{self.current_agent_name}] 对话历史已清空")
        else:
            print("❌ 没有当前 Agent")
    
    def run(self) -> None:
        """启动交互式命令行"""
        print("=" * 60)
        print("🤖 多 Agent 对话系统")
        print("=" * 60)
        print("\n可用命令:")
        print("  list              - 列出所有 Agent")
        print("  switch <name>     - 切换到指定 Agent")
        print("  clear             - 清空当前 Agent 历史")
        print("  prompt <text>     - 设置系统提示词")
        print("  quit/exit         - 退出程序")
        print("  temperature [0, 2) - 设置代理多样性"  )
        print("  <任意文字>        - 与当前 Agent 对话\n")
        
        # 如果没有 Agent，提示注册
        if not self.agents:
            print("⚠️ 还没有注册任何 Agent，请先注册 Agent\n")
        
        while True:
            try:
                # 显示当前 Agent 提示符
                prompt = f"[{self.current_agent_name}]> " if self.current_agent else "[无Agent]> "
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ["quit", "exit"]:
                    print("\n👋 再见!")
                    break
                
                elif user_input.lower() == "list":
                    self.list_agents()
                
                elif user_input.lower().startswith("switch "):
                    name = user_input[7:].strip()
                    self.switch_agent(name)
                elif user_input.lower().startswith("temperature "):
                    new_temperature = user_input[12:].strip()
                    self.current_agent.temperature = float(new_temperature)
                    print("temperature: ", new_temperature)
                elif user_input.lower() == "clear":
                    self.clear_history()
                
                elif user_input.lower().startswith("prompt "):
                    new_prompt = user_input[7:].strip()
                    self.set_system_prompt(new_prompt)
                
                else:
                    # 普通对话
                    response = self.chat(user_input)
                    if response:
                        print(f"AI: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}\n")


def main():
    """示例用法：创建并运行多个 Agent"""
    
    # 创建运行器
    runner = AgentRunner()
    
    # 创建第一个 Agent - 通用助手
    try:
        general_agent = LLMAgent(
            name="通用助手",
            model="qwen3.5-plus"
        )
        runner.register_agent("general", general_agent)
    except ValueError as e:
        print(f"⚠️ 无法创建通用助手: {e}")
    
    # 创建第二个 Agent - 代码专家（使用同样的 API Key）
    try:
        code_agent = LLMAgent(
            name="代码专家",
            model="qwen3.5-plus"
        )
        runner.register_agent("code", code_agent)
    except ValueError as e:
        print(f"⚠️ 无法创建代码专家: {e}")
    
    # 如果有可用的 Agent，默认切换到第一个
    if runner.agents:
        first_agent = list(runner.agents.keys())[0]
        runner.switch_agent(first_agent)
        
        # 启动交互
        runner.run()
    else:
        print("\n❌ 没有可用的 Agent，请检查 API Key 设置")
        print("设置环境变量: export OPENAI_API_KEY='your-api-key'")


if __name__ == "__main__":
    main()
