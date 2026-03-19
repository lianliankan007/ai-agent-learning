#!/usr/bin/env python3
"""
Day 2 Runner
参照 Day 1 的交互风格，新增：
- 参数实验
- top_p / max_tokens / retries 设置
- 多轮对话上下文查看
"""

from typing import Optional

from day2_llm_agent import LLMAgent


class Day2Runner:
    def __init__(self):
        self.agent: Optional[LLMAgent] = None
        self.system_prompt = "你是一个学习助手，回答简洁、结构化、可执行。"

    def create_agent(self) -> bool:
        try:
            self.agent = LLMAgent(name="Day2学习助手")
            print(f"✅ Agent 已创建: {self.agent.get_info()}")
            return True
        except ValueError as e:
            print(f"❌ 创建失败: {e}")
            return False

    def run_experiment(self, prompt: str) -> None:
        if not self.agent:
            print("❌ 请先创建 Agent")
            return

        print("\n🧪 参数实验开始...\n")
        results = self.agent.run_parameter_experiment(prompt=prompt, system_prompt=self.system_prompt)
        for item in results:
            print(
                f"Case {item['case']} | temperature={item['temperature']} | "
                f"top_p={item['top_p']} | max_tokens={item['max_tokens']}"
            )
            print(item["answer"])
            print(f"[length={item['length']} chars]\n")

    def chat(self, message: str) -> None:
        if not self.agent:
            print("❌ 请先创建 Agent")
            return

        try:
            print("\n🤖 思考中...")
            ans = self.agent.chat(message=message, system_prompt=self.system_prompt)
            print(f"AI: {ans}\n")
        except Exception as e:
            print(f"❌ 对话失败: {e}")

    def run(self) -> None:
        print("=" * 60)
        print("Day 2 LLM 学习控制台")
        print("=" * 60)
        print("命令：")
        print("  chat <text>                 - 对话")
        print("  experiment <text>           - 参数实验")
        print("  temp <float>                - 设置 temperature")
        print("  top_p <float>               - 设置 top_p")
        print("  max_tokens <int>            - 设置 max_tokens")
        print("  retries <int>               - 设置最大重试次数")
        print("  history                     - 查看历史条数")
        print("  clear                       - 清空历史")
        print("  prompt <text>               - 设置 system prompt")
        print("  info                        - 查看当前配置")
        print("  quit/exit                   - 退出\n")

        if not self.create_agent():
            return

        while True:
            try:
                user_input = input("[day2]> ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("\n👋 再见")
                    break

                if user_input.lower().startswith("chat "):
                    self.chat(user_input[5:].strip())
                    continue

                if user_input.lower().startswith("experiment "):
                    self.run_experiment(user_input[11:].strip())
                    continue

                if user_input.lower().startswith("temp ") and self.agent:
                    self.agent.temperature = float(user_input[5:].strip())
                    print(f"temperature => {self.agent.temperature}")
                    continue

                if user_input.lower().startswith("top_p ") and self.agent:
                    self.agent.top_p = float(user_input[6:].strip())
                    print(f"top_p => {self.agent.top_p}")
                    continue

                if user_input.lower().startswith("max_tokens ") and self.agent:
                    self.agent.max_tokens = int(user_input[11:].strip())
                    print(f"max_tokens => {self.agent.max_tokens}")
                    continue

                if user_input.lower().startswith("retries ") and self.agent:
                    self.agent.max_retries = int(user_input[8:].strip())
                    print(f"max_retries => {self.agent.max_retries}")
                    continue

                if user_input.lower() == "history" and self.agent:
                    print(f"当前历史条数: {len(self.agent.get_history())}")
                    continue

                if user_input.lower() == "clear" and self.agent:
                    self.agent.clear_history()
                    print("🗑️ 历史已清空")
                    continue

                if user_input.lower().startswith("prompt "):
                    self.system_prompt = user_input[7:].strip()
                    print("📝 system prompt 已更新")
                    continue

                if user_input.lower() == "info" and self.agent:
                    print(self.agent.get_info())
                    continue

                print("❓ 未知命令，请使用 chat/experiment/temp/top_p/max_tokens/retries/history/clear/prompt/info")

            except KeyboardInterrupt:
                print("\n\n👋 再见")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def main() -> None:
    Day2Runner().run()


if __name__ == "__main__":
    main()
