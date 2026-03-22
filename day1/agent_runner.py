#!/usr/bin/env python3
"""
Day 1 multi-agent runner.

This CLI keeps terminal output ASCII-first to avoid mojibake in GBK-based
Windows consoles.
"""

from __future__ import annotations

from typing import Dict, Optional

from llm_agent import LLMAgent


class AgentRunner:
    """Manage multiple agents and provide a simple CLI."""

    def __init__(self):
        self.agents: Dict[str, LLMAgent] = {}
        self.current_agent: Optional[LLMAgent] = None
        self.current_agent_name: str = ""
        self.system_prompt: str = "You are a helpful AI assistant. Keep answers clear and short."

    def register_agent(self, name: str, agent: LLMAgent) -> None:
        self.agents[name] = agent
        print(f"[OK] Registered agent: {agent.get_info()}")

    def switch_agent(self, name: str) -> bool:
        if name not in self.agents:
            print(f"[ERROR] Agent '{name}' not found. Available: {list(self.agents.keys())}")
            return False

        self.current_agent = self.agents[name]
        self.current_agent_name = name
        print(f"[OK] Switched to: {self.current_agent.get_info()}")
        return True

    def list_agents(self) -> None:
        print("\nAvailable agents:")
        for name, agent in self.agents.items():
            marker = " [current]" if name == self.current_agent_name else ""
            print(f"  - {name}: {agent.get_info()}{marker}")
        print()

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        print("[OK] System prompt updated")

    def chat(self, message: str) -> Optional[str]:
        if self.current_agent is None:
            print("[ERROR] Switch to an agent first with: switch <name>")
            return None

        try:
            print(f"\n[{self.current_agent_name}] Thinking...")
            return self.current_agent.chat(message, system_prompt=self.system_prompt)
        except Exception as exc:
            print(f"\n[ERROR] Request failed: {exc}")
            return None

    def clear_history(self) -> None:
        if self.current_agent:
            self.current_agent.clear_history()
            print(f"[OK] Cleared history for: {self.current_agent_name}")
        else:
            print("[ERROR] No current agent")

    def run(self) -> None:
        print("=" * 60)
        print("Day1 Multi-Agent Chat")
        print("=" * 60)
        print("\nCommands:")
        print("  list               - list all agents")
        print("  switch <name>      - switch current agent")
        print("  clear              - clear current history")
        print("  prompt <text>      - set system prompt")
        print("  temperature <num>  - set temperature")
        print("  quit/exit          - exit")
        print("  <any text>         - chat with current agent\n")

        if not self.agents:
            print("[WARN] No agents registered yet\n")

        while True:
            try:
                prompt = f"[{self.current_agent_name}]> " if self.current_agent else "[no-agent]> "
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\nBye!")
                    break
                if lower_text == "list":
                    self.list_agents()
                    continue
                if lower_text.startswith("switch "):
                    self.switch_agent(user_input[7:].strip())
                    continue
                if lower_text.startswith("temperature "):
                    if self.current_agent is None:
                        print("[ERROR] Switch to an agent first")
                        continue
                    new_temperature = float(user_input[12:].strip())
                    self.current_agent.temperature = new_temperature
                    print(f"temperature: {new_temperature}")
                    continue
                if lower_text == "clear":
                    self.clear_history()
                    continue
                if lower_text.startswith("prompt "):
                    self.set_system_prompt(user_input[7:].strip())
                    continue

                response = self.chat(user_input)
                if response:
                    print(f"AI: {response}\n")
            except KeyboardInterrupt:
                print("\n\nBye!")
                break
            except Exception as exc:
                print(f"\n[ERROR] {exc}\n")


def main() -> None:
    runner = AgentRunner()

    try:
        general_agent = LLMAgent(name="general-assistant", model="qwen3.5-plus")
        runner.register_agent("general", general_agent)
    except ValueError as exc:
        print(f"[WARN] Could not create general agent: {exc}")

    try:
        code_agent = LLMAgent(name="code-assistant", model="qwen3.5-plus")
        runner.register_agent("code", code_agent)
    except ValueError as exc:
        print(f"[WARN] Could not create code agent: {exc}")

    if runner.agents:
        first_agent = list(runner.agents.keys())[0]
        runner.switch_agent(first_agent)
        runner.run()
    else:
        print("\n[ERROR] No available agent. Check your API key setup.")
        print("Set OPENAI_API_KEY in the project .env file.")


if __name__ == "__main__":
    main()
