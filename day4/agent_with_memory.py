#!/usr/bin/env python3
"""
day4: 支持长期记忆的 Agent 示例

默认组合:
- LLM: DashScope OpenAI 兼容接口
- Embedding: Ollama 本地 embedding 模型
- Vector DB: 本地 Qdrant（默认无鉴权）

运行前可设置的环境变量:
  OPENAI_API_KEY
  OPENAI_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
  OPENAI_MODEL=qwen3.5-plus
  OLLAMA_BASE_URL=http://10.66.131.38:6333
  OLLAMA_EMBED_MODEL=nomic-embed-text
  QDRANT_HOST=localhost
  QDRANT_PORT=6333
  MEMORY_COLLECTION=day4_agent_memory
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from utils.openai_config import resolve_openai_api_key
from vector_retriever import MemorySearchResult, QdrantMemoryRetriever

class OllamaEmbedder:
    """负责把文本转换成向量。"""

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_single(self, text: str) -> List[float]:
        """调用 Ollama 的 `/api/embed` 接口生成单条文本的向量。"""
        # 这里使用短生命周期的客户端，代码更直观，也方便控制超时。
        with httpx.Client(timeout=self.timeout) as client:
            # Ollama 支持批量输入，这里虽然只传一条，也统一走 list 结构。
            resp = client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": [text]},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data["embeddings"]
            # 正常情况下应至少返回一个向量；没有时说明服务异常。
            if not embeddings:
                raise RuntimeError("Ollama 没有返回 embedding")
            # 因为只传入了一条文本，所以直接拿第 0 个向量。
            return embeddings[0]


class QdrantMemoryStore:
    """负责长期记忆的存储层。

    这个类只处理“写入记忆”和“准备检索器”，真正的检索算法被拆到了
    `vector_retriever.py`，这样职责更清晰。
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        # 下面两个对象都采用懒加载，避免构造时就立刻连接外部服务。
        self._client: Optional[QdrantClient] = None
        self._retriever: Optional[QdrantMemoryRetriever] = None

    @property
    def client(self) -> QdrantClient:
        """首次访问时创建 QdrantClient，之后复用同一个实例。"""
        if self._client is None:
            # 这个示例默认连接本地 Qdrant，因此只保留 host/port 即可。
            self._client = QdrantClient(host=self.host, port=self.port, timeout=30)
        return self._client

    @property
    def retriever(self) -> QdrantMemoryRetriever:
        """首次访问时构造检索器，并把当前 collection 信息传进去。"""
        if self._retriever is None:
            self._retriever = QdrantMemoryRetriever(
                client=self.client,
                collection_name=self.collection_name,
            )
        return self._retriever

    def ensure_collection(self, vector_size: int) -> None:
        """确保目标 collection 已存在，不存在时自动创建。"""
        collections = self.client.get_collections()
        names = [item.name for item in collections.collections]
        if self.collection_name in names:
            return
        # 这里指定余弦距离，表示检索时按向量方向相似度比较。
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def add_memory(self, content: str, vector: List[float], metadata: Dict[str, Any]) -> str:
        """把一条长期记忆写入 Qdrant。"""
        # 写入前先保证 collection 存在，而且向量维度匹配。
        self.ensure_collection(len(vector))
        # 如果调用方没有指定 id，就自动生成一个 UUID。
        memory_id = metadata.get("id") or str(uuid.uuid4())
        # 记忆正文放在 `content` 字段里，其他信息都作为 payload 元数据保存。
        payload = {"content": content, **metadata}
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        return memory_id


class MemoryAwareAgent:
    """具备短期上下文和长期记忆的 Agent。"""

    def __init__(
        self,
        embedder: OllamaEmbedder,
        memory_store: QdrantMemoryStore,
        user_id: str = "default-user",
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.4,
        max_tokens: int = 800,
    ):
        self.embedder = embedder
        self.memory_store = memory_store
        # `user_id` 用来做多用户隔离，不同用户可以共享同一个 Qdrant collection。
        self.user_id = user_id
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "请提供 api_key，"
                "或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # `messages` 保存短期对话历史，只在当前进程内生效，重启后会丢失。
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = "你是一个具备长期记忆能力的学习助手，回答时优先利用历史记忆并保持结构清晰。"
        # 默认用 hybrid 检索，因为它兼顾语义相似度、关键词和重要度。
        self.retrieval_mode = "hybrid"

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        topic: str = "general",
        tags: Optional[List[str]] = None,
        importance: float = 0.6,
    ) -> str:
        """把一段内容转成向量并写入长期记忆库。"""
        vector = self.embedder.embed_single(content)
        # 这些元数据会跟正文一起存到 Qdrant，后续检索和过滤都要用到。
        metadata = {
            "user_id": self.user_id,
            "memory_type": memory_type,
            "topic": topic,
            "tags": tags or [],
            "importance": round(importance, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return self.memory_store.add_memory(content, vector, metadata)

    def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: Optional[str] = None,
        memory_type: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """根据查询语句召回长期记忆。"""
        mode = retrieval_mode or self.retrieval_mode
        # 无论用哪种模式，第一步都需要先把查询语句转成向量。
        vector = self.embedder.embed_single(query)
        # 默认按当前用户过滤，避免不同用户的记忆混在一起。
        filters = {"user_id": self.user_id}
        if memory_type:
            filters["memory_type"] = memory_type
        if topic:
            filters["topic"] = topic

        # simple: 只看向量相似度，不加过滤。
        if mode == "simple":
            return self.memory_store.retriever.semantic_search(vector, top_k=top_k)
        # filtered: 向量检索 + 元数据过滤。
        if mode == "filtered":
            return self.memory_store.retriever.semantic_search(
                vector,
                top_k=top_k,
                filters=filters,
            )
        # hybrid: 过滤后先扩大召回，再综合多个信号重排。
        if mode == "hybrid":
            return self.memory_store.retriever.hybrid_search(
                query,
                vector,
                top_k=top_k,
                filters=filters,
            )
        raise ValueError(f"不支持的检索模式: {mode}")

    def chat(self, user_message: str) -> str:
        """完成一次完整聊天：检索记忆、调用模型、更新上下文、尝试记忆抽取。"""
        # 先从长期记忆里召回与当前问题相关的内容。
        memories = self.retrieve_memories(user_message, top_k=4, retrieval_mode=self.retrieval_mode)
        memory_context = self._build_memory_context(memories)
        # 把检索结果拼进 system prompt，让模型优先参考历史事实。
        composed_system_prompt = (
            f"{self.system_prompt}\n\n"
            "以下是检索到的长期记忆，请优先参考，但不要编造不存在的历史。\n"
            f"{memory_context}"
        )

        # 真正发起聊天请求时，会同时带上短期对话历史。
        response = self._call_api(
            messages=self._compose_messages(user_message, composed_system_prompt),
        )

        # 把本轮问答加入短期记忆，供后续连续对话使用。
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": response})

        # 最后尝试从用户输入中抽取“值得长期保存”的信息。
        extracted = self._extract_memory(user_message)
        if extracted is not None:
            self.remember(**extracted)

        return response

    def clear_history(self) -> None:
        """只清空短期对话历史，不删除 Qdrant 里的长期记忆。"""
        self.messages = []

    def _compose_messages(self, user_message: str, system_prompt: str) -> List[Dict[str, str]]:
        """按 Chat Completions 接口要求拼装消息列表。"""
        # 顺序很重要：system 在最前面，然后是历史对话，最后是当前问题。
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self.messages)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用 DashScope OpenAI 兼容接口获取模型回答。"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        # OpenAI 兼容接口通常把主回答放在 `choices[0].message.content`。
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _build_memory_context(memories: Iterable[MemorySearchResult]) -> str:
        """把检索结果整理成适合拼进提示词的纯文本。"""
        rows = list(memories)
        if not rows:
            return "暂无长期记忆。"

        lines = []
        for index, item in enumerate(rows, start=1):
            memory_type = item.metadata.get("memory_type", "fact")
            topic = item.metadata.get("topic", "general")
            lines.append(
                f"{index}. [{memory_type}/{topic}] {item.content} (score={item.score:.3f})"
            )
        return "\n".join(lines)

    def _extract_memory(self, user_message: str) -> Optional[Dict[str, Any]]:
        """用简单规则从用户输入里提取长期记忆。"""
        text = user_message.strip()
        if not text:
            return None

        # 每条规则包含：匹配正则、记忆类型、主题、重要度。
        rules = [
            (r"请记住[:：]?\s*(.+)", "instruction", "explicit_memory", 0.95),
            (r"我叫(.+)", "profile", "identity", 0.85),
            (r"我喜欢(.+)", "preference", "preference", 0.80),
            (r"我不喜欢(.+)", "preference", "dislike", 0.80),
            (r"我正在学习(.+)", "progress", "learning", 0.75),
            (r"我的项目是(.+)", "project", "project", 0.75),
        ]

        for pattern, memory_type, topic, importance in rules:
            match = re.search(pattern, text)
            if match:
                # 显式要求“请记住”的内容直接原文入库，其他规则统一加一个来源前缀。
                content = text if memory_type == "instruction" else f"用户说: {text}"
                return {
                    "content": content,
                    "memory_type": memory_type,
                    "topic": topic,
                    "importance": importance,
                    "tags": [topic],
                }
        return None


class MemoryAgentRunner:
    """Command-line runner for the Day 4 memory agent."""

    def __init__(self, agent: MemoryAwareAgent):
        self.agent = agent

    def run(self) -> None:
        """Start the interactive CLI loop."""
        print("=" * 72)
        print("Day4 Agent Memory Demo")
        print("=" * 72)
        print("\nCommands:")
        print("  help                              - show help")
        print("  mode <simple|filtered|hybrid>     - switch retrieval mode")
        print("  remember <text>                   - save a memory")
        print("  remember-as <type> <topic> <text> - save memory with metadata")
        print("  search <text>                     - search memories")
        print("  searchf <type> <text>             - filtered memory search")
        print("  user <user_id>                    - switch current user")
        print("  clear                             - clear short-term history")
        print("  prompt <text>                     - update system prompt")
        print("  quit/exit                         - exit")
        print("  <any text>                        - normal chat with memory\n")

        while True:
            try:
                user_input = input(f"[memory:{self.agent.user_id}]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\nBye!")
                    break

                if lower_text == "help":
                    print("Example: remember I prefer concise answers")
                    print("Example: mode hybrid")
                    print("Example: search what kind of answer style do I prefer\n")
                    continue

                if lower_text.startswith("mode "):
                    mode = user_input[5:].strip()
                    if mode not in {"simple", "filtered", "hybrid"}:
                        print("[ERROR] Only simple / filtered / hybrid are supported\n")
                        continue
                    self.agent.retrieval_mode = mode
                    print(f"[OK] Current retrieval mode: {mode}\n")
                    continue

                if lower_text.startswith("remember-as "):
                    parts = user_input.split(" ", 3)
                    if len(parts) < 4:
                        print("[ERROR] Usage: remember-as <type> <topic> <text>\n")
                        continue
                    memory_id = self.agent.remember(
                        content=parts[3],
                        memory_type=parts[1],
                        topic=parts[2],
                    )
                    print(f"[OK] Saved long-term memory: {memory_id}\n")
                    continue

                if lower_text.startswith("remember "):
                    content = user_input[9:].strip()
                    memory_id = self.agent.remember(content=content)
                    print(f"[OK] Saved long-term memory: {memory_id}\n")
                    continue

                if lower_text.startswith("searchf "):
                    parts = user_input.split(" ", 2)
                    if len(parts) < 3:
                        print("[ERROR] Usage: searchf <type> <text>\n")
                        continue
                    results = self.agent.retrieve_memories(
                        query=parts[2],
                        top_k=5,
                        retrieval_mode="filtered",
                        memory_type=parts[1],
                    )
                    self._print_results(results)
                    continue

                if lower_text.startswith("search "):
                    query = user_input[7:].strip()
                    results = self.agent.retrieve_memories(query=query, top_k=5)
                    self._print_results(results)
                    continue

                if lower_text == "clear":
                    self.agent.clear_history()
                    print("[OK] Cleared short-term history\n")
                    continue

                if lower_text.startswith("prompt "):
                    self.agent.system_prompt = user_input[7:].strip()
                    print("[OK] System prompt updated\n")
                    continue

                if lower_text.startswith("user "):
                    self.agent.user_id = user_input[5:].strip()
                    print(f"[OK] Switched current user to: {self.agent.user_id}\n")
                    continue

                print("\n[agent] Thinking...")
                answer = self.agent.chat(user_input)
                print(f"AI: {answer}\n")
            except KeyboardInterrupt:
                print("\n\nBye!")
                break
            except Exception as exc:
                print(f"\n[ERROR] {exc}\n")

    @staticmethod
    def _print_results(results: List[MemorySearchResult]) -> None:
        """Print formatted memory search results."""
        if not results:
            print("[INFO] No related memories found\n")
            return

        print("\nSearch results:")
        for index, item in enumerate(results, start=1):
            memory_type = item.metadata.get("memory_type", "fact")
            topic = item.metadata.get("topic", "general")
            print(f"  {index}. [{memory_type}/{topic}] score={item.score:.3f}")
            print(f"     {item.content}")
        print()

def build_agent() -> MemoryAwareAgent:
    """根据环境变量构造一个可运行的 Agent。"""
    embedder = OllamaEmbedder(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://10.66.131.38:6333"),
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )
    memory_store = QdrantMemoryStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("MEMORY_COLLECTION", "day4_agent_memory"),
    )
    return MemoryAwareAgent(
        embedder=embedder,
        memory_store=memory_store,
        user_id=os.getenv("MEMORY_USER_ID", "default-user"),
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )


def seed_demo_memories(agent: MemoryAwareAgent) -> None:
    """向记忆库注入几条演示数据，方便本地快速体验。"""
    demo_rows = [
        ("用户喜欢中文回答，并偏好结构化说明", "preference", "style", 0.9),
        ("用户正在学习 Agent Memory、Embedding 和向量检索", "progress", "learning", 0.88),
        ("当前 day4 目标是实现一个支持长期记忆的 Agent", "project", "goal", 0.86),
    ]
    for content, memory_type, topic, importance in demo_rows:
        agent.remember(
            content=content,
            memory_type=memory_type,
            topic=topic,
            importance=importance,
            tags=[topic, "demo"],
        )


def main() -> None:
    """程序入口：构造 Agent，可选注入演示数据，然后启动 CLI。"""
    agent = build_agent()
    if os.getenv("DAY4_SEED_DEMO", "false").lower() == "true":
        seed_demo_memories(agent)
        print("🌱 已写入演示记忆")

    runner = MemoryAgentRunner(agent)
    runner.run()


if __name__ == "__main__":
    main()
