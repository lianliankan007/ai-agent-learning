#!/usr/bin/env python3
"""
Day12: 知识切分（chunking）教学 Demo。

这个脚本的目标不是做生产级切分，而是帮助你观察：
1. 固定长度切分会发生什么
2. 按段落/语义块切分会发生什么
3. 不同切分方式为什么会影响后续检索质量
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


SAMPLE_TEXT = """
RAG 的核心思想，是在大模型回答之前，先从外部知识库中检索相关资料，再把资料放进 Prompt，让模型基于资料生成答案。这样做的价值在于，大模型不必只依赖参数里的旧知识，而可以利用更新、更具体的外部信息。

在一个最小 RAG 系统里，通常会经历几个步骤。第一步是准备知识库文本。第二步是把文本转换成 embedding 向量。第三步是把向量写入向量数据库。第四步是把用户问题也转换成向量。第五步是根据相似度检索出最相关的若干条知识。第六步是把这些知识拼进 Prompt，最后由大模型生成回答。

很多初学者会以为，只要接入 embedding 和向量库，RAG 的检索效果就一定很好。其实并不是这样。检索效果除了受 embedding 模型影响，还会受到知识切分方式、top_k 设置、过滤条件、Prompt 注入方式等因素影响。知识切分往往是最容易被忽视，但又非常关键的一步。

如果把一篇很长的文档整篇直接做成一个向量，问题就会出现。因为一篇长文档往往同时包含多个主题，例如 Token、Context Window、Function Calling、Agent Loop、Memory 等。用户也许只问 Agent Loop，但向量表示里却混入了很多其他主题，结果就是语义不够聚焦，召回结果不够精准。

如果切分得太细，也会带来新的问题。比如一段话的定义在前一句，关键限制条件在后一句。如果切分时把它们拆开，那么虽然某个 chunk 看起来命中了关键词，但真正被检索出来后，信息可能并不完整，模型还是无法稳定回答问题。

所以比较合理的做法，通常不是一味追求更大或更小的 chunk，而是在“语义完整”和“主题聚焦”之间找到平衡。真实项目里常见的做法包括：按自然段切分、按标题层级切分、固定长度加 overlap 切分，或者多种策略组合。chunking 本质上是在为后续 embedding 和检索服务。
""".strip()


@dataclass
class Chunk:
    """表示一条切分后的文本块。"""

    chunk_id: str
    text: str
    strategy: str


def split_paragraphs(text: str) -> List[str]:
    """按空行切分原始文本，得到自然段。"""
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def chunk_by_fixed_length(text: str, chunk_size: int = 120, overlap: int = 30) -> List[Chunk]:
    """按固定长度切分文本。

    这是很多系统的最小起点：实现简单，但容易把一句完整语义截断。
    overlap 的作用是给相邻 chunk 留一点重叠区域，减少信息硬切断的问题。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks: List[Chunk] = []
    start = 0
    chunk_index = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"fixed-{chunk_index}",
                    text=chunk_text,
                    strategy="固定长度切分",
                )
            )
            chunk_index += 1
        # 下一段向前滑动，但保留 overlap，方便上下文衔接。
        start += chunk_size - overlap

    return chunks


def chunk_by_paragraph(paragraphs: Iterable[str], max_chars: int = 180) -> List[Chunk]:
    """按段落聚合切分。

    这里不是简单“一段一个 chunk”，而是把若干自然段组合到一个合理长度内，
    这样既保留语义块，又避免 chunk 过短。
    """

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    chunks: List[Chunk] = []
    current_parts: List[str] = []
    current_length = 0
    chunk_index = 1

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # 如果当前 chunk 加上新段落后过长，就先落盘一个 chunk。
        projected_length = current_length + len(paragraph)
        if current_parts and projected_length > max_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"paragraph-{chunk_index}",
                    text="\n\n".join(current_parts),
                    strategy="按段落切分",
                )
            )
            chunk_index += 1
            current_parts = [paragraph]
            current_length = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_length = projected_length

    if current_parts:
        chunks.append(
            Chunk(
                chunk_id=f"paragraph-{chunk_index}",
                text="\n\n".join(current_parts),
                strategy="按段落切分",
            )
        )

    return chunks


def extract_keywords(query: str) -> List[str]:
    """做一个非常轻量的关键词提取，仅用于观察命中情况。"""
    separators = ["，", "。", "？", "：", "、", " ", "\n"]
    normalized = query
    for separator in separators:
        normalized = normalized.replace(separator, "|")
    return [part.strip().lower() for part in normalized.split("|") if len(part.strip()) >= 2]


def inspect_query_hits(query: str, chunks: List[Chunk]) -> None:
    """观察一个问题在不同 chunk 上的简单命中情况。"""
    keywords = extract_keywords(query)
    print(f"\n问题: {query}")
    print(f"轻量关键词: {keywords}")

    for chunk in chunks:
        lowered_text = chunk.text.lower()
        hits = [keyword for keyword in keywords if keyword in lowered_text]
        print(f"- {chunk.chunk_id} 命中词: {hits if hits else '无'}")


def print_chunks(title: str, chunks: List[Chunk]) -> None:
    """打印切分结果，方便观察。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"chunk 数量: {len(chunks)}")

    for chunk in chunks:
        print(f"\n[{chunk.chunk_id}] {chunk.strategy}")
        print(f"长度: {len(chunk.text)}")
        print(chunk.text)


def main() -> None:
    """程序入口。"""
    print("=" * 72)
    print("Day12 Chunking Demo")
    print("=" * 72)

    paragraphs = split_paragraphs(SAMPLE_TEXT)
    print(f"\n原始段落数: {len(paragraphs)}")
    print(f"原始总长度: {len(SAMPLE_TEXT)}")

    fixed_chunks = chunk_by_fixed_length(SAMPLE_TEXT, chunk_size=120, overlap=30)
    paragraph_chunks = chunk_by_paragraph(paragraphs, max_chars=180)

    print_chunks("固定长度切分结果", fixed_chunks)
    print_chunks("按段落切分结果", paragraph_chunks)

    # 用两个示例问题，帮助你初步感受“切分方式会影响后续检索观察”。
    inspect_query_hits("为什么知识切分会影响检索质量？", fixed_chunks)
    inspect_query_hits("为什么知识切分会影响检索质量？", paragraph_chunks)

    inspect_query_hits("如果 chunk 太小，会有什么问题？", fixed_chunks)
    inspect_query_hits("如果 chunk 太小，会有什么问题？", paragraph_chunks)

    print(
        "\n观察建议:\n"
        "1. 看固定长度切分是否把一句完整语义截断了\n"
        "2. 看按段落切分是否更容易保留一个完整知识点\n"
        "3. 思考：如果后面要做 embedding，这两种切分会如何影响向量表达"
    )


if __name__ == "__main__":
    main()
