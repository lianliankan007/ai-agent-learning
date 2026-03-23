#!/usr/bin/env python3
"""
Day14: RAG 问题定位教学 Demo。

这个脚本不追求真实 embedding 或真实向量库，而是把常见问题做成可观察的案例，
帮助你练习判断：
1. 这是 chunk 问题吗
2. 这是 top_k 问题吗
3. 这是 filter 问题吗
4. 检索结果到底有没有把正确知识召回出来
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DemoResult:
    """保存一次案例运行结果。"""

    case_name: str
    question: str
    docs: List[str] = field(default_factory=list)
    diagnosis: str = ""
    suggestion: str = ""


CASE_LIBRARY: Dict[str, Dict[str, object]] = {
    "bad_chunk": {
        "question": "什么是 Agent Loop？",
        "docs": [
            "长 chunk：这篇知识同时讲了 Token、Context Window、Function Calling、Agent Loop、Memory，内容很多，但和 Agent Loop 直接相关的信息只占一小部分。",
            "另一个长 chunk：这篇知识还包含 RAG、chunking、top_k、filter 等多个主题。",
        ],
        "diagnosis": "这是典型的 chunk 过大问题。主题过多导致检索虽然命中了，但信息不聚焦。",
        "suggestion": "把长文档按主题或段落切分，让一个 chunk 尽量只表达一个明确知识点。",
    },
    "topk_noise": {
        "question": "为什么 top_k 太大会有噪音？",
        "docs": [
            "正确知识：top_k 太大时，系统会把更多相似度较低的内容也一起召回。",
            "半相关知识：chunk 切分也会影响检索质量。",
            "半相关知识：Memory 系统常用向量库做长期记忆。",
            "弱相关知识：Function Calling 可以调用外部工具。",
            "弱相关知识：Context Window 会限制一次对话能看到的内容。",
        ],
        "diagnosis": "这是典型的 top_k 过大问题。正确知识虽然出现了，但混入了大量噪音。",
        "suggestion": "降低 top_k，或者增加 rerank / filter，让后面的弱相关内容不要进入 Prompt。",
    },
    "missing_filter": {
        "question": "为什么要对检索结果加 topic 过滤？",
        "docs": [
            "retrieval 主题：过滤检索可以先缩小范围，再做相似度搜索，提高精度。",
            "memory 主题：长期记忆常保存在向量库中。",
            "tool_use 主题：Function Calling 让模型决定是否调用工具。",
        ],
        "diagnosis": "这是典型的缺少 filter 问题。问题明明在 retrieval 主题下，却混入了其他主题。",
        "suggestion": "优先增加 topic、memory_type、user_id 等过滤条件，减少跨主题误召回。",
    },
    "good_retrieval_bad_generation": {
        "question": "为什么回答前要先检索资料？",
        "docs": [
            "正确知识：RAG 的核心是先检索，再把资料拼进 Prompt，最后让模型基于资料回答。",
            "补充知识：检索的价值在于引入外部最新知识，降低幻觉。",
        ],
        "diagnosis": "这个案例里检索结果基本是对的。如果最终回答还是不好，更可能是 Prompt 注入或生成阶段的问题。",
        "suggestion": "检查 Prompt 是否明确要求优先依据检索资料回答，以及模型是否真正使用了检索结果。",
    },
}


class RagDiagnosisRunner:
    """Day14 命令行入口。"""

    def run(self) -> None:
        print("=" * 72)
        print("Day14 RAG Diagnosis Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-cases              - 查看所有案例")
        print("  show <case_name>        - 查看单个案例")
        print("  demo-bad-chunk          - 演示 chunk 过大问题")
        print("  demo-topk-noise         - 演示 top_k 噪音问题")
        print("  demo-filter             - 演示缺少 filter 问题")
        print("  demo-generation         - 演示检索正确但生成阶段仍可能出问题")
        print("  review-all              - 依次查看全部案例")
        print("  quit/exit               - 退出")
        print()

        while True:
            try:
                user_input = input("[day14-rag-diagnosis]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "list-cases":
                    self._list_cases()
                    continue

                if lower_text == "demo-bad-chunk":
                    self._show_case("bad_chunk")
                    continue

                if lower_text == "demo-topk-noise":
                    self._show_case("topk_noise")
                    continue

                if lower_text == "demo-filter":
                    self._show_case("missing_filter")
                    continue

                if lower_text == "demo-generation":
                    self._show_case("good_retrieval_bad_generation")
                    continue

                if lower_text == "review-all":
                    for case_name in CASE_LIBRARY:
                        self._show_case(case_name)
                    continue

                if lower_text.startswith("show "):
                    case_name = user_input[5:].strip()
                    self._show_case(case_name)
                    continue

                print("不支持的命令，请输入 list-cases / show / review-all / quit\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")

    @staticmethod
    def _list_cases() -> None:
        print("\n当前案例:")
        for case_name in CASE_LIBRARY:
            print(f"  - {case_name}")
        print()

    @staticmethod
    def _show_case(case_name: str) -> None:
        case = CASE_LIBRARY.get(case_name)
        if not case:
            print(f"\n未找到案例: {case_name}\n")
            return

        result = DemoResult(
            case_name=case_name,
            question=str(case["question"]),
            docs=[str(item) for item in case["docs"]],
            diagnosis=str(case["diagnosis"]),
            suggestion=str(case["suggestion"]),
        )

        print("\n" + "=" * 72)
        print(f"[Case] {result.case_name}")
        print("=" * 72)
        print(f"问题: {result.question}")
        print("\n检索结果:")
        for index, doc in enumerate(result.docs, start=1):
            print(f"  {index}. {doc}")
        print(f"\n问题定位: {result.diagnosis}")
        print(f"优化建议: {result.suggestion}\n")


def main() -> None:
    runner = RagDiagnosisRunner()
    runner.run()


if __name__ == "__main__":
    main()
