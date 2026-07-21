"""Minimal end-to-end demo of the AwareLiquid Qwen adapter.

Exercises the full ingest -> retrieve -> compress -> answer loop.

The chat client fails closed, so pick one of:

* offline mock  — ``AWARELIQUID_LLM_BACKEND=mock`` **and** ``AWARELIQUID_TEST_MODE=1``
* real model    — ``AWARELIQUID_LLM_API_KEY`` (or ``DASHSCOPE_API_KEY``)

With neither set this script raises rather than silently faking answers.
"""

from awareliquid import MemoryQAAgent, RetrievalConfig, summarize_usage

# A tiny stand-in "financial report". Real documents are far longer; the adapter
# is designed for reports that do not fit in a single context window.
REPORT = """
本公司 2023 年度实现营业收入 124.5 亿元，同比增长 8.3%。
归属于母公司股东的净利润为 18.2 亿元，较上年同期增长 12.1%。
研发投入 9.7 亿元，占营业收入的 7.8%。
公司拟每 10 股派发现金红利 3.5 元（含税）。
2022 年营业收入为 114.9 亿元，净利润为 16.2 亿元。
""".strip()


def main() -> None:
    agent = MemoryQAAgent(config=RetrievalConfig(retrieval_backend="lexical"))
    n_chunks = agent.ingest_document("annual-2023", REPORT)
    print(f"ingested annual-2023 into {n_chunks} chunk(s)\n")

    questions = [
        {
            "qid": "q1",
            "question": "公司 2023 年营业收入是多少？",
            "options": ["114.9 亿元", "124.5 亿元", "18.2 亿元", "9.7 亿元"],
            "qtype": "mcq",
        },
        {
            "qid": "q2",
            "question": "2023 年净利润同比是否实现增长？",
            "options": ["是", "否"],
            "qtype": "tf",
        },
    ]

    results = []
    for q in questions:
        res = agent.answer_question(doc_ids=["annual-2023"], **q)
        results.append(res)
        print(f"[{res.qid}] answer={res.answer!r}  tokens={res.usage.total_tokens}")

    total = summarize_usage(results)
    print(f"\ntotal generation tokens: {total.total_tokens} "
          f"(prompt={total.prompt_tokens}, completion={total.completion_tokens})")


if __name__ == "__main__":
    main()
