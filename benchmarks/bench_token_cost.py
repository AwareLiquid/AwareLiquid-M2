"""Measure generation-token cost per question, without spending a cent.

The scoring rule is ``accuracy x (0.7 + 0.3 x token_efficiency)`` against a
5,000,000-token budget, so how many Qwen calls a single question triggers — and
how big each prompt is — directly moves 30% of the score.

This harness wraps the chat client in a counter and runs the real pipeline
(lexical retrieval, structured judgement, per-option evidence, …) over a
synthetic corpus. The mock backend answers instantly, so we measure **call
count and prompt size**, which dominate cost; completion tokens are capped by
config and are comparatively negligible.

    python benchmarks/bench_token_cost.py

Caveat, stated plainly: the mock always returns a canonical answer, so the
canonicality RETRY path never fires here. Measured numbers are therefore the
happy path — a real run can be higher. The retry ceiling is reported separately.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from awareliquid.adapter.qa_agent import MemoryQAAgent, RetrievalConfig
from awareliquid.adapter.qwen_client import MockChatClient

TOKEN_BUDGET = 5_000_000
QUESTION_SET_SIZE = 100  # the real group-A set


class CountingClient:
    """Delegates to the mock while recording every call's size."""

    def __init__(self) -> None:
        self._inner = MockChatClient()
        self.calls: List[Dict[str, int]] = []

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 64):
        prompt = "\n".join(m.get("content", "") for m in messages)
        result = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens)
        self.calls.append(
            {
                "prompt_chars": len(prompt),
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cap": int(max_tokens),
            }
        )
        return result

    def reset(self) -> None:
        self.calls.clear()


def _corpus() -> Dict[str, str]:
    """Two multi-section filings, long enough to exercise the context budgets."""
    years = range(2019, 2026)
    report = "\n\n".join(
        [
            "第一节 公司概况。本公司主营新能源汽车及动力电池的研发、生产与销售。",
            *[
                f"第二节 主要财务数据（{y} 年度）。{y} 年营业收入为 {700 + (y - 2019) * 130}.5 亿元，"
                f"归属于母公司股东的净利润为 {30 + (y - 2019) * 12}.4 亿元，"
                f"经营活动产生的现金流量净额为 {90 + (y - 2019) * 21}.7 亿元，"
                f"研发投入为 {40 + (y - 2019) * 15}.2 亿元，占营业收入的 {5 + (y - 2019) * 0.4:.1f}%。"
                for y in years
            ],
            "第三节 风险因素。原材料价格波动可能对毛利率造成不利影响；海外业务占比约 35%，存在汇率风险。",
            "第四节 利润分配。公司拟每 10 股派发现金红利 3.5 元（含税）。",
        ]
    )
    contract = "\n\n".join(
        [
            "第一条 保险责任。保险人对被保险人在保险期间内发生的合理施救费用承担赔偿责任。",
            "第二条 免赔额。每次事故绝对免赔额为人民币 5,000 元，或损失金额的 10%，以高者为准。",
            "第三条 除外责任。因战争、核污染、被保险人故意行为造成的损失，保险人不负赔偿责任。",
            "第四条 等待期。重大疾病保险的等待期为 90 天，等待期内发生的保险事故不予赔付。",
            "第五条 施救费用。施救费用的赔偿以保险金额为限，最高不超过人民币 200 万元。",
            "第六条 争议解决。因本合同产生的争议，双方应协商解决；协商不成的提交仲裁委员会仲裁。",
        ]
    )
    return {"annual_report": report, "insurance_contract": contract}


QUESTIONS = [
    {"qid": "t1", "qtype": "mcq", "doc": "annual_report",
     "question": "2023 年公司的营业收入是多少？",
     "options": ["1090.5 亿元", "1220.5 亿元", "960.5 亿元", "830.5 亿元"]},
    {"qid": "t2", "qtype": "tf", "doc": "annual_report",
     "question": "2024 年归母净利润较 2023 年实现增长，判断该说法是否正确。",
     "options": ["正确", "错误"]},
    {"qid": "t3", "qtype": "multi", "doc": "annual_report",
     "question": "下列关于公司经营业绩的描述，哪些是准确的？",
     "options": ["2025 年营业收入较 2024 年增长", "2025 年研发投入占比下降",
                 "2024 年经营现金流优于 2023 年", "2019 年净利润高于 2025 年"]},
    {"qid": "t4", "qtype": "mcq", "doc": "annual_report",
     "question": "2025 年与 2019 年相比，营业收入增加了多少亿元？",
     "options": ["780.0", "650.0", "910.0", "520.0"]},
    {"qid": "t5", "qtype": "mcq", "doc": "insurance_contract",
     "question": "每次事故的绝对免赔额是多少？",
     "options": ["5,000 元或损失金额的 10%，以高者为准", "1,000 元", "无免赔额", "200 万元"]},
    {"qid": "t6", "qtype": "multi", "doc": "insurance_contract",
     "question": "下列哪些属于本合同的除外责任或限制？",
     "options": ["战争造成的损失", "被保险人故意行为", "等待期 90 天内的重疾",
                 "合理的施救费用"]},
]

CONFIGS = {
    "default (as shipped)": {},
    "no calculation pass": {"calculation_judgement": False},
    "lean": {
        "calculation_judgement": False,
        "multi_option_audit": False,
        "evidence_coverage_supplement": False,
        "answer_context_budget": 4000,
        "compression_budget": 1500,
        "option_evidence_budget": 400,
    },
}


def run_config(name: str, overrides: dict) -> dict:
    client = CountingClient()
    config = replace(
        RetrievalConfig(retrieval_backend="lexical", max_chars=300, top_k=6),
        **overrides,
    )
    agent = MemoryQAAgent(config=config, chat_client=client)
    agent.ingest_documents(_corpus())

    per_question = []
    for q in QUESTIONS:
        before = len(client.calls)
        agent.answer_question(
            qid=q["qid"], question=q["question"], options=q["options"],
            qtype=q["qtype"], doc_ids=[q["doc"]],
        )
        calls = client.calls[before:]
        per_question.append(
            {
                "qid": q["qid"], "qtype": q["qtype"], "n_calls": len(calls),
                "tokens": sum(c["total_tokens"] for c in calls),
                "prompt_chars": sum(c["prompt_chars"] for c in calls),
            }
        )

    n = len(per_question)
    total_tokens = sum(p["tokens"] for p in per_question)
    total_calls = sum(p["n_calls"] for p in per_question)
    avg_tokens = total_tokens / n
    projected = avg_tokens * QUESTION_SET_SIZE
    return {
        "name": name, "per_question": per_question,
        "calls_per_q": total_calls / n, "avg_tokens": avg_tokens,
        "projected": projected, "budget_pct": 100 * projected / TOKEN_BUDGET,
    }


def main() -> None:
    print("=== generation-token cost per question (mock backend, real pipeline) ===\n")
    results = [run_config(name, ov) for name, ov in CONFIGS.items()]

    baseline = results[0]
    print(f"--- per-question breakdown: {baseline['name']} ---")
    print(f"  {'qid':<5} {'type':<6} {'calls':>5} {'tokens':>8} {'prompt chars':>13}")
    for p in baseline["per_question"]:
        print(f"  {p['qid']:<5} {p['qtype']:<6} {p['n_calls']:>5} "
              f"{p['tokens']:>8} {p['prompt_chars']:>13}")

    print(f"\n--- configuration comparison (projected over {QUESTION_SET_SIZE} questions) ---")
    print(f"  {'config':<22} {'calls/q':>8} {'tokens/q':>9} {'projected':>11} {'of budget':>10}")
    for r in results:
        print(f"  {r['name']:<22} {r['calls_per_q']:>8.1f} {r['avg_tokens']:>9.0f} "
              f"{r['projected']:>11,.0f} {r['budget_pct']:>9.1f}%")

    lean, base = results[-1], results[0]
    if base["avg_tokens"]:
        saving = 100 * (1 - lean["avg_tokens"] / base["avg_tokens"])
        print(f"\n  lean vs default: {saving:.0f}% fewer tokens per question")
    print("\n  NOTE: the mock always answers canonically, so canonicality RETRIES "
          "never fire here.\n  A real run can add up to 3 extra answer calls per "
          "question on top of these numbers.")


if __name__ == "__main__":
    main()
