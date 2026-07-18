"""End-to-end accuracy + token-cost evaluation on a labelled question set.

Runs the full ingest -> retrieve -> compress -> answer loop against questions
that carry a ground-truth answer, and reports the two things the scoring rule is
built from:

    score = accuracy x (0.7 + 0.3 x token_efficiency)
    token_efficiency = max(0, (budget - tokens_used) / budget)

Offline (mock backend) the numbers are meaningless for accuracy -- the mock just
echoes the first option -- but the harness is identical, so a real run is a
one-liner once a Qwen key is set:

    # PowerShell
    $env:AWARELIQUID_LLM_API_KEY="sk-..."; $env:AWARELIQUID_LLM_MODEL="qwen-plus"
    python examples/eval_accuracy.py

    # bash
    AWARELIQUID_LLM_API_KEY=sk-... AWARELIQUID_LLM_MODEL=qwen-plus python examples/eval_accuracy.py

The key is read from the environment; this script never prints or stores it.
"""

from __future__ import annotations

import os

from awareliquid import MemoryQAAgent, summarize_usage

# Official per-run token budget for the full test set (used only to illustrate
# the efficiency term; on this tiny demo set efficiency is ~1.0).
TOKEN_BUDGET = 5_000_000

DOCS = {
    "annual-2023": (
        "本公司 2023 年度实现营业收入 124.5 亿元，同比增长 8.3%。"
        "归属于母公司股东的净利润为 18.2 亿元，较上年同期增长 12.1%。"
        "2022 年营业收入为 114.9 亿元，净利润为 16.2 亿元。"
        "2023 年研发投入 9.7 亿元，占营业收入的 7.8%。"
        "公司拟每 10 股派发现金红利 3.5 元（含税）。"
    ),
    "bond-prospectus": (
        "本期债券名称为 2024 年度第一期中期票据，发行规模 20 亿元。"
        "债券期限为 5 年，票面利率为 3.45%，按年付息。"
        "债券的信用评级为 AAA，评级机构为中诚信国际。"
        "发行人承诺在债券存续期内合并资产负债率不超过 70%。"
    ),
}

# Each item carries the ground-truth answer letter for scoring.
QUESTIONS = [
    {"qid": "q1", "doc_id": "annual-2023", "qtype": "mcq", "answer": "B",
     "question": "公司 2023 年营业收入是多少？",
     "options": ["114.9 亿元", "124.5 亿元", "18.2 亿元", "9.7 亿元"]},
    {"qid": "q2", "doc_id": "annual-2023", "qtype": "tf", "answer": "A",
     "question": "2023 年归母净利润同比实现增长，判断该说法是否正确。",
     "options": ["正确", "错误"]},
    {"qid": "q3", "doc_id": "annual-2023", "qtype": "mcq", "answer": "C",
     "question": "2023 年公司的研发投入金额是多少？",
     "options": ["6.3 亿元", "18.2 亿元", "9.7 亿元", "3.5 亿元"]},
    {"qid": "q4", "doc_id": "bond-prospectus", "qtype": "mcq", "answer": "A",
     "question": "本期债券的票面利率是多少？",
     "options": ["3.45%", "70%", "5 年", "20 亿元"]},
    {"qid": "q5", "doc_id": "bond-prospectus", "qtype": "mcq", "answer": "B",
     "question": "本期债券的信用评级是什么？",
     "options": ["AA", "AAA", "A", "BBB"]},
    {"qid": "q6", "doc_id": "bond-prospectus", "qtype": "mcq", "answer": "B",
     "question": "本期债券的发行规模是多少？",
     "options": ["12 亿元", "20 亿元", "5 亿元", "70 亿元"]},
]


def main() -> None:
    backend = os.environ.get("AWARELIQUID_LLM_BACKEND")
    has_key = bool(os.environ.get("AWARELIQUID_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
    mode = "MOCK (offline)" if (backend == "mock" or not has_key) else \
        f"REAL Qwen ({os.environ.get('AWARELIQUID_LLM_MODEL', 'qwen-plus')})"
    print(f"=== accuracy evaluation — backend: {mode} ===\n")

    agent = MemoryQAAgent()
    agent.ingest_documents(DOCS)

    results, correct = [], 0
    for q in QUESTIONS:
        res = agent.answer_question(
            qid=q["qid"], question=q["question"], options=q["options"],
            qtype=q["qtype"], doc_ids=[q["doc_id"]],
        )
        results.append(res)
        ok = res.answer == q["answer"]
        correct += ok
        print(f"  {q['qid']}: got {res.answer!r} expected {q['answer']!r} "
              f"{'OK' if ok else 'X'}  ({res.usage.total_tokens} tok)")

    n = len(QUESTIONS)
    total = summarize_usage(results)
    accuracy = correct / n
    token_eff = max(0.0, (TOKEN_BUDGET - total.total_tokens) / TOKEN_BUDGET)
    score = accuracy * (0.7 + 0.3 * token_eff)

    print("\n--- summary ---")
    print(f"  accuracy          : {correct}/{n} ({100*accuracy:.1f}%)")
    print(f"  total tokens      : {total.total_tokens} "
          f"(prompt {total.prompt_tokens} + completion {total.completion_tokens})")
    print(f"  avg tokens / q    : {total.total_tokens / n:.0f}")
    print(f"  token efficiency  : {token_eff:.4f}  (vs {TOKEN_BUDGET:,} budget)")
    print(f"  est. score        : {score:.4f}  = accuracy x (0.7 + 0.3 x token_eff)")
    if "MOCK" in mode:
        print("\n  NOTE: mock backend echoes the first option — accuracy here is not "
              "meaningful.\n  Set AWARELIQUID_LLM_API_KEY and rerun for a real number.")


if __name__ == "__main__":
    main()
