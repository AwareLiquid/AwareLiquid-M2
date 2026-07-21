"""Score the adapter on a labelled evaluation set.

This is the only way to turn "we think it works" into a number. It runs the full
ingest -> retrieve -> compress -> answer loop over `questions.jsonl`, compares
each answer to the ground truth, and reports accuracy overall, by question type,
and by answer format — plus token cost and the resulting score under
``accuracy x (0.7 + 0.3 x token_efficiency)``.

    # offline sanity check of the harness (answers are meaningless)
    AWARELIQUID_LLM_BACKEND=mock AWARELIQUID_TEST_MODE=1 python evals/run_eval.py

    # real measurement
    AWARELIQUID_LLM_API_KEY=sk-... python evals/run_eval.py

Scope, stated plainly: the corpus and questions here were authored alongside the
system, so this measures the PIPELINE (does retrieval find the passage, does
compression keep the answer sentence, does parsing return the right letter) — not
real-world difficulty. It is a regression baseline, not a leaderboard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from awareliquid import MemoryQAAgent, RetrievalConfig  # noqa: E402
from awareliquid.adapter.qwen_client import MockChatClient  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TOKEN_BUDGET = 5_000_000


def load_corpus() -> dict:
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((EVAL_DIR / "corpus").glob("*.md"))
    }


def load_questions() -> list:
    lines = (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def options_in_order(options: dict) -> list:
    return [options[key] for key in sorted(options)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the adapter on the labelled eval set.")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--show-errors", action="store_true", help="print every wrong answer")
    args = ap.parse_args()

    corpus = load_corpus()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    mock = os.environ.get("AWARELIQUID_LLM_BACKEND") == "mock"
    print(f"corpus: {len(corpus)} documents | questions: {len(questions)}")
    print(f"backend: {'MOCK (answers meaningless)' if mock else 'REAL model'}\n")

    agent = MemoryQAAgent(
        config=RetrievalConfig(retrieval_backend="lexical"),
        chat_client=MockChatClient() if mock else None,
    )
    agent.ingest_documents(corpus)

    correct = 0
    by_type: dict = defaultdict(lambda: [0, 0])
    by_format: dict = defaultdict(lambda: [0, 0])
    prompt_tokens = completion_tokens = 0
    errors = []

    for q in questions:
        result = agent.answer_question(
            qid=q["qid"],
            question=q["question"],
            options=options_in_order(q["options"]),
            qtype=q["answer_format"],
            doc_ids=q["doc_ids"],
        )
        gold = q["answer"]
        hit = result.answer == gold
        correct += hit
        by_type[q["type"]][0] += hit
        by_type[q["type"]][1] += 1
        by_format[q["answer_format"]][0] += hit
        by_format[q["answer_format"]][1] += 1
        prompt_tokens += result.usage.prompt_tokens
        completion_tokens += result.usage.completion_tokens
        if not hit:
            errors.append((q["qid"], q["type"], gold, result.answer, q["question"][:40]))
        print(f"  {'OK ' if hit else '!! '}{q['qid']}  got={result.answer or '(empty)':<4} "
              f"gold={gold:<4} {q['type']}")

    total = len(questions)
    accuracy = correct / total if total else 0.0
    total_tokens = prompt_tokens + completion_tokens
    token_score = max(0.0, min(1.0, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET))
    final = 100 * accuracy * (0.7 + 0.3 * token_score)

    print(f"\n=== accuracy: {correct}/{total} ({100*accuracy:.1f}%) ===")
    print("\n  by question type:")
    for name, (hit, n) in sorted(by_type.items()):
        print(f"    {name:<16} {hit}/{n}  ({100*hit/n:.0f}%)")
    print("\n  by answer format:")
    for name, (hit, n) in sorted(by_format.items()):
        print(f"    {name:<6} {hit}/{n}  ({100*hit/n:.0f}%)")

    print(f"\n  tokens: {total_tokens:,} "
          f"(prompt {prompt_tokens:,} + completion {completion_tokens:,})")
    print(f"  avg/question: {total_tokens/total:.0f}" if total else "")
    print(f"  score: {final:.2f}  = 100 x {accuracy:.3f} x (0.7 + 0.3 x {token_score:.4f})")

    if args.show_errors and errors:
        print("\n  wrong answers:")
        for qid, qtype, gold, got, stem in errors:
            print(f"    {qid} [{qtype}] gold={gold} got={got or '(empty)'} — {stem}…")

    if mock:
        print("\n  NOTE: mock backend — this run only proves the harness works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
