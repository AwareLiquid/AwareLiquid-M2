"""Generate a choice-question submission CSV from a question set.

Reads a document set and a question file, runs the full memory-adapter pipeline
per question, and writes the submission CSV:

    header : qid,answer,prompt_tokens,completion_tokens,total_tokens
    row 1  : summary,,<sum prompt>,<sum completion>,<sum total>
    rows   : one per question, answer normalised per type

Answer formatting (handled by the agent's parser):
    mcq / tf -> a single letter        multi -> sorted letters, no separator

Usage:
    python submit.py --questions questions.jsonl --docs docs.json --out submission.csv

Input formats (flexible):
    --docs       {doc_id: text} JSON, a [{doc_id, text|content}] array, or a
                 directory of <doc_id>.txt files.
    --questions  JSONL or a JSON array; each item: qid, question, options
                 (dict {letter: text} or list), answer_format, doc_ids.

Set AWARELIQUID_LLM_API_KEY (+ AWARELIQUID_LLM_MODEL) to answer with a real Qwen
model; otherwise the offline mock backend runs so the pipeline is testable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from awareliquid import MemoryQAAgent

# Map the input's answer_format values to the internal question type.
_QTYPE = {
    "mcq": "mcq", "single": "mcq", "single_choice": "mcq", "单选": "mcq",
    "multi": "multi", "multiple": "multi", "multiple_choice": "multi", "多选": "multi",
    "tf": "tf", "judgment": "tf", "judgement": "tf", "boolean": "tf",
    "true_false": "tf", "判断": "tf",
}


def load_docs(path: str) -> Dict[str, str]:
    p = Path(path)
    if p.is_dir():
        return {f.stem: f.read_text(encoding="utf-8")
                for f in sorted(p.iterdir()) if f.is_file()}
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    docs: Dict[str, str] = {}
    for d in data:
        did = str(d.get("doc_id") or d.get("id"))
        docs[did] = str(d.get("text") or d.get("content") or "")
    return docs


def load_questions(path: str) -> List[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def options_to_list(options) -> List[str]:
    """Options may be a {letter: text} dict; order by letter so the agent's
    A/B/C… labels line up with the given letters. A plain list passes through
    unchanged."""
    if isinstance(options, dict):
        return [str(options[k]) for k in sorted(options.keys())]
    return [str(o) for o in (options or [])]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a choice-question submission CSV.")
    ap.add_argument("--questions", required=True, help="JSONL/JSON question file")
    ap.add_argument("--docs", required=True, help="docs JSON or directory")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (0=all)")
    args = ap.parse_args()

    has_key = bool(os.environ.get("AWARELIQUID_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
    if not has_key and os.environ.get("AWARELIQUID_LLM_BACKEND") != "mock":
        print("WARNING: no Qwen API key set — running the offline MOCK backend. "
              "Answers will not be real. Set AWARELIQUID_LLM_API_KEY for a real run.",
              file=sys.stderr)

    docs = load_docs(args.docs)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    print(f"loaded {len(docs)} docs, {len(questions)} questions", file=sys.stderr)

    agent = MemoryQAAgent()
    agent.ingest_documents(docs)

    rows: List[list] = []
    tp = tc = tt = 0
    for i, q in enumerate(questions, 1):
        qid = str(q.get("qid") or q.get("id") or i)
        fmt = str(q.get("answer_format") or q.get("qtype") or "mcq").lower()
        qtype = _QTYPE.get(fmt, "mcq")
        options = options_to_list(q.get("options"))
        doc_ids = q.get("doc_ids")
        res = agent.answer_question(
            qid=qid, question=str(q.get("question", "")),
            options=options, qtype=qtype, doc_ids=doc_ids,
        )
        u = res.usage
        tp += u.prompt_tokens
        tc += u.completion_tokens
        tt += u.total_tokens
        rows.append([qid, res.answer, u.prompt_tokens, u.completion_tokens, u.total_tokens])
        if i % 50 == 0:
            print(f"  answered {i}/{len(questions)} (tokens so far: {tt})", file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", tp, tc, tt])
        w.writerows(rows)

    empty = sum(1 for r in rows if not r[1])
    print(f"wrote {len(rows)} answers -> {args.out}", file=sys.stderr)
    print(f"total tokens: {tt} (prompt {tp} + completion {tc}); "
          f"avg {tt/max(1,len(rows)):.0f}/q; {empty} empty answer(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
