"""Generate a choice-question submission CSV from a question set.

Reads a document set and a question file, runs the full memory-adapter pipeline
per question, and writes the submission CSV:

    row 1  : summary,<budget>,<used>,<unused>
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

Formal submission requires AWARELIQUID_LLM_API_KEY and a real Qwen model.  The
offline mock backend is intentionally unavailable from this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from awareliquid import MemoryQAAgent, RetrievalConfig
from awareliquid.adapter.submission_contract import (
    SubmissionAnswer,
    ParsedQuestion,
    parse_questions,
    render_submission_csv,
)
from awareliquid.adapter.qwen_client import (
    FORMAL_NETWORK_ENV,
    formal_network_enabled,
)

# Map the input's answer_format values to the internal question type.
_QTYPE = {
    "mcq": "mcq", "single": "mcq", "single_choice": "mcq", "单选": "mcq",
    "multi": "multi", "multiple": "multi", "multiple_choice": "multi", "多选": "multi",
    "tf": "tf", "judgment": "tf", "judgement": "tf", "boolean": "tf",
    "true_false": "tf", "判断": "tf",
}


def render_csv(questions: Sequence[ParsedQuestion], answers: Sequence[SubmissionAnswer]) -> str:
    """Render only a fully validated official five-column CSV."""

    return render_submission_csv(questions, answers)


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


def _sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace *path* atomically after the complete contents are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_payload(question_payloads, docs, retrieval_backend, answers):
    return {
        "version": 1,
        "questions_sha256": _sha256_json(question_payloads),
        "docs_sha256": _sha256_json(docs),
        "retrieval_backend": retrieval_backend,
        "qids": [str(item["qid"]) for item in question_payloads],
        "answers": [
            {
                "qid": item.qid,
                "answer": item.answer,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
            }
            for item in answers
        ],
    }


def _load_checkpoint(path, question_payloads, questions, docs, retrieval_backend):
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_qids = [question.qid for question in questions]
    if data.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version in {path}")
    if data.get("questions_sha256") != _sha256_json(question_payloads):
        raise ValueError("checkpoint question set does not match current input; use --fresh")
    if data.get("docs_sha256") != _sha256_json(docs):
        raise ValueError("checkpoint documents do not match current input; use --fresh")
    if data.get("retrieval_backend") != retrieval_backend:
        raise ValueError("checkpoint retrieval backend does not match current run; use --fresh")
    if data.get("qids") != expected_qids:
        raise ValueError("checkpoint qid order does not match current input; use --fresh")
    raw_answers = data.get("answers", [])
    if len(raw_answers) > len(expected_qids):
        raise ValueError("checkpoint contains more answers than the question set")
    answers = [SubmissionAnswer(**item) for item in raw_answers]
    if [item.qid for item in answers] != expected_qids[:len(answers)]:
        raise ValueError("checkpoint answers must be a contiguous prefix in question order")
    if answers:
        render_csv(questions[:len(answers)], answers)
    return answers


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a choice-question submission CSV.")
    ap.add_argument("--questions", required=True, help="JSONL/JSON question file")
    ap.add_argument("--docs", required=True, help="docs JSON or directory")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="atomic per-question checkpoint path (default: <out>.checkpoint.json)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore an existing checkpoint and start the ordered run from zero",
    )
    ap.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="disable per-question checkpointing for an ephemeral run",
    )
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (0=all)")
    ap.add_argument(
        "--retrieval-backend",
        choices=("lexical",),
        default="lexical",
        help="formal submission retrieval backend (lexical/BM25 only)",
    )
    ap.add_argument(
        "--reset-ledger",
        action="store_true",
        help="zero the token ledger before starting (DISCARDS recorded spend)",
    )
    args = ap.parse_args()

    # Preflight: a formal run denies network egress unless deliberately armed.
    # Check it BEFORE loading/ingesting documents so an unarmed run fails in a
    # second instead of after minutes of indexing, on question 1.
    if not formal_network_enabled():
        raise SystemExit(
            f"{FORMAL_NETWORK_ENV} is not set to '1', so this formal run cannot "
            "call the Qwen API and would fail on the first question.\n"
            f"  Arm it deliberately:  export {FORMAL_NETWORK_ENV}=1   "
            f"(PowerShell: $env:{FORMAL_NETWORK_ENV}=\"1\")"
        )

    docs = load_docs(args.docs)
    question_payloads = load_questions(args.questions)
    if args.limit:
        question_payloads = question_payloads[: args.limit]
    questions = parse_questions(question_payloads)
    print(f"loaded {len(docs)} docs, {len(questions)} questions", file=sys.stderr)

    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = Path(args.checkpoint or f"{args.out}.checkpoint.json")

    formal_run_id = os.environ.get("AWARELIQUID_FORMAL_RUN_ID") or (
        f"submission-{_sha256_json(question_payloads)[:16]}"
    )
    formal_ledger_path = os.environ.get("AWARELIQUID_FORMAL_LEDGER_PATH") or (
        f"{args.out}.usage.json"
    )

    # The token ledger records REAL spend and deliberately outlives --fresh:
    # restarting the answering loop does not un-bill tokens already consumed.
    # Discarding it must therefore be an explicit choice, and carrying it over
    # must be visible (otherwise a later run dies mid-way on "budget exhausted").
    ledger_file = Path(formal_ledger_path)
    if args.reset_ledger:
        for stale in (ledger_file, ledger_file.with_name(f"{ledger_file.name}.lock")):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        print(f"reset token ledger: {ledger_file}", file=sys.stderr)
    elif ledger_file.exists():
        try:
            recorded = int(json.loads(ledger_file.read_text(encoding="utf-8")).get("used_tokens", 0))
        except (OSError, ValueError, TypeError):
            recorded = 0
        if recorded:
            carried = "CARRIED OVER into this --fresh run" if args.fresh else "in effect"
            print(
                f"NOTE: token ledger {ledger_file} already records {recorded:,} used tokens "
                f"({carried}). Pass --reset-ledger to zero it.",
                file=sys.stderr,
            )

    agent = MemoryQAAgent(
        config=RetrievalConfig(
            retrieval_backend=args.retrieval_backend,
            competition_mode=True,
            formal_run_id=formal_run_id,
            formal_ledger_path=formal_ledger_path,
        )
    )
    agent.ingest_documents(docs)

    answers: List[SubmissionAnswer] = []
    if checkpoint_path and checkpoint_path.exists() and not args.fresh:
        answers = _load_checkpoint(
            checkpoint_path, question_payloads, questions, docs, args.retrieval_backend
        )
        print(f"resuming after {len(answers)}/{len(questions)} committed questions", file=sys.stderr)

    for i, parsed in enumerate(questions[len(answers):], len(answers) + 1):
        qid = parsed.qid
        qtype = parsed.answer_format
        options = options_to_list(parsed.options)
        res = agent.answer_question(
            qid=qid, question=parsed.question,
            options=options, qtype=qtype, doc_ids=parsed.doc_ids,
            split=parsed.split,
        )
        u = res.usage
        answers.append(
            SubmissionAnswer(
                qid,
                res.answer,
                u.prompt_tokens,
                u.completion_tokens,
                u.total_tokens,
            )
        )
        if checkpoint_path:
            _atomic_write_text(
                checkpoint_path,
                json.dumps(
                    _checkpoint_payload(question_payloads, docs, args.retrieval_backend, answers),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            print(f"  committed {i}/{len(questions)} ({qid})", file=sys.stderr)
        if i % 50 == 0:
            print(f"  answered {i}/{len(questions)} (tokens so far: {sum(a.total_tokens for a in answers)})", file=sys.stderr)

    _atomic_write_text(Path(args.out), render_csv(questions, answers))

    tp = sum(a.prompt_tokens for a in answers)
    tc = sum(a.completion_tokens for a in answers)
    tt = sum(a.total_tokens for a in answers)
    print(f"wrote {len(answers)} answers -> {args.out}", file=sys.stderr)
    print(f"total tokens: {tt} (prompt {tp} + completion {tc}); "
          f"avg {tt/max(1,len(answers)):.0f}/q", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
