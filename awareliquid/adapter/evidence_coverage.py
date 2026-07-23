"""Checking that each claim actually has its evidence in context.

Two complementary gaps are detected. An *anchor* gap is an exact,
high-signal token in the claim (a year, an amount, a clause number) that is
absent from the evidence. A *slot* gap is a domain concept the claim depends
on whose whole synonym family is missing.

NOTE: ``_EVIDENCE_SLOT_GROUPS`` is Chinese-financial vocabulary and is the
one domain-locked thing in the pipeline. It lives here, alone and visible,
so it can be swapped for another domain without touching the agent.
"""

from __future__ import annotations

import re
from typing import List, Sequence


_EVIDENCE_SLOT_GROUPS = (
    ("营业收入", "营业总收入", "主营业务收入"),
    ("归母净利润", "归属于上市公司股东的净利润"),
    ("经营现金流", "经营活动产生的现金流量净额", "经营活动现金流量净额"),
    ("研发投入", "研发费用", "研发投入占营业收入比例"),
    ("现金分红", "每10股现金分红", "利润分配方案"),
    ("一般医疗", "一般医疗保险金", "住院医疗费用"),
    ("免赔额", "共享免赔额", "家庭共享免赔额"),
    ("施救费用", "施救费", "必要合理施救费用"),
    ("除外责任", "免责", "不承担责任"),
)
_EVIDENCE_ANCHOR = re.compile(
    r"(?:19|20)\d{2}\s*年?"
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:个百分点|亿元|万元|美元|港元|bp|BP|%|％|亿|万|元)"
    r"|第\s*[一二三四五六七八九十百\d]+\s*(?:条|章|节|款|项|个保单年度|保单年度)"
    r"|AAA|AA\+?"
    r"|除外|例外|但|不适用|特殊情形|免责|等待期"
    ,
)
def _normalise_evidence_anchor(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _missing_evidence_anchors(question: str, option: str, evidence: str) -> List[str]:
    """Find exact, high-signal anchors from a claim missing in its evidence."""
    anchors: List[str] = []
    seen = set()
    # Option anchors are the primary coverage contract.  Question anchors are
    # added only when they are structural facts (dates, amounts, clauses, ...),
    # not ordinary connective words such as ``但``.  This keeps a supplement
    # query targeted instead of turning every multi-choice question into a
    # broad second retrieval pass.
    for match in _EVIDENCE_ANCHOR.finditer(f"{option} {question}"):
        anchor = match.group(0).strip()
        key = _normalise_evidence_anchor(anchor)
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            anchors.append(anchor)
    evidence_text = _normalise_evidence_anchor(evidence)
    return [anchor for anchor in anchors if _normalise_evidence_anchor(anchor) not in evidence_text]


def _missing_evidence_slots(question: str, option: str, evidence: str) -> List[str]:
    source = f"{question} {option}"
    normalized = _normalise_evidence_anchor(evidence)
    missing = []
    for group in _EVIDENCE_SLOT_GROUPS:
        if not any(term in source for term in group):
            continue
        if any(_normalise_evidence_anchor(term) in normalized for term in group):
            continue
        missing.append(" ".join(group))
    return missing


def _merge_evidence_blocks(contexts: Sequence[str], max_chars: int) -> str:
    """Deduplicate source blocks and enforce one final context budget."""
    if max_chars <= 0:
        return ""
    blocks: List[str] = []
    seen = set()
    used = 0
    for context in contexts:
        for block in (part.strip() for part in (context or "").split("\n\n")):
            if not block:
                continue
            key = re.sub(r"\s+", " ", block).strip().lower()
            if key in seen:
                continue
            separator = 2 if blocks else 0
            remaining = max_chars - used - separator
            if remaining <= 0:
                return "\n\n".join(blocks)
            seen.add(key)
            selected = block[:remaining]
            blocks.append(selected)
            used += separator + len(selected)
            if len(selected) < len(block):
                return "\n\n".join(blocks)
    return "\n\n".join(blocks)
