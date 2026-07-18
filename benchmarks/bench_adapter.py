"""Performance benchmark for the AwareLiquid Qwen adapter.

The adapter's own contribution is everything *around* the frozen model:
retrieval and compression. This harness measures exactly those, on a small but
adversarial financial QA set (many distractor sentences, answers hinging on
specific numbers), and reports the metrics that map to the two things that
matter under a token budget:

* **Retrieval recall@k** — does the answer-bearing chunk make it into top-k?
* **Answer-sentence retention** — does compression KEEP the sentence that holds
  the answer? (If it drops it, accuracy dies regardless of the model.)
* **Compression ratio** — retrieved-context size vs. compressed prompt context.
* **Token efficiency vs. full-document baseline** — how many prompt tokens the
  adapter spends compared with stuffing the whole document into the prompt.
* **End-to-end validity** — every answer is a well-formed mcq/tf/multi row with
  token accounting.

Retrieval quality needs the real multilingual embedder. When it cannot be loaded
(e.g. a memory-constrained box), pass --fake to substitute a self-contained
LEXICAL encoder (hashed char n-grams) so compression and token metrics still
compute; those numbers are labelled as a lexical stand-in, not e5 semantics.

Usage:
    python benchmarks/bench_adapter.py            # real e5 embedder
    python benchmarks/bench_adapter.py --fake     # lexical stand-in (no model)
"""

from __future__ import annotations

import argparse
import sys

import torch

from awareliquid.adapter.qa_agent import MemoryQAAgent, RetrievalConfig
from awareliquid.adapter.qwen_client import MockChatClient, _estimate_tokens
from awareliquid.memory.knowledge_store import PersistentKnowledgeMemory

# --------------------------------------------------------------------------
# A small, adversarial financial corpus. Each document mixes answer-bearing
# sentences (specific numbers) with plausible distractors from the same domain.
# --------------------------------------------------------------------------
DOCS = {
    "annual-2023": (
        "第一节 公司概况。本公司是一家从事新能源装备制造的高新技术企业，总部位于深圳。"
        "第二节 主要财务数据。本公司 2023 年度实现营业收入 124.5 亿元，同比增长 8.3%。"
        "归属于母公司股东的净利润为 18.2 亿元，较上年同期增长 12.1%。"
        "2022 年营业收入为 114.9 亿元，净利润为 16.2 亿元。"
        "第三节 研发投入。公司持续加大研发力度，2023 年研发投入 9.7 亿元，占营业收入的 7.8%。"
        "研发人员数量为 3200 人，占员工总数的 21%。"
        "第四节 利润分配。公司拟每 10 股派发现金红利 3.5 元（含税），合计派发现金红利 6.3 亿元。"
        "第五节 风险因素。原材料价格波动可能对公司毛利率造成不利影响。"
        "公司出口业务占比约 35%，面临一定的汇率波动风险。"
    ),
    "bond-prospectus": (
        "本期债券基本情况。本期债券名称为 2024 年度第一期中期票据，发行规模 20 亿元。"
        "债券期限为 5 年，票面利率为 3.45%，按年付息，到期一次还本。"
        "债券的信用评级为 AAA，评级机构为中诚信国际。"
        "募集资金用途。本期募集资金拟用于偿还存量有息债务及补充流动资金。"
        "其中不超过 12 亿元用于偿还银行借款，其余用于补充营运资金。"
        "增信措施。本期债券无担保，为信用债券。"
        "财务约束条款。发行人承诺在债券存续期内合并资产负债率不超过 70%。"
    ),
    "reg-filing": (
        "关于计提资产减值准备的公告。经初步测算，公司拟对应收账款计提坏账准备 1.2 亿元。"
        "对存货计提跌价准备 0.8 亿元，合计影响 2023 年度归母净利润约 2.0 亿元。"
        "本次计提减值准备已经公司第五届董事会第十次会议审议通过。"
        "独立董事发表了同意的独立意见，认为计提依据充分、程序合规。"
        "本次计提不会对公司正常生产经营活动产生重大影响。"
    ),
}

# needle = a distinctive substring that appears in the answer-bearing sentence.
QUESTIONS = [
    {"qid": "q1", "doc_id": "annual-2023", "qtype": "mcq", "needle": "124.5",
     "question": "公司 2023 年营业收入是多少？",
     "options": ["114.9 亿元", "124.5 亿元", "18.2 亿元", "9.7 亿元"], "answer": "B"},
    {"qid": "q2", "doc_id": "annual-2023", "qtype": "mcq", "needle": "9.7",
     "question": "2023 年公司的研发投入金额是多少？",
     "options": ["6.3 亿元", "18.2 亿元", "9.7 亿元", "3.5 亿元"], "answer": "C"},
    {"qid": "q3", "doc_id": "annual-2023", "qtype": "tf", "needle": "12.1%",
     "question": "2023 年归母净利润同比实现了增长，判断该说法是否正确。",
     "options": ["正确", "错误"], "answer": "A"},
    {"qid": "q4", "doc_id": "bond-prospectus", "qtype": "mcq", "needle": "3.45%",
     "question": "本期债券的票面利率是多少？",
     "options": ["3.45%", "70%", "AAA", "5 年"], "answer": "A"},
    {"qid": "q5", "doc_id": "bond-prospectus", "qtype": "mcq", "needle": "70%",
     "question": "发行人承诺债券存续期内合并资产负债率不超过多少？",
     "options": ["35%", "70%", "7.8%", "21%"], "answer": "B"},
    {"qid": "q6", "doc_id": "reg-filing", "qtype": "mcq", "needle": "2.0 亿元",
     "question": "本次计提减值准备合计影响 2023 年度归母净利润约多少？",
     "options": ["1.2 亿元", "0.8 亿元", "2.0 亿元", "6.3 亿元"], "answer": "C"},
    # Exact-token stress cases: the answer hinges on a literal code/figure with a
    # semantically-similar distractor nearby -- where dense e5 blurs and BM25 wins.
    {"qid": "q7", "doc_id": "bond-prospectus", "qtype": "mcq", "needle": "AAA",
     "question": "本期债券的信用评级是什么？",
     "options": ["AA", "AAA", "A", "BBB"], "answer": "B"},
    {"qid": "q8", "doc_id": "bond-prospectus", "qtype": "mcq", "needle": "发行规模 20 亿元",
     "question": "本期债券的发行规模是多少？",
     "options": ["12 亿元", "20 亿元", "5 亿元", "70 亿元"], "answer": "B"},
]


class LexicalEncoder:
    """Self-contained lexical embedder (hashed char trigrams), no model download.

    A genuine bag-of-n-grams retriever -- not random -- so it exercises the store
    and retrieval logic and gives meaningful (lexical) recall. It is NOT a
    semantic model; results under --fake are labelled accordingly.
    """

    def __init__(self, dim: int = 512):
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str, is_query: bool = False) -> torch.Tensor:
        vec = torch.zeros(self._dim, dtype=torch.float32)
        t = (text or "").lower()
        for n in (2, 3):  # char bigrams + trigrams
            for i in range(len(t) - n + 1):
                vec[hash(t[i : i + n]) % self._dim] += 1.0
        norm = torch.linalg.vector_norm(vec)
        return vec / norm if norm > 0 else vec


# Realistic boilerplate distractors — the kind of text that pads a real filing
# and buries the few answer-bearing sentences. Deliberately number-free so they
# do not collide with any answer needle.
_FILLER = [
    "公司始终坚持稳健经营的方针，注重风险防控与合规管理。",
    "报告期内，公司持续完善公司治理结构，规范董事会运作。",
    "公司高度重视人才队伍建设，优化激励机制，吸引和留住核心人才。",
    "面对复杂多变的宏观经济环境，公司积极调整经营策略。",
    "公司不断加强内部控制体系建设，提升内控执行的有效性。",
    "公司积极履行社会责任，推进绿色低碳发展与节能减排。",
    "管理层认为，行业竞争格局总体稳定，但仍需关注下游需求变化。",
    "公司加强与上下游合作伙伴的战略协同，巩固供应链稳定性。",
    "公司持续推进数字化转型，提升运营效率与客户服务水平。",
    "董事会对经营层报告期内的工作给予了充分肯定。",
    "公司将继续聚焦主业，稳步推进重点项目建设。",
    "报告期内公司未发生重大诉讼、仲裁事项。",
    "公司信息披露工作严格遵循相关法律法规及交易所规则。",
    "公司审计委员会对内部审计工作进行了监督与指导。",
]


def _inflate(core: str, repeats: int = 8) -> str:
    """Bury the answer-bearing sentences among many boilerplate distractors, so
    the benchmark stresses retrieval the way a real long filing would."""
    blocks = []
    fillers = _FILLER * repeats
    # Sprinkle the core sentences through the filler at fixed offsets.
    core_sents = [s + "。" for s in core.split("。") if s.strip()]
    step = max(1, len(fillers) // (len(core_sents) + 1))
    ci = 0
    for i, f in enumerate(fillers):
        blocks.append(f)
        if i > 0 and i % step == 0 and ci < len(core_sents):
            blocks.append(core_sents[ci])
            ci += 1
    blocks.extend(core_sents[ci:])  # any remaining core sentences
    return "".join(blocks)


# Inflate each document to a realistic length (thousands of chars) that would not
# fit a modest context window, which is exactly when compression pays off.
DOCS = {did: _inflate(text) for did, text in DOCS.items()}


# --------------------------------------------------------------------------
# HARD scenario: exact-token disambiguation. Many near-identical sentences that
# differ only in the exact token (year, figure) the question hinges on -- the
# case where dense e5 blurs (every "YYYY 年营业收入为 X 亿元" embeds alike) and
# BM25's exact-term match is decisive -- the cross-figure comparison case.
# --------------------------------------------------------------------------
HARD_DOC = "\n".join(
    [f"{y} 年营业收入为 {v} 亿元。" for y, v in
     [(2018, "78.4"), (2019, "86.3"), (2020, "92.1"),
      (2021, "98.6"), (2022, "114.9"), (2023, "124.5")]]
    + [f"{y} 年归母净利润为 {v} 亿元。" for y, v in
       [(2018, "7.9"), (2019, "9.1"), (2020, "10.4"),
        (2021, "13.8"), (2022, "16.2"), (2023, "18.2")]]
    + [f"{y} 年研发投入为 {v} 亿元。" for y, v in
       [(2021, "7.2"), (2022, "8.5"), (2023, "9.7")]]
)

HARD_QUESTIONS = [
    {"qid": "h1", "needle": "98.6", "question": "2021 年公司的营业收入是多少？",
     "options": ["92.1 亿元", "98.6 亿元", "114.9 亿元", "124.5 亿元"], "qtype": "mcq"},
    {"qid": "h2", "needle": "10.4", "question": "2020 年公司的归母净利润是多少？",
     "options": ["9.1 亿元", "10.4 亿元", "13.8 亿元", "16.2 亿元"], "qtype": "mcq"},
    {"qid": "h3", "needle": "114.9", "question": "2022 年公司的营业收入是多少？",
     "options": ["98.6 亿元", "114.9 亿元", "124.5 亿元", "92.1 亿元"], "qtype": "mcq"},
    {"qid": "h4", "needle": "8.5", "question": "2022 年公司的研发投入是多少？",
     "options": ["7.2 亿元", "8.5 亿元", "9.7 亿元", "16.2 亿元"], "qtype": "mcq"},
    {"qid": "h5", "needle": "9.1", "question": "2019 年公司的归母净利润是多少？",
     "options": ["7.9 亿元", "9.1 亿元", "10.4 亿元", "13.8 亿元"], "qtype": "mcq"},
    {"qid": "h6", "needle": "78.4", "question": "2018 年公司的营业收入是多少？",
     "options": ["78.4 亿元", "86.3 亿元", "92.1 亿元", "98.6 亿元"], "qtype": "mcq"},
]


def run_hard(use_fake: bool) -> int:
    """Exact-token disambiguation: dense vs hybrid recall@1 on near-identical rows."""
    label = "lexical stand-in" if use_fake else "real e5 embedder"
    print(f"=== HARD exact-token disambiguation ({label}) ===\n")
    if use_fake:
        enc = LexicalEncoder()
    else:
        from awareliquid.memory.encoder import SentenceEncoder
        torch.set_num_threads(1)
        enc = SentenceEncoder()
    store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    # Small chunks so each fact is its OWN chunk -> retrieval must pick the exact row.
    cfg = RetrievalConfig(max_chars=24, overlap_chars=0, top_k=5)
    agent = MemoryQAAgent(encoder=enc, store=store, chat_client=MockChatClient(), config=cfg)
    n_chunks = agent.ingest_document("metrics", HARD_DOC)
    print(f"  ingested metrics -> {n_chunks} single-fact chunks\n")

    def measure() -> tuple:
        r1 = correct = 0
        rows = []
        for q in HARD_QUESTIONS:
            hits = agent.retrieve(q["question"], doc_ids=["metrics"], options=q["options"])
            top1 = bool(hits) and q["needle"] in hits[0]
            r1 += top1
            res = agent.answer_question(qid=q["qid"], question=q["question"],
                                        options=q["options"], qtype=q["qtype"], doc_ids=["metrics"])
            ok = res.answer == "B" if q["qid"] in ("h1", "h2", "h3", "h5") else True
            correct += ok
            rows.append((q["qid"], int(top1)))
        return r1, rows

    agent.config.hybrid = False
    d_r1, d_rows = measure()
    agent.config.hybrid = True
    h_r1, h_rows = measure()

    n = len(HARD_QUESTIONS)
    print("  per-question recall@1 (dense -> hybrid):")
    for (qid, dr), (_, hr) in zip(d_rows, h_rows):
        arrow = "" if dr == hr else ("  <-- recovered by BM25" if hr > dr else "  <-- REGRESSED")
        print(f"    {qid}: {dr} -> {hr}{arrow}")
    print(f"\n  recall@1 (exact row in TOP chunk): "
          f"dense {d_r1}/{n} ({100*d_r1/n:.0f}%)  ->  hybrid {h_r1}/{n} ({100*h_r1/n:.0f}%)")
    delta = h_r1 - d_r1
    print(f"  hybrid delta: {'+' if delta >= 0 else ''}{delta} question(s)")
    return 0 if h_r1 >= d_r1 else 1


def build_agent(use_fake: bool) -> MemoryQAAgent:
    if use_fake:
        enc = LexicalEncoder()
    else:
        from awareliquid.memory.encoder import SentenceEncoder
        torch.set_num_threads(1)
        enc = SentenceEncoder()
    store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    cfg = RetrievalConfig(max_chars=140, overlap_chars=30, top_k=4, compression_budget=400)
    return MemoryQAAgent(encoder=enc, store=store, chat_client=MockChatClient(), config=cfg)


def _measure(agent: MemoryQAAgent) -> dict:
    """Run all questions once under the agent's current config; return metrics."""
    recall_hits = recall1_hits = retention_hits = e2e_valid = 0
    comp_tokens, base_tokens = [], []
    per_q = []
    for q in QUESTIONS:
        retrieved = agent.retrieve(q["question"], doc_ids=[q["doc_id"]], options=q["options"])
        recall = any(q["needle"] in c for c in retrieved)
        recall1 = bool(retrieved) and q["needle"] in retrieved[0]  # answer in the TOP chunk
        recall_hits += recall
        recall1_hits += recall1
        comp = agent.compressor.compress(q["question"] + " " + " ".join(q["options"]), retrieved)
        retained = q["needle"] in comp.text
        retention_hits += retained
        comp_tokens.append(_estimate_tokens(comp.text))
        base_tokens.append(_estimate_tokens(DOCS[q["doc_id"]]))
        res = agent.answer_question(
            qid=q["qid"], question=q["question"], options=q["options"],
            qtype=q["qtype"], doc_ids=[q["doc_id"]],
        )
        e2e_valid += bool(res.answer) and res.usage.total_tokens > 0
        per_q.append((q["qid"], int(recall1), int(recall)))
    n = len(QUESTIONS)
    return {
        "recall": recall_hits, "recall1": recall1_hits, "retention": retention_hits,
        "valid": e2e_valid, "n": n,
        "comp_tokens": sum(comp_tokens), "base_tokens": sum(base_tokens), "per_q": per_q,
    }


def run(use_fake: bool) -> int:
    label = "lexical stand-in" if use_fake else "real e5 embedder"
    print(f"=== AwareLiquid adapter benchmark ({label}) ===\n")
    agent = build_agent(use_fake)
    for did, text in DOCS.items():
        n = agent.ingest_document(did, text)
        print(f"  ingested {did:<16} -> {n} chunks")
    print(f"  FTS5 lexical channel: {'available' if agent.store.fts_enabled else 'UNAVAILABLE'}\n")

    # Ablation: dense-only vs hybrid (BM25 + dense, RRF-fused) on the same store.
    agent.config.hybrid = False
    dense = _measure(agent)
    agent.config.hybrid = True
    hybrid = _measure(agent)

    n = dense["n"]
    print("  per-question recall@1 (dense -> hybrid):")
    for (qid, dr1, _), (_, hr1, _) in zip(dense["per_q"], hybrid["per_q"]):
        arrow = "" if dr1 == hr1 else ("  <-- recovered by BM25" if hr1 > dr1 else "  <-- REGRESSED")
        print(f"    {qid}: {dr1} -> {hr1}{arrow}")

    print("\n--- ablation: dense-only vs hybrid RRF ---")
    print(f"  recall@1 (answer in TOP chunk): "
          f"dense {dense['recall1']}/{n} ({100*dense['recall1']/n:.0f}%)   ->   "
          f"hybrid {hybrid['recall1']}/{n} ({100*hybrid['recall1']/n:.0f}%)")
    print(f"  recall@{agent.config.top_k}                     : "
          f"dense {dense['recall']}/{n} ({100*dense['recall']/n:.0f}%)   ->   "
          f"hybrid {hybrid['recall']}/{n} ({100*hybrid['recall']/n:.0f}%)")
    print(f"  answer retention              : "
          f"dense {dense['retention']}/{n}   hybrid {hybrid['retention']}/{n}")
    print(f"  end-to-end valid rows         : {hybrid['valid']}/{n}")
    tot_comp, tot_base = hybrid["comp_tokens"], hybrid["base_tokens"]
    print(f"  prompt context tokens         : {tot_comp} vs full-doc {tot_base} "
          f"-> {100*(1-tot_comp/tot_base):.0f}% fewer")

    ok = (hybrid["valid"] == n and hybrid["recall"] >= dense["recall"]
          and hybrid["recall1"] >= dense["recall1"])
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'} "
          f"(hybrid recall@1 {hybrid['recall1']}/{n} >= dense {dense['recall1']}/{n}, "
          f"recall@{agent.config.top_k} {hybrid['recall']}/{n} >= dense {dense['recall']}/{n})")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="use the lexical stand-in encoder (no model download)")
    ap.add_argument("--hard", action="store_true",
                    help="run the exact-token disambiguation scenario (dense vs hybrid)")
    args = ap.parse_args()
    sys.exit(run_hard(args.fake) if args.hard else run(args.fake))
