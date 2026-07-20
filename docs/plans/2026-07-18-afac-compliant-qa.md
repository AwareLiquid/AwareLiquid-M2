# AFAC 合规金融问答系统实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不使用正式路径 Embedding/Rerank/微调/任意代码执行的前提下，完成 AwareLiquid 的可追溯词法检索、结构化证据、问题路由、本地金融计算、候选答案验证、多轮证据补检和可重复评测。

**Architecture:** 离线阶段使用 SQLite/FTS5/BM25 与结构化索引保存文档、表格和来源；在线阶段由确定性路由器决定检索计划，经有限轮次的证据覆盖与缺口补检组装原文窗口，再调用 Qwen 生成答案并进行本地格式、数值和证据校验。A 榜严格限定给定 `doc_ids`，B 榜先做候选文档定位。

**Tech Stack:** Python 3.10+, SQLite FTS5/BM25, pytest, Qwen OpenAI-compatible API, 标准库正则/JSON/算术白名单。

---

## 执行门禁

每个 Step 依次创建独立的 `E-Step-N` 执行 Agent 与 `V-Step-N` 只读验收 Agent。只有验收 `PASS` 才进入下一步；`FAIL` 必须由新的 `R-Step-N` 修复 Agent 处理，再创建新的验收 Agent。每个 Step 只触碰自身范围，并记录测试、风险和报告。

## Step 0：基线和规则合规审计

- 阅读交接、规则、README、架构和核心代码，建立可重复 baseline。
- 审计 dense embedding、rerank、`doc_ids`、Qwen token 统计和压缩丢失风险。
- 仅添加审计/合规测试，不实现后续功能。
- 通过独立验收后进入 Step 1。

## Step 1：合规词法/结构化检索

- 强化 FTS5/BM25、标题/章节/页码/段落/表格标题、数字/年份/日期/单位索引。
- 在数据库查询层严格应用 `doc_ids`；区分 A 榜给定文档检索与 B 榜候选文档定位。
- 增加中英文金融 fixture 和反例测试，正式模式不加载 dense/rerank。

## Step 2：证据对象和上下文组装

- 引入带 `doc_id/page/section/source_type/text/table_id/row_id/column_ids/score` 的统一 Evidence。
- 保留原文、表头、单位、脚注、条款例外和邻近上下文，使用首尾优先和证据相邻布局控制预算。

## Step 3：问题类型路由

- 用规则支持 fact/clause/formula/comparison/multihop/mcq，输出稳定可序列化计划。
- 为计算、逐文档比较、多跳补检和选项验证选择安全路径；路由不调用 API，并提供回退。

## Step 4：金融表格和本地安全计算

- 解析表格标题/表头/行列/单位/脚注，按指标和年份定位单元格。
- 仅允许白名单加减乘除、百分比、同比、差值、比较，并保留输入证据和计算过程。
- 明确拒绝任意代码、除零、单位冲突和缺失数据。

## Step 5：候选答案验证

- 对每个选项独立召回支持/反驳证据，核对数字、年份、单位和限定词。
- 输出 `supported/refuted/insufficient` 及证据 ID、数值检查和置信度，拒绝表面词汇匹配。

## Step 6：多轮证据补检和充分性

- 实现有限状态的 Coverage → Gap → Commitment 流程，记录每轮查询、命中、证据缺口和 token。
- 具备最大轮数/单题 token 上限、去重、无新增停止、跨文档最低覆盖及安全降级。

## Step 7：评测、消融和提交集成

- 固化 baseline、E1–E5 消融，按题型统计准确率、证据覆盖率、计算/格式正确率及 token。
- 验证 A/B 榜分离、异常处理和提交文件格式，补齐回归测试与运行说明。

## Final：独立总体验收

- 只读检查全部 diff、报告、测试和输出；运行全量及关键路径测试。
- 只有规则合规、功能完整、回归通过且 A/B 榜与 token 账本均可解释时输出 `GO`。
