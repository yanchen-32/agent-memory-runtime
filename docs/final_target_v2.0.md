# Agent Memory Runtime 最终目标 v2.0

状态：冻结

冻结日期：2026-08-31

本文件将四条增强路线合并进原有目标。它不改变总体架构、核心接口和
B0/B1/B2/B3/Ours 的能力边界，也不把计划中的功能写成已经完成的结果。

## 1. 最终交付目标

交付一个可运行、可实验、可展示、可追溯并可在鲲鹏 ARM64/openEuler 上
验证的 Agent Memory Runtime 原型。系统面向长期运行的 Agent，而不是
“聊天机器人 + 向量数据库”或普通 RAG。

核心接口继续冻结为：

```text
write() / read() / retrieve() / update() / invalidate()
consolidate() / forget() / trace() / build_context()
```

原有能力继续包括：时序版本治理、冲突处理、Vector + BM25 + RRF、
Query-Adaptive Rerank、Episodic → Semantic Consolidation、生命周期管理、
Context Budget 和来源追溯。

## 2. 四条横向增强路线

### 2.1 Adaptive Consolidation

在已有 Consolidation 前增加可解释规则策略，分别决定“何时整合”和
“保留多少细节”。触发得分使用 Redundancy、Cluster Size、Age 和
Storage Pressure；粒度得分使用 Importance、Novelty、Information Density、
Access Frequency 和 Redundancy。

冲突检查必须先于 Consolidation。未解决冲突不得通过摘要合并。V1 权重和
阈值只能在 Development Set 调整，冻结后不得在 Test/Holdout 继续调参。

验收比较：Fixed Granularity vs Adaptive Granularity；同时报告 Answer F1、
Fidelity、Key Fact Recall、Semantic Tokens、Compression Ratio、Recall@K、
Latency 和 Consolidation Cost。更短但事实损失更大的摘要不算成功。

### 2.2 Formal Evaluation

Answer F1 为主要回答质量指标，Raw Exact Match、Normalized Match 和
Answer Accuracy 为辅助指标。B1 Full History 是“不降低回答质量”和 E2E
性能的主要对照组，B0/B2/B3 用于机制贡献分析。

正式结论使用相同 Case 的配对比较。每个 Query 至少重复三次，先取同一
Query 的中位数，再进行 Case 级 Paired Bootstrap。只有
`CI95(delta F1).lower >= 0` 才能表述为统计证据支持“不下降”；仅平均值
不低于 B1 时，只能表述为“点估计未观察到下降”。

完整时延定义为：

```text
T_E2E = T_memory + T_context + T_LLM + T_post
T_memory(Ours) = T_retrieval + T_rerank + T_budget
```

B1 与 Ours 按 Query 交错/反向运行，降低网络和服务端时变偏差。E3 按
Full-History Token Size 分层，目标是在长上下文中保持 F1、稳定上下文规模，
并验证相对 B1 的 E2E Latency Reduction 是否达到 50%。

### 2.3 Kunpeng Optimization

必须遵循 Profile → Bottleneck → Optimization → Controlled Benchmark。
先采集各阶段耗时、CPU、内存以及条件允许时的 perf 指标，再决定是否优化。

NUMA 仅在多 NUMA Node 硬件上验证；单节点环境只允许报告 CPU Affinity 和
ARM64 结果。NEON 仅优化已确认的 FP32 向量热点，保留 Scalar Fallback，
并验证数值容差和 Top-K 语义一致。

具备硬件条件时使用 2×2 因子实验：NUMA OFF/ON × NEON OFF/ON，分别报告
Memory Runtime 本地加速和完整问答 E2E，不能把远程 LLM 波动算作 CPU 收益。

### 2.4 Memory Observatory

Observatory 只记录真实机制决策，不参与 Ranking。统一 TraceEvent 至少包含：

```text
trace_id, query_id, stage, memory_id, version, source_ids,
score_components, rank, selected, reject_reason,
token_cost, latency_ms, timestamp
```

Demo 展示 Retrieval Trace、Memory Evolution Timeline、Consolidation Lineage
和 Context Budget Trace。拒绝原因来自运行路径，而不是 LLM 事后解释。

正式性能实验默认 `trace=off` 或轻量异步模式，并单独测量 Trace Overhead；
完整 Trace 用于 Demo。

## 3. E1–E7 最终映射

| 实验 | 核心问题 | 主要证据 |
|---|---|---|
| E1 | 长期记忆回答质量 | Answer F1；B1 vs Ours 不下降验证 |
| E2 | Update / Conflict | 可演化时序记忆、历史版本、过期泄漏 |
| E3 | Long Context Scaling | F1 / Context Token / E2E；50% 延迟目标 |
| E4 | Context Budget | Token Reduction vs F1 |
| E5 | Consolidation | Fixed vs Adaptive；Fidelity / Compression / F1 |
| E6 | Lifecycle / Forgetting | Archive、Forget、长期存储治理 |
| E7 | Kunpeng Performance | NUMA × NEON；Retrieval / Throughput / E2E |
| Demo | Memory Observatory | Version / Retrieval / Lineage / Budget Trace |

## 4. 最终证据链和完成条件

```text
企业问题 → 技术问题 → 方法设计 → 系统实现 → 鲲鹏适配
→ Benchmark → 原始实验数据 → 统计结论 → Demo → 企业价值 → 交付物
```

项目只有在以下三项都有真实、可复现数据时才达到最终完成状态：

1. 相比 B1，长期记忆压缩和检索是否维持 Answer F1；
2. 历史增长时是否减少 Context Token，并验证 50% E2E 延迟目标；
3. 鲲鹏上 NUMA/NEON 是否降低 Memory Runtime 本地计算成本。

未运行、硬件条件不具备或置信区间不支持的项目必须明确列为待验证，禁止用
目标值代替实验结果。
