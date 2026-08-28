# Agent Memory Runtime 正式实验协议 v1.0

状态：冻结版

冻结日期：2026-08-28

适用项目：基于鲲鹏平台的 Agent Memory 记忆管理系统

## 1. 协议目的

本协议统一 Agent Memory Runtime 的实验对象、数据、对照组、评价指标、运行环境、重复方式、统计方法和结果留存规则。

所有正式实验必须按照本协议执行。实验结果只能说明本次实验条件下的观察结果，不能将尚未运行的目标值描述为已取得成果。

核心问题：

> 在记忆持续增长、事实不断更新和上下文预算受限的条件下，Agent Memory Runtime 能否以更高的记忆正确性、更低的 Token 成本和可接受的系统开销，持续提供当前正确且历史可追溯的记忆。

## 2. 实验对象与方法版本

正式对比对象固定为以下五种系统：

| 编号 | 方法 | 记忆能力 | 允许使用的组件 |
|---|---|---|---|
| B0 | No Memory | 不使用长期记忆 | 当前查询、LLM |
| B1 | Full History | 将完整历史对话放入上下文 | 历史对话、LLM |
| B2 | Vector Memory | 向量召回 Top-K | Embedding、Vector Index、LLM |
| B3 | Hybrid Memory | 向量 + BM25 + RRF | Vector、BM25、RRF、LLM |
| Ours | Agent Memory Runtime | 完整记忆治理 | Write、Hybrid、Temporal、Conflict、Consolidation、Lifecycle、Budget、LLM |

实验约束：

1. B0、B1、B2、B3 不得调用 Ours 的治理能力。
2. Ours 的实验开关必须写入运行配置。
3. 各方法使用同一 LLM、同一 Embedding、同一数据集、同一查询顺序和同一生成参数。
4. B0、B1 没有检索过程时，Recall@K 和 MRR 记录为 null，不人为记为 0。
5. 运行失败、超时或模型异常必须记录原因，不得删除失败样本后重新计算结果。

## 3. Benchmark 版本与数据划分

### 3.1 Benchmark 版本

| 版本 | 用途 |
|---|---|
| v0.1 | 早期回归测试，不能作为最终结论的唯一依据 |
| v0.2 | 当前功能联调与流程验证 |
| v1.0 | 正式实验评测集，完成数据审核后冻结 |

v0.2 不得为了提高某个方法的结果而修改。正式申报材料的主结果使用 v1.0；v0.2 仅用于回归和开发记录。

### 3.2 正式场景类别

Benchmark v1.0 固定包含以下十二类：

```text
fact_recall
semantic_recall
temporal
update
conflict
long_context
noise
abstention
budget
multi_hop
forgetting
consolidation
```

每条 Case 至少包含：

```json
{
  "case_id": "",
  "category": "",
  "conversation": [],
  "query": "",
  "expected_memory_ids": [],
  "expected_answer": "",
  "answer_aliases": [],
  "expected_version": "",
  "query_time": "",
  "difficulty": "",
  "token_budget": null,
  "forbidden_memory_ids": [],
  "metadata": {}
}
```

### 3.3 数据规模

长期记忆实验固定测试：

```text
10 turns
20 turns
50 turns
100 turns
```

规模扩展实验固定测试：

```text
100 memories
1,000 memories
10,000 memories
100,000 memories
```

正式 v1.0 至少包含 Development、Test 和 Holdout 三部分。建议划分为 60%、20%、20%，使用固定随机种子生成并保存 Case ID 清单。

### 3.4 Ground Truth 规则

每条 Case 由两名成员独立审核，分歧由第三名成员裁决。审核内容包括：

- 正确答案及允许别名；
- 正确 Memory ID；
- 正确版本；
- 查询时刻；
- 不应召回的过期或无关记忆；
- Multi-hop 所需的完整记忆集合；
- Forgetting 场景中应保留和应归档的记忆。

测试集审核结束后禁止根据模型结果反向修改 Ground Truth。若发现 Ground Truth 错误，必须递增 Benchmark 版本并重新运行受影响实验。

## 4. 统一运行环境

每次正式运行必须保存：

```text
实验协议版本
代码 commit SHA
Benchmark 版本和 SHA
操作系统与版本
CPU 型号、核心数、内存
Python 版本
依赖锁定文件
Embedding 模型与版本
LLM 模型与版本
Tokenizer 与版本
Prompt 模板版本
数据库与索引参数
随机种子
运行时间
```

开发阶段可使用 x86 环境完成流程验证；正式性能实验必须增加 Kunpeng ARM64/AArch64、openEuler 和 openGauss 或明确记录的对照存储后端。

ARM64 实验中，Memory Write、Memory Search、Rerank、Update、Consolidation 和 Agent Runtime 必须在 ARM64 环境执行。

正式实验应优先使用真实 Embedding 和真实 LLM。所有方法使用同一 temperature、top_p、最大输出长度、system prompt、answer format、timeout 和 retry 配置。

离线 RuleBasedClient 和 HashEmbeddingModel 只能用于单元测试、回归测试和流程冒烟，不能作为最终比赛性能结论。

## 5. 实验运行规则

### 5.1 预热与计时

延迟实验采用：

- 预热 30 次，不计入结果；
- 正式测量至少 300 次 Query；
- 规模实验每个规模至少 1,000 次检索请求；
- 保存每次请求的原始延迟。

LLM 网络延迟与 Memory Runtime 延迟必须分开记录。

### 5.2 重复实验

准确率类实验每个方法、每个实验条件至少重复 3 次，正式主结果建议重复 5 次。

协议冻结值：

```text
正式主结果：5 repeats
资源不足时的最低要求：3 repeats
性能延迟测试：每个条件 300 次以上请求
规模测试：每个规模 1,000 次以上检索请求
```

重复实验使用固定种子序列：

```text
202601
202602
202603
202604
202605
```

不允许只报告表现最好的一次运行。

### 5.3 参数冻结

参数调优只能在 Development 集完成。Test 和 Holdout 运行前必须冻结：

- top_k；
- RRF 参数；
- Reranker 权重；
- Importance、Recency 和 Utility 权重；
- Context Token Budget；
- Forgetting threshold；
- Consolidation threshold；
- 索引参数；
- Prompt 模板。

正式实验后修改参数，必须新增协议或实验版本，并完整重跑受影响的对照组。

## 6. 指标定义

### 6.1 Agent 效果指标

#### Answer Accuracy

```text
Answer Accuracy = 正确回答的 Query 数 / 可回答 Query 总数
```

主指标采用规范化 Exact Match。规范化只允许去除首尾空格、统一大小写、去除规定回答格式前缀和使用预先审核的 answer_aliases。

不得根据预测结果临时增加答案别名。

对于 Multi-hop、Update、Conflict 和 Consolidation 场景，只有答案、版本和必要记忆链均正确时才算 Task Success。

### 6.2 检索指标

对存在检索过程的方法计算：

```text
Recall@K = Top-K 中命中相关 Memory ID 的比例
MRR = 正确 Memory 首次出现位置倒数的平均值
Precision@K = Top-K 中相关 Memory 的比例
```

固定报告 Recall@1/5/10、MRR 和 Precision@1/5/10。

Multi-hop Case 使用全部必需 Memory ID 作为相关集合，并额外报告完整链命中率。

### 6.3 动态记忆指标

```text
Update Accuracy
    = 正确采用最新事实的 Query 数 / 更新 Query 总数

Conflict Resolution Accuracy
    = 正确识别并处理冲突的 Case 数 / 冲突 Case 总数

Temporal QA Accuracy
    = 当前查询和历史查询均使用正确版本的 Case 数 / Temporal Case 总数

Stale Memory Retrieval Rate
    = 当前状态查询错误召回过期版本的次数 / 当前状态 Query 总数
```

历史查询额外报告：

- Historical Query Accuracy：最终答案是否正确；
- Historical Retrieval Accuracy：正确历史版本是否进入结果；
- Superseded Version Leakage：当前查询中返回过期版本的比例。

### 6.4 Token 与预算指标

记录 prompt_tokens、context_tokens、candidate_memory_tokens、budget_before_prompt_tokens、budget_after_prompt_tokens 和 budget_accuracy_delta。

```text
Token Reduction =
(Full History Token - Ours Token) / Full History Token
```

预算实验必须同时报告预算前后 Token、Token 减少比例、预算前后准确率、准确率变化百分点和是否满足预算。

### 6.5 生命周期与演化指标

固定报告：

```text
Redundancy Ratio
Compression Ratio
Consolidation Fidelity
Archive Accuracy
```

```text
Compression Ratio
    = 1 - Managed Memory Size / Raw Memory Size
```

Consolidation Fidelity 同时评价语义记忆是否保留原始核心事实，以及是否引入原始 Episodic Memory 中不存在的新事实。

### 6.6 延迟与资源指标

延迟拆分为：

1. Candidate Retrieval Latency：Query Encoding → Search → Fusion → Validity Filter；
2. Memory Read Latency：Candidate Retrieval → Rerank → Compression → Context Pack；
3. End-to-End Agent Latency：Query → Memory Runtime → LLM → Response。

每类延迟统一报告 Mean、Std、P50、P95 和 P99。

资源指标包括 Search QPS、Write QPS、CPU Utilization、Peak RSS Memory、Storage Usage、Index Size、Index Build Time 和 Average Bytes per Memory。

## 7. E1–E7 正式实验矩阵

### E1：Memory Baseline

问题：长期记忆是否改善 Agent 的回答能力？

对比：B0、B1、B2、B3、Ours。

场景：fact_recall、semantic_recall、noise、abstention、multi_hop。

主指标：Answer Accuracy、Recall@5、MRR、prompt_tokens 和 End-to-End P95。

### E2：Retrieval Comparison

问题：不同检索策略的召回质量和延迟有什么差异？

固定同一 Memory Store 和 Query 集，比较：

```text
R0 Vector Only
R1 BM25 Only
R2 Vector + BM25 + RRF
R3 R2 + Reranker
```

主指标：Recall@1/5/10、MRR、Precision@5 和 Candidate Retrieval P50/P95。

### E3：Long-term Memory

问题：对话长度增加后，系统是否仍能记住早期信息？

规模：10、20、50、100 turns。

主指标：Memory Accuracy、100-turn Accuracy 和 Accuracy Drop。

```text
Accuracy Drop = Accuracy@20 - Accuracy@100
```

### E4：Update and Conflict

问题：事实更新、冲突和历史查询是否能够正确处理？

每组至少包含三版本链：

```text
V1 → V2 → V3
```

主指标：Update Accuracy、Conflict Resolution Accuracy、Temporal QA Accuracy、Stale Memory Retrieval Rate 和 Historical Query Accuracy。

### E5：Ablation

问题：完整方案的提升来自哪些模块？

固定 Full Ours，分别移除：

```text
Ours - Temporal Versioning
Ours - Conflict Governance
Ours - Consolidation
Ours - Context Budget
Ours - Hybrid Retrieval
Ours - Lifecycle
```

每个消融组保持其他配置不变，报告 Accuracy、Recall@5、Token、P95 Latency 和 Memory Footprint。

### E6：Kunpeng Performance

问题：系统是否真正适配并优化鲲鹏？

阶段一：在 ARM64/openEuler 上完成核心功能兼容性测试。

阶段二：记录 Kunpeng Baseline。

阶段三：针对 Profile 发现的瓶颈进行优化，并在同一硬件、同一数据和同一参数下进行前后对比。

主指标：Write QPS、Candidate Retrieval P95、Memory Read P95、Peak RSS、CPU Utilization 和 Recall@5。

### E7：Scalability

问题：Memory 数量增加后，系统是否保持可用？

规模：100、1K、10K、100K memories。

主指标：Recall@5、Candidate Retrieval P95、Memory Read P95、QPS、RAM、Storage 和 Index Build Time。

## 8. 目标与判定规则

以下是项目工程目标，不表述为华为官方硬性指标：

| 指标 | 工程目标 |
|---|---:|
| Recall@5 | ≥85%，且优于 B2 |
| Update Accuracy | ≥90% |
| Conflict Resolution Accuracy | ≥90% |
| Stale Memory Retrieval Rate | ≤5% |
| 100-turn Memory Accuracy | ≥80% |
| 相对 Full History 的 Token Reduction | ≥30% |
| Token 优化后的 Accuracy 损失 | ≤2 个百分点 |
| Memory Read P95@10K | ≤300 ms |
| Candidate Retrieval P95@10K | ≤100 ms |
| ARM64 核心功能通过率 | 100% |
| 鲲鹏至少一项关键性能收益 | ≥20% |
| 优化后 Recall@5 损失 | ≤1 个百分点 |

目标值用于项目管理和结果解释。未达到目标时必须报告真实结果、失败原因和改进方向，不能反向修改指标定义。

## 9. 统计与报告规则

准确率和 Token 指标报告：

- 每个重复实验的结果；
- Mean；
- Std；
- 95% Bootstrap Confidence Interval；
- 方法间绝对差值和相对差值。

延迟指标报告：

- 原始 Query 级延迟；
- 每次重复的 Mean；
- 跨重复 Mean/Std；
- 全部请求的 P50/P95/P99。

主结果使用配对比较：不同方法在相同 Case、相同 Seed 下运行，避免数据难度差异影响比较。

不得只报告均值。若重复实验差异较大，必须保留原始结果并解释方差来源。

## 10. 结果留存与命名

每次正式实验至少生成：

```text
manifest.json
raw_rows.jsonl
summary.json
summary.csv
run.log
environment.txt
figures/
```

统一目录：

```text
results/formal/
├── E1_baseline/
├── E2_retrieval/
├── E3_long_memory/
├── E4_update_conflict/
├── E5_ablation/
├── E6_kunpeng/
└── E7_scalability/
```

文件名必须包含实验编号、方法、数据集、硬件、日期和协议版本。

例如：

```text
E4_Ours_benchmark-v1.0_kunpeng_20260828_protocol-v1.0.json
```

## 11. 冻结与变更规则

协议冻结后，以下内容不得无记录修改：

- Benchmark Test/Holdout Case；
- 指标定义；
- 对照组能力边界；
- Prompt；
- 模型版本；
- 统计方法；
- 目标值；
- 结果筛选规则。

以下情况必须递增协议版本：

1. 增加或删除主要对照组；
2. 修改主指标公式；
3. 修改 Test 集；
4. 更换 LLM 或 Embedding；
5. 修改核心 Prompt；
6. 修改硬件比较方式；
7. 改变重复次数或统计方法。

普通 Bug 修复不改变协议，但必须记录旧版本运行失败原因，并重新运行全部受影响的实验。

## 12. 当前执行状态

截至协议冻结时：

- B0、B1、B2、Ours 已有统一离线运行器；
- Temporal、Budget、Multi-hop、Forgetting 已进入 Benchmark v0.2；
- E1/E2 阶段性脚本已完成三次重复流程验证；
- 单元测试已通过；
- 当前离线生成结果不作为最终比赛数据；
- B3 独立基线、Consolidation 正式实验、E5–E7 和真实 Kunpeng 结果仍待完成。

当前项目状态应表述为：

> 实验基础设施和阶段性验证已完成，正式真实模型实验与鲲鹏性能证据仍在执行中。

