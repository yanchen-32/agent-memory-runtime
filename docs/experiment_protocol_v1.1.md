# Agent Memory Runtime 正式实验协议 v1.1

状态：冻结版

冻结日期：2026-08-31

取代：v1.0

最终目标：`docs/final_target_v2.0.md`

## 1. 对照组与公平性

| 编号 | 方法 | 能力边界 |
|---|---|---|
| B0 | No Memory | 当前 Query + LLM |
| B1 | Full History | 完整历史 + LLM，无检索和治理 |
| B2 | Vector Memory | Vector Top-K |
| B3 | Hybrid Memory | Vector + BM25 + RRF |
| Ours | Complete Runtime | Hybrid + Temporal + Conflict + Rerank + Budget + Consolidation + Lifecycle |

所有方法使用同一 Benchmark、LLM、Embedding、Prompt answer format、
temperature、top_p、thinking 模式、最大输出长度、timeout 和 retry 设置。
B2/B3 不得使用 Ours 的治理能力。失败和超时必须留在原始记录中。

正式运行采用逐行追加 Checkpoint；恢复时只允许配置指纹和 Benchmark SHA256
完全一致，并跳过已经完成的 `(Case ID, Agent, Repeat)`。暂态 HTTP 错误允许按
冻结的指数退避策略重试；401 等认证错误不重试。最终失败保留为失败行，不能
静默删除、当作零分混入配对统计，或在失败后临时更换某个 Agent 的配置。

正式生成模型第一组配置冻结为 non-thinking、temperature=0。模型或
Embedding 变更必须作为新实验配置完整重跑所有对照组。API Key 只允许从
环境变量或命令行注入，不得写入结果、日志、代码或 Git。

## 2. 数据划分与参数冻结

正式 Benchmark v1.0 必须有 Development、Test、Holdout 固定 Case ID 清单。
Normalization、Aliases、Ground Truth 和 Token 分层边界在 Test 前冻结。

所有策略权重和阈值，包括 Adaptive Consolidation 的
`alpha/beta/gamma/delta/lambda/tau_c`，只允许在 Development Set 调整。
Test/Holdout 禁止继续调参。Ground Truth 变更必须递增 Benchmark 版本并重跑。

## 3. Answer F1

规范化顺序冻结为：Unicode NFKC、去首尾空格、英文小写、删除预先列出的
回答前缀、移除 Unicode 标点。Tokenizer 将英文/数字连续串作为一个 Token，
将每个中文字符作为一个 Token。计算多重集 Token overlap：

```text
P = overlap / prediction_tokens
R = overlap / gold_tokens
F1 = 2PR / (P + R)
```

存在预审 `answer_aliases` 时，取与任一冻结 Gold/Alias 的最高 F1。禁止从
Test 预测结果反向增加 Alias。

每条结果同时保存：Raw Exact Match、Normalized Match、Answer Accuracy、
Answer Precision、Answer Recall、Answer F1、原始 Prediction 和匹配的 Gold。
Answer Accuracy 采用规范化全等，或对明确的冻结回答前缀提取第一个答案分句
后全等；不采用任意子串命中，避免旧值与新值同时出现时产生错误判定。

## 4. 配对统计与不下降表述

每个 Query 至少重复 3 次，正式主结果建议 5 次。先取同一 Case 各重复的
中位数，再计算：

```text
delta_i = F1_ours_i - F1_B1_i
delta_F1 = mean(delta_i)
```

使用固定种子 202601，进行 10,000 次 Case 级 Paired Bootstrap，保存
`CI95(delta_F1)`。

- `mean(delta_i) >= 0`：可写“点估计未观察到准确率下降”；
- `CI95.lower >= 0`：可写“配对 Bootstrap 支持准确率不下降”；
- 其他情况必须报告下降或证据不足。

不得把离线 RuleBasedClient + HashEmbedding 结果用于上述正式结论。

## 5. 端到端时延

```text
T_E2E = T_memory + T_context + T_LLM + T_post
```

逐请求保存 setup、memory、context build、LLM、post、E2E；API 支持时保存
TTFT 和实际 prompt/completion/total tokens。汇总保存 Mean、Std、P50、P95、
P99。Setup 不计入 Query E2E，但单独报告。

B1 和 Ours 必须在 Query 级交错：奇数 Query 正序，偶数 Query 反序；不得先
跑完整 B1 再跑完整 Ours。每个 Query 使用重复中位数进行配对时延比较。

```text
LatencyReduction = 1 - T_ours / T_B1
```

50% 是 E3 长上下文工程目标，不是未实验即可宣称的结果。E3 根据冻结的
Full-History Token 分位数分为 Short / Medium / Long / Very Long，并分别
报告 F1、Context Tokens、E2E 和 Reduction。

## 6. CPU Microbenchmark

Retrieval、NUMA、NEON 等 CPU Microbenchmark 必须预热并至少进行几十次
重复；正式规模测试建议 300 次以上请求。记录 CPU affinity、NUMA topology、
向量维度、Memory Scale、线程数、编译器/库版本和原始 latency。

NUMA 结论只允许来自多 NUMA Node 硬件。NEON 与 Scalar/NumPy 使用相同输入，
预先冻结浮点误差阈值，并验证 Top-K Recall 不发生实质变化。

## 7. E1–E7

实验编号和目标严格采用 `docs/final_target_v2.0.md` 第 3 节。消融至少包括：

```text
Ours - Temporal Versioning
Ours - Conflict Governance
Ours - Reranker
Ours - Context Budget
Ours - Consolidation
Fixed vs Adaptive Consolidation
NUMA OFF/ON × NEON OFF/ON
```

Observatory 不做回答准确率消融，只测试 Trace Completeness 和 Runtime
Overhead。正式 50% E2E 主结果使用 `trace=off` 或冻结的轻量异步模式。

## 8. 结果留存

每次正式运行至少生成：

```text
manifest.json
raw_rows.jsonl
summary.json
summary.csv
run.log
environment.txt
figures/
```

Manifest 必须保存协议版本、Git commit、Benchmark SHA256、Agent 列表、运行
顺序策略、重复次数、所有非秘密模型配置、Embedding、OS、CPU 架构、Python
和依赖版本。逐 Case 原始记录不得只保留聚合结果。

## 9. 当前适用边界

截至冻结日，当前服务器为 openEuler 22.03 LTS、aarch64、Kunpeng-920、
4 CPU、单 NUMA Node。它可用于 ARM64 兼容性、Profiling、CPU Affinity 和
NEON 实验，但不能作为多 NUMA 优化已经验证的证据。
