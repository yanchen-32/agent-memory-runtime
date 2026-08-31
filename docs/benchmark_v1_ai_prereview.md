# Benchmark v1.0 AI 技术预审记录

日期：2026-08-31

结论：Development、Test、Holdout 共 36 条案例全部通过六项技术预审。此记录
本身不是人工签字；随后人工 Reviewer `Zhang` 已完成全部 36 条签字，Benchmark
已生成 `frozen_manifest.json`。Reviewer 内容由人工填写，AI 预审未代填或覆盖。

## 预审范围

- 问题是否单义，能否只依赖给定记忆回答；
- 标准答案是否被给定证据直接支持；
- `expected_memory_ids` 是否完整且只含必要证据；
- `forbidden_memory_ids` 是否正确覆盖旧版本、未来版本或应遗忘内容；
- `valid_from`、`query_time`、`memory_query_time` 是否带时区且区间正确；
- `answer_aliases` 是否语义等价，无 Alias 的案例是否确实无需补充。

同时复核了三组拆分的场景族隔离、Update/Conflict 的
`subject + predicate` 一致性、三条多跳题的双证据完整性、三条弃答题和三条
遗忘题的 `UNKNOWN` 边界，以及三个 80-token Budget 案例。

## 逐案例结论

下表的 `6/6` 表示六个清单字段均通过；人工 Reviewer 通过 `reviewer` 字段承担
最终签字责任。

| Split | Case | 结果 |
| --- | --- | --- |
| Development | `dev_update_db_current_001` | 6/6 通过 |
| Development | `dev_update_db_history_001` | 6/6 通过 |
| Development | `dev_conflict_platform_current_001` | 6/6 通过 |
| Development | `dev_conflict_platform_history_001` | 6/6 通过 |
| Development | `dev_fact_name_001` | 6/6 通过 |
| Development | `dev_abstention_owner_phone_001` | 6/6 通过 |
| Development | `dev_semantic_memory_type_001` | 6/6 通过 |
| Development | `dev_noise_dimension_001` | 6/6 通过 |
| Development | `dev_budget_topk_001` | 6/6 通过 |
| Development | `dev_multihop_deploy_001` | 6/6 通过 |
| Development | `dev_forgetting_port_001` | 6/6 通过 |
| Development | `dev_long_context_codename_001` | 6/6 通过 |
| Test | `test_update_db_current_001` | 6/6 通过 |
| Test | `test_update_db_history_001` | 6/6 通过 |
| Test | `test_conflict_platform_current_001` | 6/6 通过 |
| Test | `test_conflict_platform_history_001` | 6/6 通过 |
| Test | `test_fact_name_001` | 6/6 通过 |
| Test | `test_abstention_budget_001` | 6/6 通过 |
| Test | `test_semantic_memory_type_001` | 6/6 通过 |
| Test | `test_noise_batch_001` | 6/6 通过 |
| Test | `test_budget_recall_001` | 6/6 通过 |
| Test | `test_multihop_quality_001` | 6/6 通过 |
| Test | `test_forgetting_code_001` | 6/6 通过 |
| Test | `test_long_context_codename_001` | 6/6 通过 |
| Holdout | `holdout_update_db_current_001` | 6/6 通过 |
| Holdout | `holdout_update_db_history_001` | 6/6 通过 |
| Holdout | `holdout_conflict_platform_current_001` | 6/6 通过 |
| Holdout | `holdout_conflict_platform_history_001` | 6/6 通过 |
| Holdout | `holdout_fact_name_001` | 6/6 通过 |
| Holdout | `holdout_abstention_owner_001` | 6/6 通过 |
| Holdout | `holdout_semantic_memory_type_001` | 6/6 通过 |
| Holdout | `holdout_noise_window_001` | 6/6 通过 |
| Holdout | `holdout_budget_candidates_001` | 6/6 通过 |
| Holdout | `holdout_multihop_production_001` | 6/6 通过 |
| Holdout | `holdout_forgetting_token_001` | 6/6 通过 |
| Holdout | `holdout_long_context_codename_001` | 6/6 通过 |

## Update/Conflict 专项复核

六个治理 Pair 均有一个当前态问题和一个历史态问题。两版记忆按
`valid_from` 递增，当前题期望新值并禁用旧值，历史题期望旧值并禁用未来值；
结构化抽取后的 `subject + predicate` 在每个 Pair 内相同。

| Pair | 旧版本有效起点 | 新版本有效起点 | 当前/历史禁用方向 |
| --- | --- | --- | --- |
| `dev_update_db_001` | 2026-01-01 | 2026-02-01 | 旧 / 新 |
| `dev_conflict_platform_001` | 2026-01-10 | 2026-02-10 | 旧 / 新 |
| `test_update_db_001` | 2026-03-01 | 2026-04-01 | 旧 / 新 |
| `test_conflict_platform_001` | 2026-03-10 | 2026-04-10 | 旧 / 新 |
| `holdout_update_db_001` | 2026-05-01 | 2026-06-01 | 旧 / 新 |
| `holdout_conflict_platform_001` | 2026-05-10 | 2026-06-10 | 旧 / 新 |

## 数据哈希

技术预审没有修改三个 JSONL，候选哈希保持为：

- Development: `6bdfb653ec170343bc4f7ac8983fa541c5717eab26c1bdbc495cfc7426693b34`
- Test: `4580c3968475dfe3726767e6a3859d21a7d3f9861bfbbf71ab050aaf76ceddf8`
- Holdout: `c8bac78e2a2c47b7aaef4009a786ef48af779a45239c1959b92801714b697733`

## 人工门禁

人工 Reviewer `Zhang` 已确认本记录和清单所代表的预审结论，并完成全部 36
条签字。冻结脚本随后成功生成 `frozen_manifest.json`；该清单同时锁定三个
Split 哈希和 `review_checklist.csv` 的签字哈希。任何数据或审核清单变更都会使
正式运行器拒绝执行，必须重新审核和冻结。
