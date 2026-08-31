# Benchmark v1.0 人工审核指南

状态：候选集待人工审核

当前 E1/E2 候选集包含 Development、Test、Holdout 各 12 条。机器校验只能
检查结构、时间区间、ID、场景族隔离以及 Update/Conflict 的结构化事实键，
不能代替人对题意和答案的判断。

## 审核边界

审核者逐行查看对应 JSONL，并在
`benchmark/data/v1.0/review_checklist.csv` 填写：

- `question_clear=yes`：问题单义，不依赖未给出的常识；
- `answer_correct=yes`：标准答案由给定记忆直接支持；
- `evidence_ids_correct=yes`：所有必要证据均列入，且没有遗漏多跳证据；
- `forbidden_ids_correct=yes`：旧版本、未来版本或已遗忘记忆标注正确；
- `timestamps_correct=yes`：写入时间、当前查询和历史查询构成正确区间；
- `aliases_checked=yes`：Alias 仅包含审核前认可的等价答案；没有 Alias 也需确认；
- `reviewer`：填写真实审核者姓名或团队内唯一标识；
- `decision=approved`：六项全部通过后才可填写；否则保持 `pending` 并写 Notes。

建议由两名成员分别审核，出现分歧时先修订候选 JSONL、递增候选修订记录并
重新运行机器校验。当前冻结脚本要求至少一份完整签字；比赛最终版本建议在
Notes 中保留第二审核者和仲裁记录。

## 审核顺序

1. 先审核 Development，可在此修订题型和 Alias；
2. 冻结题型规则后审核 Test，不根据模型预测修改答案；
3. Holdout 最后审核，审核完成后不得用于调参；
4. 执行 `python -m benchmark.validate_splits benchmark/data/v1.0`；
5. 执行 `python -m benchmark.freeze_v1 --data-dir benchmark/data/v1.0 --reviews benchmark/data/v1.0/review_checklist.csv`；
6. 将生成的 `frozen_manifest.json` 与数据文件一起提交 Git。

冻结脚本会验证全部 36 个 Case、拒绝空 Reviewer 或未通过字段，并将三个
Split 的 SHA256 写入冻结清单。之后任何数据修改都会使正式运行器拒绝执行。
