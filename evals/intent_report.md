# 意图识别离线评测报告

- 生成时间：2026-11-02 16:39:00
- 评测集：`evals\dataset.json`（20 题）
- 训练集：`evals\intent_dataset.json`（72 条）
- 路由模式：L1+L2（确定性离线路径）

## 结论

- 准确率：**95.0%**（19/20）；目标 ≥ 85.0%：**达标 ✅**
- macro F1：0.9495
- 分层命中：L1 17 题 / L2 3 题 / L3 0 题 / L4 澄清 0 题
- 路由覆盖率：100.0%（20/20 题被 L1/L2/L3 成功分流）
- 分类耗时：平均 0.030 ms / 最大 0.124 ms（另有 L2 首次冷启动训练 1.7 ms，不计入逐题均值）

## 分层命中与覆盖率

| 层 | 说明 | 题数 | 占比 |
| --- | --- | --- | --- |
| L1-rule | L1 规则 | 17 | 85.0% |
| L2-classifier | L2 分类器 | 3 | 15.0% |
| L3-llm | L3 LLM | 0 | 0.0% |
| L4-clarify | L4 澄清 | 0 | 0.0% |

| 指标 | 数值 |
| --- | --- |
| L1 规则命中 | 17 |
| L2 分类器命中 | 3 |
| L3 LLM 兜底 | 0 |
| L4 兜底（需澄清） | 0 |

## 混淆矩阵（行 = 真实，列 = 预测）

| 真实 \ 预测 | database（数据分析） | document（内部知识） | web（市场情报） | mixed（综合） | 合计 |
| --- | --- | --- | --- | --- | --- |
| database | 5 | 0 | 0 | 0 | 5 |
| document | 0 | 5 | 0 | 0 | 5 |
| web | 0 | 0 | 5 | 0 | 5 |
| mixed | 1 | 0 | 0 | 4 | 5 |
| 合计 | 6 | 5 | 5 | 4 | 20 |

## 每类指标

| 意图 | 中文名 | precision | recall | f1 | support |
| --- | --- | --- | --- | --- | --- |
| database | 数据分析 | 0.8333 | 1.0000 | 0.9091 | 5 |
| document | 内部知识 | 1.0000 | 1.0000 | 1.0000 | 5 |
| web | 市场情报 | 1.0000 | 1.0000 | 1.0000 | 5 |
| mixed | 综合 | 1.0000 | 0.8000 | 0.8889 | 5 |

## 逐题明细

| 编号 | 意图 | 预测 | 层 | 置信度 | 耗时(ms) | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| db-01 | database | database | L1-rule | 0.8600 | 0.038 | ✅ |
| db-02 | database | database | L1-rule | 0.8600 | 0.020 | ✅ |
| db-03 | database | database | L1-rule | 0.8200 | 0.018 | ✅ |
| db-04 | database | database | L1-rule | 0.8600 | 0.016 | ✅ |
| db-05 | database | database | L1-rule | 0.8200 | 0.015 | ✅ |
| rag-01 | document | document | L1-rule | 0.8600 | 0.016 | ✅ |
| rag-02 | document | document | L1-rule | 0.8600 | 0.014 | ✅ |
| rag-03 | document | document | L1-rule | 0.8600 | 0.014 | ✅ |
| rag-04 | document | document | L1-rule | 0.8200 | 0.016 | ✅ |
| rag-05 | document | document | L1-rule | 0.8200 | 0.014 | ✅ |
| web-01 | web | web | L1-rule | 0.8200 | 0.015 | ✅ |
| web-02 | web | web | L2-classifier | 1.0000 | 0.124 | ✅ |
| web-03 | web | web | L1-rule | 0.8200 | 0.032 | ✅ |
| web-04 | web | web | L2-classifier | 0.9967 | 0.091 | ✅ |
| web-05 | web | web | L2-classifier | 1.0000 | 0.080 | ✅ |
| mix-01 | mixed | mixed | L1-rule | 0.9400 | 0.020 | ✅ |
| mix-02 | mixed | mixed | L1-rule | 0.8600 | 0.016 | ✅ |
| mix-03 | mixed | mixed | L1-rule | 0.8600 | 0.015 | ✅ |
| mix-04 | mixed | database | L1-rule | 0.8200 | 0.016 | ❌ |
| mix-05 | mixed | mixed | L1-rule | 0.9000 | 0.015 | ✅ |

## 误判 / 需澄清

| 编号 | 问题 | 预期 | 预测 | 层 | 置信度 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| mix-04 | orders 表里最早和最晚的订单分别发生在哪一天？ | mixed | database | L1-rule | 0.8200 | 误判 |

## 说明

- L2 为纯 Python 字符 1-3 gram 朴素贝叶斯，拉普拉斯平滑 alpha=1.0；接受条件为 confidence ≥ 0.55 且 margin ≥ 0.15。
- 训练集与评测集无重叠原句，但覆盖同一领域词汇；短文本下朴素贝叶斯后验容易饱和，因此 L3/L4 主要作用于词表外或表述模糊的问句。
- mix-* 题的 expected_subagents 是按题面实际需要标注的子集，本报告只评 intent 标签本身；子智能体召回率由 evals/run_eval.py 的 routing 指标覆盖。
