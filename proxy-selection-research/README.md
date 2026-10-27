# proxy-selection-research

数字代理评测与真实翻拍选型 — DeepSeek Harness 执行工作区（EXAM01—09）。

本目录是**新增**工作区，不修改执行包中的任何原始文件（原始 56 个文件的 SHA-256 已登记并与
`migration/迁移清单.json` 逐项核对一致，见 `ops/logs/06_input_verify.log`）。

## 目录

| 路径 | 用途 |
|---|---|
| `protocols/` | 冻结的范围、数据角色、数字族、选择规则、容差、先导与确认合同（YAML） |
| `manifests/` | 来源/谱系/显示/捕获/样本清单（parquet）与模型注册表 |
| `scores/EXAMxx/<run_id>/` | 逐图评分（schema 见 `contracts/schema.py`） |
| `selections/EXAMxx/` | 选择日志（冻结后写入） |
| `metrics/EXAMxx/` | 统计估计与区间 |
| `results/RESULTxx.json`、`reports/RESULTxx.md` | 阶段结果记录与报告 |
| `src/proxy_selection/` | 研究代码（合同、适配器、指标、选择规则、评分、日志） |
| `tests/` | 单元测试（合成夹具只在 `tests/` 内） |
| `ops/` | 日志、命令台账、产物哈希台账、归档、决策记录、工具 |
| `data/raw/` | 原始/技术样本（只读，写入即登记哈希） |
| `models/` | 候选检测器审计与权重缓存（`hf_cache/`，不纳入版本） |
| `paper/` | 论文稿、主张—证据索引 |

## 运行

```powershell
# 环境（一次性）
python -m venv .venv
.venv\Scripts\python -m pip install <wheels> transformers timm datasets ...

# 测试
.venv\Scripts\python -m pytest tests -q

# 技术样本（EXAM01，TECH 角色，非研究数据）
.venv\Scripts\python scripts\fetch_technical_samples.py

# 最小真实链：真实检查点 → 官方预处理 → 分数 → AUC
$env:HF_ENDPOINT="https://hf-mirror.com"
.venv\Scripts\python scripts\run_exam01_chain.py
```

## 留痕纪律

1. 所有写入经 `write_new`（存在即拒绝）或 `replace_with_archive`（先归档旧字节）。
2. 每个 parquet/json 产物登记到 `ops/manifests/artifacts.csv`（路径、SHA-256、字节数、run_id）。
3. 每次运行写 `ops/logs/runs/<run_id>.jsonl`（环境、输入哈希、产物、失败、结束状态）。
4. 不删除任何产物；失败日志与失败产物同样保留。
5. 原始文件覆盖次数、越权读取次数是本项目的工程验收指标，必须为 0。

## 状态

见 `reports/` 与 `ops/logs/WORKLOG.md`。数值参数仍为 `DRAFT_UNAPPROVED`，未经研究负责人批准。
