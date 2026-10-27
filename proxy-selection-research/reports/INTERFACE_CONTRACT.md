# 接口合同（EXAM01）

> 版本 `contracts_v1`，2026-10-26。对应源码：`src/proxy_selection/contracts/`、`scoring.py`、`adapters/`。
> 本文件描述**已实现并测试**的接口；不是计划。

## 1. 标识符合同

| 实体 | 前缀 | 生成方式 | 语义 |
|---|---|---|---|
| source | `src_` | UUIDv5(lineage_key,label,provenance_uri) | 一个原始来源（摄影或生成），标签由来源记录决定 |
| split_group | `grp_` | UUIDv5(lineage_key,generation_family) | 谱系/划分单位；同一谱系不得拆分到不同角色 |
| display_asset | `dsp_` | UUIDv5(source_id,display_spec_hash) | 送入屏幕显示的固定文件；TECH 用 identity 规格 |
| capture | `cap_` | UUID4（事件） | 一次物理拍摄（含重复、失败） |
| sample | `smp_` | UUIDv5(branch,transform_id,parent) | 一次推理单元；digital 父为 dsp，physical 父为 cap |
| model | `mdl_` | UUIDv5(repo,revision,weight_sha256) | 冻结检查点；换权重即换 id |
| rule | `rul_` | UUIDv5(rule_name,spec) | 选择规则实例（含冻结参数） |
| run | `run_` | 时间戳+随机后缀 | 一次执行；所有产物带此 id |

校验函数 `require_id(value, kind)` 强制前缀与字符集；错误类型 `IdError`。

## 2. 三角色访问表

| 逻辑角色 | 可读数据角色 | 禁止 |
|---|---|---|
| development | C_pool, D_sel, C_confirm, D_sel_confirm, TECH | T_pilot, T_confirm, EXTERNAL |
| evaluation | 全部 | ——（但必须先冻结选择再揭示） |
| audit | 全部（只读） | 不得拟合/选择 |

实现：`AccessPolicy.check_read/check_write` 记录每次尝试并累计 `violations`；越权抛
`AccessViolation`。工程验收要求未授权跨角色读取次数为 0。

## 3. 评分行 schema（逐图）

必填字段（与主大纲 §7.2 一致）：

```text
run_id, sample_id, source_id, split_role, environment_id, model_id,
image_branch, transform_id, raw_score, score_orientation,
checkpoint_hash, preprocess_hash, status, failure_code, is_fixture
```

约束：

- `status ∈ {ok, failed, skipped}`。
- `status=ok` ⇒ `raw_score` 为 [0,1] 内有限浮点，`failure_code=None`。
- `status≠ok` ⇒ `raw_score` **必须为 None**，`failure_code` 必须属于预定义集合。
  **禁止用 0 或 0.5 填补缺失分数**（单元测试 `test_failed_row_requires_null_score_and_code`）。
- `is_fixture=True` 的行只能来自 `tests/`，不得进入研究评分或论文主表。
- `score_orientation ∈ {higher_is_ai, higher_is_real}`；适配器在进入指标层之前完成定向。

## 4. 模型适配器合同

- 输入：单张 RGB `PIL.Image`；**不接收**标签、设备、来源、角色。
- 预处理：必须来自模型自带 `AutoImageProcessor`（官方配置），不手写 resize/normalize。
- 方向：由 `config.id2label` 关键字确定 AI 正类；无法唯一确定则 `OrientationUndetermined`，
  需在 registry 中给出**有据可查的精确 `ai_label`**，不得猜测。
- 哈希：`checkpoint_hash` = 权重文件字节哈希集合的规范 JSON 哈希；`preprocess_hash` =
  预处理配置哈希。两者写入每一行评分。
- 失败：显式 `failure_code`（`FILE_MISSING / DECODE_ERROR / MODEL_ERROR / OOM / NON_FINITE_SCORE / …`）。
- 数值重复性：同一输入两次推理须满足 `|s1-s2| <= 1e-6 + 1e-5*|s2|`（FP32 合同）。

## 5. 评分与统计分离

- `scoring.score_rows()` 只做「图像 → 分数」；标签在**推理完成之后**由
  `join_labels()` 从清单合并，并在合并时校验 `source_id` 唯一。
- `scoring.compute_auc_table()` 从**落盘的 parquet** 重新读取并计算 AUC；这是"独立评分接口"。
- AUC 使用 mid-rank 处理并列（`metrics/auc.py`），`y=1` 为 AI 生成。
- 同模型差值严格为 0（同一实现、同一数据），不依赖容差。

## 6. 覆盖与不可变

- `io.write_new()`：目标存在即拒绝，并登记 `guard_violations.csv`。
- `io.replace_with_archive()`：先复制旧字节到 `ops/archive/`，再替换。
- 每个产物登记 `ops/manifests/artifacts.csv`：路径、SHA-256、字节数、kind、run_id。
- `ops/logs/runs/<run_id>.jsonl` 记录环境、输入哈希、产物、失败与结束状态。

## 7. 已知边界

- 当前未实现"物理捕获"环节（需要人工设备）；`captures` 表为空是**真实状态**，不是遗漏。
- `S_quality`、`S_residual` 为计划项；`S_dig/S_real/S_shrink/S_weight` 已实现并有测试。
- 数字测试族参数（F1—F4）尚未冻结，等待 EXAM03 前负责人审查。
