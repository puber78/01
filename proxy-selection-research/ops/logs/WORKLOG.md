# WORKLOG — 数字代理选型（DeepSeek Harness 执行）

> 规则：只追加，不删除。每个条目带时间戳（Asia/Shanghai）、命令证据文件、产物哈希登记位置。
> 原始执行包文件永不修改；所有派生结果写在本目录所属的 `proxy-selection-research/` 下。

## 2026-10-26 16:17 — 开工：读取执行包

- 读取：README、DeepSeek_Harness_执行说明、首轮启动指令、主大纲（721 行）、九阶段一览、迁移说明、来源与交付范围、验收参数、EXAM01 工作卡、PREDICTION01、RESULT01 模板、切口 v2 原文。
- 结论：本轮按工作卡与首轮指令只启动 EXAM01 的获准工作；EXAM02—09 依赖人工实拍与负责人批准，按阶段推进。

## 2026-10-26 16:19 — 环境审计（实际测量，非假设）

| 项 | 实测结果 | 证据 |
|---|---|---|
| OS / Python | Windows，Python 3.11.9（另装 3.13，未使用） | `ops/logs/00_venv_create.log` |
| GPU | NVIDIA GeForce RTX 4060 Laptop，8 GB，驱动 616.92 / CUDA UMD 13.4 | `nvidia-smi` 输出（见下） |
| 磁盘 | C: 178.2 GB 可用 / 451.2 GB（D: 177.4 GB 可用） | `[System.IO.DriveInfo]` |
| 系统 TLS | schannel 失败（SEC_E_NO_CREDENTIALS），PowerShell/curl 无法 HTTPS | `pwsh` 诊断 |
| Python TLS | 正常（OpenSSL），PyPI / download.pytorch.org / github / hf-mirror 可达 | 实测 200 |
| huggingface.co | 直连超时；镜像 `hf-mirror.com` 可达（需 User-Agent） | 实测 200 |
| 原始包完整性 | 56 个文件，全部与 `migration/迁移清单.json` 记录一致 | `ops/logs/06_input_verify.log` |

GPU 原始输出（节选）：

```
NVIDIA-SMI 616.92   Driver Version: 616.92   CUDA Version: 13.4
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 49C, 6W / 72W
```

## 2026-10-26 16:20 — 建立留痕工作区

新建 `proxy-selection-research/`（与执行包同级，位于包内），包含：

```
ops/{logs,manifests,archive,decisions,tools}   # 日志、哈希清单、归档、决策记录、工具
protocols/  manifests/  predictions/  scores/  selections/  metrics/  results/  reports/
src/proxy_selection/  tests/  paper/  data/{raw,processed}  models/  tmp/
```

- 输入哈希清单：`ops/manifests/inputs_sha256.csv`（56 行，SHA-256）。
- 覆盖保护：所有写入经 `proxy_selection.io.write_new`，已存在即拒绝；替换必须先归档到 `ops/archive/`。
- 每阶段日志：`ops/logs/runs/<run_id>.jsonl`；命令台账：`ops/logs/command_ledger.csv`。

## 2026-10-26 16:21 — 环境安装（含一次失败与修复）

- 首次 `pip install torch` 在 2.5 GB 轮子下载处停滞（13 分钟仅 6 MB），已终止并保留日志 `ops/logs/01_pip_install.log`。
- 修复：改用 `ops/tools/download_wheels.py` 以 HTTP Range 分块下载 + zip 完整性校验，再本地 `pip install`。下载日志 `ops/logs/02_torch_download.log`、`03_torchvision_download.log`。
- 该失败与修复过程本身作为工程留痕保留，不删除。


## 2026-10-26 17:05 — EXAM01 完成（工程 PASS）

- 环境：torch 2.6.0+cu124 / CUDA 12.4 / RTX 4060；transformers 5.20.2；pandas 3.1.0；numpy 2.4.6。证据 `ops/logs/08_env_check.log`、`ops/manifests/pip_freeze.txt`。
- 单元测试：68 passed / 0 failed（首轮 `ops/logs/09_pytest_exam01.log`，复核 `ops/logs/18_pytest_confirm.log`）。
- 技术样本：80 张（40 COCO 真实摄影 + 40 ELSA1M 扩散生成），provenance 覆盖率 100%，内容哈希 100% 一致（`data/raw/TECH/`）。
- 真实链：6 个公开检查点 × 80 图 = 480 行评分，失败 0；产出 `scores/EXAM01/run_20261026T090345Z_acb8682b/`。
- 重复推理：固定批次下 6/6 模型最大差 = 0.0（容差 1e-6+1e-5|s2|）。
- 结果：`results/RESULT01.json`、`reports/RESULT01.md`、`reports/INTERFACE_CONTRACT.md`；五条硬条件全部 PASS。
- 技术样本 AUC（仅审计）：swinv2 0.921 / siglip 0.849 / swin_sdxl 0.827 / vit 0.524 / convnext 0.508 / resnet 0.312（方向可疑，已标记）。
- 决策记录：`ops/decisions/DEC-0001_exam01.md`。

### 当前阻塞（EXAM02）

EXAM02 要求 20 次技术捕获（含 ≥4 次重复）与 240 组先导实拍，必须由人工用相机/手机拍摄屏幕显示文件。
本环境无相机、无屏幕采集权限，**未执行任何伪造或模拟拍摄**。需人工采集路线决定后才能继续 EXAM02 的物理分支。

### 未批准事项

δ=ε=0.01、b=40、200 次重放、2000 次 bootstrap 仍为 `DRAFT_UNAPPROVED`；`approved_by` 为空。

## 2026-10-26 18:00 — EXAM02 准备完成（等待人工实拍）

负责人选择 **A 路线：人工实拍**。目标环境与设备已登记：

- 主环境 e1：R27qa 2560×1440 显示器 + Redmi K80；备选 e2（仅 EXAM08 外部面板）：华硕天选四 NE156FHM-NX6 1920×1080 + Redmi K90。
- 室内顶灯、JPEG 原生输出、固定 45 cm 垂直拍摄（详见 `captures/kit/capture_protocol.md`）。

已实际完成（非计划）：

1. 源池：440 个来源组 —— 220 真实摄影（COCO）+ 220 扩散生成（ELSA1M, SD 2.1）。
   分配：C_pool 80（40+40）、T_pilot 160（80+80）、D_sel 200（100+100，仅数字）。
   依据文件：`manifests/EXAM02/run_20261026T095*/sources.parquet` 等。
2. 显示资产：256 个 2560×1440 PNG（C_pool 80 + T_pilot 160 + TECH 16），内容矩形已登记。
3. 拍摄队列：技术批 20（16 唯一 + 4 重复）、主批 264（240 唯一 + 24 预定重复），分 4 段各 66。
   队列校验通过：文件齐全、类别平衡、重复项可追溯、序号唯一（`ops/logs/53_verify_kit.log`）。
4. 采集工具：全屏显示程序 `tools/capture_viewer.py`（断点续拍、进度 JSON）、一键启动 `captures/kit/run_*.bat`。
5. 导入工具：`scripts/import_captures.py` —— ORB+单应配准、内容矩形裁剪、模糊/曝光 QC、显式失败码；
   合成图单测已验证配准能恢复透视（`tests/test_registration.py`，73 项测试全部通过）。
6. 决策记录：`ops/decisions/DEC-0002_exam02_capture_design.md`。

**当前状态**：EXAM02 工程侧就绪，物理侧等待人工拍摄技术批 20 张。
未拍任何照片，未把模拟图当作实拍数据。EXAM04 及以后的真实翻拍结论在收到照片前一律为空。

## 2026-10-27 09:10 — 规模决策：240 → 1000（负责人决定）

- 负责人询问"200 多张是否太少、1000 张是否更稳"。做了**基于实测的精度分析**
  （`docs/precision_memo.md`、`metrics/EXAM02_precision/`）：
  - 240 张（T 160）：强模型对成对 AUC 差 95% 半宽约 ±0.10–0.13；
  - 1000 张（每类 500）：约 ±0.04–0.05；
  - 要到 ±0.01 需每类约 8 千–1.3 万张，人工不可行；δ=0.01 只能作实质意义阈值。
  - 先导的职责是筛出**大而具体**的错选，确认阶段再用新来源买精度。
- 负责人选择 **1000 张方案**：C_pool 200、T_pilot 700、预定重复 100；D_sel 500（仅数字）。
- 已重建：源池 1400 组（700 真实摄影 + 700 SD2.1 生成），显示资产 900 张 2560×1440（2.6 GB），
  主批队列 10 段 × 100 张，重复 100 个，队列校验通过（`ops/logs/68_verify_kit_1000.log`）。
- 决策记录：`ops/decisions/DEC-0003_scale_1000.md`。
- 仍未解决：生成器单一（SD 2.1）；跨生成器泛化留给 EXAM08 外部面板。
