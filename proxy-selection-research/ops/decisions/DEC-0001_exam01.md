# DEC-0001 — EXAM01 执行决策记录

- 日期：2026-10-26
- 状态：已执行（工程），待研究负责人复核
- 关联：RESULT01、PREDICTION01、`protocols/scope.yaml`

## 决策

1. **独立命名空间**：新建 `proxy-selection-research/`，不修改执行包任一原始文件。原始 56 个文件在
   t0 建立 SHA-256 基线（`ops/manifests/inputs_sha256.csv`），完成时全量复核一致
   （`results/RESULT01.json` → `checks.A1`）。
2. **技术样本来源**（TECH 角色，非研究数据）：
   - 真实摄影：`detection-datasets/coco`（COCO 真实照片，40 张，最小边 ≥480px）。
   - AI 生成：`elsaEU/ELSA1M_track1`（扩散生成，40 张；本批全部为 `stabilityai/stable-diffusion-2-1-base`）。
   - 理由：两者均有可核验来源记录、许可与逐条 provenance；读取仅取所需行（HTTP range）。
   - 限制：只有单一生成器进入本批技术样本；EXAM03 的 D_sel 需补充多生成器分层。
3. **候选池**：注册 6 个真实可运行公开检查点（Swin / SigLIP / ConvNeXt / ResNet / SwinV2 / ViT），
   覆盖率 100%，全部真实运行。候选**尚未冻结**，EXAM03 前由负责人确认。
4. **容差测量口径**：FP32 重复推理必须在**相同批次构成**下比较。首次测量因重复子集批次形状不同
   （16 vs 8）出现 5.4e-4 差异，修正后 6 个模型全部为 0.0。该教训写入 RESULT01 偏离 D3。
5. **方向审计**：`resnet_ai` 在技术样本上 AUC=0.312（低于 0.5），均值显示它给真实照片更高的
   "FAKE" 分数。**不据技术样本翻转方向**；标记 `BELOW_CHANCE_DIRECTION_SUSPECT`，留待 EXAM03
   候选审查决定保留/替换（替换理由必须记录为技术资格问题，并保留原记录）。

## 未决

- 物理拍摄设备、屏幕、几何与光照参数（EXAM02 的硬前置）。
- 研究负责人对 δ=ε=0.01、b=40、200 重放、2000 bootstrap 的批准（当前仍 `DRAFT_UNAPPROVED`）。
- 数字测试族 F1—F4 参数冻结（EXAM03 前）。
