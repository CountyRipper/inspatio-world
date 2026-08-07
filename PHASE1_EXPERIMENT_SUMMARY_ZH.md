# Phase 1 shared-memory 宽视角实验总结

## 1. 研究问题与范围

本阶段验证一个冻结 InSpatio-World backbone、仅训练轻量 memory adapter 的方案，能否从同一份历史 projected latent 中读取可识别的内容约束，并把这种能力从 exact revisit 扩展到 medium 和 wide near-view，同时观察 overlap 外的污染。

- 场景：S0、S1。
- 每个场景只 capture 一次历史 A，并共享同一份 `z_A`、latent prefix、last prediction、KV cache 和 RNG continuation state。
- 查询视角：0°、±10°、±20°；S0 正向 20° 无有效 overlap，按预定规则改为 +15°，仍为 0% coverage，因此只作为 hard-bypass diagnostic。
- 性质：两个场景上的 capacity/overfit 探索，不代表 held-out 泛化。

## 2. 最终方法

### Shared A 与空间投影

- A 使用 lossless render、mask、depth、K、c2w 和 latent，不为不同 query 重复编解码公共历史。
- S0、S1 的 A capture count 均为 1；各场景全部 query 引用同一个 `z_A` hash，分叉前 causal tensor/state 一致。
- backprojection 始终使用相同的 A-depth、K、A-c2w 和 `z_A`，只改变 target c2w。
- 非 exact query 的 projection 均为真实非 identity 投影；0 overlap 的 S0 +15° 被明确排除出内容恢复统计。

### Memory adapter 与 hard gate

- 保留 122,880 参数的 Conv3d memory adapter，没有增加可训练参数。
- projected occupancy 使用严格 0/1 mask，并按 patch embedding 的 `(1, 2, 2)` kernel/stride 做 max-pool，得到 0/1 `G_patch`。
- 注入顺序为：adapter 生成 residual → residual 乘 `G_patch` → 加入 DiT embedding。
- correct、mask-only、wrong-same-mask 共享完全相同的 gate；no-memory 使用 hard bypass。
- gate 外 residual 的 `max_abs=0`、`count_nonzero=0`；全零 occupancy 时四个条件精确退化到相同 no-memory 输出。
- 只有 adapter 权重参与梯度更新，DiT、VAE、text encoder 全部冻结。

### 训练与评测

- 初始化：`artifacts/phase1_lsm/train/fixed8_projected/memory_adapter.safetensors`。
- 一个 adapter 联合训练两个场景的 9 个 eligible query。
- balanced cyclic sampling，AdamW，lr=1e-3，500 steps，preservation weight=0.5。
- overlap 内优化 projected-memory content loss；overlap 外使用 prediction 相对 detached no-memory 的 strict raw L1。
- 保持仅最后一个 denoise step 反传。
- 训练后冻结 adapter，每个 query 固定评测 no-memory、correct、mask-only、wrong-same-mask 四组。

曾以同一批 shared-A 样本定向尝试 preservation weight=1.0。它没有改善最终目标：S0 spill 仅小幅变化，而 S1 spill 由约 0.115 上升到约 0.135，同时 valid gain 下降，因此最终保留 0.5。

## 3. 视角与投影 coverage

| Scene | Query | Actual yaw | Coverage | 说明 |
|---|---:|---:|---:|---|
| S0 | exact | 0° | 26.3% | eligible |
| S0 | +10 | +10° | 8.0% | eligible，最低非零 coverage |
| S0 | -10 | -10° | 22.2% | eligible |
| S0 | +15 | +15° | 0.0% | low/no-overlap diagnostic |
| S0 | -20 | -20° | 20.8% | eligible wide |
| S1 | exact | 0° | 42.2% | eligible |
| S1 | +10 | +10° | 25.6% | eligible |
| S1 | -10 | -10° | 35.3% | eligible |
| S1 | +20 | +20° | 12.9% | eligible wide |
| S1 | -20 | -20° | 32.6% | eligible wide |

## 4. 核心量化结果

- Eligible query：9/10。
- correct 的 overlap latent raw-L1 在 9/9 eligible query 中优于 no-memory。
- correct 在 9/9 eligible query 中同时优于 mask-only 和 wrong-same-mask。
- 两个场景的 exact 和 ±10° valid gain 方向全部一致为正。
- 三个 eligible wide query（S0 -20°、S1 ±20°）均保持正向内容恢复收益。

| Scene | Mean valid gain | Mean invalid spill | Mean net gain |
|---|---:|---:|---:|
| S0 | 0.1615 | 0.0572 | -0.0180 |
| S1 | 0.1833 | 0.1149 | -0.0338 |

代表性 decoded overlap pixel-L1：

- S0 -20°：no-memory 0.075，correct 0.019。
- S1 +20°：no-memory 0.081，correct 0.036。
- S0 +15°：occupancy 全零，所有 memory 条件精确 hard-bypass，不计内容指标。

量化说明 adapter 确实利用了正确 memory content；同时两场景 aggregate net gain 仍为负，表明注入后的 DiT 空间混合会把影响传播到 gate 外，invalid preservation 尚未完全解决。

## 5. 定性观察与分析

### 正向证据

- S0 的 exact、±10°和 -20°中，correct 保持了食物数量、托盘/桌缘、杯具位置和局部纹理。
- S1 的 exact、±10°和 ±20°中，correct 保持了瓶子排列、架上物、窗框、台面和壶的位置。
- mask-only 普遍产生低频灰雾或半透明结构，说明 occupancy/layout 本身不能解释 correct 的恢复。
- wrong-memory 会引入与错误场景内容相关的杯盘、窗框或台面结构，并产生明显重影，说明网络真正读取了 memory content。
- 从 exact 到 medium/wide，correct 的绝对 overlap error 总体随 coverage 降低而上升，没有离开 exact pose 后立即崩溃。

### 主要失败模式

- correct 相对 no-memory 的肉眼增益小于它相对 mask-only/wrong-memory 的差异；no-memory 本身已能生成不少合理内容。
- S1 +10°和 +20°的 invalid spill 与 temporal flicker 偏高，是当前最弱方向。
- hard gate 保证注入瞬间 gate 外 residual 为零，但不能阻止后续 DiT block 的空间 mixing 把影响扩散到非 overlap 区域。
- preservation weight 从 0.5 提升到 1.0 没有稳定改善该问题，说明当前瓶颈不只是 loss 权重不足。

## 6. 结论与 Phase 2 决策

本次实验已经证明当前轻量 adapter 具备初步可见的 shared-memory 内容读取能力，而不仅是利用 mask/layout：correct 在两个场景、exact/medium 和三个 eligible wide query 上都提供一致的 overlap 恢复；wrong-memory 的内容相关破坏进一步构成因果对照。wide-view 没有立即崩溃，且视觉上的非 overlap 污染尚未压倒 overlap 内收益。

因此本轮建议：

**ENTER_PHASE2**

这里只给出决策，没有实际启动 Phase 2。进入下一阶段后最重要的问题是降低后续 DiT mixing 造成的 invalid spill；若回到 Phase 1 继续优化，优先检查四个 denoise steps 是否应完整参与反传，而不是先解冻 backbone。

## 7. 未证明边界

- 不证明 held-out 场景或更多相机轨迹上的泛化。
- 不证明动态 memory 更新、长期 rollout 或完整 LSM 系统效果。
- 不证明四个 denoise steps 完整反传的收益。
- 不证明当前 negative aggregate net gain 已解决。
- projected-A composite reference 只是 overlap 内 projected `z_A`、外部 no-memory 的可视化参考，不是真实 A′ ground truth。

## 8. 结果与复现入口

- 最终完整结果：`artifacts/phase1_lsm/phase1_exploratory_wide/final/`
- 详细实验报告：`artifacts/phase1_lsm/phase1_exploratory_wide/RESULT_ZH.md`
- 核心指标：`artifacts/phase1_lsm/phase1_exploratory_wide/final/aggregate_metrics.json`
- 逐 query 指标：`artifacts/phase1_lsm/phase1_exploratory_wide/final/metrics_per_query.csv`
- 主 montage：`artifacts/phase1_lsm/phase1_exploratory_wide/final/montages/`
- 四条件视频：`artifacts/phase1_lsm/phase1_exploratory_wide/final/videos/`
- 代表性 wide crop：`artifacts/phase1_lsm/phase1_exploratory_wide/final/overlap_crops/`
- 最终 adapter：`artifacts/phase1_lsm/phase1_exploratory_wide/final/memory_adapter.safetensors`
- 主运行入口：`scripts/phase1_lsm/run_wide_sharedA_hardgate.py`
- 训练评测入口：`scripts/phase1_lsm/train_eval_wide_sharedA_hardgate.py`

## 9. 运行与验证状态

- 成功使用物理 GPU 0；没有发生 CUDA OOM。
- 最终产物包含 40 个四条件 full-context 视频和 20 张主/overlap-error montage。
- Phase 1 CPU tests：17/17 passed。
- 相关 Python 文件全部通过 `py_compile`。
- `git diff --cached --check` 通过。
- base checkpoint 与初始化 adapter 的 SHA256 在运行前后保持不变。
