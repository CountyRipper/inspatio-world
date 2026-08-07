# Phase 2 在线 latent memory self-rollout 结果

## 1. 结论

本阶段已完成 Phase 2A 小规模闭环和 Phase 2B 扩展训练/留出评测。最终方案保持 InSpatio-World 1.3B backbone、VAE、text encoder 全冻结，只训练现有 122,880 参数 memory adapter，并使用强度 0.35 的 masked timestep-consistent latent anchoring；没有启用 LoRA。

最终 trajectory-held-out 结果为：

- 6 组留出轨迹、24 个返回点中，correct memory 在 24/24 返回点优于 no-memory。
- correct memory 在 24/24 返回点优于 wrong memory。
- 平均 overlap latent L1：no-memory 0.25235，correct 0.08899，wrong 0.24781，mask-only 0.44184。
- 平均 correct-vs-no-memory gain 为 0.16337；平均 wrong-minus-correct gap 为 0.15882。
- correct overlap feature cosine 为 0.98805；correct overlap 外 spill 为 0.22599。
- outward-return、branch-return、multi-memory 三类轨迹以及 S0/S1 两个场景的平均 gain 全部为正。
- pose/FOV 检索、非恒等几何投影、A/B 写回后的 v2 再读在全部组通过。

因此结论是：当前轻量 adapter 加 anchoring 已经形成可运行的在线多记忆长时回访闭环，且在本地可用数据范围内通过强因果对照；无需解冻 backbone 或增加 LoRA。

## 2. 固定模型与数据边界

- Base checkpoint：`/data4/daixiangting/inspatio-world/checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors`
- Base SHA256：`ec60de7789df514c2bc85c7e11fa76af575396ea6d0ccb136520a32384630441`
- Phase 1 final adapter SHA256：`5583e8082596a0b560b3b884854a08d5863c8f19e4dbf01bda7db25b7950adb6`
- Phase 2A adapter SHA256：`7dfdb67396a80dc91a5985288711408942f62b51542c0ef00fcf7fe4250288a7`
- Phase 2 final adapter SHA256：`b5a5afdf418638056f92beb0178f3a3ecabb728a46fbaf8634e456d9f1249824`

本机只有 S0、S1 两个场景具备完整 240 帧 RGB/depth/camera geometry，因此 manifest 使用轨迹级留出，而不是场景级留出。共 30 组：24 train、6 heldout；三类轨迹各 10 组。这个结果不能解释为 unseen-scene 泛化。

## 3. 最终方法

### 在线 memory bank

- 轨迹按 20 个 latent blocks 执行，显式写入 A/B/C 多个 memory ID。
- 每条记录保存模型生成的 detached clean latent、c2w、K、depth、occupancy/confidence、FOV、ID 和 version。
- memory 内容只来自当前模型已经生成的历史块；输入视频/reference 只作为生成条件，不会把未来 return target 写进 memory。
- 检索先按 pose/FOV 距离排序并取 top-k，再用 Phase 1 的几何投影把历史 latent 投到 query view；所有评测投影均验证为非 identity。
- return_write 会用当前生成结果更新 A/B 为 v2，后续 return 必须重新检索并读到 v2。

### Anchoring 与注入

- 在每个 denoise timestep 对 projected latent 做一致的 scheduler re-noising。
- 只在有效 projected occupancy 内以 0.35 强度混合；occupancy 外不直接覆盖。
- correct、wrong 使用相同注入接口；mask-only 保留 mask 但 latent 内容置零；no-memory 完全 bypass。

### Self-rollout 训练边界

- Phase 2B 的 24 组训练 rollout 全部由 Phase 2A 当前模型在线生成，共 96 个返回记录。
- 96/96 correct replay 与原 rollout 达到 `torch.equal`，确保四步输入、transition noise、history 与 KV 语义一致。
- 历史 latent/KV state 全部 detach；只在当前返回块的 4 个 denoise steps 打开计算图。
- 首步梯度审计只有 `model.memory_adapter.proj.weight`，可训练参数 122,880；backbone、VAE、text encoder 均冻结。
- overlap 内使用 projected-memory content loss；overlap 外使用相对 detached no-memory 的 strict raw-L1 preservation。
- 最终配置：AdamW，lr=1e-4，300 steps，preservation weight=0.25，history truncation probability=0.25，anchoring=0.35。

## 4. Phase 2A 与迭代过程

| 阶段 | Groups / returns | Mean correct gain | Mean wrong gap | 两项胜率 | Mean correct spill |
|---|---:|---:|---:|---:|---:|
| Phase 2A：Phase 1 adapter，无 anchoring | 2 / 8 | 0.13990 | 0.14015 | 100% / 100% | 未聚合 |
| Phase 2A：Phase 1 adapter，anchor 0.35 | 2 / 8 | 0.14313 | 0.14416 | 100% / 100% | 未聚合 |
| Phase 2A：200-step adapter，anchor 0.35 | 2 / 8 | 0.14693 | 0.14910 | 100% / 100% | 未聚合 |
| Phase 2A adapter：6-group heldout baseline | 6 / 24 | 0.15824 | 0.15460 | 100% / 100% | 0.22708 |
| Phase 2B 首次 500-step 尝试 | 6 / 24 | 0.14137 | 0.15255 | 100% / 100% | 0.28305 |
| Phase 2B 最终 300-step 修正版 | 6 / 24 | 0.16337 | 0.15882 | 100% / 100% | 0.22599 |

首次扩展训练沿用 lr=5e-4、500 steps、preservation weight=0.5。它虽然保持所有返回点方向正确，但相对 Phase 2A baseline 同时降低 gain 并增加 spill，因此被淘汰。诊断发现 preservation 项的量级长期高于 memory 项，且 adapter 权重相对 Phase 2A 明显漂移。

修正版从 Phase 2A adapter 重新初始化，降低 lr 和 preservation 权重。前/后 96 步平均训练 loss 从 0.05561 降到 0.04863，最终权重范数 9.363 与 Phase 2A 的 9.360 基本一致。g24/g25 快速筛选中，gain 从 0.14910 提升到 0.15416，同时 spill 从 0.25876 降到 0.24952，随后才执行完整留出评测。

## 5. 最终留出结果

### 绝对指标

| Condition | Mean overlap L1 | Mean feature cosine | Mean invalid spill |
|---|---:|---:|---:|
| no-memory | 0.25235 | 0.88094 | 0.00000 |
| correct | 0.08899 | 0.98805 | 0.22599 |
| wrong | 0.24781 | 0.89322 | 0.24763 |
| mask-only | 0.44184 | 0.84502 | 0.26160 |

mask-only 明显差于 correct，说明收益不能由 occupancy/layout 单独解释；wrong memory 接近 no-memory 且显著差于 correct，构成内容因果对照。

### 按轨迹族

| Family | Groups / returns | Mean correct gain | Mean wrong gap |
|---|---:|---:|---:|
| outward-return | 2 / 8 | 0.14668 | 0.15017 |
| branch-return | 2 / 8 | 0.16242 | 0.14181 |
| multi-memory | 2 / 8 | 0.18101 | 0.18447 |

### 按场景

| Scene | Groups / returns | Mean correct gain | Mean wrong gap |
|---|---:|---:|---:|
| S0 | 3 / 12 | 0.15246 | 0.14890 |
| S1 | 3 / 12 | 0.17428 | 0.16874 |

## 6. 定性与视频检查

- S0 中 correct 保持杯具、托盘、桌缘、食物数量与位置；wrong 出现局部内容/位置偏差；mask-only 呈明显灰雾和低频覆盖。
- S1 中 correct 保持人物、瓶子阵列、窗框与台面结构；wrong 的瓶子/背景局部不一致更明显；mask-only 同样退化。
- 已人工检查 S0/S1 代表性 montage，并对 full correct/no-memory 视频做均匀时间抽帧，没有发现破图、突然清空或只在 return keyframe 正常的现象。
- 最终产物包含 24 个视频与 6 张 montage。全部视频经 ffprobe 验证为 832x480、24 fps、237 frames。

## 7. 运行审计

- Phase 2A capture：2 groups、8 records、87.3 s、peak 35.68 GiB。
- Phase 2A train：200 steps、371.8 s、peak 6.49 GiB。
- Phase 2A 四条件评测：258.5 s、peak 35.69 GiB。
- Phase 2B capture：24 groups、96 records、738.2 s、peak 35.70 GiB。
- Phase 2B 最终 train：300 steps、811.3 s、peak 6.49 GiB。
- Phase 2B 最终四条件 heldout：6 groups、713.1 s、peak 35.71 GiB。
- 96/96 replay exact；24/24 训练组历史 detach；未来 return GT 使用次数为 0；24/24 训练组写回再读通过。
- Base checkpoint、Phase 1/Phase 2A 初始化 adapter 在各次运行前后 SHA256 保持不变。
- Phase 2 CPU tests：5/5 passed；Phase 1 regression tests：17/17 passed；Phase 2 源码与测试通过 compileall。
- GPU launcher 只检查物理 GPU 0/1/2，选择空闲显存最多者，不终止其他进程；并行评测暴露的固定 rendezvous port 冲突已修复为动态空闲端口。

## 8. 产物与复现入口

- 最终 adapter：`artifacts/phase2_memory_selfrollout/final/memory_adapter.safetensors`
- 最终训练摘要：`artifacts/phase2_memory_selfrollout/final/training_summary.json`
- 最终聚合指标：`artifacts/phase2_memory_selfrollout/final/heldout_eval/aggregate_metrics.json`
- 最终 24 个视频：`artifacts/phase2_memory_selfrollout/final/heldout_eval/groups/*/videos/*.mp4`
- 最终 6 张 montage：`artifacts/phase2_memory_selfrollout/final/heldout_eval/groups/*/montage.png`
- Phase 2A 证据：`artifacts/phase2_memory_selfrollout/final/phase2a_eval/`
- 数据/轨迹清单：`configs/phase2_memory_manifest.json`
- 运行入口：`scripts/phase2_memory/{capture_rollouts,train_adapter,evaluate,launch_on_best_gpu}.py`
- CPU tests：`scripts/phase2_memory/run_cpu_tests.py`

README 已给出最终 Phase 2B 复现命令。所有 rollout、checkpoint、视频、图和日志均位于被 gitignore 的 artifact 目录；只暂存源码、配置、测试和文档。

## 9. 未证明边界

- 只有两个 geometry-complete 场景，因此只证明 trajectory-held-out，不证明 unseen-scene 泛化。
- correct 的 overlap 外 spill 仍为 0.22599；虽然最终版略优于 Phase 2A baseline，但远未消除 DiT 空间混合带来的污染。
- Phase 2B 使用一次完整 self-rollout 采集后固定 replay bank 训练，没有在每个 optimizer step 后重新生成全轨迹。
- 检索目前是 pose/FOV 粗排序，没有学习式 content reranker。
- no-memory 本身已经较强，因此部分定性差异比 latent 指标更细微。
