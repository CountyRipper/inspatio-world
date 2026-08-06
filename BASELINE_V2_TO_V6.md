# InSpatio-World Baseline V2–V6 简要汇总

本文按当前仓库中的入口脚本、实现分支和测试整理，描述的是实现方案，不代表效果排名或 benchmark 结果。

## 总体演进

V2–V5 都在 InSpatio 的分块生成过程中维护历史 RGB 点记忆：先按计划相机位姿渲染已有地图，与离线 reference 按“reference 有效像素优先”融合并编码为条件；当前块生成完成后，再估计几何并写回地图。

V6 系列改用 SLAM3R，并作为已生成视频的离线重建/二次生成流程：V6 保留因果的“先读旧图、再写当前块”，V6.2 则用完整序列构建固定地图后执行 InSpatio Pass 2。

## 版本对照

| 版本 | 几何后端 | 地图与写入 | 条件读取 | 相对上一版的重点变化 |
|---|---|---|---|---|
| V2（legacy） | 单帧 DA3 | `RGBPointMemory`；默认每块最后一个 keyframe；固定体素和点数上限，仅写 reference 未覆盖区域 | 当前块计划位姿渲染，reference 严格优先 | 最基础的有界历史点云模式。当前仓库没有独立 V2.0 脚本，对应统一入口的默认 `bounded_voxel` 模式 |
| V2.1 | DA3 block depth | `DenseGeneratedPointMemory`；append-only 双层记忆；可写每个 latent keyframe（237 帧视频共 60 个）或全部 237 帧 | 当前块读取；reference 层与 generated 层严格融合 | 引入 `dense_two_layer`，用 reference depth 校准生成深度尺度，并保存置信度，不再只保留 reference 空洞处的生成点 |
| V2.2 | 持久化 Align3R worker | 延续 `dense_two_layer`，固定为 `full_block`，写入全部 237 帧 | 与 V2.1 相同 | 用 Align3R 替换生成块的 DA3 深度；复用已验证的 V2.1 reference geometry，通常要求 InSpatio/Align3R 分占两张 GPU，并增加结果审计 |
| V3 | DA3 小窗口 + robust Sim(3) | 增量 `IncrementalVoxelSurfelMemory`；每块 3 个 latent keyframe；固定 voxel `0.02`、上限 `500k` | 当前块计划位姿渲染 | 首块对齐 immutable reference；后续块带上一个“上一块最后 keyframe”作 anchor，把局部 DA3 点注册到 canonical world；拒绝高 RMSE 或尺度跳变 |
| V3.1 | 与 V3 相同 | 增量 voxel surfel；根据 reference 中值深度和焦距自适应 voxel，目标投影间距 3 px；上限 `3M`、3×3 splat | 当前块读取；增加 GPU map 连续性检查 | 保留 V3 单 anchor/Sim(3)，主要改进尺度自适应、覆盖密度和纯 GPU 读写；在线读路径无 PLY round-trip |
| V3.2（测试） | 单帧 DA3 + immutable-reference Sim(3) | 延续 V3.1 自适应 voxel surfel，但整段只在第一个精确 `yaw=45°` 帧估计并写入一次，之后地图固定 | 写入前为空图，写入后各块只读取同一份静态地图 | 面向纯 yaw `0→45→0→45` 的消融实验；不再使用逐块 latent keyframe 或 anchor 链，第二次到达 45° 不重建 |
| V4 | DA3 全历史 prefix + robust Sim(3) | 每块重新解码并重建全部历史 latent keyframes；reference-aware geometry admission；自适应 voxel surfel | 从第 0 帧到当前块末尾的 full-prefix 渲染、融合和编码 | 由 V3 的相邻块增量注册改为全前缀重建；在 reference 覆盖区只接收几何一致点，在新视区接收有效点，降低累计漂移 |
| V5 | MapAnything | 保存通过质量门限的当前块点；从所有已接受 frame chunks 重建有界 voxel map | 沿用 V4 full-prefix 条件读取 | 用 pose-conditioned MapAnything 替换 DA3；比较 `pred_only` 与 `source+pred paired` 两个分支，按 immutable reference 几何误差选优，只保留预测视图点，并按一致率决定是否写入 |
| V6 | 增量 SLAM3R（I2P + L2W） | 离线重放已生成视频；自适应 voxel surfel；置信度过滤和 V4 geometry admission | 每块先用旧地图渲染并融合，再注册当前 latent keyframes | 不再对每块运行 DA3/MapAnything；先以若干 reference 对应拟合并冻结 SLAM3R→canonical Sim(3)，保持 `read-before-write` 的因果顺序和原计划轨迹 |
| V6.2 | 官方 SLAM3R offline pipeline | 对完整 latent-keyframe 序列重建；冻结 Sim(3) 后按 voxel 保留最高置信度观测，形成固定地图 | 固定地图沿完整计划轨迹渲染，与 reference 严格融合 | 从 V6 的因果增量 replay 改为非因果的完整序列固定地图；随后生成 Pass 2 输入并再次运行 InSpatio，形成完整二次生成流程 |

## 主要入口

| 版本 | 入口 |
|---|---|
| V2 legacy | `run_scripts/run_test_pipeline.sh`，启用 `--historical_memory`，使用默认 `bounded_voxel` |
| V2.1 | `run_scripts/run_dense_memory_baseline_v2_1.sh` |
| V2.2 | `run_scripts/run_dense_memory_baseline_v2_2.sh` |
| V3 | `run_scripts/run_overlap_voxel_v3_server15.sh` |
| V3.1 | `run_scripts/run_overlap_voxel_v3_1_server15.sh` |
| V3.2（测试） | `run_scripts/run_overlap_voxel_v3_2_server15.sh`；example0/1 的唯一 keyframe 分别为 82/100 |
| V4 | `run_scripts/run_overlap_voxel_v4_server15.sh` |
| V5 | 无独立 run 脚本；通过 `run_test_pipeline.sh` 选择 `overlap_voxel_v5`、`mapanything` 和本地模型路径 |
| V6 | `run_scripts/run_slam3r_offline_v6.sh` |
| V6.2 | `run_scripts/run_slam3r_offline_v6_2_full.sh`；内部串联 keyframe 导出、官方 SLAM3R、固定地图构建和 InSpatio Pass 2 |

## 核心实现位置

- V2–V5 的统一读写控制：`pipeline/historical_memory_controller.py`
- 点记忆、voxel surfel、dense chunks 和融合函数：`utils/historical_point_memory.py`
- V3/V4 的 DA3 Sim(3) 与几何准入：`utils/overlap_da3_registration.py`
- V5 MapAnything 封装：`utils/mapanything_estimator.py`
- V6 增量 SLAM3R adapter：`utils/slam3r_incremental.py`
- V6 离线因果 replay：`scripts/run_slam3r_offline_v6_impl.py`
- V6.2 固定地图构建：`scripts/build_slam3r_offline_v6_2.py`

## 注意事项

- 当前实现没有名为 V6.1 的版本。
- V2.0 没有独立命名入口；上表的 V2 指当前代码中保留的 legacy/default `bounded_voxel` 模式。
- V2–V5 的 historical-memory controller 目前要求 batch size 为 1。
- V5、V6 和 V6.2 依赖额外的外部模型/代码仓库；V6/V6.2 明确要求 CUDA。
- `run_align3r_inspatio_example0_1.sh` 和 `run_align3r_full_frames_example0_1.sh` 是 Align3R 对照/辅助实验，不是新的独立版本号。
