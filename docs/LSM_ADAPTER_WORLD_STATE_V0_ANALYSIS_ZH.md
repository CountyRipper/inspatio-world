# LSM-adapter 与 World State-v0 代码分析

> 分析快照：当前工作区提交 `8934dc1`（`feat: add continuous revisit evaluation`）。当前 `codex/worldstate-reader-v1` 与 `codex/worldstate-reader-v0` 都指向该提交。本文只读分析现有源码与已有实验产物；除本报告外未修改源码、配置、模型、数据或产物。

> 并发工作区说明：分析末期出现了一个并非本次创建的 untracked 文件 `world_state/domains.py`。它的 docstring 明确指向 Reader v1，且当前没有任何文件 import/调用它，因此不属于本文所分析的 v0 有效调用链；本次保持该文件原样。

## 1. 先给结论

仓库里没有 `LSMAdapter` 类，也没有出现字符串 `LSM`。结合分支、文档、配置和调用链，用户所称 **LSM-adapter** 对应仓库中的 `LatentMemoryAdapter`，即 **exact-pose implicit latent memory adapter/readout**。

它和 World State-v0 是两套层级不同、且在单次模型调用中被代码强制互斥的机制：

| 对比项 | LSM-adapter（实际类名 `LatentMemoryAdapter`） | World State-v0（实际名 `WorldStateReader/Teacher v0`） |
|---|---|---|
| 定位 | 精确同位姿 clean latent 读取基线 | 只读、可见性驱动、查询相关的局部世界记忆读取器 |
| 记忆载体 | 单个 `[B,F,16,H,W]` clean latent block | 不可变 observation bank：保守 source observation + 一个 generated `M40_A/B` |
| 几何 | 无投影；要求 exact pose | 相机角度检索 + pure-rotation homography 投影 |
| 注入位置 | 直接加到 DiT patch embedding | 在指定 DiT block 的原生 self-attention 前做局部 cross-attention residual |
| 是否改 noisy latent | 否 | 否 |
| 是否进入原生 KV/ST cache | 直接 residual 会参与当前 noisy block 的计算，但不会在 context-fill 提前写入；最终 `x0` 通过既有 STAR 路径间接进入下一块 | World token/KV 从不写入原生 cache；只有生成的 clean `x0` 通过既有 STAR 路径间接进入下一块 |
| 可训练部分 | 一个无 bias 的 `Conv3d`，122,880 参数 | Encoder + 3 个 Reader + 可选 Q/O LoRA；当前最终 sidecar 为 10,141,705 个参数值 |
| 当前结论 | exact-pose 身份读取成功 | 两场景结果为 `PARTIAL`，能连续、内容特异地读取，但仍偏模糊 |

两者最重要的关系是：**World State-v0 会加载已经训练好的 LSM sidecar，但只把其中的 Conv3d 当作冻结的内容 patch encoder；formal Teacher 不把它的输出直接加到 DiT patch embedding。** 对应代码见 [`world_state/runtime.py:97-124`](../world_state/runtime.py#L97-L124) 与 [`world_state/encoder.py:27-45`](../world_state/encoder.py#L27-L45)。

---

## 2. 共同的 InSpatio / Wan 基础路径

理解两套机制前，需要先固定基础张量流：

- Wan 1.3B 的 hidden dimension 是 `1536`，有 `30` 个 Transformer blocks、`12` 个原生 self-attention heads，patch size 是 `(1,2,2)`：[`wan/configs/wan_t2v_1_3B.py:19-25`](../wan/configs/wan_t2v_1_3B.py#L19-L25)。
- 推理按每 block `3` 个 latent frames 进行，共 `4` 个 denoise steps：[`configs/inference_1.3b.yaml:1-11`](../configs/inference_1.3b.yaml#L1-L11)。
- 480×832 视频经 VAE 后的 latent 空间是 60×104。一个 3-frame block 经空间 patchify 后是 `3×30×52=4,680` 个 token，即每 latent frame `1,560` 个 token。
- noisy latent 是 16 通道。模型把它与 render condition 拼成 36 通道，再过基础 `patch_embedding`：[`wan/modules/causal_model.py:422-435`](../wan/modules/causal_model.py#L422-L435)。
- `WanDiffusionWrapper` 把外部布局 `[B,F,C,H,W]` 转成模型内部布局 `[B,C,F,H,W]`，最后将 flow prediction 换算成 clean `x0`：[`utils/wan_wrapper.py:367-449`](../utils/wan_wrapper.py#L367-L449)。
- `denoise_block` 先做一次 context/KV-fill，再做四步 denoise：[`pipeline/causal_inference.py:10-95`](../pipeline/causal_inference.py#L10-L95)。

后文中的记忆机制都不直接覆盖 noisy latent，而是在 DiT 内部影响其 clean `x0` 预测。

---

## 3. LSM-adapter 的实现原理与代码

### 3.1 核心结构

核心类只有一个卷积：

```python
self.proj = nn.Conv3d(
    in_channels=20,
    out_channels=model_dim,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
    bias=False,
)
```

对应 [`world_memory/latent_adapter.py:14-37`](../world_memory/latent_adapter.py#L14-L37)。对 1.3B 模型：

```text
参数量 = 20 × 1536 × 1 × 2 × 2 = 122,880
```

已有 sidecar `artifacts/exact_identity/adapter.safetensors` 也确实只有一个 `proj.weight`，shape 为 `[1536,20,1,2,2]`，文件 metadata 格式为 `inspatio_exact_pose_latent_memory_adapter_v1`。

#### 20 通道记忆条件

外部输入布局为：

```text
memory_condition: [B,F,20,H,W]
                 = [4-channel all-valid mask ; 16-channel clean memory latent]
memory_occupancy: [B,F, 1,H,W]
```

精确位姿实验把 mask 和 occupancy 都置为 1，见 [`scripts/world_memory/common.py:112-128`](../scripts/world_memory/common.py#L112-L128)。mask 是卷积的输入特征，occupancy 则是独立的硬门控，两者不是同一个张量。

#### 硬 occupancy gate

occupancy 用 `(1,2,2)` 的 `max_pool3d` patchify，再转为布尔 gate：

```text
G = 1[max_pool3d(occupancy) > 0]
R_mem = Conv3d(memory_condition) ⊙ G
```

这样只有原始 2×2 latent patch 内至少一个像素被占用时，整个 patch residual 才能通过。对应 [`world_memory/latent_adapter.py:40-72`](../world_memory/latent_adapter.py#L40-L72)。

#### 注入公式与位置

基础路径先做 36 通道 patch embedding，随后在 flatten token 前相加：

```text
H_patch = PatchEmbed([x_t ; render_condition_20ch])
H_patch' = H_patch + Conv3d(memory_condition) ⊙ PatchOccupancy
tokens = flatten(H_patch')
```

实际注入点在 [`wan/modules/causal_model.py:431-451`](../wan/modules/causal_model.py#L431-L451)，加法实现位于 [`world_memory/latent_adapter.py:75-97`](../world_memory/latent_adapter.py#L75-L97)。它没有额外 normalization、learned scale 或 attention；因此这是一个很直接的 additive residual baseline。

### 3.2 “关闭时严格保持原路径”

当 `memory_condition is None` 时，`add_gated_memory_residual` 直接返回原来的 `base_embeddings` 对象，不计算 adapter，也不创建零 residual：[`world_memory/latent_adapter.py:81-90`](../world_memory/latent_adapter.py#L81-L90)。测试还检查了对象 identity 和逐元素相等：[`tests/test_latent_memory_adapter.py:51-70`](../tests/test_latent_memory_adapter.py#L51-L70)。

这使 memory-off 是结构性 bypass，而不是“经过 adapter 后乘 0”的数值近似路径。

### 3.3 Pipeline 调用链

端到端调用顺序如下：

```text
memory_provider(block_index, latent_start, block_size)
  └─ 每个 query block 调一次，返回 condition + occupancy 或 None
      └─ denoise_block
          ├─ context/KV-fill：不传 memory，始终 memory-off
          └─ 4 个 denoise steps：复用同一组 memory 输入
              └─ WanDiffusionWrapper：BFCHW -> BCFHW
                  └─ CausalWanModel：基础 patch embedding 后加 residual
                      └─ 输出 clean x0
                          └─ block_output_callback 保存 detached clone
```

关键代码：

- 可选 hook 定义：[`pipeline/causal_inference.py:152-168`](../pipeline/causal_inference.py#L152-L168)。
- 每 block 读取 provider：[`pipeline/causal_inference.py:225-246`](../pipeline/causal_inference.py#L225-L246)。
- 同一 memory 传给四步 denoise：[`pipeline/causal_inference.py:263-275`](../pipeline/causal_inference.py#L263-L275)。
- context pass 没有 memory kwargs：[`pipeline/causal_inference.py:43-55`](../pipeline/causal_inference.py#L43-L55)。
- wrapper 布局转换：[`utils/wan_wrapper.py:389-419`](../utils/wan_wrapper.py#L389-L419)。
- clean `x0` callback：[`pipeline/causal_inference.py:278-286`](../pipeline/causal_inference.py#L278-L286)。

因此 memory 不会提前污染当前 block 的 STAR/KV prefix；但它会影响 query block 四步 denoise 的每一步。

### 3.4 Attach 与 sidecar

`attach_latent_memory_adapter` 必须在基础 checkpoint 加载后调用。它：

1. 从 `model.dim` 得到输出维度；
2. 校验 adapter 与模型维度一致；
3. 默认跟随基础 `patch_embedding.weight` 的 device/dtype；
4. 以 `model.memory_adapter` 注册为独立子模块。

见 [`world_memory/latent_adapter.py:100-123`](../world_memory/latent_adapter.py#L100-L123)。

保存和加载只处理 adapter 的独立 safetensors，不混入冻结的 1.3B checkpoint：[`world_memory/latent_adapter.py:126-165`](../world_memory/latent_adapter.py#L126-L165)。

### 3.5 数据捕获、训练与评估协议

配置入口是 [`configs/world_memory/exact_identity.yaml`](../configs/world_memory/exact_identity.yaml)：

- rotation-only 轨迹：`0° → +40° → 0° → +40°`；
- block size = 3；
- 首次 +40° 写 block = 6；
- 最终同位姿回访 block = 19；
- A/B 只改变生成 noise seed，query 状态、位姿、occupancy、prompt 与 metadata 保持一致。

#### 捕获

[`scripts/world_memory/capture_exact_pairs.py`](../scripts/world_memory/capture_exact_pairs.py) 完成三件事：

1. 构造 dense yaw 轨迹并断言 write/read 的逐帧位姿精确相等：[`capture_exact_pairs.py:58-121`](../scripts/world_memory/capture_exact_pairs.py#L58-L121)。
2. 在 memory-off 下分别生成 World A/B，通过 callback 捕获首次 +40° block 的最终 clean `x0`，保存为 `M_A/M_B`：[`capture_exact_pairs.py:223-275`](../scripts/world_memory/capture_exact_pairs.py#L223-L275)。
3. 固定 query context、query noise、A/B target、denoise steps 和随机种子，形成单 block 训练样本：[`capture_exact_pairs.py:293-343`](../scripts/world_memory/capture_exact_pairs.py#L293-L343)。

#### 训练

[`scripts/world_memory/train_exact_adapter.py`](../scripts/world_memory/train_exact_adapter.py) 冻结整个 InSpatio backbone，只训练 122,880 个 adapter 参数，并显式断言没有其他参数可训练：[`train_exact_adapter.py:57-89`](../scripts/world_memory/train_exact_adapter.py#L57-L89)。

训练分两阶段：

- Stage A：只过拟合 A，120 steps；
- Stage B：A/B 交替，180 steps；
- loss：最终 clean latent 对对应 target 的全局 L1；
- optimizer：AdamW，默认 LR `7.5e-4`；
- adapter 参数保持 FP32，前向置于 BF16 autocast；
- `full_denoise_grad` 未开启，所以前三个 denoise steps 在 `no_grad` 下执行，只有最后一步保留反向图。对应 [`pipeline/causal_inference.py:57-83`](../pipeline/causal_inference.py#L57-L83) 与 [`train_exact_adapter.py:128-150`](../scripts/world_memory/train_exact_adapter.py#L128-L150)。

核心训练循环见 [`train_exact_adapter.py:184-260`](../scripts/world_memory/train_exact_adapter.py#L184-L260)。

#### 评估

[`scripts/world_memory/eval_exact_revisit.py`](../scripts/world_memory/eval_exact_revisit.py) 固定所有非 memory 变量，比较：

- `no_memory`；
- 最终 block 只读 `memory_A`；
- 最终 block 只读 content-swap `memory_B`。

provider 只在 return block 19 返回 memory：[`eval_exact_revisit.py:164-197`](../scripts/world_memory/eval_exact_revisit.py#L164-L197)。评估同时验证 block 19 前三条分支逐元素一致：[`eval_exact_revisit.py:213-236`](../scripts/world_memory/eval_exact_revisit.py#L213-L236)。

当前本地已有产物给出的结果是：

| 指标 | 数值 |
|---|---:|
| memory-A → A latent L1 | 0.043979 |
| no-memory → A latent L1 | 0.551891 |
| memory-B → B latent L1 | 0.043809 |
| memory-B → A latent L1 | 0.549475 |
| return block 之前 prefix 是否逐元素一致 | true |

证据文件是 `artifacts/exact_identity/simple_metrics.json` 与 `artifacts/exact_identity/RESULT_ZH.md`；该目录被 `.gitignore` 忽略，不属于提交源码。

### 3.6 LSM-adapter 的能力边界

这套实现证明的是一个很窄的命题：冻结的 InSpatio-World 1.3B 可以通过小型 sidecar 在 **完全相同的相机位姿** 读取 clean-latent identity。当前实现没有：

- near-view / translation / 6DoF 投影；
- observation 检索、冲突解决、submap 或长期 bank；
- online write、merge、eviction；
- RGB hard composite；
- 自然 handoff 或实时性保证。

因此 LSM-adapter 本身不能称为完整 World State。

---

## 4. World State-v0 的整体实现原理

### 4.1 设计目标与硬约束

World State-v0 的正式名称是 **Read-only WorldState Teacher v0**。其目标不是把旧 latent 直接“贴回”当前结果，而是：

1. 离线构建一个之后不再改写的 observation bank；
2. 根据当前真实相机参数检索和投影历史 observation；
3. 保留 source/generated 候选的独立性与 authority；
4. 让当前 noisy hidden 和 denoise timestep 通过局部 attention 自己决定读哪个候选或 null；
5. 只把 attention 结果作为小 residual 加入指定 DiT blocks；
6. 不把 World token 写入原生 ST/KV cache。

formal Teacher 明确禁止同时挂载 direct residual adapter：[`world_state/runtime.py:97-112`](../world_state/runtime.py#L97-L112)。此外，pipeline、wrapper 和 causal model 三层都检查 direct memory 与 world context 互斥：

- [`pipeline/causal_inference.py:38-41`](../pipeline/causal_inference.py#L38-L41)
- [`utils/wan_wrapper.py:393-403`](../utils/wan_wrapper.py#L393-L403)
- [`wan/modules/causal_model.py:431-432`](../wan/modules/causal_model.py#L431-L432)

### 4.2 World State 数据协议

#### 轨迹与 block 对齐

配置入口是 [`configs/world_teacher/teacher_v0.yaml`](../configs/world_teacher/teacher_v0.yaml)。block index 为零基：

```text
0°
  → blocks 2–4：第一次 0→40 traversal
  → blocks 5–6：40° hold；在 block 6 写 M40_A / M40_B
  → blocks 7–9：40→0
  → blocks 10–12：0° hold
  → block 12 后保存公共 STAR/KV/last-pred/noise snapshot
  → blocks 13–15：逐帧完全相同的第二次 0→40 traversal
  → blocks 16–18：40° continuous-read hold
```

轨迹构造和两段 ramp 完全一致的断言见 [`training/world_teacher/build_paired_records.py:63-85`](../training/world_teacher/build_paired_records.py#L63-L85)。

每个 latent pose 使用实际 pixel frame `4*i`；相机内参缩放到 104×60 latent 网格；世界坐标固定为首个 target camera 的 `W0`：

```text
c2w_W0[t] = inverse(target_c2w[0]) @ target_c2w[t]
```

实现见 [`build_paired_records.py:108-131`](../training/world_teacher/build_paired_records.py#L108-L131)。代码还断言两次 latent traversal 的相机矩阵逐元素相等，并断言 rotation-only target 没有平移：[`build_paired_records.py:354-372`](../training/world_teacher/build_paired_records.py#L354-L372)。

#### A/B paired worlds

同一场景使用相同 prompt、source、render condition 与轨迹，只用不同 noise seed 生成 A/B 两个历史世界：[`build_paired_records.py:374-405`](../training/world_teacher/build_paired_records.py#L374-L405)。这样在训练和评估时，可以把 query state 固定为 A，只替换 bank 中的 `M40_A/M40_B`，检验读取是否真的具有内容特异性。

#### Bank 中究竟有什么

每个 A 或 B bank **只包含两个 observation**：

1. `source_static`：来自 source/reference latent 的保守静态区域；
2. `M40_A` 或 `M40_B`：第一次 traversal 在 block 6 生成的 clean latent。

沿途 blocks 2–6 的 A/B 输出只是监督 target，不会被写入 bank。bank 构造见 [`training/world_teacher/dataset.py:53-82`](../training/world_teacher/dataset.py#L53-L82)。

source static confidence 来自时间维 latent variation：偏离时间中位数越大，confidence 越低；超过阈值的区域才获得 source validity/authority：[`world_state/source_truth.py:8-24`](../world_state/source_truth.py#L8-L24) 与 [`build_paired_records.py:408-427`](../training/world_teacher/build_paired_records.py#L408-L427)。generated observation 也用三帧稳定性估计 confidence。

注意：代码虽然保存了 source depth，但当前 `RotationProjector` 没有读取 depth；v0 的正式投影仍是纯旋转 homography。

### 4.3 核心数据结构

张量契约集中在 [`world_state/types.py`](../world_state/types.py)：

| 类型 | 关键字段/shape | 含义 |
|---|---|---|
| `CameraBatch` | `K [B,F,3,3]`，`c2w_W0 [B,F,4,4]` | 当前 query 的真实逐 latent-frame 相机 |
| `WorldObservation` | `clean_latent [F,16,H,W]`、相机、depth、valid、两类 confidence | 单个 source/generated 历史观测 |
| `WorldReadPacket` | `candidate_20ch [B,Kobs,F,20,H,W]` 及 valid/authority/confidence/pose/offset/provenance | 投影后但尚未局部邻域展开的独立候选 |
| `EncodedWorldTokens` | `tokens [B,L,Klocal,512]` | 每个 query patch 的局部 memory candidates |
| `WorldLayerContext` | K/V `[B,L,Klocal,heads,head_dim]` | 某个 Reader layer 预计算后的 memory K/V |
| `WorldBlockContext` | layer→context、coverage、observation IDs | 一个 query block 在四步 denoise 中复用的完整上下文 |

`WorldObservation`、`FixedWorldBank` 等使用 `@dataclass(frozen=True)`，并且 bank 内是 tuple，正常流程没有 writer：[`world_state/types.py:21-98`](../world_state/types.py#L21-L98)、[`world_state/fixed_bank.py:19-31`](../world_state/fixed_bank.py#L19-L31)。但这是 **Python 字段/API 层的结构性只读**；PyTorch tensor 本身仍可被原地修改，代码没有做 deep immutable copy。

### 4.4 端到端读取链路

```text
query CameraBatch
   │
   ▼
FixedWorldBank.retrieve
   │  按 mean view angle、authority、observation_id 排序，取 top-2
   ▼
RotationProjector
   │  每 observation 独立投影；source conflict 覆盖 generated validity
   ▼
WorldReadPacket [B,2,F,20,H,W]
   │
   ▼
WorldTokenEncoder
   │  frozen LSM Conv3d 做内容 patchify
   │  + confidence/pose/subpixel/authority/provenance embedding
   │  + 每 observation 的 3×3 patch neighborhood
   │  + learned null candidate
   ▼
EncodedWorldTokens [B,L,19,512]
   │
   ▼
3 × LocalWorldReader.precompute
   │  为 layers 8/14/20 各算一次 world K/V
   ▼
四个 denoise steps 复用 K/V
   │  每步由 current hidden + timestep 重新计算 Q
   ▼
hidden += layer_scale × ReaderAttention(Q,K,V)
   │
   ▼
原生 self-attention（可条件启用 Q/O LoRA）
   │
   ▼
clean x0；下一 block 再由既有 STAR context 路径间接提交
```

#### 1）相机检索

`FixedWorldBank.retrieve` 计算 observation 与 query 的逐帧相对旋转角，再按以下 key 排序：

```text
(mean_view_angle 升序, authority 降序, observation_id 升序)
```

只支持 batch size 1，默认取 top-2：[`world_state/fixed_bank.py:11-53`](../world_state/fixed_bank.py#L11-L53)。这不是语义检索、空间索引或 submap retrieval；在当前实验里 bank 本来就只有两个 observation。

#### 2）pure-rotation 投影与 visibility

对每帧，代码先算：

```text
T_q<-m = inverse(c2w_query) @ c2w_memory
R_q<-m = T_q<-m[:3,:3]
H_m->q = K_query @ R_q<-m @ inverse(K_memory)
```

为了用 `grid_sample` 做 backward sampling，再用 `inverse(H_m->q)` 将 query pixel 映回 memory pixel。对应 [`world_state/projector.py:33-108`](../world_state/projector.py#L33-L108)。

有效性只依赖：

- 投影坐标 finite 且在 memory 图像范围内；
- memory observation 自己的 valid；
- source authority conflict。

没有 yaw threshold、coverage threshold、时间 ramp 或全局开关。exact pose 时如果 `K` 与完整 `c2w` 都在容差内相等，直接返回原 latent、valid 和 confidence，不做 resampling：[`world_state/projector.py:43-55`](../world_state/projector.py#L43-L55)。

每个 observation 独立投影，不先平均 raw latent。若可信 source 在某像素有效，则同像素 generated candidate 被置 invalid；其余 generated 区域保持独立：[`world_state/projector.py:160-188`](../world_state/projector.py#L160-L188)。最后每个候选形成 `[valid×4 ; projected_latent×16]` 的 20 通道输入：[`world_state/projector.py:179-200`](../world_state/projector.py#L179-L200)。

#### 3）候选 token 编码

`WorldTokenEncoder` 做两类信息融合。

内容路径：

```text
20ch projected candidate
  → frozen LatentMemoryAdapter Conv3d
  → LayerNorm(1536)
  → Linear(1536,512)
```

注意，Conv3d 对每个 observation 独立运行，并且被永久冻结：[`world_state/encoder.py:27-45`](../world_state/encoder.py#L27-L45)、[`world_state/encoder.py:86-97`](../world_state/encoder.py#L86-L97)。

metadata 路径把 confidence、6DoF relative pose + view angle、subpixel offset、authority、provenance 编码后相加；confidence/authority/provenance 还产生 attention bias：[`world_state/encoder.py:98-119`](../world_state/encoder.py#L98-L119)。其中 relative translation 虽作为 metadata 输入，但几何投影本身仍忽略 translation。

随后每个 observation 在 patch grid 上展开 3×3 邻域。默认 top-2 observation，因此每个 query patch 有：

```text
2 observations × 9 neighboring patches + 1 null = 19 candidates
```

实现见 [`world_state/encoder.py:120-155`](../world_state/encoder.py#L120-L155)。对于 3×30×52 的 query grid，输出 shape 是 `[1,4680,19,512]`。

#### 4）LocalWorldReader

当前配置在 zero-based DiT layers `8,14,20` 各挂一个 width-512、8-head Reader，head dim 为 64：[`configs/world_teacher/teacher_v0.yaml:38-44`](../configs/world_teacher/teacher_v0.yaml#L38-L44)。

每层先预计算：

```text
K_world = Linear_K(world_tokens)
V_world = Linear_V(world_tokens)
V_null = 0
```

见 [`world_state/local_reader.py:54-74`](../world_state/local_reader.py#L54-L74)。四个 denoise steps 中，每一步重新计算：

```text
Q = Linear_Q(LayerNorm(current_hidden)) + Linear_timestep(timestep_embedding)
logits = QK / sqrt(64) + metadata_attention_bias
logits[invalid] = -inf
update = softmax(logits) @ V
hidden' = hidden + learned_residual_scale × Linear_O(update)
```

对应 [`world_state/local_reader.py:76-95`](../world_state/local_reader.py#L76-L95)。Reader 被放在每个原生 block 的 native self-attention 之前：[`wan/modules/causal_model.py:153-195`](../wan/modules/causal_model.py#L153-L195)。

null candidate 的 value 被强制为 0，Reader output linear 没有 bias，所以某 patch 只有 null 有效时，输出 residual 严格为 0：[`world_state/local_reader.py:45-52`](../world_state/local_reader.py#L45-L52)、[`tests/test_world_state_reader.py:126-142`](../tests/test_world_state_reader.py#L126-L142)。当前 loader 还会丢弃早期 v0 checkpoint 中违反该语义的 legacy output bias：[`world_state/runtime.py:176-196`](../world_state/runtime.py#L176-L196)。

#### 5）条件 Q/O LoRA

每个选中层的原生 self-attention Q 和 O 各挂一个 rank-8 LoRA：[`world_state/runtime.py:44-62`](../world_state/runtime.py#L44-L62)。

LoRA 只有在以下条件同时成立时才生效：

- 当前 block 的 `world_reader_context` 非空；
- runtime 已启用 LoRA；
- context 标记 `enable_lora=True`。

原生 self-attention 的条件调用见 [`wan/modules/causal_model.py:54-112`](../wan/modules/causal_model.py#L54-L112)。memory-off/context-fill 调用不会走 LoRA。

#### 6）每 block 一次 K/V，四步重算 Q

`WorldStateRuntime.precompute(packet)` 对整个 block 只编码一次候选，并为三个 Reader layers 各生成一份 K/V：[`world_state/runtime.py:64-89`](../world_state/runtime.py#L64-L89)。同一个 `WorldBlockContext` 随后传给四步 denoise；causal model 在每层按 block index 取对应 context：[`wan/modules/causal_model.py:477-490`](../wan/modules/causal_model.py#L477-L490)。

所以：

- world K/V 在一个 block 的四步 denoise 间复用；
- Q 依赖当前 noisy hidden 与 timestep，每步都会变化；
- context/KV-fill 不带 world context；
- World token 不写入原生 KV cache；
- 只有最后生成的 clean `x0` 会作为下一 block 的 `last_pred` 进入既有 context/STAR 路径。

### 4.5 当前参数量与 sidecar

按当前无 output bias 的代码和本地最终 checkpoint：

| 组件 | 参数量 | 是否训练/保存到 Teacher sidecar |
|---|---:|---|
| 冻结 LSM content Conv3d | 122,880 | 冻结；不保存到 Teacher sidecar，运行时从 direct adapter sidecar 单独加载 |
| WorldTokenEncoder（不含上述 Conv3d） | 1,329,670 | 是 |
| 3 × LocalWorldReader | 8,664,579 | 是 |
| 3 层 Q/O rank-8 LoRA | 147,456 | exact-reader 阶段否；之后是 |
| Reader-only 合计 | 9,994,249 | 是 |
| 完整 Teacher 合计 | **10,141,705** | 是 |

本地 `teacher_final.safetensors` 有 66 个 tensors，约 38.7 MiB，metadata 标明：

```text
format=inspatio_worldstate_reader_v0
selected_layers=[8,14,20]
world_width=512
heads=8
lora_rank=8
stage=two-block
step=50
null_semantics=output_bias_removed
```

仓库中的旧 `training_*.json` 参数量比当前代码多 4,608，正好对应三个 Reader 的旧 `1536` 维 output biases；当前实现已去掉这些 bias，最终 checkpoint 也已迁移。

Teacher sidecar 的保存范围只包含 encoder（排除 frozen content adapter）、Reader、Q/O LoRA：[`world_state/runtime.py:134-173`](../world_state/runtime.py#L134-L173)。

### 4.6 训练目标与四个阶段

#### ownership mask

投影后将 query latent pixels 分为：

- `source_owned`：有效 source candidate 覆盖；
- `generated_memory_owned`：有效 generated candidate 且 confidence ≥ `0.35`，再排除 source conflict；
- `unknown`：其余区域。

实现见 [`training/world_teacher/dataset.py:143-160`](../training/world_teacher/dataset.py#L143-L160)。低稳定性的 generated 人物/液体等区域不会被当作长期 identity target。

#### loss

known 区域以第一次 traversal 的 paired block 为 target：

```text
L_known = 1.0 × masked_latent_L1
        + 0.05 × masked_channel_cosine_loss
```

unknown 区域约束为贴近同 query block 的 no-memory A 轨迹：

```text
L_unknown = 0.01 × masked_L1(prediction, no_memory_target)
L_total = L_known + L_unknown
```

见 [`training/world_teacher/train.py:59-68`](../training/world_teacher/train.py#L59-L68) 与 [`training/world_teacher/train.py:235-249`](../training/world_teacher/train.py#L235-L249)。

#### 四阶段课程

| 阶段 | 默认 steps / LR | query→target | LoRA | 目的 |
|---|---:|---|---|---|
| `exact-reader` | 120 / 2e-4 | block 16 → block 6 | 关 | 先让 Encoder+Reader 学会 exact M40 读取 |
| `exact-lora` | 160 / 2e-4 | block 16 → block 6 | 开 | 加入条件 Q/O LoRA，提高 exact identity 适配 |
| `continuous` | 300 / 1e-4 | blocks 13–18 → blocks 2,3,4,5,6,6 | 开 | 训练不同 overlap 下的连续回访 |
| `two-block` | 50 / 5e-5 | 配置中的相邻双 block windows | 开 | 把前一预测作为下一 block STAR context，训练跨 block 连续性 |

阶段配置见 [`configs/world_teacher/teacher_v0.yaml:46-73`](../configs/world_teacher/teacher_v0.yaml#L46-L73)，样本选择见 [`training/world_teacher/train.py:251-264`](../training/world_teacher/train.py#L251-L264)。

所有阶段都冻结 1.3B backbone 与 LSM content Conv3d；Teacher 参数保持 FP32，在 BF16 autocast 中训练：[`training/world_teacher/train.py:91-121`](../training/world_teacher/train.py#L91-L121)。不同于 LSM baseline，formal Teacher 设置 `full_denoise_grad=True`，四个 denoise steps 全部保留反向图：[`training/world_teacher/train.py:204-233`](../training/world_teacher/train.py#L204-L233)。

`two-block` 阶段还把第一块 prediction 作为第二块的 `last_pred` context，并允许梯度穿过第二块的 context/KV-fill；两个 block loss 取平均：[`training/world_teacher/train.py:310-333`](../training/world_teacher/train.py#L310-L333)。

### 4.7 推理/评估路径

#### 通用 pipeline hook

`CausalInferencePipeline.inference` 已提供：

```python
world_context_provider(block_index, latent_start, query_pose)
```

pipeline 会从总 `CameraBatch` 切出当前 block 相机，再获取 `WorldBlockContext`：[`pipeline/causal_inference.py:237-246`](../pipeline/causal_inference.py#L237-L246)。

但当前仓库中没有默认 demo/CLI 调用这个 provider，也没有默认路径自动 attach WorldStateReader。实际 Teacher 训练和评估脚本都是直接构造 packet/context 后调用 `denoise_block`。因此它目前是 **实验 side path**，不是 `run_test_pipeline.sh` 默认启用的产品路径。

#### 公共 snapshot 与六分支评估

数据构建在第二次 traversal 前保存共同 prefix、`last_pred`、后续 noise 和完整 KV cache：[`training/world_teacher/build_paired_records.py:458-507`](../training/world_teacher/build_paired_records.py#L458-L507)。评估的六个分支从完全相同的 snapshot 出发：

1. no memory；
2. frozen direct adapter exact one-shot；
3. Teacher A exact one-shot；
4. Teacher B exact one-shot；
5. Teacher A continuous blocks 13–18；
6. Teacher B continuous blocks 13–18。

分支逻辑见 [`training/world_teacher/evaluate.py:139-218`](../training/world_teacher/evaluate.py#L139-L218)，指标计算见 [`training/world_teacher/evaluate.py:256-329`](../training/world_teacher/evaluate.py#L256-L329)。

值得注意的是，评估脚本为了同进程比较，会先 attach formal runtime，再 attach direct adapter；但是每次 `denoise_block` 仍只传 direct memory 或 world context 之一，因此没有绕过逐调用互斥约束：[`training/world_teacher/evaluate.py:91-119`](../training/world_teacher/evaluate.py#L91-L119)。

### 4.8 当前实验结论

本地已有两个场景 S0/S1 的最终原生 VAE decode 与 metrics，结论为 `PARTIAL`：

- 公共 snapshot 前缀在六分支间逐元素一致；
- direct residual 关闭时，Teacher 能在 exact/HOLD 恢复 A/B 不同 identity；
- continuous read 从 first-visible 到 exact/HOLD 都能按投影可见性工作，没有 yaw/coverage 人工开关；
- first-visible 未发生 direct one-shot 式的全局亮度塌陷；
- HOLD 三个 blocks 没有逐块恶化；
- 但 near/exact 结果仍比第一次 traversal reference 模糊，尤其 S1，因此不能判定为 `WORKS`。

代表性数值：

| 场景 | no-memory→A exact L1 | Teacher-A→A one-shot exact L1 | Teacher-B→B exact L1 | Teacher-B→A exact L1 |
|---|---:|---:|---:|---:|
| S0 | 0.6075 | 0.2487 | 0.2682 | 0.5635 |
| S1 | 0.4438 | 0.2775 | 0.2688 | 0.3701 |

S0 的 inclusive timing 约为 no-memory `1974 ms/block`、continuous Teacher-A `2020 ms/block`，Teacher peak allocated VRAM 约 `6.0–6.5 GiB`。这些是当前本地 ignored artifacts 中的测量值，不是根据代码推算的理论值。

证据位于：

- `artifacts/world_teacher_v0/RESULT_ZH.md`
- `artifacts/world_teacher_v0/S0/evaluation/simple_metrics.json`
- `artifacts/world_teacher_v0/S1/evaluation/simple_metrics.json`

---

## 5. 关键代码地图

### LSM-adapter

| 文件 | 职责 |
|---|---|
| [`world_memory/latent_adapter.py`](../world_memory/latent_adapter.py) | 20→1536 Conv3d、occupancy gate、attach、sidecar IO |
| [`world_memory/__init__.py`](../world_memory/__init__.py) | 对外 API |
| [`pipeline/causal_inference.py`](../pipeline/causal_inference.py) | memory provider / block callback / 四步 denoise hook |
| [`utils/wan_wrapper.py`](../utils/wan_wrapper.py) | 外部/内部张量布局转换与参数透传 |
| [`wan/modules/causal_model.py`](../wan/modules/causal_model.py) | patch embedding 后的实际 residual 注入点 |
| [`configs/world_memory/exact_identity.yaml`](../configs/world_memory/exact_identity.yaml) | exact-pose 实验配置 |
| [`scripts/world_memory/capture_exact_pairs.py`](../scripts/world_memory/capture_exact_pairs.py) | 捕获 A/B clean latent 与固定 query state |
| [`scripts/world_memory/train_exact_adapter.py`](../scripts/world_memory/train_exact_adapter.py) | 只训练 adapter |
| [`scripts/world_memory/eval_exact_revisit.py`](../scripts/world_memory/eval_exact_revisit.py) | no-memory / A / B content-swap 评估 |
| [`tests/test_latent_memory_adapter.py`](../tests/test_latent_memory_adapter.py) | 参数量、硬门控、sidecar round trip、off-path 测试 |

### World State-v0 核心

| 文件 | 职责 |
|---|---|
| [`world_state/types.py`](../world_state/types.py) | 相机、observation、packet、token/context 张量契约 |
| [`world_state/fixed_bank.py`](../world_state/fixed_bank.py) | 不可变 bank 与 camera-angle top-k retrieval |
| [`world_state/source_truth.py`](../world_state/source_truth.py) | 保守 source static confidence/validity |
| [`world_state/projector.py`](../world_state/projector.py) | exact fast path、旋转 homography、可见性与 source conflict |
| [`world_state/encoder.py`](../world_state/encoder.py) | frozen LSM 内容编码、metadata embedding、3×3 邻域、null |
| [`world_state/local_reader.py`](../world_state/local_reader.py) | 局部 cross-attention Reader 与 ConditionalLoRA |
| [`world_state/runtime.py`](../world_state/runtime.py) | attach、per-layer K/V precompute、trainable 参数筛选、sidecar IO |
| [`wan/modules/causal_model.py`](../wan/modules/causal_model.py) | Reader-before-self-attention、条件 Q/O LoRA、逐层 context |
| [`pipeline/causal_inference.py`](../pipeline/causal_inference.py) | 通用 world context provider hook |

### World State-v0 数据、训练、评估

| 文件 | 职责 |
|---|---|
| [`configs/world_teacher/teacher_v0.yaml`](../configs/world_teacher/teacher_v0.yaml) | 轨迹、层、宽度、LoRA、loss、训练阶段、评估 blocks |
| [`training/world_teacher/build_paired_records.py`](../training/world_teacher/build_paired_records.py) | 轨迹/相机、A/B 世界、source truth、paired record、公共 KV snapshot |
| [`training/world_teacher/dataset.py`](../training/world_teacher/dataset.py) | bank、block example、ownership masks |
| [`training/world_teacher/train.py`](../training/world_teacher/train.py) | 四阶段训练与 loss |
| [`training/world_teacher/evaluate.py`](../training/world_teacher/evaluate.py) | 六分支 continuous revisit 评估、metrics、timing |
| [`training/world_teacher/visualize.py`](../training/world_teacher/visualize.py) | native decode 对比视频和 montage |
| [`tests/test_world_state_reader.py`](../tests/test_world_state_reader.py) | 不可变字段、投影、authority、unknown、19 candidates、null no-op、sidecar、四步梯度测试 |

---

## 6. 当前实现中必须牢记的边界

1. **它是 read-only Teacher，不是完整 online World State。** bank 在数据准备阶段离线构建；没有 writer、merge、update、eviction 或持久化策略。
2. **仅支持 rotation-only。** `WorldObservation` 虽保存 depth，relative pose 也包含 translation，但 projector 只用 rotation homography；translation/6DoF 和 occlusion reasoning 未实现。
3. **检索很轻量。** 只是 mean view angle 排序，batch size 固定为 1；没有大规模索引、语义检索或 submap。
4. **“immutable”不是 tensor 深冻结。** frozen dataclass 防止字段重新赋值，流程中也没有写操作；但 tensor 仍可被原地修改。
5. **source geometry confidence 当前基本是 1。** source depth 被记录但不参与 projector，真实几何置信与遮挡还未接入。
6. **World State-v0 依赖已训练的 LSM sidecar。** Teacher checkpoint 不包含 frozen content Conv3d，部署时两份 sidecar 缺一不可。
7. **默认 demo 未启用。** 当前 `run_test_pipeline.sh` 不 attach runtime，也不提供 `world_context_provider`；正式路径只在专用训练/评估脚本中使用。
8. **实验规模很小。** 结论来自 S0/S1 两个固定场景、固定 40° rotation-only protocol，不代表大规模泛化。
9. **没有 RGB compositing。** 所有正式结果都是 DiT latent 输出经 native VAE decode，没有把历史 RGB/novel-view render 硬合成到结果中。
10. **当前质量结论仍是 PARTIAL。** continuous read 和 A/B identity 方向成立，但投影 token 的锐度和自然、无冲突融合还没有解决。

---

## 7. 本次只读核验

本次没有重跑 GPU 训练或生成流程，只做了源码、checkpoint metadata 和已有 metrics 的读取核验。相关 CPU 单元测试结果：

```text
tests/test_latent_memory_adapter.py: 4 passed
tests/test_world_state_reader.py:   9 passed, 1 skipped
```

跳过项是需要 CUDA 的“四个 denoise steps 全部保留梯度图”测试；其代码断言位于 [`tests/test_world_state_reader.py:212-243`](../tests/test_world_state_reader.py#L212-L243)。测试以 `PYTHONDONTWRITEBYTECODE=1` 运行，未改动仓库源码或产物。
