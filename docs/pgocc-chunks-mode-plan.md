# PG-Occ chunks mode data loading adaptation plan

## 1. 背景和目标

本项目当前沿用 GaussTR 原生的文件级数据加载方式：每个 sample 在训练热路径中分别读取 nuScenes 图像、Metric3D 深度、FeatUp 特征、Occupancy GT，并通过 sweep 管线组织 8 帧时序输入。云服务器训练时，数据位于 TOS 对象存储桶，经 FUSE 或远程挂载访问，文件级多源读取会放大小文件和跨目录 IO 成本。

GaussTR 已经实现并验证了 fused training chunks 模式。PG-Occ 本次只复用已有预处理结果，不新增 chunk 预处理流程。当前 PG-Occ 工作区内：

```text
data/gausstr_chunks -> /home/C_UserData/dongzhipeng/Datasets/nuScenes_GaussTR_chunks
```

已可见的 profile/split：

```text
data/gausstr_chunks/train/featup_metric3d_sam2/
data/gausstr_chunks/val/featup_metric3d_sam2/
```

当前设备上可见 chunks 不是完整数据集。`profile.json` 记录的 producer command 均包含 `--ratio 0.01`，`index.json` 当前统计为：

```text
train: 321 samples
val:    80 samples
```

用户补充说明：云服务器上已有全量 chunks。本机 `0.01` chunks 仅用于方案开发、schema 微验证和 mini smoke test；正式训练、训练时评估、独立单卡评估、独立多卡评估的目标环境是云服务器全量 chunks。云端没有 `test` split，本次训练时评估和独立评估都固定使用 `val` split。

当前 profile 字段表：

| split | raw image | depth | feats/text_vision | sem_seg | occ_gt |
| --- | --- | --- | --- | --- | --- |
| train | true | metric3d | featup | grounded_sam2 | false |
| val | true | metric3d | featup | grounded_sam2 | true |
 
`train occ_gt=false` 不影响当前 PG-Occ 2D training loss 主路径，但意味着不能用 train chunks 直接做需要 occupancy GT 的训练内诊断。评估固定使用带 `occ_gt=true` 的 `val` chunks。

因此本机这些 chunks 只能用于方案开发和微验证，不能直接作为正式训练/正式评估的数据规模依据。正式训练前应在云服务器确认全量 split/profile 的路径、字段 schema 与本机 mini chunks 一致。

本阶段目标是先完成设计文档，待审核后再实现：

- 训练适配 chunks 加载。
- 训练时评估适配 chunks 加载。
- 独立单卡评估适配 chunks 加载。
- 独立多卡评估适配 chunks 加载。
- 保证三种评估模式的样本覆盖、顺序截断、指标聚合结果一致。
- chunks 模式关闭 PG-Occ 默认 8f 时序输入，只使用当前关键帧的 6 个相机视角。

本次目标就是实现 1f chunks 训练/评估。关闭 8f 时序后，chunks 1f 指标与原生 raw 8f 指标不同属于预期内；一致性要求只约束 chunks 1f 的训练时 val、独立单卡 val、独立多卡 val 三者内部一致。

## 2. 已确认的现状

### 2.1 PG-Occ 数据和入口

PG-Occ 使用 MMCV/MMDetection3D 1.x 风格：

- 训练入口：[train.py](/home/dzp62442/Projects/PG-Occ/train.py)
- 评估入口：[val.py](/home/dzp62442/Projects/PG-Occ/val.py)
- 数据集：`NuSceneOVOcc` / `NuSceneOcc`
- dataloader builder：[loaders/builder.py](/home/dzp62442/Projects/PG-Occ/loaders/builder.py)
- 主配置：[configs/pgocc.py](/home/dzp62442/Projects/PG-Occ/configs/pgocc.py)

原生 PG-Occ 配置要点：

```python
_num_frames_ = 8
ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "H": 900,
    "W": 1600,
    "bot_pct_lim": (0.0, 0.0),
}
render_conf = {
    "render_h": 180,
    "render_w": 320,
}
```

虽然 `resize_lim` 是范围，当前 `RandomTransformImage(..., training=False)` 在 train/test pipeline 中都走确定性 resize/crop：

```text
original: 1600 x 900
resize:   max(256 / 900, 704 / 1600) = 0.44
resized:  704 x 396
crop:     x = 0, y = 140, w = 704, h = 256
final:    704 x 256
```

PG-Occ 原生 8f 管线依赖：

- `LoadMultiViewImageFromMultiSweeps(sweeps_num=7)`
- `GenerateRenderImageFromMultiSweeps(sweeps_num=7)`
- 模型 transformer/head 中的 `num_frames=8`
- `t0_2_x_geo` 和 `render_gt` 用于 temporal depth warping loss
- `simple_test_online()` 在单卡评估时按帧切分并缓存图像特征

### 2.2 GaussTR chunks 实现

GaussTR 使用 MMEngine/MMDetection3D 2.x 风格，不能直接拷贝到 PG-Occ，但可以复用数据格式和设计。

关键文件：

- `gausstr/datasets/chunked_nuscenes_occ.py`
- `gausstr/datasets/transforms.py`
- `tools/build_training_chunks.py`
- `configs/gausstr_featup_chunks.py`
- `docs/chunk-eval-consistency-plan.md`

GaussTR chunks dataset 的核心策略：

- `IterableDataset` 内部负责 distributed rank 和 worker 分片。
- dataloader 配置 `sampler=None`，避免 sampler 和 dataset 双重分片。
- train split 可 shuffle chunk 和 chunk 内 sample。
- val/test split 不 shuffle。
- 每个 chunk 一次 `torch.load` 后连续产出 sample。
- metric 使用每 sample confusion matrix + `sample_idx` 覆盖检查，避免多卡评估只统计 rank0 或重复样本。

### 2.3 GaussTR-FeatUp 分辨率

GaussTR FeatUp profile：

```text
profile: featup_metric3d_sam2
input_size: 432 x 768
resize_scale: 0.48
crop: x = 0, y = 0
```

对应原始 1600x900 图像：

```text
1600 x 900 --resize 0.48--> 768 x 432
```

GaussTR Talk2DINO profile：

```text
profile: talk2dino_metric3d
input_size: 504 x 896
resize_scale: 0.56
crop: x = 0, y = 0
```

当前设备可见 chunks 只有 `featup_metric3d_sam2`，且是 `--ratio 0.01` 的 mini 产物，因此本项目第一阶段在本机只适配 FeatUp chunks 的数据闭环和一致性微验证，不把本机 mini chunks 视为正式训练数据。正式训练以云服务器全量 chunks 为准。

## 3. 关键适配判断

### 3.1 不新增预处理

实现时不得调用或改造 `tools/build_training_chunks.py` 来重新生成 PG-Occ 专属 chunks。PG-Occ 只读取：

```text
data/gausstr_chunks/{train,val}/featup_metric3d_sam2/
```

当前设备可见 train/val chunks 均为 `--ratio 0.01` 产物；用户说明云服务器已有全量 chunks，且没有 `test` split。后续正式训练和评估只需在云服务器确认完整 `train/featup_metric3d_sam2` 与 `val/featup_metric3d_sam2` 的目录和 manifest/index/profile。本次不适配 test/OV split，也不在 PG-Occ 中新增预处理。

### 3.2 分辨率等效适配

PG-Occ 原生图像输入等效变换是：

```text
original 1600x900 -> resize 0.44 -> 704x396 -> bottom crop y=140 -> 704x256
```

GaussTR FeatUp chunk 图像已经 materialize 为：

```text
original 1600x900 -> resize 0.48 -> 768x432
```

如果 chunk 图像已经是 GaussTR FeatUp profile 的 `432x768` materialized image，要从 chunk 图像得到与 PG-Occ 原生加载一致的图像，应再做一个二阶段确定性变换：

```text
chunk 768x432 -> resize 704/768 = 0.9166667 -> 704x396 -> bottom crop y=140 -> 704x256
```

组合后：

```text
0.48 * 0.9166667 = 0.44
```

因此 materialized chunk 图像路径可以和 PG-Occ 原生路径保持几何等效。实现时不能直接把现有 `RandomTransformImage` 的 `H=900,W=1600` 用在 432x768 chunk 图像上，否则实际 resize/crop 会错误。

但当前可见 manifest 已经显示 `depth` 和 `sem_seg` 的 shape summary 为 `900x1600`，不是 `432x768`。第一轮专家抽查还指出当前样本图像可能仍是原始分辨率且没有 `materialized/img_aug_mat`。因此实现主路径不能只假设 materialized input，而应按字段 metadata 和实际 shape 分派：

```text
image/depth/sem_seg shape == 432x768 and materialized=True
  -> 使用 GaussTR profile 到 PG-Occ 的二阶段变换

image/depth/sem_seg shape == 900x1600 or materialized missing/False
  -> 使用 PG-Occ 原生 0.44 resize + bottom crop 变换
```

实现前必须在 `pgocc` conda 环境中做微验证，读取一个 chunk 样本，确认每个字段的真实 shape、`materialized` 标记和是否存在 `img_aug_mat`。这一步是阻断项，不通过则不进入训练实现。

FeatUp `text_vision` 对应 GaussTR chunk 的 `feats` 字段，manifest 显示为 `512x27x48`。该字段不是 image-space dense map，不参与 image resize/crop，加载后应按 PG-Occ 当前 `text_vision` 使用方式保持 shape 兼容。

### 3.3 chunks 模式关闭时序

用户要求 chunks 模式不使用时序信息。建议为 chunks 配置单独设置：

```python
_num_frames_ = 1
model.pts_bbox_head.transformer.num_frames = 1
```

同时 chunks train pipeline 不再生成：

- sweep 图像
- sweep `lidar2img`
- sweep `ego2lidar`
- `t0_2_x_geo`
- 多帧 `render_gt`
- `auxi_img`

由此带来的模型/损失调整：

- sampling/mixing 中 `num_points * num_frames` 自动退化为单帧。
- `PGOcc.simple_test_online()` 在单卡评估中 `num_frames=len(filename)//6=1`，缓存逻辑仍可运行，但无需多帧重组。
- `PGOcc.simple_test_online()` 使用 `filename` 作为 cache key。chunks 模式如果复用原始 filename 但图像内容经过 chunk decode/二次变换，应在微验证中确认不会跨 raw/chunk 配置或跨 sample 污染缓存；必要时在 chunks 模式禁用该 cache 或把 key 扩展为 `chunk:{profile}:{sample_idx}:{camera}`。
- `loss_depth_warping` 依赖过去/未来帧，chunks 模式不能只把权重置 0。当前 [models/pgocc_head.py](/home/dzp62442/Projects/PG-Occ/models/pgocc_head.py:258) 会无条件读取 `kwargs['t0_2_x_geo']` 和 `kwargs['render_gt']` 并调用 `calc_time_warping_loss`，因此必须加代码级条件分支：当 `depth_warping <= 0` 或缺少时序字段时跳过该分支。
- `render_gt` 仍可能用于可视化或其他分支时，应只保留当前 6 张相机图的 `render_h x render_w` 版本。
- `depth_foundation` loss 可保留，因为它只依赖当前帧深度。
- OV feature loss 可保留，因为它只依赖当前帧 `text_vision`。

另一个必须修复的代码点是 [loaders/pipelines/loading.py](/home/dzp62442/Projects/PG-Occ/loaders/pipelines/loading.py:432)：`GenerateRenderImageFromMultiSweeps.__call__()` 在 `sweeps_num == 0` 时直接 `return`，会让 pipeline 返回 `None`。chunks 模式不能复用该行为，应新增单帧 render transform 或修正为返回 `results`。

建议新增 chunks 专用配置，不改默认 `configs/pgocc.py`：

```text
configs/pgocc_chunks.py
```

## 4. 设计方案

### 4.1 新增 PG-Occ chunk dataset

新增文件建议：

```text
loaders/chunked_nuscenes_ov_occ_dataset.py
```

职责：

- 读取 `chunk_manifest.json`、`index.json`、`profile.json`。
- 以 `IterableDataset` 方式按 chunk 流式产出 sample。
- 复用 GaussTR 的重试读取、可选 bytes cache、prefetch 设计。
- 内部完成 rank/worker 分片。
- 输出 PG-Occ pipeline 需要的 `results` 字典。

输出字段应贴近 `NuSceneOVOcc.get_data_info()`：

```python
{
    "sample_idx": token_or_source_sample_idx,
    "token": token,
    "scene_name": scene,
    "img_filename": [...6...],
    "feature_names": [...6...],
    "img_timestamp": [...6...],
    "lidar2img": [...6...],
    "ego2lidar": [...6...],
    "ego2img": [...6...],
    "cam2ego": [...6...],
    "cam2global": [...6...],
    "ori_k": [...6...],
    "occ_path": ".../labels.npz",
    "_chunk_sample": sample,
}
```

如果 chunk 的 `meta/images` 字段来自 OpenMMLab v2 格式，而 PG-Occ sweep pkl 来自当前项目自定义格式，需要在 dataset 内做字段映射，不应把映射逻辑散落在 transforms 里。

相机顺序必须以 camera name 显式重排，不能依赖 dict 插入顺序。PG-Occ 原生顺序为：

```text
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

GaussTR profile 中记录的 `camera_order` 为：

```text
CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT
```

schema adapter 必须按 PG-Occ 期望顺序同时重排 image、depth、text_vision、sem_seg、`lidar2img`、`ego2img`、`cam2ego`、`ori_k`，并断言每个 camera name 的字段齐全。

### 4.2 新增 chunk pipeline transforms

新增或扩展文件：

```text
loaders/pipelines/loading.py
loaders/pipelines/transforms.py
```

建议新增 transforms：

- `LoadMultiViewImageFromChunks`
- `LoadChunkFeature`
- `LoadChunkOccGT`
- `RandomTransformChunkImageForPGOcc`
- `GenerateCurrentRenderImage`

职责划分：

- `LoadMultiViewImageFromChunks`：从 `sample["image_bytes"]` 解码 6 视角图像，不访问原始图片路径。
- `LoadChunkFeature(key="depth")`：从 `sample["depth"]` 读取深度，并按 materialized 状态选择二阶段变换或 PG-Occ 原生变换。
- `LoadChunkFeature(key="feats", out_key="text_vision")`：从 `sample["feats"]` 读取 FeatUp 特征，不做 image-space resize。
- `LoadChunkOccGT`：从 `sample["occ_gt"]` 读取 `semantics/mask_camera`；如果 train chunks 没有 `occ_gt`，训练管线不使用该 transform。
- `RandomTransformChunkImageForPGOcc`：对已经 432x768 的 chunk 图像执行等效二阶段几何变换，并更新 `lidar2img`。
- `GenerateCurrentRenderImage`：只生成当前 6 视角 `render_gt`，并按 PG-Occ `render_conf` 处理深度到 `180x320`。

### 4.3 dataloader builder 适配

PG-Occ 当前 `loaders/builder.py` 总是创建 sampler。chunks dataset 是 `IterableDataset`，需要避免 sampler。

建议扩展 builder：

- 如果 dataset 暴露 `is_chunk_iterable = True`，则强制 `sampler=None`。
- 分布式和 worker 分片由 dataset 内部处理。
- train split 可 `chunk_shuffle=True, sample_shuffle=True`。
- val/test split 必须 `chunk_shuffle=False, sample_shuffle=False`。
- `shuffle` 参数只对 map-style dataset 生效。

### 4.4 配置拆分

新增：

```text
configs/pgocc_chunks.py
```

建议继承或复制 `pgocc.py` 的主体配置，并显式覆盖：

```python
dataset_type = "NuSceneOVOccChunk"
_num_frames_ = 1
loss_weights["depth_warping"] = 0.0
model.pts_bbox_head.transformer.num_frames = 1
```

train dataset：

```python
data.train = dict(
    type="NuSceneOVOccChunk",
    chunk_root="data/gausstr_chunks",
    split="train",
    profile="featup_metric3d_sam2",
    chunk_shuffle=True,
    sample_shuffle=True,
    skip_padding=False,
    pipeline=train_chunk_pipeline,
    test_mode=False,
)
```

val/test dataset：

```python
data.val = dict(
    type="NuSceneOVOccChunk",
    chunk_root="data/gausstr_chunks",
    split="val",
    profile="featup_metric3d_sam2",
    chunk_shuffle=False,
    sample_shuffle=False,
    skip_padding=True,
    pipeline=test_chunk_pipeline,
    test_mode=True,
)
```

用户已确认独立评估就是评 `val`。实现时建议在 `configs/pgocc_chunks.py` 中让 `data.test` 显式复用 `data.val`，或让 `val.py` 只构建 `data.val`，避免误用不存在的 test/OV split。

## 5. 评估一致性方案

PG-Occ 的 `evaluate()` 当前按 `self.data_infos` 顺序重新加载 GT，并使用 `occ_results[data_id]` 对齐预测。这对 map-style dataset 可行，但 chunks iterable dataset 的多卡结果收集容易出现以下风险：

- rank 分片后结果顺序不等同于 `data_infos` 全局顺序。
- sampler padding 或 dataset padding 导致重复 sample。
- `multi_gpu_test(..., gpu_collect=True)` 收集后可能按 rank 拼接，不保证满足当前 `occ_results[data_id]` 假设。
- 训练入口使用项目自定义 [loaders/builder.py](/home/dzp62442/Projects/PG-Occ/loaders/builder.py:14)，独立评估入口 [val.py](/home/dzp62442/Projects/PG-Occ/val.py) 使用 `mmdet3d.datasets.build_dataloader`，两条 dataloader/collect 路径必须分别验证。

建议实现评估一致性的最小闭环：

1. 模型输出中保留唯一 `sample_idx`。chunks 模式统一使用 chunk sample 的 `source_sample_idx`/nuScenes token 作为全局样本 id，不使用 rank-local offset；`global_offset` 只用于排序和分片诊断。
2. dataset evaluate 先将 `occ_results` 转为 `{sample_idx: prediction}`。
3. 用 expected sample list 校验：
   - 无重复。
   - 无缺失。
   - 无 unexpected sample。
4. 再按 expected sample list 顺序读取 GT 和预测，计算 mIoU/RayIoU/depth。
5. 训练时评估、单卡评估、多卡评估使用同一套 expected sample list 和同一套 evaluate 逻辑。

为了尽量少改模型，可优先在 `val.py`/test hook 结果收集后，为每个 result 注入或保留 `sample_idx`。如果当前模型 result 没有 meta，需要在 `PGOcc.forward_test()` 或 `merge_occ_pred()` 的返回 dict 中附加：

```python
"sample_idx": img_metas[b]["sample_idx"]
```

对 chunks dataset：

- `__len__` 在 val/test 中应返回全局真实样本数，便于 collect/truncate。
- `__iter__` 只产出本 rank/worker 的样本。
- expected sample list 来自 `index.json`，按 `global_offset` 排序。
- dataset 仍需维护完整 `data_infos` 或等价 `eval_infos`，供 evaluate 按 expected sample list 读取 GT、ego pose 和 scene 信息；不能只保留 rank-local chunk 样本。

`val.py` 当前使用 `mmdet3d.datasets.build_dataloader`，不是项目自定义 `loaders.builder`。实现时必须确认该 builder 对 `IterableDataset` 不会再叠加 sampler/padding；若无法保证，应让独立评估也走统一的 chunks-aware dataloader builder。

对 raw dataset：

- 可保留现有 sampler 分片。
- evaluate 也可以逐步迁移到 sample_idx 对齐，以便 raw 和 chunks 使用同一评估安全机制。

目标一致性：

```text
chunk training-time val == chunk standalone single-GPU val == chunk standalone multi-GPU val
```

上述一致性比较只在 chunks 1f 的 `val` split 内部进行。不要求 chunks 1f 指标等于原生 raw 8f 指标。

一致性定义为：

- expected sample set hash 相同。
- prediction sample set hash 相同。
- confusion matrix hash 相同。
- 最终 mIoU/IoU/per-class IoU 数值一致。

## 6. 微验证计划

需要微验证时，按用户要求激活 `pgocc` conda 环境，但不跑完整训练流程。

环境要求：

```bash
conda activate pgocc
```

如果本机需要项目自带激活脚本，应先确认脚本路径有效。当前工作区的 `haoce_activate.sh` 引用 `/vepfs-mlp2/...`，在本机静态尝试中路径不存在，因此后续微验证应以实际服务器环境为准。

允许的微验证类型：

- 读取一个 chunk 的 metadata，不跑模型。
- 构建 chunks dataset，取 1 到 2 个 sample，检查 keys/shape/dtype。
- 构建 dataloader，取 1 个 batch，检查 collate 后字段。
- 检查 camera name 顺序、图像/特征/矩阵对应关系、组合后的 `lidar2img` 与 raw pipeline 是否数值一致或在容差内一致。
- 在有 checkpoint 且用户许可时，单 batch 前向验证 shape，不启动 epoch 训练。
- 单卡/多卡评估一致性可先用极小 mini subset 做 dry-run。

微验证通过标准是硬门槛：

- 真实 chunk 字段 shape、dtype、`materialized` 状态记录清楚。
- camera name 顺序和 image/depth/text_vision/matrix 一一对应。
- 单样本 pipeline 输出不为 `None`，且 keys 满足模型 forward/loss 需求。
- 单 batch collate 后 `img/depth/text_vision/render_gt/img_metas` shape 符合 chunks 1f 配置。
- 单卡 mini val 与多卡 mini val 的 expected sample hash、prediction sample hash、confusion matrix hash 和最终指标一致。
- 任一项失败即停止进入训练实现，先修 schema adapter 或评估协议。

禁止在本阶段执行：

- 完整训练。
- 全量 train epoch。
- 未经确认的全量多卡评估。
- 重新生成 chunks。

建议微验证命令模板：

```bash
conda activate pgocc
python - <<'PY'
import importlib
from mmcv import Config
from mmdet3d.datasets import build_dataset

importlib.import_module("loaders")
cfg = Config.fromfile("configs/pgocc_chunks.py")
dataset = build_dataset(cfg.data.train)
it = iter(dataset)
sample = next(it)
print(sample.keys())
PY
```

## 7. 实施步骤

审核通过后建议按以下顺序实现：

1. 新增 `NuSceneOVOccChunk`，只完成 manifest/index/profile 读取和单样本产出。
2. 在 `pgocc` 环境中读取一个 chunk，确认真实字段 shape、materialized 状态、camera order 和样本规模。
3. 新增 chunk 图像、depth、text_vision、occ_gt transforms。
4. 新增 `configs/pgocc_chunks.py`，设置 `_num_frames_=1`，并用代码级条件跳过 temporal warping。
5. 修正单帧 render transform，避免 `sweeps_num=0` 返回 `None`。
6. 修改 dataloader builder 支持 iterable chunks dataset 不使用 sampler，并确认 `val.py` 独立评估路径也不会套 sampler。
7. 修改评估结果对齐方式，引入 `sample_idx` coverage check。
8. 做单样本 dataset 微验证。
9. 做单 batch dataloader 微验证。
10. 做单卡 mini val 微验证。
11. 做多卡 mini val 微验证，并比对 single/multi coverage hash 和指标。
12. 最后再进行用户批准的正式训练或评估。

用户审核时需要明确裁决：

- 本机当前 `--ratio 0.01` chunks 仅作为 smoke test。
- 云服务器全量 chunks 是正式训练/评估目标数据源。
- 实现完成后，需要在云服务器对全量 chunks 先做同一套 schema/coverage 微验证，再启动正式训练或正式评估。
- 评估 split 固定为 `val`；本次不适配不存在的 `test` split。
- chunks 模式为 1f 实现，指标不与原生 raw 8f 加载方式做等价承诺。

## 8. 风险和待确认项

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| 当前设备只看到 train/val `featup_metric3d_sam2` mini chunks，且 producer command 为 `--ratio 0.01` | 本机不能支撑正式训练/正式评估 | 本机 mini chunks 只用于 smoke test；正式训练/评估转到云服务器全量 chunks，并先复用微验证 |
| 当前 manifest 显示 depth/sem_seg 为 `900x1600`，专家抽查指出 image 也可能未 materialized | 分辨率适配路径不能只按 materialized 假设写 | 实现前必须在 `pgocc` 环境读取真实 sample，并按 shape/metadata 分派变换 |
| GaussTR chunks 使用 OpenMMLab v2 meta，PG-Occ 使用自定义 sweep pkl meta | 字段名和矩阵定义可能不完全一致 | dataset 内集中做 schema adapter |
| chunks 模式关闭 8f 时序 | 与原生 8f 训练行为不完全等价 | 配置上显式命名和记录，不与 raw 8f 指标直接混淆 |
| temporal depth warping loss 失效，且当前代码无条件读取时序字段 | 训练 loss 组成变化或直接 KeyError | 代码级跳过 warping 分支，不只改权重 |
| 多卡评估结果顺序不稳定 | mIoU 可能错误或不可复现 | sample_idx 对齐 + coverage hard check |
| 相机顺序不同 | 图像、特征、矩阵错配但不一定报错 | 以 camera name 显式重排并断言 |

## 9. 当前结论

PG-Occ 可以复用 GaussTR 的 chunks 物理格式，但不能直接复用 GaussTR 的 dataset/config 代码，因为两个项目的数据栈分别是 MMCV 1.x 和 MMEngine 2.x。正确适配应在 PG-Occ 内新增 chunk dataset 和 transforms，并围绕 `sample_idx` 重做评估对齐。

分辨率上，GaussTR FeatUp chunks 的 `432x768` 图像可以通过二阶段 `resize=704/768` 和 `crop_y=140` 等效得到 PG-Occ 原生 `256x704` 输入。时序上，chunks 模式应明确退化到 1f，并关闭依赖过去/未来帧的 temporal warping loss。
