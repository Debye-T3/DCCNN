# ARPES 二维 Cut 降噪优化设计

日期：2026-07-27
状态：已完成分节确认，等待书面设计复核

## 1. 背景与目标

现有 DCCNN 项目可以完成 ARPES 数据转换、训练和推理，但源码、原始数据、派生 H5、模型权重、指标和预览图混放，环境依赖未锁定，训练与验证按裁剪样本随机拆分，训练和推理的归一化不一致，推理还会向输入 H5 追加结果。这些问题使现有指标难以证明模型能泛化到未见谱图。

本轮工作的首要目标是：

> 对从未参与训练的新二维 ARPES cut 稳定降噪，同时尽量保留真实能带、峰位置、峰宽、积分强度和温度演化。

第一阶段仅处理二维 `cut` 和 `fineCut`。三维 `map`、`fastmap` 和体数据在数据清单中登记，但不进入第一阶段训练；待二维方案通过验收后，再设计逐切片或三维扩展。

## 2. 设计原则

1. 原始实验数据只读，任何转换、训练和推理均不得修改原始文件。
2. 先建立可信的数据划分和评估基线，再比较网络结构。
3. 配对数据优先，重复扫描次之，单张数据用于补充物理多样性。
4. 不把温度、光子能量、偏振、样品位置、沉积或解离造成的真实变化当作噪声。
5. 模型优化必须同时证明噪声降低和物理结构保真。
6. 所有训练均可复现，依赖、数据清单、配置、随机种子和代码版本可追溯。
7. 迁移过程不自动删除或覆盖旧数据、旧权重和已有实验结果。

## 3. 工程边界与目录

采用源码仓库与机器学习工作区并列的结构：

```text
D:\Projects\dccnn\
├─ dccnn-arpes-main\        # Git 源码仓库
├─ workspace\
│  ├─ manifests\            # 原始数据索引、配对审核、数据划分
│  ├─ converted\            # convert 输出的 xarray/HDF5
│  ├─ cache\                # 可重建的裁剪索引与统计缓存
│  └─ splits\               # train/val/test 清单
├─ outputs\
│  ├─ checkpoints\          # 模型权重
│  ├─ experiments\          # 训练配置、指标和运行元数据
│  ├─ inference\            # 新谱图的降噪结果
│  └─ previews\             # 对比图、差值图和谱线图
└─ legacy_archive\          # 当前项目中的旧实验结果
```

`D:\Data\ARPES` 保持为原始实验档案。`D:\Projects\dccnn\workspace` 保存可重新生成的机器学习派生数据，`outputs` 保存训练与推理结果。源码仓库只保存源码、配置、测试、文档和不含敏感绝对路径的版本化数据清单。

### 3.1 统一数据格式

`D:\Projects\convert` 项目输出的 xarray/HDF5 是 DCCNN 新流程唯一的标准交换格式。文件虽然使用 `.h5` 扩展名，但其内部必须是由 `xarray.to_netcdf()` 通过 `h5netcdf` 或 `netCDF4` 写出的 NetCDF4/HDF5 结构，而不是任意的 `h5py` 数据集集合。

第一阶段二维 cut 的规范对象是一个 `xarray.DataArray`：

- 维度为 `("eV", "alpha")`；
- `eV` 和 `alpha` 是与维度绑定的一维坐标，不是普通数据变量；
- 强度数组为二维数值数据；
- 样品、采集条件和仪器参数保存在 `attrs` 中；
- 文件能由 `xr.load_dataarray()` 直接读取；
- 加载后允许按名称转置到 `("eV", "alpha")`，但禁止仅凭数组下标猜测坐标含义。

DCCNN 的训练、评估和推理入口只消费该规范对象。进入 PyTorch 前，数据加载层负责校验维度、坐标、有限值和单调性，再提取 NumPy 数组并转换为 tensor；模型层不直接解析 HDF5。

map 扩展允许使用 `xarray.DataArray`、`Dataset` 或 `DataTree` 表达额外扫描维度，但不属于第一阶段实现。遇到包含多个候选信号的 `Dataset` 或 `DataTree` 时，必须由配置显式指定变量或节点，不能自动选择。

旧 DCCNN H5 不是标准格式。它只通过独立的只读兼容适配器加载，显式执行 `spectrum → DataArray`、`energy → eV`、`thetax → alpha` 映射。适配结果必须通过与标准文件相同的校验；旧文件不在原位改写。需要长期保留的数据可另存为新的标准 xarray/HDF5 文件。

推理不得向输入文件追加数据集。每张降噪 cut 保存为独立的 `<原文件名>_denoised.h5`，仍是可由 `xr.load_dataarray()` 直接读取的 `DataArray`，并保持原始维度、坐标和实验属性；模型名称、checkpoint 标识、归一化参数和运行时间作为新增属性记录。输入文件的校验和必须保持不变。

源码仓库整理为：

```text
dccnn-arpes-main/
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ src/dccnn_arpes/
│  ├─ data/
│  ├─ io/                    # xarray 标准读取、校验与旧 H5 适配
│  ├─ models/
│  ├─ training/
│  ├─ inference/
│  └─ evaluation/
├─ configs/
├─ manifests/
├─ tests/
└─ outputs/                  # Git 忽略
```

所有路径由配置文件或命令行参数提供，不硬编码用户名、Python 路径或盘符。默认配置采用上述 Windows 路径，但允许覆盖。

## 4. 数据发现与清单

数据发现程序只读扫描 `D:\Data\ARPES`，解析：

- `.pxt`
- `.txt`
- `.bin` 及其 `.ini`
- `.ibw`
- `.zip`
- 实验记录 `.xlsx`

扫描结果写入 `D:\Projects\dccnn\workspace\manifests`，不回写原始目录。

原始谱图到标准 xarray/HDF5 的转换由 `D:\Projects\convert` 负责。DCCNN 不复制 PXT、TXT、BIN 或 IBW 的转换实现，只索引原始记录、关联 convert 产物并校验标准文件；这样避免两套转换器生成语义不同但扩展名相同的 H5。

标准数据清单至少包含：

- `record_id`
- `source_path`
- `source_format`
- `file_id`
- `sample_name`
- `sample_id`
- `session_id`
- `acquisition_group`
- `scan_type`
- `temperature_K`
- `photon_energy_eV`
- `polarization`
- `position_x`
- `position_y`
- `position_z`
- `position_polar`
- `position_tilt`
- `position_azimuth`
- `energy_axis`
- `angle_axis`
- `acquisition_time_s`
- `sweep_count`
- `pair_type`
- `pair_id`
- `review_status`
- `split`
- `quality_flag`
- `exclusion_reason`
- `notes`

Excel 中的空白字段可能表示沿用上一条记录，也可能表示未记录。扫描程序可以生成候选继承值，但不能未经审核直接把缺失值视为相同实验条件。

## 5. 配对规则

### 5.1 A 级：短采集与长采集

同一配对必须满足：

- 同一样品和物理状态
- 相同温度
- 相同光子能量和偏振
- 相同位置与角度
- 相同扫描范围、shape 和坐标轴
- 差异仅来自采集时长、sweep 数量或计数统计

输入使用短采集或部分 sweep，目标使用长采集或全部 sweep。二者转换为可比较的计数率。A 级配对拥有最高训练采样权重。

### 5.2 B 级：重复扫描

相同实验条件下的两次独立扫描可用于 Noise2Noise。存在三次以上重复时，可用一次作为输入，其余重复的平均作为参考。候选配对必须经过文件头、坐标轴、shape、采集设置和实验备注复核。

### 5.3 C 级：单张谱图

单张谱图用于扩大材料、温度、光子能量和能带结构覆盖。第一版根据真实数据估计泊松噪声、背景噪声和条纹噪声，并动态生成训练输入。复杂 Blind-Spot 自监督不进入第一版，以避免在基线尚未可信时增加难以解释的变量。

### 5.4 明确禁止配对

以下变化前后的数据禁止组成降噪配对：

- 不同温度
- 不同光子能量
- 不同偏振
- 不同样品位置或角度
- 钾沉积或其他表面处理
- 样品解离、老化或相变
- 扫描范围或坐标轴不一致
- `cut` 与 `map` 类型不同

不满足规则的候选记录排除原因，不通过自动插值强行组成配对。

## 6. 数据划分与采样

在生成裁剪之前，先按照样品和完整测量批次划分 `train`、`val` 和 `test`。同一原始谱图、同一配对组及其所有裁剪只能出现在一个 split 中。

测试集包括：

1. 未见测量批次；
2. 至少一组未见样品或材料；
3. 配对定量测试数据；
4. 变温物理保真数据；
5. 高质量谱图恒等测试数据。

默认训练采样比例为：

```text
A 级短/长配对：50%
B 级重复扫描：30%
C 级单张数据：20%
```

实际比例在数据清单完成后可通过配置调整，但测试集保持锁定。每个 epoch 重新采样裁剪和合成噪声；固定随机种子时，完整实验仍可复现。

## 7. 预处理与归一化

输入与目标不得分别独立归一化。处理流程为：

1. 根据采集时间或 sweep 数量转换为计数率；
2. 对同一配对使用共同的尺度；
3. 在保留 `eV`、`alpha` 坐标和原始属性的前提下，采用稳定的 `log1p` 或配置指定的强度变换；
4. 使用由输入或配对共同确定的稳健缩放参数；
5. 保存逆变换所需统计量；
6. 推理输出使用同一统计量恢复物理强度尺度。

如果采集时间和 sweep 数量缺失，则记录缺失状态，并仅在人工确认比例关系后进入 A 级配对；否则降级为 B 级或 C 级数据。

## 8. 模型

### 8.1 LegacyCCNN 基线

LegacyCCNN 保持现有 7 层、64 通道、3×3 卷积和 PReLU 结构不变。它用于测量仅修复数据、归一化和训练流程后的收益，并保持旧权重的加载能力。旧权重结果与新锁定测试集结果分开标记。

### 8.2 ResidualDenoiser2D 候选

候选模型采用轻量残差网络：

- 单通道二维输入；
- 3×3 卷积映射到 64 通道；
- 8 个保持分辨率的残差块；
- 不使用池化和上采样；
- 不使用 BatchNorm；
- 网络预测噪声；
- 输出为 `denoised = input - predicted_noise`；
- 参数量控制在约百万级。

第一版不使用大型 U-Net、Transformer 或扩散模型，避免在数据与评估体系刚建立时引入过度生成和难以归因的复杂度。

## 9. 损失函数

默认损失为：

```text
总损失 =
0.80 × Charbonnier 像素损失
+ 0.15 × (1 - MS-SSIM)
+ 0.05 × 能量/动量方向梯度损失
```

每个组成项独立记录。约 10% 的训练样本使用“干净输入到自身”的恒等约束，降低模型破坏本来较干净谱图的风险。

关键依赖缺失时训练直接失败并报告错误，不提供静默近似实现。

## 10. 训练与实验记录

每次训练保存：

- 完整配置；
- 随机种子；
- 数据清单哈希；
- split 清单哈希；
- Git commit；
- Python、PyTorch、CUDA 和依赖版本；
- 每个 epoch 的训练与验证指标；
- 最佳 checkpoint；
- 最后 checkpoint；
- 固定验证样本的对比图、差值图和 EDC/MDC；
- 运行开始、结束时间和设备信息。

最佳模型依据验证集复合损失保存。训练遇到 NaN、Inf、shape 不一致、坐标轴不一致、数据为空或缺失关键依赖时立即停止并给出明确错误。

## 11. 评估体系

### 11.1 配对定量测试集

评价指标包括：

- MAE
- NRMSE
- PSNR
- SSIM
- EDC/MDC 相关性
- 峰位置偏差
- FWHM 偏差
- 积分强度偏差
- 无信号区域噪声降低比例

比较对象包括：

- 未降噪输入；
- 高斯滤波；
- 中值滤波；
- LegacyCCNN；
- ResidualDenoiser2D。

报告每张谱图结果、汇总统计和最差案例，不删除失败样本。

### 11.2 变温物理保真测试集

变温序列逐张独立处理，检查：

- 峰位置温度趋势；
- 费米边与峰宽的温度展宽；
- 积分强度趋势；
- 相变、解离、沉积和老化信号是否被误删；
- 模型输出是否产生不连续跳变。

### 11.3 高质量恒等测试集

对本来较干净的谱图检查模型是否产生新的条纹、峰或能带，或明显改变强度、峰宽和背景。

### 11.4 可视化

统一生成：

- 固定色标的 input/denoised/reference 对比图；
- 输入与输出差值图；
- 代表性 EDC；
- 代表性 MDC；
- 每个指标的逐文件表格。

## 12. 第一版通过标准

ResidualDenoiser2D 必须同时满足：

1. 至少 80% 的配对测试谱图，其 NRMSE 优于未降噪输入；
2. 配对测试集中位 NRMSE 比 LegacyCCNN 至少降低 10%；
3. 峰位置偏差不超过一个能量或动量采样步长；
4. FWHM 相对高信噪比参考的偏差不超过 10%；
5. 曝光归一化后的积分强度偏差不超过 5%；
6. 高质量恒等测试集不出现新的条纹、峰或能带；
7. 变温趋势不被明显压平、反转或产生不连续跳变；
8. 所有失败样本保留并分析。

未同时满足上述条件时，候选模型不得替代 LegacyCCNN 基线。

## 13. 环境与运行入口

环境采用：

- `uv`
- Python 3.12
- 项目内 `.venv`
- `pyproject.toml`
- `uv.lock`
- 与 RTX 5080 匹配的官方 PyTorch CUDA wheel

核心依赖包括 PyTorch、xarray、h5netcdf、netCDF4、NumPy、SciPy、h5py、pandas、PyYAML、scikit-image、matplotlib、pytorch-msssim 和 tqdm。`h5py` 仅用于旧 DCCNN H5 兼容适配及底层诊断，不用于写出新的标准数据。PySide6 作为 GUI 可选依赖，不进入最小训练环境。

提供以下统一入口：

```text
dccnn-data scan
dccnn-data pairs
dccnn-data validate
dccnn-train
dccnn-eval
dccnn-denoise
```

## 14. 测试

自动测试至少覆盖：

- 归一化与反归一化往返；
- 配对条件和禁止配对规则；
- group-level split 无泄漏；
- 动态增强跨 epoch 变化；
- 固定种子可复现；
- convert 输出的 DataArray 可经 `xr.load_dataarray()` 往返读取；
- 标准二维 cut 的维度为 `("eV", "alpha")`，坐标与数据 shape 一致；
- 坐标或维度缺失、重复、非有限或语义不明确时明确失败；
- 旧 DCCNN H5 只读适配为规范 DataArray，且不修改源文件；
- 标准文件与旧格式适配结果经过同一数据校验；
- 模型输入输出 shape；
- 损失有限并能反向传播；
- checkpoint 保存与恢复；
- 推理前后原始文件校验和不变；
- 小样本 CPU 训练与推理；
- RTX 5080 CUDA smoke test；
- 固定小样本的指标回归测试。

## 15. 旧项目安全迁移

迁移顺序为：

1. 保存当前 Git 状态、diff、文件清单和校验和；
2. 不删除现有权重、H5、CSV 或预览图，并区分标准 xarray/HDF5 与旧 DCCNN H5；
3. 将旧实验产物归入 `D:\Projects\dccnn\legacy_archive`；
4. 清理 Git 对大体积原始数据和派生产物的跟踪；
5. 更新 `.gitignore`；
6. 建立最小可运行环境；
7. 将 convert 输出设为数据入口，并建立旧 DCCNN H5 只读适配器；
8. 在新结构中复现 LegacyCCNN；
9. 运行锁定测试集；
10. 训练和评估 ResidualDenoiser2D；
11. 归档校验通过后，另行决定是否删除重复文件。

同名不同内容、无法匹配实验记录或被旧推理修改过的 H5 一律标记为待审核，不自动覆盖。

## 16. 后续三维 Map 扩展

二维 cut 方案通过第一版标准后，再评估：

1. 使用同一二维模型对 map 逐切片处理；
2. 利用相邻切片作为多通道上下文；
3. 数据量与显存允许时建立三维模型。

三维扩展使用独立设计、数据划分和验收标准，不在本轮实现范围内。
