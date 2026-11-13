# pMF-DDS（mean-flow 先验 + 每步 CG）

> 本卡片由 `scripts/gen_algorithm_cards.py` 根据 `inv_framework/registry.py`、示例数据的 `meta` 和 `algorithms/results.json` 生成；请修改源头后重新生成。

## 1. 算法身份

- **名称**：pMF-DDS（mean-flow 先验 + 每步 CG）；注册名 `pmf_dds`
- **一句话简介**：2048² pixel MeanFlow 先验的 DDS 采样，每步做 Tikhonov-CG 数据一致性。
- **所属任务**：CT：二维平行束（同步辐射）、二维扇束（实验室锥束中心层）
- **方法类型**：学习类（使用预训练模型）
- **适用范围**：适用于 2048² 单通道二维 CT，包括内部断层（物体大于探测器视场的截断投影）与稀疏角扫描；先验在 NaSn 电池 FullCT 重建切片上训练。需要 CUDA GPU。网格必须为 2048²；跨材料/跨设备数据属于分布外，需配合 LoRA 适配或重新训练；不支持三维体数据。
- **对应论文**：Chung, Lee & Ye, DDS, ICLR 2024；Geng et al., Mean Flows for One-step Generative Modeling, 2025；Lu et al., One-step Latent-free Image Generation with Pixel Mean Flows, 2026（arXiv:2601.22158）
- **代码来源**：迁移自 peterw634/nasn-truncated-ct（commit 35bb708，BSD-3-Clause）；pMF 模型保留上游 MIT 许可；权重 nasn-pmf-2048-epoch449-ema（model-v0.1.0，MIT）

## 2. 输入输出

示例数据位于 `examples/data/`，字段、单位与许可见 [数据说明](../../examples/data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_nasn_pmf_2048.npz` | 二维平行束内部断层（同步辐射，NaSn 几何） | pMF 无条件生成的电池切片（不含 NaSn 扫描数据），NaSn 合成 benchmark 几何：2048 个角度，仅中心 512/2048 列有测量（内部断层），无噪声。`image` 2048×2048，`sinogram` 2048×2048（角度×探测器） | `image = (pMF 样本 + 1) / 2`（float16 存储）；未测量列存为 0 |
| `toy_nasn_pmf_fan_2048.npz` | 二维扇束（实验室锥束中心层，ASTRA） | 同一 pMF 切片，实验室扇束 180 个角度（0–360°），SOD 4096、SDD 6144，2304 个探测器，全视场，无噪声。`image` 2048×2048，`sinogram` 180×2304（角度×探测器） | 同上；`sinogram` 由 ASTRA 扇束正投影生成 |

运行后在 `--output` 目录生成：`reconstruction.npy`（重建结果，与 `image` 同尺寸、同单位，float32）、`preview.png`（真值｜重建｜4 倍放大的绝对误差；三维为中心轴向＋冠状切片，可直接打开查看）、`summary.json`（实际参数、耗时、指标、输入的 SHA-256）。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_nasn_pmf_2048.npz` | 2048×2048 | 44.54 | 0.996 | 64.05 | 0.0002 | 65.2 | cuda |
| `toy_nasn_pmf_fan_2048.npz` | 2048×2048 | 66.24 | 1.000 | — | 0.0000 | 15.3 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。结果与上表相差在 1 dB 内即可视为正常（随机方法与 GPU 型号会带来小幅差异）。

说明：toy 真值是该先验自身的无条件样本（分布内），因此近乎完美的恢复是预期结果，不代表真实数据质量；NaSn 真实合成验证集上 16 步的 ROI PSNR 为 36.6 dB（`docs/nasn_ct.md`）。

## 3. 运行方式

```bash
pip install -r requirements.txt && pip install -e .
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python scripts/download_weights.py pmf
python examples/run.py --algo pmf_dds --input examples/data/toy_nasn_pmf_2048.npz --output outputs/pmf_dds/toy_nasn_pmf_2048 --weights weights/nasn-pmf-2048-epoch449-ema.safetensors --device cuda
python examples/run.py --algo pmf_dds --input examples/data/toy_nasn_pmf_fan_2048.npz --output outputs/pmf_dds/toy_nasn_pmf_fan_2048 --weights weights/nasn-pmf-2048-epoch449-ema.safetensors --device cuda
```

- 设备：需要 CUDA GPU。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：NaSn pMF-B/128 epoch-449 EMA 权重（502 MiB，MIT），来自 https://github.com/peterw634/nasn-truncated-ct/releases/tag/model-v0.1.0 ，脚本按清单校验 SHA-256；同名 `.json` 配置必须放在权重旁边。
- 权重与数据的约定：模型在 2048²、[-1, 1] 值域上训练；toy 包用 `x_phys = 0.5·x_model + 0.5`（与 NaSn 合成 benchmark 相同），此时推荐 λ=5.0。真实 NaSn 测量使用 `physical_image × scale` 的标定且不做中心化，对应 λ=0.05（见 `inv-nasn-ct reconstruct`）。网格必须为 2048²。
- 参数可用 `--set 名称=值` 覆盖（值按 JSON 解析，如 `--set min_value=null`）；`python examples/run.py --list` 以 JSON 列出全部算法与默认值。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_inference_steps` | mean-flow 采样步数 | `16` | nasn-truncated-ct 发布配置（合成 benchmark 门槛 16 步） | 1–20 的整数；超过 20 步时 t<0.05，pMF 的 x0 估计不再精确 | 16 或 20 |
| `dc_lam` | CG 中的 Tikhonov 权重 λ（模型域） | `5` | nasn-truncated-ct 合成 benchmark | 正实数；随测量尺度变化 | 合成数据 5.0；真实 NaSn 数据 0.05 |
| `dc_iter` | 每步 CG 迭代次数 | `10` | nasn-truncated-ct 发布配置 | 正整数 | 5–20 |
| `eta` | 每步注入新噪声的比例 | `0` | nasn-truncated-ct 发布配置（确定性采样） | [0, 1] |  |
| `omega` | pMF 的 CFG 系数 | `1` | nasn-truncated-ct 发布配置 | 正实数；无条件模型取 1 |  |
