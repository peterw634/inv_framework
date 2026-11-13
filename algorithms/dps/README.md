# DPS（Diffusion Posterior Sampling）

> 本卡片由 `scripts/gen_algorithm_cards.py` 根据 `inv_framework/registry.py`、示例数据的 `meta` 和 `algorithms/results.json` 生成；请修改源头后重新生成。

## 1. 算法身份

- **名称**：DPS（Diffusion Posterior Sampling）；注册名 `dps`
- **一句话简介**：在 DDPM 反向采样中注入测量残差梯度。
- **所属任务**：CT：二维平行束（同步辐射）、二维扇束（实验室锥束中心层）
- **方法类型**：学习类（使用预训练模型）
- **适用范围**：适用于与预训练先验同网格（像素模型 512²、潜空间模型 512² 图像）的二维 CT，任意提供正投影与伴随的线性几何（平行束、扇束）；先验在 LoDoInd 工业 CT 切片上训练，跨数据域需要重新训练或微调。不支持三维体数据；所有结果保持模型域 [-1, 1] 的尺度约定。
- **对应论文**：Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023（arXiv:2209.14687）
- **代码来源**：改编自 DM4CT（https://github.com/DM4CT/DM4CT ，commit 49b3e59，pipelines.py / condition_methods.py）；预训练权重 jiayangshi/lodoind_*_diffusion（MIT）

## 2. 输入输出

示例数据位于 `examples/data/`，字段、单位与许可见 [数据说明](../../examples/data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_lodoind_512.npz` | 二维平行束（同步辐射） | LoDoInd 测试切片 2750（工业 CT），40 个角度（0–180°）稀疏角，无噪声（DM4CT 设置）。`image` 512×512，`sinogram` 40×512（角度×探测器） | DM4CT `preprocess_lodoind.py`（百分位截断、映射到 [-1, 1]、缩放到 512²）后取 `(x + 1) / 2` |
| `toy_lodoind_fan_512.npz` | 二维扇束（实验室锥束中心层，ASTRA） | 同一 LoDoInd 切片，实验室扇束 80 个角度（0–360°），SOD 1024、SDD 1536，576 个探测器，无噪声。`image` 512×512，`sinogram` 80×576（角度×探测器） | 同上；`sinogram` 由 ASTRA 扇束正投影生成 |

运行后在 `--output` 目录生成：`reconstruction.npy`（重建结果，与 `image` 同尺寸、同单位，float32）、`preview.png`（真值｜重建｜4 倍放大的绝对误差；三维为中心轴向＋冠状切片，可直接打开查看）、`summary.json`（实际参数、耗时、指标、输入的 SHA-256）。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_lodoind_512.npz` | 512×512 | 22.25 | 0.579 | — | 0.0118 | 485.0 | cuda |
| `toy_lodoind_fan_512.npz` | 512×512 | 23.14 | 0.557 | — | 0.0190 | 483.6 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。结果与上表相差在 1 dB 内即可视为正常（随机方法与 GPU 型号会带来小幅差异）。

说明：DM4CT 原设置 scale=10 在本仓库 LoDoInd toy 数据上发散（平行束 10.8 dB），默认值改为 1。归一化损失把恰好为 0 的测量视为未测量，扇束数据（探测器超出体积）不再被放大步长。

## 3. 运行方式

```bash
pip install -r requirements.txt && pip install -e .
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python scripts/download_weights.py lodoind-pixel
python examples/run.py --algo dps --input examples/data/toy_lodoind_512.npz --output outputs/dps/toy_lodoind_512 --weights weights/lodoind_pixel_diffusion --device cuda
python examples/run.py --algo dps --input examples/data/toy_lodoind_fan_512.npz --output outputs/dps/toy_lodoind_fan_512 --weights weights/lodoind_pixel_diffusion --device cuda
```

- 设备：需要 CUDA GPU。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：DM4CT 在 LoDoInd 上训练的像素域 UNet（`unet/`、`scheduler/`，455 MB，MIT，固定版本 jiayangshi/lodoind_pixel_diffusion@0f4a0dc），下载到 `weights/lodoind_pixel_diffusion/`。
- 权重与数据的约定：模型在 512² 网格、[-1, 1] 值域上训练；`examples/run.py` 按数据包的 `image_scale=0.5`、`image_offset=0.5` 把测量换到模型域（`(y − A·0.5)/0.5`），重建后再映射回 [0, 1]。网格必须与模型一致（512²），否则报错。
- 参数可用 `--set 名称=值` 覆盖（值按 JSON 解析，如 `--set min_value=null`）；`python examples/run.py --list` 以 JSON 列出全部算法与默认值。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_inference_steps` | DDPM 采样步数 | `1000` | DM4CT 原实验脚本（reverse_sample_*_ct.py，commit 49b3e59） | 正整数，不超过训练步数 1000 |  |
| `scale` | 测量引导步长（NormalizedPosteriorSampling） | `1` | 本仓库 toy 数据扫描（LoDoInd 40 角平行束，scale 0.3/1/3/10/30 → 21.6/22.2/16.1/10.8/9.0 dB）；DM4CT 原设置为 10 | 正实数 | 0.3–1；3 以上明显发散 |
