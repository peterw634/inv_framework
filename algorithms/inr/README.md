# INR（SIREN 隐式神经表示）

> 本卡片由 `scripts/gen_algorithm_cards.py` 根据 `inv_framework/registry.py`、示例数据的 `meta` 和 `algorithms/results.json` 生成；请修改源头后重新生成。

## 1. 算法身份

- **名称**：INR（SIREN 隐式神经表示）；注册名 `inr`
- **一句话简介**：用坐标 MLP 表示图像，逐例拟合测量。
- **所属任务**：CT：二维平行束（同步辐射）、二维扇束（实验室锥束中心层）
- **方法类型**：非学习类（不使用预训练模型）
- **适用范围**：适用于没有训练数据的二维重建：用 SIREN 坐标网络表示图像并逐例拟合测量，平滑区域效果好。不使用预训练模型（非学习类）。每个样本单独拟合一个网络；对学习率敏感；不支持三维。
- **对应论文**：Sitzmann et al., Implicit Neural Representations with Periodic Activation Functions, NeurIPS 2020
- **代码来源**：本仓库编写（inv_framework）

## 2. 输入输出

示例数据位于 `examples/data/`，字段、单位与许可见 [数据说明](../../examples/data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 二维平行束（同步辐射） | 改进 Shepp-Logan 模体，30 个角度（0–180°）稀疏角，加性高斯噪声 σ=0.5。`image` 128×128，`sinogram` 30×128（角度×探测器） | 无；`sinogram` 已是线积分（负对数后）投影 |
| `toy_lodoind_fan_512.npz` | 二维扇束（实验室锥束中心层，ASTRA） | 同一 LoDoInd 切片，实验室扇束 80 个角度（0–360°），SOD 1024、SDD 1536，576 个探测器，无噪声。`image` 512×512，`sinogram` 80×576（角度×探测器） | 同上；`sinogram` 由 ASTRA 扇束正投影生成 |

运行后在 `--output` 目录生成：`reconstruction.npy`（重建结果，与 `image` 同尺寸、同单位，float32）、`preview.png`（真值｜重建｜4 倍放大的绝对误差；三维为中心轴向＋冠状切片，可直接打开查看）、`summary.json`（实际参数、耗时、指标、输入的 SHA-256）。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 128×128 | 27.09 | 0.495 | — | 0.0525 | 44.9 | cuda |
| `toy_lodoind_fan_512.npz` | 512×512 | 24.40 | 0.699 | — | 0.0075 | 161.3 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。结果与上表相差在 1 dB 内即可视为正常（随机方法与 GPU 型号会带来小幅差异）。

## 3. 运行方式

```bash
pip install -r requirements.txt && pip install -e .
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python examples/run.py --algo inr --input examples/data/toy_parallel_ct_128.npz --output outputs/inr/toy_parallel_ct_128 --device cpu
python examples/run.py --algo inr --input examples/data/toy_lodoind_fan_512.npz --output outputs/inr/toy_lodoind_fan_512 --device cuda
```

- 设备：二维平行束示例可在 CPU 上运行，GPU 更快。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：不需要。
- 参数可用 `--set 名称=值` 覆盖（值按 JSON 解析，如 `--set min_value=null`）；`python examples/run.py --list` 以 JSON 列出全部算法与默认值。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_iterations` | Adam 迭代次数 | `2000` | 本框架默认值 | 正整数 | 1000–5000 |
| `lr` | Adam 学习率 | `1e-4` | 本框架默认值；SIREN 论文常用 1e-4 | 正实数 |  |
| `hidden_features` | 隐藏层宽度 | `256` | 本框架默认值 | 正整数 |  |
| `hidden_layers` | 隐藏层数 | `4` | 本框架默认值 | 正整数 |  |
