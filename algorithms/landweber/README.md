# Landweber 迭代

> 本卡片由 `scripts/gen_algorithm_cards.py` 根据 `inv_framework/registry.py`、示例数据的 `meta` 和 `algorithms/results.json` 生成；请修改源头后重新生成。

## 1. 算法身份

- **名称**：Landweber 迭代；注册名 `landweber`
- **一句话简介**：对 ||Ax−y||² 做固定步长梯度下降。
- **所属任务**：CT：二维与三维平行束（同步辐射）、二维扇束与三维锥束（实验室）
- **方法类型**：非学习类（不使用预训练模型）
- **适用范围**：适用于任意线性 CT 几何的最小二乘重建，步长按 1/||A||² 自动取，保证收敛；可加非负约束。无需训练。收敛比 SIRT 慢；不含先验。
- **对应论文**：Landweber, An iteration formula for Fredholm integral equations of the first kind, Am. J. Math. 1951
- **代码来源**：本仓库编写（inv_framework）

## 2. 输入输出

示例数据位于 `examples/data/`，字段、单位与许可见 [数据说明](../../examples/data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 二维平行束（同步辐射） | 改进 Shepp-Logan 模体，30 个角度（0–180°）稀疏角，加性高斯噪声 σ=0.5。`image` 128×128，`sinogram` 30×128（角度×探测器） | 无；`sinogram` 已是线积分（负对数后）投影 |
| `toy_parallel_ct_3d.npz` | 三维平行束（同步辐射体数据，ASTRA） | 与锥束包相同的合成圆柱电池，同步辐射式三维平行束 90 个角度（0–180°），每个探测器行对应一个切片，高斯噪声 σ=0.5%·max。`image` 64×128×128，`sinogram` 64×90×128（探测器行×角度×探测器列） | 无；`sinogram` 为线积分，ASTRA 顺序（探测器行, 角度, 探测器列）；探测器像素 = 体素 |
| `toy_cone_ct_3d.npz` | 三维锥束（实验室，ASTRA） | 合成圆柱电池（钢壳、卷绕层、极柱、端盖、夹杂），圆轨道锥束 120 个角度（0–360°），SOD 400、SDD 600（放大 1.5 倍），高斯噪声 σ=0.5%·max。`image` 64×128×128，`sinogram` 80×120×144（探测器行×角度×探测器列） | 无；`sinogram` 为线积分，ASTRA 顺序（探测器行, 角度, 探测器列）；长度以体素为单位 |

运行后在 `--output` 目录生成：`reconstruction.npy`（重建结果，与 `image` 同尺寸、同单位，float32）、`preview.png`（真值｜重建｜4 倍放大的绝对误差；三维为中心轴向＋冠状切片，可直接打开查看）、`summary.json`（实际参数、耗时、指标、输入的 SHA-256）。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 128×128 | 24.80 | 0.698 | — | 0.0161 | 12.1 | cpu |
| `toy_parallel_ct_3d.npz` | 64×128×128 | 29.52 | 0.921 | — | 0.0121 | 7.3 | cuda |
| `toy_cone_ct_3d.npz` | 64×128×128 | 27.11 | 0.932 | — | 0.0121 | 10.4 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。结果与上表相差在 1 dB 内即可视为正常（随机方法与 GPU 型号会带来小幅差异）。

## 3. 运行方式

```bash
pip install -r requirements.txt && pip install -e .
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python examples/run.py --algo landweber --input examples/data/toy_parallel_ct_128.npz --output outputs/landweber/toy_parallel_ct_128 --device cpu
python examples/run.py --algo landweber --input examples/data/toy_parallel_ct_3d.npz --output outputs/landweber/toy_parallel_ct_3d --device cuda
python examples/run.py --algo landweber --input examples/data/toy_cone_ct_3d.npz --output outputs/landweber/toy_cone_ct_3d --device cuda
```

- 设备：二维平行束示例可在 CPU 上运行，GPU 更快。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：不需要。
- 参数可用 `--set 名称=值` 覆盖（值按 JSON 解析，如 `--set min_value=null`）；`python examples/run.py --list` 以 JSON 列出全部算法与默认值。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_iterations` | 迭代次数 | `400` | 本仓库 toy 数据上的经验值（200→800 次 PSNR 23.9→25.1 dB） | 正整数 | 200–800；收敛比 SIRT 慢 |
| `step_size` | 步长；None 时取 1/||A||²（幂迭代估计） | `None` | Landweber 收敛条件 0<α<2/||A||² | None 或 (0, 2/||A||²) 内的实数 |  |
| `min_value` | 每次迭代后的下限截断 | `0` | 经验值：衰减系数非负 | None 或实数 |  |
| `max_value` | 每次迭代后的上限截断 | `None` | 不截断 | None 或实数 |  |
