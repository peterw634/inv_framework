# 示例数据（toy data）

本目录的 8 个 `.npz` 是整理好的最小示例数据，`examples/run.py` 可以直接读取，
用于每个算法的 smoke 测试与“正常结果”核对。每种算法都提供两类几何：

- **同步辐射平行束**：二维平行束（含内部断层/截断投影），以及三维平行束体数据；
- **实验室锥束**：三维圆轨道锥束，以及它的中心层二维扇束（2D 先验只能用扇束）。

| 文件 | 几何 | 内容 | 大小 | 许可 | 用于 |
| --- | --- | --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 二维平行束，30 角 / 180°，128 探测器 | Shepp-Logan 128²，加性高斯噪声 σ=0.5 | 0.02 MB | CC0（生成数据） | FBP、SIRT、Landweber、CG、DIP、INR |
| `toy_truncated_ct_128.npz` | 二维平行束内部断层，90 角，仅中心 48/128 列有测量 | Shepp-Logan 128²，无噪声 | 0.02 MB | CC0 | 截断几何下的迭代方法对比 |
| `toy_parallel_ct_3d.npz` | 三维平行束（ASTRA `parallel3d_vec`），90 角 / 180°，探测器 64×128 | 同一个合成圆柱电池 64×128×128，高斯噪声 | 2.8 MB | CC0 | FBP、SIRT、Landweber、CG（同步辐射体数据） |
| `toy_cone_ct_3d.npz` | 三维锥束（ASTRA `cone_vec`），120 角 / 360°，探测器 80×144 | 合成圆柱电池（钢壳、卷绕层、极柱、端盖、夹杂）64×128×128，高斯噪声 | 5.1 MB | CC0 | FDK、SIRT、Landweber、CG |
| `toy_lodoind_512.npz` | 二维平行束，40 角 / 180°，512 探测器（DM4CT 设置） | LoDoInd 测试切片 2750，无噪声 | 1.0 MB | CC BY-NC 4.0 | 10 个 DM4CT 扩散方法 |
| `toy_lodoind_fan_512.npz` | 二维扇束（ASTRA `fanflat_vec`），80 角 / 360°，576 探测器 | 同上切片 | 1.1 MB | CC BY-NC 4.0 | DM4CT 方法、DIP、INR 的实验室几何 |
| `toy_nasn_pmf_2048.npz` | 二维平行束内部断层（NaSn 合成 benchmark 几何），2048 角，仅中心 512/2048 列有测量 | pMF 无条件生成的电池切片 2048²（不含 NaSn 扫描数据） | 8.5 MB | MIT（发布权重的输出） | pMF-DDS、pMF-LoRA-DDS |
| `toy_nasn_pmf_fan_2048.npz` | 二维扇束，180 角 / 360°，2304 探测器，全视场 | 同上切片 | 6.6 MB | MIT | pMF 方法的实验室几何 |

校验值见 [SHA256SUMS](SHA256SUMS)（`sha256sum -c SHA256SUMS`）。

## 字段与单位

所有数据包字段一致（`np.load(path)`）：

| 字段 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `image` | 二维 `(H, W)`；三维 `(Z, Y, X)`；float32（pMF 包为 float16） | 真值。单位为“每像素（体素）衰减”，范围约 [0, 1]（电池模体到 1.2） |
| `sinogram` | 二维 `(角度, 探测器)`；三维平行束/锥束 `(探测器行, 角度, 探测器列)`（ASTRA 顺序）；float32 | 已取负对数的线积分投影，单位为像素长度 × 每像素衰减，即无量纲 −log(I/I0)。截断包中未测量的列存为 0 |
| `theta` | `(角度,)` float32 | 投影角（平行束）或光源角（扇束/锥束），**弧度** |
| `det_mask` | `(探测器列,)` bool | 有测量的探测器列（中心连续带，`left = (W − inner) // 2`） |
| `center` | 标量 | 旋转中心所在的探测器列坐标（像素） |
| `operator` | 字符串 | `parallel_torch`（纯 torch 平行束）、`tomopy_cuda`（NaSn CUDA 投影器）、`astra_parallel3d`、`astra_fan`、`astra_cone` |
| `sod`, `sdd`, `voxel_size`, `det_pixel`, `det_offset_u` | 标量；三维 `det_pixel` 为 `(dv, du)` | 仅 ASTRA 几何（三维平行束的 `sod`/`sdd` 为 0）：光源–物体、光源–探测器距离、体素与探测器像素尺寸（均以体素为长度单位，`voxel_size = 1`），探测器沿 u 的偏移（像素） |
| `image_scale`, `image_offset` | 标量 | 学习类先验的模型域映射 `x_phys = image_scale · x_model + image_offset`（DM4CT 与 pMF 为 0.5、0.5，即模型域 [-1, 1]）；非学习方法直接在 `image` 单位下重建 |
| `noise_sigma` | 标量 | 加性高斯噪声标准差（0 表示无噪声） |
| `meta` | JSON 字符串 | 描述、来源、许可、预处理、几何、随机种子 |

几何约定：平行束探测器像素与图像像素等宽；三维平行束、扇束与锥束采用 ASTRA 的
`parallel3d_vec`/`fanflat_vec`/`cone_vec`，角度 0 时光源（射线）位于 −y 方向、探测器 u 沿 +x，与
`inv_framework.operators.ct.parallel_beam_3d_geometry` / `fan_beam_geometry` / `circular_cone_geometry` 一致。
三维平行束的每个探测器行对应一个切片，因此与“逐层二维平行束”是同一个问题，但可以整体做三维迭代与三维正则。
各数据包的具体预处理与几何参数写在 `meta` 里，例如
`python -c "import numpy as np, json; print(json.dumps(json.loads(str(np.load('examples/data/toy_cone_ct_3d.npz')['meta'])), indent=2, ensure_ascii=False))"`。

## 预处理说明

- Shepp-Logan 与电池模体：直接生成，投影即线积分，没有负对数以外的处理。
- LoDoInd：Zenodo 记录 10391412 的 `Ref.zip` 参考重建，经 DM4CT
  `preprocess_lodoind.py` 处理（0.5/99.5 百分位截断、去均值、线性映射到 [-1, 1]、LANCZOS 缩放到 512²），
  取测试切片 2750，再映射到 [0, 1]：`image = (stored + 1) / 2`。
- pMF：用发布的 `nasn-pmf-2048-epoch449-ema` 无条件采样（16 步，种子 20261113）作为真值，
  `image = (sample + 1) / 2`。重建时必须使用其他噪声种子（`examples/run.py` 默认 0）。
  该切片由模型生成，不含 NaSn 原始扫描数据。

## 重新生成

```bash
python scripts/make_toy_data.py --output examples/data --device cuda \
  --lodoind-slice /path/to/LoDoInd_preprocessed/ref_normalized/test/2750.tif \
  --pmf-weights weights/nasn-pmf-2048-epoch449-ema.safetensors
```

不带 `--device cuda` 时只生成两个纯 torch 二维平行束包；三维平行束、锥束与扇束包需要 astra-toolbox 与 CUDA GPU。
生成过程是确定性的，同一环境下重新生成得到相同的 SHA-256。

## 完整数据集的获取

示例数据只用于跑通流程。完整数据：

| 数据 | 内容 | 获取方式 | 许可 |
| --- | --- | --- | --- |
| LoDoInd | 工业 CT（不同剂量 Noise1–5 与参考 Ref），每个 zip 约 22.5 GB | Zenodo [10356955](https://zenodo.org/records/10356955)（Noise1/2）、[10391277](https://zenodo.org/records/10391277)（Noise3/4）、[10391412](https://zenodo.org/records/10391412)（Noise5、Ref） | CC BY-NC 4.0 |
| DM4CT 同步辐射岩石 | 真实平行束同步辐射投影（F3_1、F3_2） | Zenodo [15420527](https://zenodo.org/records/15420527) | CC BY 4.0 |
| NaSn 截断视场 CT | 4 个目标扫描（约 37 GiB）与 4 个 FullCT 训练扫描（约 28 GiB） | 正迁移到私有 Hugging Face Dataset，需向仓库所有者申请权限；拿到权限后设置 `HF_TOKEN` 并在清单中填入地址，再用 `inv-nasn-ct data fetch --profile quickstart --extract` | CC BY 4.0（发布时） |
| BIL（Battery Imaging Library） | 电池 CT 原始投影：22 个同步辐射平行束扫描、20 个实验室锥束扫描（Diondo D5） | 各扫描的 Zenodo DOI，见 [docs/industrial_datasets.md](../../docs/industrial_datasets.md#按几何分类的重建数据算法接入用) | CC BY 4.0 |
| RWTH Samsung 35E | 4 颗 18650 的重建体（初始/老化），无投影，用于模拟 | Hugging Face `bsebench-org/rwth-jellyroll-deformation-2021-raw` | CC BY 4.0 |
| CWI Walnuts | 42 个核桃的实验室锥束投影与参考重建 | Zenodo [2686726](https://zenodo.org/records/2686726)（样本 1–8）及同系列记录 | CC BY 4.0 |
| 其他工业数据 | 电池分割、复合材料等 | 见 [docs/industrial_datasets.md](../../docs/industrial_datasets.md) | 各数据页 |

LoDoInd 参考重建的下载示例（只需 `Ref.zip`，约 22.4 GB）：

```bash
curl --fail --location --retry 3 --continue-at - -o Ref.zip \
  'https://zenodo.org/records/10391412/files/Ref.zip?download=1'
echo 'd72d0e247c70b82e39ba087ab3d3ebca  Ref.zip' | md5sum -c -
```

预训练权重的下载见 `python scripts/download_weights.py --help` 与各算法卡片。
