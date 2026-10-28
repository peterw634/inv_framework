# NaSn 电池截断视场工业 CT

本应用把原 [nasn-truncated-ct 项目](https://github.com/peterw634/nasn-truncated-ct)
接入 `inv_framework`，保留数据预处理、pMF 训练、EMA 权重导出、真实截断重建与合成评估流程。
主重建路径为 **2048 × 2048 单通道 pMF-B/128 先验 + mean-flow DDS + 严格伴随 CG 数据一致性**。
安装本框架即可使用，无需另外安装原项目的 `nasn_ct` 包。

这是研究用途的重建流程。真实 NaSn 目标数据没有配对真值，真实重建报告测量残差和耗时；
PSNR／SSIM 用于具有参考图像的合成实验。

相关文档：[迁移与数值验证](../../../docs/nasn_ct.md) ·
[各 diffusion 方法与论文](../../solvers/diffusion/README.md) ·
[工业数据资源与下载说明](../../../docs/industrial_datasets.md)。

## 项目组成

```text
nasn_ct/
├── cli.py                  # 数据、权重、重建、评估命令入口
├── config.py               # YAML 配置读取及区间约定
├── measurement.py          # 真实 NaSn 测量几何和尺度适配
├── reconstruct.py          # 真实切片的 mean-flow DDS 重建
├── benchmark.py            # 合成截断测量与质量评价
├── weights.py              # EMA 导出及 pMF 权重加载
├── random.py               # 按样本独立的 CPU 随机数流
├── data/
│   ├── download.py         # 按清单下载、SHA-256 校验和解压
│   └── preprocess.py       # 平暗场、PAD-PA、Rivers、Gridrec、语料构建
├── training/
│   ├── dataset.py          # TIFF 数据集和可选增强
│   └── train.py            # pMF 单卡／分布式训练
├── configs/                # 扫描与实验参数
└── manifests/              # 数据、模型、源文件清单
```

核心模型和算子放在框架公共目录：

| 功能 | 实现位置 |
|---|---|
| pMF 模型 | [models/pmf](../../models/pmf/) |
| CUDA 正投影、严格伴随与 Gridrec | [nasn_cuda.py](../../operators/ct/nasn_cuda.py)、[radon_cuda](../../operators/ct/radon_cuda/) |
| mean-flow 模型适配 | [PMFVelocityPrior](../../solvers/diffusion/pmf_compat.py) |
| DDS 与 mean-flow 时间调度 | [dds.py](../../solvers/diffusion/dds.py)、[scheduler.py](../../solvers/diffusion/scheduler.py) |
| CG 数据一致性 | [conditioning.py](../../solvers/diffusion/conditioning.py)、[cg.py](../../solvers/cg.py) |
| 在环 LoRA 适配 | [dds_lora.py](../../solvers/diffusion/dds_lora.py) |

## 1. 安装与环境

以下命令均从 **`inv_framework` 仓库根目录**执行，替换示例中的 `/path/to/...`。

```sh
pip install -e '.[nasn,test]'
inv-nasn-ct doctor
inv-nasn-ct --help
```

需要 Python 3.10+。真实重建、FullCT Gridrec、目标尺度标定和训练需要 NVIDIA CUDA GPU。
首次调用 CUDA 算子会编译扩展，需要与 PyTorch 匹配的 CUDA 工具链和 C++ 编译器；
可设置 `MAX_JOBS=2` 限制编译并行数。`doctor` 报告运行环境，不代替 CUDA 编译测试。

未使用安装后的命令入口时，可通过模块或示例脚本调用：

```sh
python -m inv_framework.applications.nasn_ct.cli --help
python examples/nasn_ct_reconstruct.py --help
```

## 2. 数据与模型准备

### 模型权重

原项目的 [model-v0.1.0 Release](https://github.com/peterw634/nasn-truncated-ct/releases/tag/model-v0.1.0)
提供 epoch-449 EMA 推理权重，约 502 MiB。需要同时保存同名 JSON 配置：

```sh
mkdir -p outputs/weights
curl --fail --location --retry 3 --continue-at - \
  --output outputs/weights/nasn-pmf-2048-epoch449-ema.safetensors \
  'https://github.com/peterw634/nasn-truncated-ct/releases/download/model-v0.1.0/nasn-pmf-2048-epoch449-ema.safetensors'
curl --fail --location --retry 3 \
  --output outputs/weights/nasn-pmf-2048-epoch449-ema.json \
  'https://github.com/peterw634/nasn-truncated-ct/releases/download/model-v0.1.0/nasn-pmf-2048-epoch449-ema.json'
```

Linux 下可按 [模型清单](manifests/model-v0.1.0.json)核验 SHA-256：

```sh
printf '%s  %s\n' \
  '22a96d87cefcf34364768399fbf83ba6bbf7f180ef0de96d4e8dcf99ef87a70a' \
  'outputs/weights/nasn-pmf-2048-epoch449-ema.safetensors' \
  '071d71ca998cf5912d37cf8da6423a4f99a284fedf8dc587113a62b6a31da930' \
  'outputs/weights/nasn-pmf-2048-epoch449-ema.json' | sha256sum --check -
```

默认加载权重旁同名 `.json`；配置放在其他位置时，用 `--model-config` 指定。
这里需要的是带 `model` 字段的导出 JSON，不能直接替换成 `configs/pmf_2048.yaml`。

### 数据状态与输入目录

当前随仓库保存的 [数据清单](manifests/data-v0.1.0.json)为 `published=false`，
各数据包尚未配置直接下载地址。因此 `inv-nasn-ct data fetch --profile quickstart --extract`
会提示未发布并返回退出码 `2`。需要已有的本地数据，或后续由数据所有者提供的访问方式。
公开模型权重与原始扫描数据的访问状态分别管理。

原始扫描按以下结构放置；文件前缀、投影区间和切片区间由对应 YAML 配置确定：

```text
/path/to/nasn-data/raw/
├── target/scan1/Projection/       # NaSn 截断目标扫描
│   ├── tomo_*.tif
│   ├── flat_*.tif
│   └── dark_*.tif                # 若扫描提供暗场
└── training/scan1/Projection/     # FullCT；scan2、scan3、scan4 同理
    ├── tomo_*.tif
    └── flat_*.tif
```

预处理结果采用 `sinograms_rivers/sino_001150.tif` 等名称。
若已有预处理后的 sinogram、正确的尺度 JSON 和推理权重，可直接进入第 4 节，
将命令中的输入路径指向已有文件。

## 3. 目标扫描预处理与尺度标定

流程为平暗场校正 → PAD-PA 相位恢复 → 按切片组织 sinogram → Rivers 条纹校正。
下例处理 scan1 的默认切片范围，并按原始旋转中心 1172.0 重新标定尺度：

```sh
inv-nasn-ct data prepare-target \
  --raw-root /path/to/nasn-data/raw/target/scan1 \
  --output outputs/target/scan1 --device cuda

inv-nasn-ct data calibrate-target \
  --sinogram-dir outputs/target/scan1/sinograms_rivers \
  --output outputs/target/scan1/scale_stats_center1172.json \
  --center 1172 --samples 80
```

默认配置是 [target_legacy.yaml](configs/target_legacy.yaml)，选择 900 个投影、
2560 个探测器列以及 550–1750 号切片。切片编号与配置区间均为 **1-based、两端包含**；
可用 `--slice-start`、`--slice-stop` 成对覆盖切片范围。

预处理中间结果保存在输出目录的 `.nasn-work/`，同配置重复运行可续接。
更换原始扫描、物理参数或切片范围时，应使用新的输出目录，避免复用旧中间结果。

## 4. 真实 NaSn 重建

```sh
inv-nasn-ct reconstruct \
  --weights outputs/weights/nasn-pmf-2048-epoch449-ema.safetensors \
  --sinogram-dir outputs/target/scan1/sinograms_rivers \
  --scale-stats outputs/target/scan1/scale_stats_center1172.json \
  --slices 1150 --num-steps 16 --dc-lam 0.05 --dc-iter 10 \
  --seed 0 --device cuda:0 --output outputs/nasn-real
```

| 参数 | 含义／默认值 |
|---|---|
| `--slices` | 原始切片编号，可给多个；默认 `1150` |
| `--num-steps` | mean-flow DDS 步数，默认 `16` |
| `--dc-lam` | CG 先验锚点权重，默认 `0.05` |
| `--dc-iter` | 每步 CG 迭代上限，默认 `10` |
| `--seed` | 初始噪声种子，默认 `0` |
| `--omega` | pMF guidance 参数，默认 `1.0` |
| `--warm-start --t-start 0.5` | 使用 Gridrec 初值热启动；热启动要求 `0 < t_start < 1` |

输出为 `recon_slice_001150.tif` 等 float32 图像，以及记录配置、已测区域相对残差、
重建耗时和强度统计的 `summary.json`。图像保留训练尺度，未做显示归一化；
如需恢复物理图像尺度，应除以对应的 `final_scale`。

### 几何与数值约定

- 输入 sinogram 为 `900 × 2560`。探测器轴用 OpenCV AREA 缩至 410 列，再向左右各补
  819 列零，形成 `900 × 2048` 测量；算子仅约束有效探测器区域。
- 原始中心 `1172.0` 按像素中心约定映射到 `1006.283203125`。
  `load_scale_stats` 会检查 `center_2560`，不接受其他中心生成的尺度文件。
- 真实目标采用 `image_model = image_physical * final_scale`，不减均值、不额外平移到 `[-1, 1]`。
- CG 使用正投影的严格伴随。Gridrec 是 `pseudo_inverse` 初值路径，不用作 CG 的伴随。
- 主路径调用框架的 `DDSSolver`／`MeanFlowDDSSolver`，不是 ε 预测模型的 `DDIMDDSSolver`。
  每个样本使用独立 CPU 随机数流生成初始噪声，再传入求解器。

## 5. FullCT 训练语料构建

对四个训练扫描分别进行预处理和 Gridrec 重建。下例要求原始目录已经就绪：

```sh
for scan in 1 2 3 4; do
  inv-nasn-ct data prepare-fullct \
    --config "inv_framework/applications/nasn_ct/configs/fullct_scan${scan}.yaml" \
    --raw-root "/path/to/nasn-data/raw/training/scan${scan}" \
    --output "outputs/fullct/scan${scan}" --device cuda
done

inv-nasn-ct data build-corpus \
  --train outputs/fullct/scan1/recon_gridrec \
          outputs/fullct/scan2/recon_gridrec \
          outputs/fullct/scan3/recon_gridrec \
  --val outputs/fullct/scan4/recon_gridrec \
  --output outputs/corpus
```

默认完整扫描范围对应 **3003 张训练切片、1001 张验证切片**；若只处理部分切片，
实际数量随输入变化。构建结果包含 `train/`、`val/`、`scale_stats.json` 和 `manifest.csv`。
线性缩放统计仅由训练扫描计算，验证扫描共用该尺度；图像调整为 `2048 × 2048`。

`prepare-fullct` 默认执行全部步骤；只需要预处理 sinogram 时使用 `--stage sinograms`。
此处的训练语料尺度与第 3 节真实目标标定分别记录，不应混用两个 scale 文件。

## 6. pMF 训练与 EMA 导出

训练入口在 [training/train.py](training/train.py)。需要额外安装训练依赖，
并提供上游 pMF 预训练参数文件；`--pretrained` 读取 PyTorch 参数字典，
不是第 2 节的推理 safetensors。

```sh
pip install -e '.[nasn,nasn-train]'

torchrun --nproc_per_node=4 \
  -m inv_framework.applications.nasn_ct.training.train \
  --data-root outputs/corpus \
  --pretrained /path/to/upstream-pmf-B-16.pt \
  --workdir outputs/train-pmf --batch-size 16 --epochs 500
```

`--batch-size` 是全局 batch，必须能被 GPU 数量整除。单卡运行可将
`--nproc_per_node` 改为 `1`，也可使用安装后的 `inv-nasn-ct-train` 入口。
默认训练 `pmfDiT_B_128`，使用 AdamW、warmup + cosine 学习率、EMA 和 LPIPS 损失；
配置摘要见 [pmf_2048.yaml](configs/pmf_2048.yaml)。

当前训练入口的模型和损失配置由 Python 代码固定，**不会自动读取该 YAML**；
命令行可调整学习率、epoch、batch、EMA 等已暴露参数。训练仅遍历 `train/`，
没有内置逐 epoch 验证循环。`--augment` 可启用翻转与 90° 旋转增强。

默认每 50 个 epoch 保存一次 checkpoint，编号从 0 开始，因此完成第 450 个 epoch
后生成 `ckpt_epoch_449.pt`。续训时在原训练命令中加入 `--resume /path/to/checkpoint.pt`，
仍需保留 `--pretrained` 参数。导出推理 EMA：

```sh
inv-nasn-ct weights export \
  --checkpoint outputs/train-pmf/ckpt_epoch_449.pt \
  --output outputs/weights-trained/nasn-pmf-2048-epoch449-ema.safetensors
```

导出会同时生成同名 JSON，排除训练损失网络和训练专用参数。
使用新训练的模型时，将重建／评估命令的 `--weights` 指向 `outputs/weights-trained/` 下的导出文件。
完整训练尚未作为迁移检查重新运行，耗时和显存应以实际 GPU 配置为准。

## 7. 合成截断评估

提供与原实验存储尺度一致的验证 TIFF，如 `000991.tif` 至 `001000.tif`：

```sh
inv-nasn-ct benchmark \
  --weights outputs/weights/nasn-pmf-2048-epoch449-ema.safetensors \
  --val-root /path/to/synthetic_val \
  --slices 991 992 993 994 995 996 997 998 999 1000 \
  --steps 16 32 --device cuda:0 --output outputs/nasn-benchmark
```

该评估使用固定的 `2048²` 图像、2048 个角度、512 个有效探测器和半径 256 的 ROI。
在投影／CG 前应用 `(x+1)/2`，CG 后恢复模型尺度，默认 `dc_lam=5.0`、CG 10 次。
它与真实扫描的几何、尺度和 λ 不同，不能直接交换参数。
输入不会自动读取训练语料的 `scale_stats.json`，使用其他验证集前需要确认强度约定。

输出 `progress.json` 与 `summary.json`，包含逐切片及聚合指标；当前该命令不保存重建 TIFF。
PSNR 在圆形 ROI 内计算，SSIM 在 ROI 的中心方形裁剪区域计算。
脚本保留原实验的 16／32 步指标门槛，未达到时返回非零退出码；有指标输出不等于验收通过。

多种 diffusion 方法的统一图像保存和比较入口见
[examples/diffusion_ct_compare.py](../../../examples/diffusion_ct_compare.py)。

## 8. Python 接口与扩展

下例对应真实切片重建，复用上述权重和预处理结果：

```python
from inv_framework.applications.nasn_ct.measurement import TruncatedMeasurement, load_scale_stats
from inv_framework.applications.nasn_ct.random import BatchGenerator
from inv_framework.applications.nasn_ct.weights import load_pmf
from inv_framework.solvers.diffusion import MeanFlowDDSSolver, PMFVelocityPrior

device = 'cuda'
scale = load_scale_stats('outputs/target/scan1/scale_stats_center1172.json')
measurement = TruncatedMeasurement('outputs/target/scan1/sinograms_rivers', scale)
A = measurement.operator(device=device)
y = measurement.load(1150, device=device)
pmf = load_pmf('outputs/weights/nasn-pmf-2048-epoch449-ema.safetensors', device=device)
solver = MeanFlowDDSSolver(
    PMFVelocityPrior(pmf), num_inference_steps=16, dc_lam=0.05, dc_iter=10,
)
noise = BatchGenerator(device, [0]).randn((1, *A.domain_shape))
x = solver.solve(y, A, initial_noise=noise)
```

LoRA 适配通过公共 `LoRAAdaptiveDDSSolver` 使用，需要先注入 LoRA 并配置适配参数；
当前 `inv-nasn-ct reconstruct` 没有 LoRA 开关。
方法说明和相关论文见 [diffusion README](../../solvers/diffusion/README.md)。

## 验证与来源

接口检查与 CUDA 算子检查可分别运行：

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest tests/nasn_ct -q
python -m pytest tests/nasn_ct/test_cuda_integration.py -q
```

已有真实切片迁移检查通过预设门槛；合成实验的严格跨运行指标对照尚未通过。
完整训练、全验证集和真实 2048 LoRA 实验不属于已完成的迁移验证范围。
结果、数值约定及 CUDA 原子累加造成的波动见 [迁移记录](../../../docs/nasn_ct.md)，
其他 diffusion 方法的检查见 [验证记录](../../../docs/diffusion_validation.md)。

本应用源于原项目提交 `35bb7088273dddb07495cafbf9fcc612c07ae930`。
NaSn、pMF、TomoPy 及数据清单的来源和许可见
[THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md)。
本文命令与模型 Release 元数据核对日期：2026-10-28；文档编写未执行数据下载、训练或真实重建。
