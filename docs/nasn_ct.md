# NaSn 项目按 inv_framework 接口迁移

来源是 `peterw634/nasn-truncated-ct` 的提交
`35bb7088273dddb07495cafbf9fcc612c07ae930`。本次接入保留公开的 2048²
pMF + 严格伴随 DDS 主路径，以及预处理、训练、权重导出和合成评估入口。
运行时不需要安装或引用原 `nasn_ct` 包。

## 与框架模板的对应关系

| 原项目 | 框架中的位置和接口 |
|---|---|
| `operators/ct.py` 的正投影、伴随 | `operators/ct/nasn_cuda.py:TomoPyCUDARadon2D(LinearOperator)` |
| CUDA projector、BP、Gridrec | `operators/ct/radon_cuda/`，保留原数值实现，首次调用时编译 |
| `models/pmf/` | `models/pmf/`，模型结构及参数名称保持不变 |
| pMF 与采样器的连接 | 现有 `PMFVelocityPrior`，提供 `model(x_t,t) -> u` |
| `solvers/dds.py` 的主重建循环 | 现有 `DDSSolver` + `MeanFlowSchedule` |
| `solvers/cg.py` 的数据一致性 | 现有 `cg_data_consistency` + `CGDataConsistency` |
| 真实测量、预处理、训练、权重与 CLI | `applications/nasn_ct/` |

新增算子暴露 `domain_shape=(1,N,N)`、`range_shape=(1,A,N)`、`forward`、
`adjoint` 和 `pseudo_inverse`。正投影的反传等于伴随，伴随的反传等于正投影，
使可微 CG 能将梯度传回先验参数。`pseudo_inverse` 使用独立 Gridrec；
`adjoint` 始终使用不带滤波的严格 BP。框架的 `FBPSolver` 仍使用自己的
Ram-Lak FBP 公式；复现 NaSn 的 Gridrec 初值应调用 `A.pseudo_inverse(y)`。

核心框架导入不加载 CUDA 扩展、pMF 模型、OpenCV 或训练损失网络。
算子为单通道二维平行束，CUDA 内核采用 float32；真实 2048 重建需要 CUDA。

## 安装与入口

```bash
pip install -e '.[nasn,test]'
inv-nasn-ct doctor
inv-nasn-ct --help

# 在源码目录也可以直接使用：
python examples/nasn_ct_reconstruct.py --help
python -m inv_framework.applications.nasn_ct.cli --help
```

首次 CUDA 调用需要与 PyTorch 兼容的 CUDA 编译工具链及 C++ 编译器。
扩展写入 PyTorch 缓存；`MAX_JOBS=2` 可以限制并行编译数量。

## 标准求解器调用

```python
from inv_framework.applications.nasn_ct.measurement import TruncatedMeasurement, load_scale_stats
from inv_framework.applications.nasn_ct.weights import load_pmf
from inv_framework.applications.nasn_ct.random import BatchGenerator
from inv_framework.solvers.diffusion import DDSSolver, PMFVelocityPrior

scale = load_scale_stats('/path/to/scale_stats_center1172.json')
measurement = TruncatedMeasurement('/path/to/sinograms_rivers', scale)
A = measurement.operator(device='cuda')
y = measurement.load(1150, device='cuda')
pmf = load_pmf('/path/to/nasn-pmf-2048-epoch449-ema.safetensors')
model = PMFVelocityPrior(pmf)

solver = DDSSolver(model, num_inference_steps=16, dc_lam=0.05, dc_iter=10)
noise = BatchGenerator('cuda', [0]).randn((1, *A.domain_shape))
x = solver.solve(y, A, initial_noise=noise)
```

权重旁须有同名 `.json` 配置，或通过 CLI 的 `--model-config` 指定配置。
支持原项目的 epoch-449 safetensors，无需转换参数名。

## 真实数据重建

```bash
inv-nasn-ct reconstruct \
  --weights /path/to/artifacts/nasn-pmf-2048-epoch449-ema.safetensors \
  --sinogram-dir /path/to/target/sinograms_rivers \
  --scale-stats /path/to/target/scale_stats_center1172.json \
  --slices 1150 --num-steps 16 --dc-lam 0.05 --dc-iter 10 \
  --device cuda:0 --output outputs/nasn-real
```

原始 sinogram 为 `900×2560`，OpenCV AREA 沿探测器轴缩至 410，然后两侧
各补 819 列零。原始旋转中心 1172.0 按像素中心约定映射为
`819+(1172+0.5)*410/2560-0.5 = 1006.283203125`。CLI 会拒绝其他中心
生成的 scale 文件，避免混用旧标定。

默认参数为 16 步、每步 10 次 CG、λ=0.05、seed=0、确定性采样。
`--warm-start --t-start 0.5` 使用 Gridrec 初值。输出包括训练尺度 TIFF
和 `summary.json`。真实数据没有真值，测量残差仅反映已测射线的一致性。

## 复现时必须保持的数值约定

1. 真实数据使用 `physical_image * scale`，不进行中心化或 [-1,1] 转换。
2. 原合成 benchmark 在 CG 前使用 `(x+1)/2`，CG 后恢复 `2*x-1`。
   这里由 `CGDataConsistency(image_scale=0.5, image_offset=0.5)` 实现；
   λ=5.0、2048 个角度、512 个有效 detector，区别于真实数据配置。
3. 原项目使用每个样本独立的 CPU 随机数流。新增 `initial_noise` 参数让
   框架重用这份噪声，再乘模型的 `noise_scale`，不会偷偷改用 CUDA RNG。
4. `MeanFlowSchedule` 使用原始时间网格的下一个点，并直接重编码校正后的
   x0，避免两个较大图像相减导致的精度损失。

## 数据处理与训练

配置和下载清单随包安装在 `applications/nasn_ct/configs/`、`manifests/`。
目标预处理的默认配置路径与当前工作目录无关。

```bash
inv-nasn-ct data prepare-target --raw-root /path/to/raw/target/scan1 \
  --output outputs/target/scan1
inv-nasn-ct data calibrate-target \
  --sinogram-dir outputs/target/scan1/sinograms_rivers \
  --output outputs/target/scan1/scale_stats_center1172.json --center 1172

inv-nasn-ct data prepare-fullct \
  --config inv_framework/applications/nasn_ct/configs/fullct_scan1.yaml \
  --raw-root /path/to/raw/training/scan1 --output outputs/fullct/scan1

# 四个 FullCT scan 准备好后：
inv-nasn-ct data build-corpus \
  --train outputs/fullct/scan1/recon_gridrec outputs/fullct/scan2/recon_gridrec outputs/fullct/scan3/recon_gridrec \
  --val outputs/fullct/scan4/recon_gridrec --output outputs/corpus

pip install -e '.[nasn,nasn-train]'
torchrun --nproc_per_node=4 -m inv_framework.applications.nasn_ct.training.train \
  --data-root outputs/corpus --pretrained /path/to/upstream-pmf-B-16.pt \
  --workdir outputs/train-pmf --batch-size 16 --epochs 500

inv-nasn-ct weights export --checkpoint outputs/train-pmf/ckpt_epoch_449.pt \
  --output outputs/weights/nasn-pmf-2048-epoch449-ema.safetensors
```

保留平暗场校正 → PAD-PA → Rivers → Gridrec，以及 3003 张训练切片 /
1001 张验证切片的划分。完整训练不属于迁移回归测试的运行内容。
数据下载清单保留来源仓库的 `published=false` 状态；需要已有本地数据
或更新后的授权下载地址，迁移不会自动发布、下载或复制原始数据。

## 评估和迁移检查

```bash
inv-nasn-ct benchmark \
  --weights /path/to/nasn-pmf-2048-epoch449-ema.safetensors \
  --val-root /path/to/synthetic_val --steps 16 32 \
  --output outputs/nasn-benchmark

# CPU 回归；CUDA 测试明确跳过：
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q

# CUDA 算子、双向 autograd、可微 CG、Gridrec/TomoPy 对照：
python -m pytest tests/nasn_ct/test_cuda_integration.py -q

# 使用外部原项目作为参考，分别运行同一张真实切片：
python scripts/verify_nasn_migration.py \
  --reference-root /path/to/nasn-truncated-ct \
  --weights /path/to/artifacts/nasn-pmf-2048-epoch449-ema.safetensors \
  --sinogram-dir /path/to/artifacts/quickstart-v0.1.0/target/sinograms_rivers \
  --scale-stats /path/to/artifacts/quickstart-v0.1.0/target/scale_stats_center1172.json \
  --slice 1150 --num-steps 16 --dc-iter 10 --output outputs/nasn-migration
```

对照脚本保存双方的 TIFF、summary 和 `comparison.json`。预设门槛为图像
相对 L2 ≤1e-3、投影残差绝对差 ≤1e-4。CUDA BP 使用原子累加，独立运行
不要求逐位相等。Gridrec 对照测试在未安装 TomoPy 时会明确跳过。

主路径使用框架已有 DDS/CG，未另复制旧 `dds_sv_generate`。旧采样器中
非默认的非伴随 Gridrec 校正、monotone safeguard 和 pedestal 选项仍留在
原项目中。在线 LoRA 可复用框架现有求解器；其完整 2048 实验尚未纳入
本次基准。源代码及第三方许可见根目录 `THIRD_PARTY_NOTICES.md`。

## 本机迁移验证记录（2026-10-28）

- CPU：39 项通过；CUDA：4 项通过，含 Gridrec/TomoPy 数值对照。
- wheel 构建成功；在源码目录外检查了 CUDA 源文件、配置、许可及两个
  CLI 入口，未依赖原 `nasn_ct` 包。
- 真实切片 1150，16 步 / CG 10 次：原项目测量残差
  `0.02729678899`，框架 `0.02729744464`；图像相对 L2 差异
  `0.000668712`，通过预设迁移门槛。
- 合成切片 991，16 步：框架得到 36.6072 dB / 0.81288 SSIM，
  原项目独立运行得到 36.2930 dB / 0.80194 SSIM。
  **这项严格的跨运行指标对照没有通过门槛，不能声称逐次运行等价。**
- 隔离检查中，原始投影、模型输出及尺度转换逐值一致；记录并回放同一组
  CUDA 算子响应后，24 次算子调用的输入，以及校正后的下一步图像，
  最大绝对差均为 0。见 `scripts/verify_nasn_cg_replay.py`。
- 原项目自身重复同一 CG 校正也出现了约 0.24% 和 1.22% 的相对 L2
  波动。保留的 float32 CUDA BP 使用原子累加，数值变化会被 CG 放大；
  本次没有修改该底层数值实现或放宽原先的合成对照门槛。

完整记录在本机 `outputs/nasn-migration/validation.json`；独立结果及失败
的合成跨运行对照也一并保留。完整 500-epoch 训练、全部验证切片和真实
2048 LoRA 适配没有重新运行。
