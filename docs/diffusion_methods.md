# Diffusion 重建方法

全部求解器使用 `solve(measurement, operator, **kwargs) -> Tensor`。
输入包含 batch 维，输出为模型图像域的 `(B,C,H,W)` 张量，保留计算设备，
不执行显示用的截断、PIL 转换或逐图 min/max 归一化。
首轮复现基准为 batch=1、float32。模型应事先放到与测量相同的设备。

## 方法与调用

| 方法 | 入口 | 调度器/依赖 |
|---|---|---|
| DPS | `DPSSolver` + `NormalizedPosteriorSampling` | DDPM，显式选择参考损失 |
| MCG | `MCGSolver` | DDPM；线性算子 `project` |
| PGDM | `PGDMSolver` | DDIM；线性算子 `pseudo_inverse` |
| DDS | `DDIMDDSSolver` | DDIM；线性算子和原版 CG |
| DMPlug | `DMPlugSolver` | DDIM；贯穿采样轨迹优化初始噪声 |
| RED-Diff | `REDDiffSolver` | VP 系数；伪逆初始化与 Adam |
| HybridReg | `HybridRegSolver` | VP 系数；相关噪声与 Adam |
| PSLD | `PSLDSolver` | DDPM、VQCodec、线性投影 |
| ReSample | `ReSampleSolver` | ReSample DDIM、VQCodec |
| DiffStateGrad | `DiffStateGradSolver` | ReSample + SVD 梯度投影 |
| pMF DDS | `MeanFlowDDSSolver` | `PMFVelocityPrior`、MeanFlowSchedule；`DDSSolver` 仍有效 |
| pMF LoRA DDS | `LoRAAdaptiveDDSSolver` | 在环 LoRA，沿用现有接口 |

```bash
pip install -e '.[diffusion,test]'
python examples/diffusion_ct_compare.py --smoke --output outputs/diffusion-smoke
```

smoke 使用未训练小网络和精确矩阵算子，只验证接口、优化和评价链路。
正式使用时传入预训练模型及 CT 算子：

```python
from diffusers import UNet2DModel, DDIMScheduler
from inv_framework.solvers.diffusion import DDIMDDSSolver

model = UNet2DModel.from_pretrained(weights, subfolder="unet").to(y.device).eval()
schedule = DDIMScheduler.from_pretrained(weights, subfolder="scheduler")
solver = DDIMDDSSolver(model, schedule, num_inference_steps=100, eta=0.85,
                      cg_inner=5, gamma=0)
x = solver.solve(y, A, generator=generator)
```

新求解器接受原方法的超参数，既可在构造器设置，也可在 `solve` 覆盖。
未知参数会报错。DPS 沿用原有构造器接口；旧默认损失保持不变。
需要归一化参考损失时，显式传入 `NormalizedPosteriorSampling(A, scale=...)`。

潜空间模型使用 `inv_framework.models.latent_codec.VQCodec(vqvae)`。
其 encode/decode 返回原始张量，缩放因子由采样算法使用。
潜空间初始化遵循 CPU 随机数流，因此其 `generator` 应为 CPU generator；
像素域使用与模型设备对应的 generator。完全复现还须重置全局 torch RNG，
因为参考算法的部分随机操作使用 `torch.randn_like`。对比入口已处理这些步骤。

`DiffusersScheduleAdapter.step()` 保留二元组返回，`step_output()` 提供完整预测。
新增采样器使用 VP/DDPM/DDIM 系列；mean-flow 使用原有独立接口。
不支持直接将 epsilon 模型放进 mean-flow 调度器。

MCG/PGDM/PSLD 等方法的参考算子行为由
`SIRTProjectionOperator(A, iterations=100)` 提供：正投影复用 A，投影采用 R/C
行列权重，伪逆采用逐样本 SIRT。默认 `LinearOperator.project` 和 Landweber
伪逆仍然可选，但它们对应不同实验设置。DDS 保留独立 CG 计算，未替换为
NaSn 的 Tikhonov CG。

## 统一比较

```bash
python examples/diffusion_ct_compare.py \
  --case examples/configs/diffusion_nasn_real.json --device cuda \
  --output outputs/diffusion-real
```

先替换配置中的权重与数据路径。权重加载仅访问本地文件；目录中包含
`unet/`、`scheduler/`，潜空间方法另需 `vqvae/`。NaSn 权重沿用 safetensors
及同名 JSON。模型目录与网络配置定义了实际使用的 checkpoint。

`methods` 的键为 `dps, mcg, pgdm, dds, dmplug, reddiff, hybridreg, psld,
resample, diffstategrad, meanflow, meanflow_lora`；每项配置：

- `weights`：权重目录或 pMF 文件；`parameters`：方法超参数。
- `image_scale`、`image_offset`：`x_phys = scale*x_model + offset`，默认 1、0。
- `model_grid`：仅针对全卷积 UNet 的显式实验分辨率覆盖，不改变磁盘权重。
  在新尺寸/新数据域上运行属于迁移实验，质量结果不能当作原论文复现。
- `sirt_iterations`：默认 100；`seed` 可覆盖全局 seed。
- `meanflow_lora` 支持 `lora_rank`，`parameters.adapt_lr` 默认 1e-4。

统一输入由 `data.kind` 指定：`nasn-real`、`nasn-synthetic` 或 `measurement`。
前两者沿用 NaSn 发布几何；后者接受 `geometry`（TomoPyCUDARadon2D 构造参数）、
测量 TIFF 路径及可选真值 TIFF 路径。均不自动调整图像尺寸。

线性测量按 `(y - A(offset*1))/scale` 转换到模型域，重建后转换回公共图像域。
NaSn 合成真值取 `(stored_image+1)/2`，pMF 的映射为 `(0.5,0.5)`；
其他权重的映射须根据其训练数据约定显式设置。示例中的默认值不代表跨域标定。
真实测量使用原有标定，没有真实真值，报告测量残差而不报告 PSNR/SSIM。

结果写入各方法 TIFF 和 `summary.json`，包括配置、版本、seed、耗时、网络调用
次数、残差及可用的图像指标。ROI 的 PSNR 使用圆形掩膜，SSIM 使用同一掩膜下的
局部 SSIM 图平均值；data_range 取固定参考图的 max-min。
某方法失败时记录原因并继续其余方法，进程最终返回非零状态。

原实验常用预设（不同于部分函数构造默认值）：

| 方法 | 参考实验参数 |
|---|---|
| DPS / MCG | 1000 步，scale 分别 10 / 2 |
| PGDM | 100 步，scale=0.3 |
| DDS | 100 步，eta=0.85，cg_inner=5，gamma=0 |
| DMPlug | 3 步，epochs=1000，lr=0.005；完整反传显存需求较高 |
| RED-Diff / HybridReg | 200 步，sigma=1e-4，measurement weight=0.5，noise weight=10000，lr=0.099；HybridReg beta=0.9999 |
| PSLD | 1000 步，gamma=0.8，omega=0.2 |
| ReSample / DiffStateGrad | 200 步，pixel_lr=1e-4，latent_lr=0.1；后者 var_cutoff=0.99 |

## 数值一致性与已知行为

```bash
python scripts/verify_diffusion_methods.py --reference /path/to/reference-checkout \
  --deep --output outputs/diffusion-validation/deep.json
DIFFUSION_REFERENCE=/path/to/reference-checkout python -m pytest tests/diffusion -q
```

验证工具仅在测试进程加载外部参考实现，安装后的框架完全独立。
深分支测试使用 30 步采样，并将两边内部优化限制为 2 次，以覆盖像素、潜空间
优化及重采样。比较调度器输出和最终图像，CPU 门槛 rtol=1e-5、atol=1e-6。
真实权重测试使用 `--checkpoint`；潜空间另加 `--latent-only`。
`--repeat-reference` 用于测量参考实现自身的重复波动。
真实权重 DMPlug 检查限制为 1 个扩散步、2 次优化，以降低显存；并非完整实验。

以下行为为保持参考结果而保留：测量归一化损失的规约顺序；全零测量报错；
PGDM 的初始化噪声系数和末步负索引；DDS 的 gamma 方程及全 batch 点积；
PSLD 的投影目标与潜变量缩放；ReSample 的 `t//2` 索引及 pseudo-clean 公式；
DiffStateGrad 只对首个 batch 求 SVD，并使用固定除数 3 计算秩。
这些限制不能被理解为对所有 batch、尺度和时间网格的正确性保证。

本次接入的实际结果与 GPU 限制见 [验证记录](diffusion_validation.md)。

## 实现来源

参考实现快照：
[研究代码，49b3e59](https://github.com/DM4CT/DM4CT/tree/49b3e5907178b56338d55c17944de17168a7a0a1)。
各方法归属于其原作者；代码来源和调度器许可见根目录第三方说明。
算法参考包括 [PSLD](https://github.com/LituRout/PSLD)、
[ReSample](https://github.com/soominkwon/resample)、
[DiffStateGrad](https://diffstategrad.github.io/)。
