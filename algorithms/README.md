# 算法索引

每个算法一张卡片（`<注册名>/README.md`），按《算法提交规范》包含算法身份、输入输出、运行方式与参数。
统一入口为 `examples/run.py`；程序化接口为 `inv_framework.registry`（`ALGORITHMS`、`defaults(name)`、`build(name, operator, params)`）。

| 算法 | 注册名 | 类型 | 预训练权重 | 同步辐射平行束 toy PSNR (dB) | 实验室锥/扇束 toy PSNR (dB) |
| --- | --- | --- | --- | --- | --- |
| [FBP / FDK（滤波反投影）](fbp/README.md) | `fbp` | 非学习类 | 无 | 17.92 `toy_parallel_ct_128`<br>25.77 `toy_parallel_ct_3d` | 20.42 `toy_cone_ct_3d` |
| [SIRT（联合迭代重建）](sirt/README.md) | `sirt` | 非学习类 | 无 | 24.16 `toy_parallel_ct_128`<br>29.00 `toy_parallel_ct_3d` | 25.73 `toy_cone_ct_3d` |
| [Landweber 迭代](landweber/README.md) | `landweber` | 非学习类 | 无 | 24.80 `toy_parallel_ct_128`<br>29.52 `toy_parallel_ct_3d` | 27.11 `toy_cone_ct_3d` |
| [CG-Tikhonov](cg_tikhonov/README.md) | `cg_tikhonov` | 非学习类 | 无 | 20.25 `toy_parallel_ct_128`<br>27.55 `toy_parallel_ct_3d` | 22.62 `toy_cone_ct_3d` |
| [DIP（Deep Image Prior）](dip/README.md) | `dip` | 非学习类 | 无 | 25.81 `toy_parallel_ct_128` | 23.78 `toy_lodoind_fan_512` |
| [INR（SIREN 隐式神经表示）](inr/README.md) | `inr` | 非学习类 | 无 | 27.09 `toy_parallel_ct_128` | 24.40 `toy_lodoind_fan_512` |
| [DPS（Diffusion Posterior Sampling）](dps/README.md) | `dps` | 学习类 | dm4ct-pixel | 22.25 `toy_lodoind_512` | 23.14 `toy_lodoind_fan_512` |
| [MCG（Manifold-Constrained Gradient）](mcg/README.md) | `mcg` | 学习类 | dm4ct-pixel | 20.35 `toy_lodoind_512` | 23.61 `toy_lodoind_fan_512` |
| [ΠGDM（Pseudoinverse-Guided Diffusion）](pgdm/README.md) | `pgdm` | 学习类 | dm4ct-pixel | 22.72 `toy_lodoind_512` | 24.06 `toy_lodoind_fan_512` |
| [DDS（Decomposed Diffusion Sampler，DDIM 版）](ddim_dds/README.md) | `ddim_dds` | 学习类 | dm4ct-pixel | 23.05 `toy_lodoind_512` | 25.09 `toy_lodoind_fan_512` |
| [DMPlug](dmplug/README.md) | `dmplug` | 学习类 | dm4ct-pixel | 19.83 `toy_lodoind_512` | 19.78 `toy_lodoind_fan_512` |
| [RED-diff](reddiff/README.md) | `reddiff` | 学习类 | dm4ct-pixel | 21.74 `toy_lodoind_512` | 21.97 `toy_lodoind_fan_512` |
| [HybridReg](hybridreg/README.md) | `hybridreg` | 学习类 | dm4ct-pixel | 21.60 `toy_lodoind_512` | 21.74 `toy_lodoind_fan_512` |
| [PSLD（潜空间后验采样）](psld/README.md) | `psld` | 学习类 | dm4ct-latent | 19.44 `toy_lodoind_512` | 16.97 `toy_lodoind_fan_512` |
| [ReSample](resample/README.md) | `resample` | 学习类 | dm4ct-latent | 19.89 `toy_lodoind_512` | 8.50 `toy_lodoind_fan_512` |
| [DiffStateGrad](diffstategrad/README.md) | `diffstategrad` | 学习类 | dm4ct-latent | 20.44 `toy_lodoind_512` | 15.12 `toy_lodoind_fan_512` |
| [pMF-DDS（mean-flow 先验 + 每步 CG）](pmf_dds/README.md) | `pmf_dds` | 学习类 | pmf | 44.54（ROI 64.05） `toy_nasn_pmf_2048` | 66.24 `toy_nasn_pmf_fan_2048` |
| [pMF-LoRA-DDS（在环 LoRA 自适应）](pmf_lora_dds/README.md) | `pmf_lora_dds` | 学习类 | pmf | 59.17（ROI 67.65） `toy_nasn_pmf_2048` | 62.01 `toy_nasn_pmf_fan_2048` |

数值来自 `scripts/run_toy_benchmarks.py`（推荐默认参数、种子 0），原始记录见 [results.json](results.json)。toy 数据仅用于 smoke 测试，不同数据包之间的数值不可直接比较。
