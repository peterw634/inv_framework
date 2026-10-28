# Diffusion 逆问题求解器

本目录提供以预训练 diffusion／mean-flow 模型为先验的重建方法，统一通过
`x = solver.solve(y, A)` 调用。`y` 是观测，`A` 是前向算子，返回值为模型图像域的张量。
下面按当前实现介绍 **13 个求解器入口**，并给出各方法的原始文献。
`MeanFlowDDSSolver` 是 `DDSSolver` 的别名，计作同一种实现。

安装与统一实验配置见 [Diffusion 方法与统一对比](../../../docs/diffusion_methods.md)，
工业 CT 数据、权重及训练入口见 [NaSn 接入说明](../../../docs/nasn_ct.md)。

## 方法总览

| 方法 | 类名与源码 | 模型／采样空间 | 当前实现的数据一致性方式 |
|---|---|---|---|
| DPS | `DPSSolver` · [dps.py](dps.py) | 像素域 ε 预测 | 对去噪估计的测量误差反传，修正采样状态 |
| MCG | `MCGSolver` · [mcg.py](mcg.py) | 像素域 ε 预测，DDPM | 梯度引导后投影到带噪测量约束 |
| ΠGDM／PGDM | `PGDMSolver` · [pgdm.py](pgdm.py) | 像素域 ε 预测，DDIM | 通过伪逆构造引导方向 |
| DDS（DDIM） | `DDIMDDSSolver` · [ddim_dds.py](ddim_dds.py) | 像素域 ε 预测，DDIM | 对干净图像估计做 CG 校正，再更新采样状态 |
| DMPlug | `DMPlugSolver` · [dmplug.py](dmplug.py) | 像素域 ε 预测，DDIM | 对完整生成轨迹反传，优化初始噪声 |
| RED-Diff | `REDDiffSolver` · [reddiff.py](reddiff.py) | 像素域 ε 预测，VP 噪声系数 | 用测量损失和扩散正则项直接优化图像 |
| HybridReg | `HybridRegSolver` · [hybridreg.py](hybridreg.py) | 像素域 ε 预测，VP 噪声系数 | 结合相关噪声与混合正则化优化图像 |
| PSLD | `PSLDSolver` · [psld.py](psld.py) | 潜空间 ε 预测，DDPM + codec | 解码后的测量误差与潜变量一致性联合引导 |
| ReSample | `ReSampleSolver` · [resample.py](resample.py) | 潜空间 ε 预测，专用 DDIM + codec | 内层数据一致性优化与随机重采样 |
| DiffStateGrad | `DiffStateGradSolver` · [diffstategrad.py](diffstategrad.py) | 本实现基于 ReSample | 用中间状态的低秩子空间投影内层优化梯度 |
| mean-flow DDS | `DDSSolver`／`MeanFlowDDSSolver` · [dds.py](dds.py) | 平均速度预测，`MeanFlowSchedule` | 带先验锚点的 CG 校正与 mean-flow 更新 |
| mean-flow LoRA DDS | `LoRAAdaptiveDDSSolver` · [dds_lora.py](dds_lora.py) | mean-flow + 在环 LoRA | 通过测量残差适配模型，再执行 DDS 校正 |
| Budgeted DPS | `BudgetedDPSSolver` · [budget_dps.py](budget_dps.py) | ε 预测，DPS 工程扩展 | 按采样阶段选择测量源子集，控制梯度计算成本 |

**命名区别：**`DDIMDDSSolver` 使用噪声预测模型与 DDIM；`DDSSolver` 使用
平均速度模型与 `MeanFlowSchedule`。两者的模型输出含义、时间参数和 CG 形式不同，
不能只替换类名就共用同一套模型／调度器。

## 统一接口与输入约定

- 观测包含 batch 维，形状为 `(B, *A.range_shape)`；图像输出为 `(B, *A.domain_shape)`，
  当前图像流程通常为 `(B, C, H, W)`。模型、codec 和观测应位于相同设备。
- 噪声模型预测 `eps = model(x_t, t)`；mean-flow 模型预测 `u = model(x_t, t)`。
  原生 Diffusers UNet 用于 `DPSSolver` 时应显式包装为 `DiffusersUNetWrapper`，
  原生调度器应包装为 `DiffusersScheduleAdapter`。
- 继承 `SamplingSolver` 的方法自动完成 Diffusers 包装；其采样参数可在构造器或
  `solve` 中设置。DPS、mean-flow DDS、LoRA DDS、Budgeted DPS 的主要参数在构造器设置。
- 潜空间方法另需 `codec`，例如 [VQCodec](../../models/latent_codec.py)，提供
  `encode`、`decode` 和 `scaling_factor`；测量算子仍作用于解码后的图像。
- MCG、PGDM、DDIM DDS、RED-Diff、HybridReg、PSLD 在本实现中要求
  [LinearOperator](../../operators/base.py)。mean-flow 的默认 CG 校正及 LoRA DDS
  同样需要线性算子和显式伴随；DPS、DMPlug、ReSample 等梯度路径需要可微前向计算。
- 返回结果保留模型图像尺度。比较不同先验时，应显式对齐图像与观测的尺度，
  不能通过逐图归一化代替标定。DMPlug 内部保留了 `[-1, 1]` 截断，其权重需匹配该约定。
- 潜空间方法的初始噪声在 CPU 生成，应传 CPU `torch.Generator`；像素域方法使用
  与模型设备一致的 generator。部分方法内部仍使用全局 RNG，复现实验还需设置全局随机种子。

## 像素域梯度引导

### 1. DPS：Diffusion Posterior Sampling

先由扩散模型给出干净图像估计，再计算它与测量之间的残差，并把残差对当前带噪状态的
梯度注入反向采样。它可作为可微测量算子的通用重建基线。

本目录默认使用 `PosteriorSampling`，损失是 `||y - A(x0_hat)||₂`；
需要现有对比实验的测量归一化损失时，显式选择 `NormalizedPosteriorSampling`。
两种损失对应的 `scale` 不宜直接照搬。当前默认条件项不会仅凭传入 `noiser`
就自动切换成论文中的不同噪声似然。

主要参数：`num_inference_steps`、`scale`、`eta`、`conditioning`。

文献：Hyungjin Chung 等，
[Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)，
ICLR 2023。

### 2. MCG：Manifold-Constrained Gradient

在扩散采样中加入经过去噪模型反传的测量梯度，随后执行测量一致性投影，
以兼顾生成先验和观测约束。本实现使用 `ManifoldConstraintGradient`，
并通过 `operator.project` 向当前时间步的带噪测量投影。

主要参数：`num_inference_steps`、`scale`。需要 DDPM 调度器与线性投影算子；
`project` 的具体实现会影响结果。

文献：Hyungjin Chung、Byeongsu Sim、Dohoon Ryu、Jong Chul Ye，
[Improving Diffusion Models for Inverse Problems using Manifold Constraints](https://arxiv.org/abs/2206.00941)，
NeurIPS 2022。

### 3. ΠGDM／PGDM：Pseudoinverse-Guided Diffusion

利用测量模型的伪逆构造引导方向，使去噪图像的预测测量更接近观测。
本实现用 `A.pseudo_inverse(y)` 初始化，并通过 `PseudoInverseGuidance` 比较
观测和预测测量在伪逆映射后的差异，采用随时间变化的引导系数。

主要参数：`num_inference_steps`、`scale`。需要 `LinearOperator.pseudo_inverse`；
原论文覆盖更广的测量模型，本分支入口限定为线性算子。

文献：Jiaming Song、Arash Vahdat、Morteza Mardani、Jan Kautz，
[Pseudoinverse-Guided Diffusion Models for Inverse Problems](https://jankautz.com/publications/PiGDM_ICLR23.pdf)，
ICLR 2023。

## 像素域分解采样与优化

### 4. DDS：Decomposed Diffusion Sampler（DDIM 版本）

把扩散去噪与数据一致性分开：模型先预测干净图像，CG 在图像域校正测量残差，
随后用校正图像完成 DDIM 更新。模型预测阶段不需要像 DPS 那样对 UNet 输入反传。

入口为 `DDIMDDSSolver`，需要正投影与伴随。主要参数：`num_inference_steps`、
`eta`、`cg_inner`、`cg_eps`、`gamma`。这里保留独立的 CG 方程和整批点积行为，
其中 `gamma` 不等同于 mean-flow DDS 的 `dc_lam`。

文献：Hyungjin Chung、Suhyeon Lee、Jong Chul Ye，
[Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems](https://arxiv.org/abs/2303.05754)，
ICLR 2024。

### 5. DMPlug：把生成过程作为可微映射

将整条反向扩散轨迹视为从初始噪声到图像的生成映射，根据最终图像的测量误差，
反向传播到初始噪声并用 Adam 更新。它优化的是生成输入，模型参数不是内层优化器的更新对象。

主要参数：`num_inference_steps`、`epochs`、`lr`、`criterion`。
`epochs` 表示初始噪声的优化次数；每次都执行完整采样轨迹。
轨迹反传的显存需求随步数增长，应结合 [统一实验文档](../../../docs/diffusion_methods.md)
中的短轨迹配置使用。

文献：Hengkang Wang 等，
[DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models](https://arxiv.org/abs/2405.16749)，
NeurIPS 2024。

### 6. RED-Diff：扩散先验正则化

将逆问题写成变分／随机优化问题，用不同噪声等级的去噪模型提供图像正则化方向。
本实现从 `A.pseudo_inverse(y)` 初始化待优化图像 `mu`，把测量损失与按 SNR 加权的
噪声预测正则项组合，用 Adam 更新 `mu`；噪声预测在构造正则方向时停止梯度。

主要参数：`num_inference_steps`、`sigma`、`loss_measurement_weight`、
`loss_noise_weight`、`lr`。虽然复用 DDIM 调度器提供 VP 系数，主循环执行的是图像优化。
`sigma` 控制当前图像估计的随机扰动，不能直接解释为自动估计的测量噪声标准差。

文献：Morteza Mardani、Jiaming Song、Jan Kautz、Arash Vahdat，
[A Variational Perspective on Solving Inverse Problems with Diffusion Models](https://arxiv.org/abs/2305.04391)，
2023 年预印本。

### 7. HybridReg：Hybrid Regularization

结合去噪正则化（DR）与一致性正则化（CR）。本实现采用与 RED-Diff 相似的图像优化结构，
并将上一步预测噪声与新噪声混合，再构造下一步模型输入，使不同时间步之间产生联系。

主要参数：`num_inference_steps`、`sigma`、`loss_measurement_weight`、
`loss_noise_weight`、`lr`、`beta`。代码中混合项为
`sqrt(1-beta) * last_noise + sqrt(beta) * noise_xt`，因此 `beta` 应在 `[0, 1]` 内。
本分支使用线性算子与伪逆初始化；论文讨论的适用范围更广。

文献：Hongkun Dou 等，
[Hybrid Regularization Improves Diffusion-based Inverse Problem Solving](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fa354c1e18eb6dbea4be8735d8f34a53-Abstract-Conference.html)，
ICLR 2025。

## 潜空间方法

### 8. PSLD：潜空间后验采样

在潜空间执行扩散采样，将干净潜变量解码成图像后计算测量残差，同时引入
编码／解码及投影所形成的潜变量一致性约束，再把联合误差反传到当前潜变量。

主要参数：`num_inference_steps`、`gamma`、`omega`；其中 `omega` 加权测量项，
`gamma` 加权潜变量项。需要潜空间噪声模型、codec 和线性投影算子。
当前代码保留特定投影目标和缩放顺序，复现实验时应保持这些约定。

文献：Litu Rout 等，
[Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models](https://arxiv.org/abs/2307.00619)，
NeurIPS 2023。

### 9. ReSample：硬数据一致性与重采样

在潜空间采样过程中，按阶段优化像素图像或潜变量，使解码图像更符合测量，
再通过随机重采样将校正结果映射回带噪状态。文献中的“硬数据一致性”在当前实现中
通过有限次内层优化近似，停止精度取决于迭代上限与误差阈值。

主要参数：`num_inference_steps`、`inter_timesteps`、`pixel_lr`、`latent_lr`、
`pixel_maxiters`、`latent_maxiters`、`pixel_eps`、`latent_eps`。
需要 codec；公共采样层会使用或转换为 [ReSampleDDIMScheduler](resample_schedule.py)。
当前流程含 `t // 2` 时间索引和固定触发条件，不能把任意时间网格视为等价配置。

文献：Bowen Song 等，
[Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency](https://arxiv.org/abs/2307.08123)，
ICLR 2024。

### 10. DiffStateGrad：扩散状态引导的梯度投影

从中间扩散状态的 SVD 构造低秩子空间，把测量梯度投影到该空间后再更新，
以过滤偏离当前状态结构的梯度成分。原论文将其作为可接入多类求解器的模块。

本目录的 `DiffStateGradSolver` 继承 `ReSampleSolver`，在像素／潜空间内层优化中
应用梯度投影。主要参数：`var_cutoff`（保留的奇异值能量比例）、`period`
（梯度投影频率）、`pixel_lr`、`inter_timesteps`、`num_inference_steps`。
当前 SVD 仅取 batch 中第一个样本，秩计算保留固定除数 `3`，应优先使用 batch=1；
不能将其视为已验证的任意 batch／通道数实现。

文献：Rayhan Zirvi、Bahareh Tolooshams、Anima Anandkumar，
[Diffusion State-Guided Projected Gradient for Inverse Problems](https://arxiv.org/abs/2410.03463)，
ICLR 2025。

## 工业 CT 的 mean-flow 适配

### 11. mean-flow DDS：平均速度先验 + CG

`DDSSolver`（别名 `MeanFlowDDSSolver`）将 DDS 的“先去噪、再做数据一致性”思路
用于平均速度模型。`MeanFlowSchedule` 根据 `u` 得到 `x0_hat = x_t - t*u`，
默认 `CGDataConsistency` 近似求解：

```text
(AᵀA + λI) x_dc = Aᵀy + λ x0_hat
```

随后把校正结果重新写入下一时间步。该流程是本分支的 mean-flow 工业 CT 适配。
原生 pMF 网络使用图像预测参数化，`PMFVelocityPrior` 通过其 `u_fn` 暴露平均速度接口，
因此应使用适配器连接求解器。

主要参数：`num_inference_steps`、`dc_lam`、`dc_iter`、`eta`、`t_start`、`dc_mode`。
`dc_mode='dds'` 每步校正，`'post'` 只在末尾校正，`'none'` 不做数据一致性。
`x_init` 可配合 `t_start < 1` 热启动；`initial_noise` 可复用外部标准正态噪声。

相关文献分别对应采样、流模型和像素先验基础：

- Hyungjin Chung 等，[Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems](https://arxiv.org/abs/2303.05754)，ICLR 2024。
- Zhengyang Geng 等，[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)，2025。
- Yiyang Lu 等，[One-step Latent-free Image Generation with Pixel Mean Flows](https://arxiv.org/abs/2601.22158)，2026。

### 12. mean-flow LoRA DDS：采样过程中的先验适配

在指定采样步骤内，用若干次优化更新 LoRA 等选定参数。损失通过可微 CG 后的
测量残差构造，也可加入分布内样本的正则项；随后用更新后的模型执行 DDS。
该方法用于先验训练分布与当前扫描对象存在差异的情况。

需要先向 `model.net` 注入 LoRA，并传入 `adapt_params` 与 `adapt_optim_factory`。
主要参数：`adapt_K`、`adapt_at_steps`、`reset_optim_per_timestep`、`reg_gamma`、
`id_sampler`、`use_scale_toggle`。`adapt_at_steps=None` 表示每步适配，空列表表示关闭适配。
模型参数会在调用中被更新；独立比较不同扫描或方法时，应重新加载或复制初始权重。

这里采用 DDIP／D3IP 的测试时适配思路，并结合本分支的 mean-flow 与 CG 流程；
不代表完整实现了原论文的三维联合重建、全部适配策略或元学习实验。

相关文献：

- Hyungjin Chung、Jong Chul Ye，[Deep Diffusion Image Prior for Efficient OOD Adaptation in 3D Inverse Problems](https://arxiv.org/abs/2407.10641)，ECCV 2024。
- Edward J. Hu 等，[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)，2021 年预印本；作为低秩适配机制的基础引用。

## 计算预算扩展

### 13. Budgeted DPS：按阶段分配测量源预算

将反向采样分为若干阶段，每个阶段选择一个测量源子集，只针对该子集计算残差和梯度。
本模块来自框架的 USCT／FTTT 预算实验，要求算子实现
[SubsettableForwardOperator](../../operators/budget.py) 的 `subset` 与 `subset_measurement`。

主要参数：`schedule_actions`、`key_indices`、`num_inference_steps`、`scale`。
`make_equispaced_schedule` 可构造均匀覆盖孔径的子集计划，`grad_cost` 估算相对源数量成本，
`last_conditioning` 保存实际源数统计和残差；这些统计不等于实测耗时。
`reformulated_gradient` 提供原始／归一化梯度变换工具，但不会由 `BudgetedDPSSolver`
自动调用，需要实验流程显式接入。

这是基于 DPS 的工程扩展，当前代码未标注独立发表论文。基础文献沿用 Chung 等的
[Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)。
当前统一比较脚本的 12 个方法键不包含此预算扩展。

## 调用示例

以下示例从仓库根目录运行。`y`、`A` 和权重由调用方提供；模型分辨率、训练尺度与
算子图像域应匹配。

```sh
pip install -e '.[diffusion]'
```

DDIM DDS 示例：

```python
from diffusers import DDIMScheduler, UNet2DModel
from inv_framework.solvers.diffusion import DDIMDDSSolver

weights = '/path/to/diffusion-checkpoint'
model = UNet2DModel.from_pretrained(weights, subfolder='unet').to(y.device).eval()
model.requires_grad_(False)
schedule = DDIMScheduler.from_pretrained(weights, subfolder='scheduler')
solver = DDIMDDSSolver(
    model, schedule, num_inference_steps=100, eta=0.85,
    cg_inner=5, gamma=0,
)
x = solver.solve(y, A)
```

mean-flow DDS 示例（`pmf` 为已加载权重、置于目标设备的 pMF 模型）：

```python
from inv_framework.solvers.diffusion import MeanFlowDDSSolver, PMFVelocityPrior

prior = PMFVelocityPrior(pmf)
solver = MeanFlowDDSSolver(
    prior, num_inference_steps=16, dc_lam=0.05, dc_iter=10,
)
x = solver.solve(y, A)
```

涉及梯度引导、DMPlug 轨迹优化或 LoRA 适配的调用需要保留 autograd；
`model.eval()` 与禁止梯度是不同操作。

## 支持模块与基础文献

| 文件 | 职责 |
|---|---|
| [sampling.py](sampling.py) | 新采样器的公共封装、形状检查、参数检查与 codec 接口 |
| [scheduler.py](scheduler.py) | `NoiseSchedule`、`VPSchedule`、`MeanFlowSchedule` |
| [diffusers_compat.py](diffusers_compat.py) | Diffusers 模型输出与调度器适配 |
| [conditioning.py](conditioning.py) | `PosteriorSampling`、`CGDataConsistency` |
| [guidance.py](guidance.py) | 归一化 DPS、MCG、伪逆引导条件项 |
| [resample_schedule.py](resample_schedule.py) | 带 pseudo-clean 预测的 ReSample DDIM 调度器 |
| [velocity_prior.py](velocity_prior.py) | 速度先验协议与仅供连通性验证的 `TinyVelocityPrior` |
| [pmf_compat.py](pmf_compat.py) | `PMFVelocityPrior`，连接真实 pMF 与 mean-flow 求解器 |
| [lora.py](lora.py) | Linear／Conv2d LoRA 注入、pMF 适配配方及 LoRA 状态保存／加载 |
| [__init__.py](__init__.py) | 公共导出与兼容别名 |

调度器的基础文献：Jonathan Ho 等，
[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)（2020）；
Jiaming Song、Chenlin Meng、Stefano Ermon，
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)（ICLR 2021）。
mean-flow 的基础文献见上文；支持模块本身不另计为独立重建方法。

## 验证与实现范围

统一示例可验证像素域、潜空间、mean-flow 与 LoRA 的接口连通性：

```sh
python examples/diffusion_ct_compare.py --smoke --output outputs/diffusion-smoke
```

该命令使用小模型与简化算子，不是使用真实权重的图像质量评测。
已有 CPU 对照、真实权重 GPU 对照及未通过的严格一致性项目见
[验证记录](../../../docs/diffusion_validation.md)。
归一化测量损失对全零观测的限制、潜变量缩放、时间索引及 batch 限制见
[完整实现说明](../../../docs/diffusion_methods.md)。

本页引用各算法的原始论文；具体迁移代码和第三方许可保留在
[第三方来源说明](../../../THIRD_PARTY_NOTICES.md)。
文献题录与实现接口核对日期：2026-10-28。
