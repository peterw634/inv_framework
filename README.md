# inv_framework

一个轻量的反问题(inverse problem)通用求解框架。支持线性反问题(CT、MRI、去模糊、超分辨等)与非线性反问题(相位恢复、饱和探测、盲去模糊等),解耦"前向算子"、"噪声模型"与"求解器"三层抽象。CT 作为内置示例完整提供。扩散类求解器与 HuggingFace `diffusers` 完全兼容(可直接用 `UNet2DModel` / `DDIMScheduler`)。

---

## 目录

- [设计理念](#设计理念)
- [安装](#安装)
- [文件结构](#文件结构)
- [核心抽象](#核心抽象)
- [快速开始](#快速开始)
- [求解器一览](#求解器一览)
- [与 diffusers 集成](#与-diffusers-集成)
- [扩展:添加新的反问题](#扩展添加新的反问题)
- [测试](#测试)

---

## 设计理念

反问题统一形式为 `y = A(x) + n`,其中:
- `A` 是**前向算子**(forward operator),线性或非线性
- `n` 是噪声(Gaussian / Poisson / 自定义)
- 目标:从 `y` 估计 `x`

本框架的两层算子抽象:

```
ForwardOperator (ABC)          ← 通用基类(线性 + 非线性)
  └── LinearOperator (ABC)     ← 线性算子,附带 adjoint / pseudo_inverse
        ├── ParallelBeamRadon2D
        └── ASTRAOperator3D
```

求解器据其需求选取算子类型:

| 求解器类别 | 接受类型 | 原因 |
|---|---|---|
| FBP / SIRT / Landweber | `LinearOperator` | 显式需要 `A^T` |
| DIP / INR / DPS | 任意 `ForwardOperator` | 只用 `forward` + autograd 求梯度 |

新增反问题只需写一个 `ForwardOperator` 子类,**求解器代码不需要改动**。

---

## 安装

依赖:
- 必需: `torch >= 2.0`
- 可选: `matplotlib`(画图)、`diffusers >= 0.30`(扩散类求解器换更强 UNet / scheduler 时需要)、`astra-toolbox`(3D CT 后端)

```bash
cd /path/to/inv_framework
# 直接以源码方式运行 (项目内自动注入 sys.path)
python examples/ct_demo.py

# 或安装为包(可选)
pip install -e .   # 当前仓库未提供 pyproject.toml,需自行添加
```

---

## 文件结构

```
inv_framework/
├── README.md                                    # 本文档
├── inv_framework/                               # 主包
│   ├── __init__.py
│   ├── operators/                               # 前向算子层
│   │   ├── base.py                              # ForwardOperator + LinearOperator (ABCs)
│   │   ├── noise.py                             # NoiseModel + GaussianNoise / PoissonLogDomainNoise / NoNoise
│   │   └── ct/                                  # CT 域算子
│   │       ├── radon_torch.py                   # ParallelBeamRadon2D (纯 torch,默认)
│   │       └── astra_adapter.py                 # ASTRAOperator3D (可选,需 astra-toolbox)
│   ├── solvers/                                 # 求解器层
│   │   ├── base.py                              # InverseProblemSolver (ABC)
│   │   ├── classical.py                         # fbp / sirt / landweber 函数 + 同名 Solver 类
│   │   ├── dip.py                               # DIPSolver
│   │   ├── inr.py                               # INRSolver
│   │   └── diffusion/                           # 扩散类方法(diffusers 风格 API)
│   │       ├── scheduler.py                     # NoiseSchedule (ABC) + VPSchedule
│   │       ├── conditioning.py                  # ConditioningMethod (ABC) + PosteriorSampling (DPS)
│   │       ├── dps.py                           # DPSSolver
│   │       └── diffusers_compat.py              # DiffusersScheduleAdapter + DiffusersUNetWrapper
│   ├── models/                                  # 网络模块(被求解器内部调用)
│   │   ├── siren.py                             # SIREN (INR)
│   │   ├── skip_unet.py                         # SkipUNet (DIP)
│   │   └── tiny_unet.py                         # TinyUNet (扩散先验)
│   └── utils/
│       └── metrics.py                           # psnr / ssim
├── examples/
│   ├── ct_demo.py                               # 线性 CT:FBP / SIRT / DIP / INR / DPS (支持 --dps-use-diffusers)
│   ├── train_diffusion_for_dps.py               # 训练 DPS 用的 ε-预测器 (支持 --use-diffusers)
│   └── nonlinear_demo.py                        # 非线性 CT 演示(饱和探测器,tanh)
└── tests/
    ├── test_radon_adjoint.py                    # backward(forward) == adjoint;FBP 重建圆盘
    └── test_nonlinear_operator.py               # 非线性算子端到端 + 经典求解器正确拒绝
```

---

## 核心抽象

### `ForwardOperator` (operators/base.py)

```python
class ForwardOperator(ABC):
    domain_shape: tuple   # x 的单样本形状,不含 batch
    range_shape:  tuple   # y 的单样本形状

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def __call__(self, x): return self.forward(x)
```

**要求**:`forward` 必须 autograd-traceable(支持反传),这样 DIP / INR / DPS 才能用自动微分求 `dL/dx`。

### `LinearOperator (ForwardOperator)`

```python
class LinearOperator(ForwardOperator):
    @abstractmethod
    def adjoint(self, y: Tensor) -> Tensor: ...        # 显式 A^T

    def pseudo_inverse(self, y, **kwargs): ...         # 默认 Landweber
    def project(self, x, y, step_size=None): ...       # 数据一致投影
```

**约定**:子类应保证 `forward` 的 autograd backward **等于** `adjoint`(本框架的 `ParallelBeamRadon2D` 用自定义 `torch.autograd.Function` 强制成立,见 `tests/test_radon_adjoint.py`)。这样基于梯度的求解器与基于显式 `A^T` 的求解器一致。

### `NoiseModel` (operators/noise.py)

```python
class NoiseModel(nn.Module):
    @abstractmethod
    def forward(self, y_clean: Tensor) -> Tensor: ...
```

内置:`NoNoise`、`GaussianNoise(sigma)`、`PoissonLogDomainNoise(photon_count, transmittance_target)`(对数域泊松)。

### `InverseProblemSolver` (solvers/base.py)

```python
class InverseProblemSolver(ABC):
    @abstractmethod
    def solve(self, measurement, operator, **kwargs) -> Tensor: ...
```

所有求解器接口统一,可以互换比较。

### `NoiseSchedule` / `ConditioningMethod` (solvers/diffusion/)

扩散方法的两个 hook,API 与 HuggingFace `diffusers` 一致:

```python
class NoiseSchedule(ABC):                            # VP / VE / EDM 等
    num_train_timesteps: int
    timesteps: torch.Tensor                          # 反向过程时间步序列

    @abstractmethod
    def set_timesteps(self, num_inference_steps, device=None): ...
    @abstractmethod
    def add_noise(self, x0, noise, t): ...
    @abstractmethod
    def step(self, eps, t, x_t, eta=0.0):            # 返回 (x_prev, x_0_pred)
        ...

class ConditioningMethod(ABC):                       # DPS / MCG / PiGDM / RED / ...
    @abstractmethod
    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs): ...
```

内置:
- `VPSchedule`:DDPM 线性 beta,无第三方依赖
- `PosteriorSampling`:DPS (Chung et al. 2023)
- `DiffusersScheduleAdapter`:将任意 `diffusers` scheduler 包装成 `NoiseSchedule`
- `DiffusersUNetWrapper`:将 `diffusers.UNet2DModel` 适配 `(x, t) → eps` 张量返回

DPSSolver 同时接受两种 schedule,代码无需改动。

---

## 快速开始

### 1. 线性 CT 演示

```bash
python examples/ct_demo.py --image-size 128 --num-angles 30 --noise-sigma 0.5
```

输出(GPU,约 1 分钟):

```
Method              PSNR (dB)     SSIM
FBP                     19.68     0.25
SIRT                    22.72     0.66
DIP                     20.24     0.46
INR                     22.38     0.43
```

可选:训练扩散先验后使用 DPS

```bash
# 使用内置 TinyUNet + VPSchedule(无第三方依赖)
python examples/train_diffusion_for_dps.py --steps 5000 --out trained_tiny_unet.pt
python examples/ct_demo.py --dps-checkpoint trained_tiny_unet.pt

# 或使用 diffusers UNet2DModel + DDPMScheduler(质量更高)
python examples/train_diffusion_for_dps.py --use-diffusers --steps 5000 --out trained_diffusers_unet
python examples/ct_demo.py --dps-use-diffusers --dps-checkpoint trained_diffusers_unet
```

### 2. 非线性 CT 演示(饱和探测器)

```bash
python examples/nonlinear_demo.py --alpha 2.0 --num-angles 60
```

输出:

```
Trying FBP on the non-linear operator (should raise)...
  caught TypeError: fbp requires a LinearOperator ...

Method              PSNR (dB)     SSIM
DIP                     20.36     0.49
INR                     24.93     0.42
```

FBP / SIRT 自动拒绝非线性算子并提示用户改用 DIP / INR / DPS。

### 3. 编程式接口

```python
import torch
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.operators.noise import GaussianNoise
from inv_framework.solvers.classical import FBPSolver, SIRTSolver
from inv_framework.solvers.dip import DIPSolver

device = 'cuda'
A = ParallelBeamRadon2D(image_size=128, num_angles=30, device=device)
noiser = GaussianNoise(sigma=0.5)

x_true = my_phantom().to(device)              # (1, 1, 128, 128)
y = noiser(A(x_true))                          # (1, 1, 30, 128)

x_fbp  = FBPSolver().solve(y, A)
x_sirt = SIRTSolver(num_iterations=100, min_value=0, max_value=1).solve(y, A)
x_dip  = DIPSolver(num_iterations=1000, lr=1e-3).solve(y, A)
```

---

## 求解器一览

| 类 | 文件 | 关键超参 | 备注 |
|---|---|---|---|
| `FBPSolver` | `solvers/classical.py` | `scale` | 仅适用 parallel-beam Radon 及自定义 BP 算子 |
| `SIRTSolver` | 同上 | `num_iterations`, `min_value`, `max_value` | 经典迭代,行/列归一化 |
| `LandweberSolver` | 同上 | `num_iterations`, `step_size` | 普通梯度下降 |
| `DIPSolver` | `solvers/dip.py` | `num_iterations`, `lr`, `input_channels`, `model_factory` | 默认用 `SkipUNet`,可注入自定义网络 |
| `INRSolver` | `solvers/inr.py` | `num_iterations`, `lr`, `hidden_features`, `hidden_layers` | 默认用 `SIREN` |
| `DPSSolver` | `solvers/diffusion/dps.py` | `model`, `schedule`, `num_inference_steps`, `scale`, `eta` | 需要训练好的 ε-预测器 |

### DIP / INR 自定义网络

```python
from inv_framework.solvers.dip import DIPSolver

def my_factory():
    return MyCustomCNN(...)            # 返回 nn.Module: z -> x

solver = DIPSolver(model_factory=my_factory, num_iterations=2000, lr=1e-3)
x_rec = solver.solve(y, A)
```

### DPS 自定义 conditioning

```python
from inv_framework.solvers.diffusion.conditioning import ConditioningMethod
from inv_framework.solvers.diffusion.dps import DPSSolver

class MyMCG(ConditioningMethod):
    def apply(self, x_t, x_0_hat, measurement, x_prev, **kw):
        # 梯度项 + 投影项 ...
        return x_t_new, residual

solver = DPSSolver(model=my_eps, conditioning=MyMCG(A, scale=1.0))
x_rec = solver.solve(y, A)
```

---

## 与 diffusers 集成

扩散类求解器与 HuggingFace `diffusers` 兼容,使用方式上有两条路径。

### 路径 A:用 diffusers 的 scheduler 与 UNet 直接做 DPS

```python
import torch
from diffusers import UNet2DModel, DDIMScheduler
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.diffusion.dps import DPSSolver
from inv_framework.solvers.diffusion.diffusers_compat import (
    DiffusersScheduleAdapter, DiffusersUNetWrapper,
)

device = 'cuda'
A = ParallelBeamRadon2D(image_size=128, num_angles=30, device=device)
y = ...                                                # 测量数据 (1, 1, 30, 128)

unet = UNet2DModel.from_pretrained('path/to/trained_unet').to(device)
ddim = DDIMScheduler(num_train_timesteps=1000)

solver = DPSSolver(
    model=DiffusersUNetWrapper(unet),                  # 把 .sample 输出适配成张量
    schedule=DiffusersScheduleAdapter(ddim),           # 把 diffusers scheduler 适配成 NoiseSchedule
    num_inference_steps=200, scale=1.0, eta=0.0,
)
x_rec = solver.solve(y, A)
```

### 路径 B:训练脚本一键切换

`examples/train_diffusion_for_dps.py` 加 `--use-diffusers` 即用 `UNet2DModel + DDPMScheduler`(默认 ~24M 参数;块结构含 attention),否则用内置 `TinyUNet + VPSchedule`(~1M 参数,纯 torch 无依赖)。

```bash
# diffusers 后端(模型保存为目录,可被 .from_pretrained 读取)
python examples/train_diffusion_for_dps.py --use-diffusers --steps 5000 --out trained_diffusers_unet
python examples/ct_demo.py --dps-use-diffusers --dps-checkpoint trained_diffusers_unet

# 内置后端(模型保存为 .pt)
python examples/train_diffusion_for_dps.py --steps 5000 --out trained_tiny_unet.pt
python examples/ct_demo.py --dps-checkpoint trained_tiny_unet.pt
```

### 适配器细节

- `DiffusersScheduleAdapter(scheduler)`:转发 `set_timesteps` / `add_noise` / `step`,从 `step` 输出读取 `pred_original_sample` 作为 Tweedie 估计(若 scheduler 不输出则用 `alphas_cumprod` 手算)。支持 DDIM 的 `eta` 参数。
- `DiffusersUNetWrapper(unet)`:把 `UNet2DModel.forward(...).sample` 包装成纯张量返回,并在 t 是 scalar 时自动广播到 batch 维度。
- `DPSSolver` 代码不需要任何分支:对内置 `VPSchedule` 与 `DiffusersScheduleAdapter` 表现一致,因为两者都满足同一个 `NoiseSchedule` ABC。

---

## 扩展:添加新的反问题

### 情形 A:新的线性算子

例:MRI(欠采样傅里叶)

```python
import torch
from inv_framework.operators.base import LinearOperator

class UnderSampledFourier2D(LinearOperator):
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask                                       # (H, W) bool
        self.domain_shape = (2, image_size, image_size)        # real + imag
        self.range_shape  = (2, image_size, image_size)

    def forward(self, x):
        z = torch.complex(x[:, 0], x[:, 1])
        Y = torch.fft.fft2(z) * self.mask
        return torch.stack([Y.real, Y.imag], dim=1)

    def adjoint(self, y):
        Y = torch.complex(y[:, 0], y[:, 1]) * self.mask
        z = torch.fft.ifft2(Y) * (Y.numel())                   # 与 FFT 一致的转置约定
        return torch.stack([z.real, z.imag], dim=1)
```

写完即可被 `FBPSolver` 之外所有求解器使用(FBP 是 CT 专属滤波)。

### 情形 B:非线性算子

例:相位恢复 `y = |F x|²`

```python
import torch
from inv_framework.operators.base import ForwardOperator

class PhaseRetrieval(ForwardOperator):
    def __init__(self, image_size):
        self.domain_shape = (1, image_size, image_size)
        self.range_shape  = (1, image_size, image_size)

    def forward(self, x):
        return torch.fft.fft2(x).abs() ** 2
```

无需任何 `adjoint`,直接用 `DIPSolver` / `INRSolver` / `DPSSolver`。`FBPSolver` 等线性求解器会自动报错并提示。

### 情形 C:新的噪声模型

```python
from inv_framework.operators.noise import NoiseModel
import torch

class SaltPepperNoise(NoiseModel):
    def __init__(self, p): super().__init__(); self.p = p
    def forward(self, y):
        mask = torch.rand_like(y) < self.p
        flip = torch.rand_like(y) < 0.5
        return torch.where(mask, flip.float(), y)
```

### 情形 D:新的扩散 conditioning

继承 `ConditioningMethod`,实现 `apply`,见上一节"DPS 自定义 conditioning"。

---

## 测试

```bash
python tests/test_radon_adjoint.py
python tests/test_nonlinear_operator.py
```

测试内容:

- **`test_radon_adjoint.py`**
  - `backward(forward) == adjoint` (相对误差应为 0)
  - FBP 在圆盘 phantom 上 PSNR > 20 dB(实测 ~29 dB)

- **`test_nonlinear_operator.py`**
  - 非线性 `ForwardOperator` 子类可正常实例化与前传
  - `fbp` / `sirt` / `landweber` 对非线性算子均抛出 `TypeError` 并附说明
  - `DIPSolver` 在非线性算子上能将测量残差降低 10× 以上

---

## License

MIT
