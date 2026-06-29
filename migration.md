# USCT / FTTT Budgeted DPS 接入说明

本文档说明 `inv_framework` 中新增的 USCT 与 FTTT 后端接口。范围只包含
**DPS 无监督采样**、source-budgeted DPS 以及 reformulated-gradient 变体。

## 1. 新增模块

后端场景 registry：

```python
from inv_framework.applications.boed_budget_ladder import (
    list_scenarios,
    get_scenario,
    list_algorithms,
    get_algorithm,
    build_external_job,
    run_external_job,
)
```

预算版 DPS 原语：

```python
from inv_framework.solvers.diffusion.budget_dps import (
    BudgetedDPSSolver,
    BudgetedPosteriorSampling,
    equispaced_subset,
    make_equispaced_schedule,
    grad_cost,
    reformulated_gradient,
)
```

source-subset 算子抽象：

```python
from inv_framework.operators import (
    SubsettableForwardOperator,
    CallableSubsetForwardOperator,
)
```

`SubsettableForwardOperator` 是给 USCT / FTTT 这类“source/shot 可选”的问题用的：

```python
class SubsettableForwardOperator(ForwardOperator):
    num_sources: int
    source_axis: int = 1

    def subset(self, source_indices): ...
    def subset_measurement(self, measurement, source_indices): ...
```

DPS solver 不需要知道 CBS 或 pykonal 的细节，只需要在每个 reverse step 调用当前 source subset 对应的 forward operator。

## 2. 暴露给 inverse_problem_agent 的接口

Agent 侧推荐只包装下面几个后端函数。

| Agent tool name | 后端函数 |
|---|---|
| `list_inverse_problem_scenarios` | `list_scenarios()` |
| `describe_inverse_problem_scenario` | `get_scenario(name)` |
| `list_inverse_problem_algorithms` | `list_algorithms()` |
| `build_inverse_problem_job` | `build_external_job(...)` |
| `run_inverse_problem_job` | `run_external_job(...)` |

`build_external_job` 只负责构造现有 `Diffusion_Policy` 推理脚本命令。

USCT 推理 job：

```python
from inv_framework.applications.boed_budget_ladder import build_external_job

job = build_external_job(
    "usct",
    task="infer",
    sample=24,
    split="eval",
    lambda_g=3e-5,
    J=4,
    cbs_iters=300,
    stop_tol=0.03,
)
print(job["command"])
```

FTTT 推理 job：

```python
job = build_external_job(
    "fttt",
    task="infer",
    sample="data_0",
    lambda_g=5e-5,
    J=6,
    n_workers=48,
    stop_tol=0.05,
)
```

执行 job：

```python
from inv_framework.applications.boed_budget_ladder import run_external_job

result = run_external_job(job, dry_run=True)
```

`dry_run=False` 会真的调用外部推理脚本：USCT 会占 GPU，FTTT 会占 CPU worker。Agent 侧应放在高风险执行确认流程里。

## 3. 已注册场景

### 3.1 USCT

USCT 的未知量是二维声速场 `c(r)`。频域声波满足 Helmholtz 方程：

```text
(nabla^2 + omega^2 / c(r)^2) u(r; omega) = -s(r; omega)
```

给定声速场 `c` 和发射源 `s_i`，CBS/Helmholtz solver 解出波场 `u_i`，再在接收器位置采样：

```text
y_i = P_rx u_i(c)
A(c) = { y_i } for selected transmitters i
```

反问题：

```text
given sparse transmitter/receiver measurements y, recover c(r)
```

当前 registry 参数：

| 项 | 值 |
|---|---|
| 图像网格 | `240 x 240` |
| 候选 Tx pool | `128` |
| Rx | `128` |
| 频率 | `250 kHz` |
| dense teacher | `k=64` |
| deploy ladder | `k = 8 -> 16 -> 32`, 可选 top-up `64` |
| key segments | `6` |
| DPS reverse steps | `30` |

物理特征：USCT 在当前设定下比较欠定，很多高对比样本会 prior-limited；增加 source 不一定线性提升重建质量。

### 3.2 FTTT

FTTT 使用 first-arrival travel time。走时 `T(r)` 满足 Eikonal 方程：

```text
|nabla T(r)|^2 = 1 / c(r)^2
T(source) = 0
```

前向算子是在接收器位置读取走时：

```text
y_i = { T_i(r_j) } over receivers j
A(c) = travel times for selected sources
```

反问题：

```text
given sparse source/receiver travel times y, recover the Marmousi medium image
```

当前 registry 参数：

| 项 | 值 |
|---|---|
| 图像网格 | `128 x 128` |
| 候选 source pool | `32` |
| dense teacher | `k=32` |
| deploy ladder | `k = 4 -> 8 -> 16` |
| key segments | `8` |
| DPS reverse steps | `40` |
| 噪声 | `sigma_y = 0.01` |

物理特征：FTTT 更 measurement-limited，通常在 `k≈8` 到 `k≈16` 附近趋于饱和；难样本在低档加 source 的收益更明显。

## 4. 已注册 DPS 算法 / baseline

```python
from inv_framework.applications.boed_budget_ladder import list_algorithms

for alg in list_algorithms():
    print(alg["name"])
```

当前有：

| name | 说明 |
|---|---|
| `full_dps` | dense teacher，全 source / 全梯度 DPS |
| `fixed_subset_dps` | 固定 equispaced subset baseline |
| `budgeted_dps` | 每个 key segment 使用指定 source subset schedule 的无监督 DPS |
| `policy_budget_ladder_dps` | 已训练 scalar_div policy 选择 sub_k_t，仍是部署推理，不是 RL |
| `dps_raw_gradient` | raw physical likelihood gradient guidance |
| `dps_normalized_drift` | adopted reformulated-gradient guidance |

## 5. Budgeted DPS 接口

标准 DPS 在每个 reverse step 做一次 measurement guidance：

```text
x_{t-1} = DDIM_step(x_t, eps_theta)
x_{t-1} <- x_{t-1} - scale * grad_x || A(x0_hat) - y ||
```

Budgeted DPS 把 `A` 替换成当前 segment 的 source subset：

```text
segment n uses B_n subset {1, ..., pool_size}
guidance uses A_{B_n}(x0_hat) and y_{B_n}
C_grad = sum_n |B_n| / (NSEG * k)
```

构造 equispaced subset：

```python
from inv_framework.solvers.diffusion.budget_dps import (
    equispaced_subset,
    make_equispaced_schedule,
    grad_cost,
)

subset = equispaced_subset(pool_size=32, k=8, sub_k=4)
# (0, 8, 16, 24)

schedule = make_equispaced_schedule([8, 4, 4, 2], k=8, pool_size=32)
cost = grad_cost(schedule, k=8)
```

`equispaced_subset` 保证低预算 source 仍覆盖全孔径，不使用 prefix subset，避免只覆盖单侧 aperture。

`BudgetedPosteriorSampling` 可以直接作为 conditioning hook：

```python
conditioning = BudgetedPosteriorSampling(
    operator=subsettable_operator,
    schedule=schedule,
    scale=1.0,
)
```

或者用 `BudgetedDPSSolver` 包装：

```python
solver = BudgetedDPSSolver(
    model=score_model,
    schedule_actions=schedule,
    num_inference_steps=40,
    scale=1.0,
)
x_rec = solver.solve(y, subsettable_operator)
```

## 6. Reformulated Gradient 变体

本次只保留 DPS guidance 相关的 reformulated-gradient 选项。

```python
from inv_framework.solvers.diffusion.budget_dps import reformulated_gradient
```

支持两种模式：

| mode | 说明 |
|---|---|
| `raw_gradient` | 直接注入原始物理 likelihood gradient，可作为 ablation |
| `normalized_drift` | Frobenius 归一化后的注入 drift，当前 USCT/FTTT 采用 |

示例：

```python
raw = reformulated_gradient(
    grad,
    mode="raw_gradient",
    source_scale=1.0,
)

drift = reformulated_gradient(
    grad,
    t=t,
    mode="normalized_drift",
    max_step=5.0,
    schedule="lin_t",
)
```

`normalized_drift` 对每个 batch sample 做 Frobenius 归一化，并把注入步长固定成：

```text
2 * max_step * f(t)
```

这样削 source 时改变的是梯度方向和物理信息量，而不是简单因为梯度范数变小导致 step 变小。

## 7. 与 tutorial.ipynb 的接口对应关系

| 教程概念 | 本次新增接口 | USCT / FTTT 对应 |
|---|---|---|
| `ForwardOperator` | `SubsettableForwardOperator` | 非线性 PDE 前向算子，可按 source subset 评估 |
| `NoiseModel` | 场景元信息中的 `default_noise` | USCT 使用预计算 dobs，FTTT 使用 Gaussian `sigma_y=0.01` |
| `DPSSolver` | `BudgetedDPSSolver` | 每个 key segment 换一个 source subset |
| `ConditioningMethod` | `BudgetedPosteriorSampling` | 只对当前 subset 做 measurement guidance |
| 正则 / 先验 | diffusion score model | `score240` 或 seismic `128` score |
| 计算成本 | `grad_cost` | `sum(sub_k_t)/(NSEG*k)` |
| Agent tool | `build_external_job` | 生成可运行 USCT/FTTT 推理命令 |

## 8. 最小验证

```bash
/home/caoxiang/anaconda3/envs/im2_torch1.13_cu116/bin/python tests/test_budget_ladder_interface.py
```

覆盖：

1. equispaced subset 是否覆盖全孔径；
2. `raw_gradient` / `normalized_drift` reformulated-gradient helper；
3. `CallableSubsetForwardOperator + BudgetedPosteriorSampling` 的 measurement slice 与 autograd；
4. USCT/FTTT registry 和 inference job builder。
