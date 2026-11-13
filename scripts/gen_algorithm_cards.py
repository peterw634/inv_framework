"""Generate algorithms/<name>/README.md and algorithms/README.md.

    python scripts/gen_algorithm_cards.py            # write the cards
    python scripts/gen_algorithm_cards.py --check    # exit 1 if a card is stale

Sources: inv_framework/registry.py (identity, defaults), the toy packs' ``meta``
(input description) and algorithms/results.json (measured "normal results",
written by scripts/run_toy_benchmarks.py). Edit those, not the cards.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inv_framework import registry  # noqa: E402

GEOMETRY = {
    "parallel_torch": "二维平行束（同步辐射）",
    "tomopy_cuda": "二维平行束内部断层（同步辐射，NaSn 几何）",
    "astra_fan": "二维扇束（实验室锥束中心层，ASTRA）",
    "astra_cone": "三维锥束（实验室，ASTRA）",
    "astra_parallel3d": "三维平行束（同步辐射体数据，ASTRA）",
}
WEIGHTS = {
    "dm4ct-pixel": dict(
        fetch="python scripts/download_weights.py lodoind-pixel",
        arg="weights/lodoind_pixel_diffusion",
        text="DM4CT 在 LoDoInd 上训练的像素域 UNet（`unet/`、`scheduler/`，455 MB，MIT，"
             "固定版本 jiayangshi/lodoind_pixel_diffusion@0f4a0dc），下载到 `weights/lodoind_pixel_diffusion/`。",
        convention="模型在 512² 网格、[-1, 1] 值域上训练；`examples/run.py` 按数据包的 `image_scale=0.5`、"
                   "`image_offset=0.5` 把测量换到模型域（`(y − A·0.5)/0.5`），重建后再映射回 [0, 1]。"
                   "网格必须与模型一致（512²），否则报错。"),
    "dm4ct-latent": dict(
        fetch="python scripts/download_weights.py lodoind-latent",
        arg="weights/lodoind_latent_diffusion",
        text="DM4CT 在 LoDoInd 上训练的潜空间模型（`unet/`、`vqvae/`、`scheduler/`，676 MB，MIT，"
             "固定版本 jiayangshi/lodoind_latent_diffusion@6f35a9b），下载到 `weights/lodoind_latent_diffusion/`。",
        convention="VQ 解码后的图像为 512²、[-1, 1]；测量换算同像素模型（`image_scale=0.5`、`image_offset=0.5`）。"
                   "潜变量初始噪声在 CPU 上生成。"),
    "pmf": dict(
        fetch="python scripts/download_weights.py pmf",
        arg="weights/nasn-pmf-2048-epoch449-ema.safetensors",
        text="NaSn pMF-B/128 epoch-449 EMA 权重（502 MiB，MIT），来自 "
             "https://github.com/peterw634/nasn-truncated-ct/releases/tag/model-v0.1.0 ，脚本按清单校验 SHA-256；"
             "同名 `.json` 配置必须放在权重旁边。",
        convention="模型在 2048²、[-1, 1] 值域上训练；toy 包用 `x_phys = 0.5·x_model + 0.5`（与 NaSn 合成 benchmark 相同），"
                   "此时推荐 λ=5.0。真实 NaSn 测量使用 `physical_image × scale` 的标定且不做中心化，对应 λ=0.05"
                   "（见 `inv-nasn-ct reconstruct`）。网格必须为 2048²。"),
}
PACK_TEXT = {
    "toy_parallel_ct_128.npz": (
        "改进 Shepp-Logan 模体，30 个角度（0–180°）稀疏角，加性高斯噪声 σ=0.5",
        "无；`sinogram` 已是线积分（负对数后）投影"),
    "toy_truncated_ct_128.npz": (
        "改进 Shepp-Logan 模体，90 个角度，内部断层（仅中心 48/128 列有测量），无噪声",
        "无；未测量列存为 0 并由 `det_mask` 标出"),
    "toy_parallel_ct_3d.npz": (
        "与锥束包相同的合成圆柱电池，同步辐射式三维平行束 90 个角度（0–180°），"
        "每个探测器行对应一个切片，高斯噪声 σ=0.5%·max",
        "无；`sinogram` 为线积分，ASTRA 顺序（探测器行, 角度, 探测器列）；探测器像素 = 体素"),
    "toy_cone_ct_3d.npz": (
        "合成圆柱电池（钢壳、卷绕层、极柱、端盖、夹杂），圆轨道锥束 120 个角度（0–360°），"
        "SOD 400、SDD 600（放大 1.5 倍），高斯噪声 σ=0.5%·max",
        "无；`sinogram` 为线积分，ASTRA 顺序（探测器行, 角度, 探测器列）；长度以体素为单位"),
    "toy_lodoind_512.npz": (
        "LoDoInd 测试切片 2750（工业 CT），40 个角度（0–180°）稀疏角，无噪声（DM4CT 设置）",
        "DM4CT `preprocess_lodoind.py`（百分位截断、映射到 [-1, 1]、缩放到 512²）后取 `(x + 1) / 2`"),
    "toy_lodoind_fan_512.npz": (
        "同一 LoDoInd 切片，实验室扇束 80 个角度（0–360°），SOD 1024、SDD 1536，576 个探测器，无噪声",
        "同上；`sinogram` 由 ASTRA 扇束正投影生成"),
    "toy_nasn_pmf_2048.npz": (
        "pMF 无条件生成的电池切片（不含 NaSn 扫描数据），NaSn 合成 benchmark 几何：2048 个角度，"
        "仅中心 512/2048 列有测量（内部断层），无噪声",
        "`image = (pMF 样本 + 1) / 2`（float16 存储）；未测量列存为 0"),
    "toy_nasn_pmf_fan_2048.npz": (
        "同一 pMF 切片，实验室扇束 180 个角度（0–360°），SOD 4096、SDD 6144，2304 个探测器，全视场，无噪声",
        "同上；`sinogram` 由 ASTRA 扇束正投影生成"),
}
NOTES = {
    "dps": "DM4CT 原设置 scale=10 在本仓库 LoDoInd toy 数据上发散（平行束 10.8 dB），默认值改为 1。"
           "归一化损失把恰好为 0 的测量视为未测量，扇束数据（探测器超出体积）不再被放大步长。",
    "mcg": "扇束结果对应修正后的归一化损失（修正前恰好为 0 的射线使步长放大约 19 倍，扇束仅 13.1 dB）。",
    "diffstategrad": "基于 ReSample，扇束上同样受 DM4CT 默认参数影响（toy 上 15.1 dB，平行束 20.4 dB），需要针对扇束另行调参。",
    "resample": "DM4CT 默认参数在扇束数据上发散（toy 上约 8.5 dB，图像四角位于视场外），需要针对扇束另行调参；"
                "平行束结果正常。单次运行约 30–45 分钟，本次未调参。",
    "pmf_dds": "toy 真值是该先验自身的无条件样本（分布内），因此近乎完美的恢复是预期结果，不代表真实数据质量；"
               "NaSn 真实合成验证集上 16 步的 ROI PSNR 为 36.6 dB（`docs/nasn_ct.md`）。",
    "pmf_lora_dds": "toy 真值是 pMF 自身的无条件样本（分布内），数值偏高，只用于验证流程，不代表真实或分布外数据上的质量；"
                    "LoRA 适配针对的是分布外数据。运行时间约为 pMF-DDS 的 14 倍（每步 4 次 LoRA 更新）。",
    "dip": "DIP 的结果随随机种子波动（toy 上 3000 次迭代、不同种子约 24.6–25.8 dB）。",
}


def fmt(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
        return f"{value:g}"
    return str(value)


def pack_info(pack):
    with np.load(ROOT / pack) as data:
        meta = json.loads(str(data["meta"]))
        image = data["image"].shape
        sinogram = data["sinogram"].shape
        kind = str(data["operator"])
        measured = int(data["det_mask"].sum())
    return dict(meta=meta, image=image, sinogram=sinogram, kind=kind, measured=measured)


def shape(dims):
    return "×".join(str(d) for d in dims)


def result_rows(spec, results):
    rows = []
    for pack in spec.toy_data:
        run = results.get(f"{spec.name}:{Path(pack).name}")
        if not run or run.get("status") != "passed":
            rows.append((pack, None))
            continue
        rows.append((pack, run))
    return rows


def metric(run, key, digits=2):
    value = run["metrics"].get(key)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def card(spec, results, environment):
    lines = [f"# {spec.title}", "",
             "> 本卡片由 `scripts/gen_algorithm_cards.py` 根据 `inv_framework/registry.py`、示例数据的 `meta` "
             "和 `algorithms/results.json` 生成；请修改源头后重新生成。", "",
             "## 1. 算法身份", "",
             f"- **名称**：{spec.title}；注册名 `{spec.name}`",
             f"- **一句话简介**：{spec.summary}",
             f"- **所属任务**：{spec.task}",
             f"- **方法类型**：{'学习类（使用预训练模型）' if spec.learned else '非学习类（不使用预训练模型）'}",
             f"- **适用范围**：{spec.scope}",
             f"- **对应论文**：{spec.paper}",
             f"- **代码来源**：{spec.code}", "",
             "## 2. 输入输出", "",
             "示例数据位于 `examples/data/`，字段、单位与许可见 [数据说明](../../examples/data/README.md)。"
             "每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`"
             "与几何参数：", "",
             "| 文件 | 几何 | 内容 | 预处理 |", "| --- | --- | --- | --- |"]
    for pack in spec.toy_data:
        info = pack_info(pack)
        dims = ("探测器行×角度×探测器列" if len(info["sinogram"]) == 3 else "角度×探测器")
        text, preprocessing = PACK_TEXT.get(Path(pack).name, (info["meta"]["description"],
                                                               info["meta"]["preprocessing"]))
        content = f"{text}。`image` {shape(info['image'])}，`sinogram` {shape(info['sinogram'])}（{dims}）"
        lines.append(f"| `{Path(pack).name}` | {GEOMETRY[info['kind']]} | {content} | {preprocessing} |")
    lines += ["",
              "运行后在 `--output` 目录生成：`reconstruction.npy`（重建结果，与 `image` 同尺寸、同单位，float32）、"
              "`preview.png`（真值｜重建｜4 倍放大的绝对误差；三维为中心轴向＋冠状切片，可直接打开查看）、"
              "`summary.json`（实际参数、耗时、指标、输入的 SHA-256）。", ""]
    rows = result_rows(spec, results)
    measured = [(pack, run) for pack, run in rows if run is not None]
    if measured:
        gpu = environment.get("gpu") or "GPU"
        lines += [f"**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 {gpu}）：", "",
                  "| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for pack, run in measured:
            info = pack_info(pack)
            lines.append(f"| `{Path(pack).name}` | {shape(info['image'])} | {metric(run, 'psnr_db')} | "
                         f"{metric(run, 'ssim', 3)} | {metric(run, 'roi_psnr_db')} | "
                         f"{metric(run, 'relative_measurement_l2', 4)} | {run['seconds']:.1f} | {run['device']} |")
        lines += ["", "PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。"
                  "结果与上表相差在 1 dB 内即可视为正常（随机方法与 GPU 型号会带来小幅差异）。"]
    missing = [pack for pack, run in rows if run is None]
    if missing:
        lines += ["", "尚未在本仓库实测：" + "、".join(f"`{Path(p).name}`" for p in missing) + "。"]
    if spec.name in NOTES:
        lines += ["", f"说明：{NOTES[spec.name]}"]

    weights = WEIGHTS.get(spec.weights)
    needs_cuda = spec.device == "cuda"
    lines += ["", "## 3. 运行方式", "",
              "```bash",
              "pip install -r requirements.txt && pip install -e .",
              "conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要"]
    if weights:
        lines.append(weights["fetch"])
    base = f"python examples/run.py --algo {spec.name}"
    tail = (f" --weights {weights['arg']}" if weights else "")
    for pack in spec.toy_data:
        cpu = not needs_cuda and pack_info(pack)["kind"] == "parallel_torch"
        lines.append(f"{base} --input {pack} --output outputs/{spec.name}/{Path(pack).stem}{tail} "
                     f"--device {'cpu' if cpu else 'cuda'}")
    lines += ["```", "",
              f"- 设备：{'需要 CUDA GPU。' if needs_cuda else '二维平行束示例可在 CPU 上运行，GPU 更快。'}"
              "三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；"
              "`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。",
              "- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。"]
    if weights:
        lines += [f"- 权重：{weights['text']}", f"- 权重与数据的约定：{weights['convention']}"]
    else:
        lines.append("- 权重：不需要。")
    lines += ["- 参数可用 `--set 名称=值` 覆盖（值按 JSON 解析，如 `--set min_value=null`）；"
              "`python examples/run.py --list` 以 JSON 列出全部算法与默认值。", "",
              "## 4. 参数", "",
              "| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in spec.params:
        lines.append(f"| `{p.name}` | {p.help} | `{fmt(p.default)}` | {p.source} | {p.constraint} | "
                     f"{p.recommended or ''} |")
    lines.append("")
    return "\n".join(lines)


def index(results):
    lines = ["# 算法索引", "",
             "每个算法一张卡片（`<注册名>/README.md`），按《算法提交规范》包含算法身份、输入输出、运行方式与参数。",
             "统一入口为 `examples/run.py`；程序化接口为 `inv_framework.registry`"
             "（`ALGORITHMS`、`defaults(name)`、`build(name, operator, params)`）。", "",
             "| 算法 | 注册名 | 类型 | 预训练权重 | 同步辐射平行束 toy PSNR (dB) | 实验室锥/扇束 toy PSNR (dB) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, spec in registry.ALGORITHMS.items():
        cells = {"synchrotron": [], "lab": []}
        for pack in spec.toy_data:
            run = results.get(f"{name}:{Path(pack).name}")
            if run and run.get("status") == "passed":
                roi = run["metrics"].get("roi_psnr_db")
                value = f"{run['metrics']['psnr_db']:.2f}" + (f"（ROI {roi:.2f}）" if roi else "")
            else:
                value = "—"
            group = "lab" if pack_info(pack)["kind"] in ("astra_fan", "astra_cone") else "synchrotron"
            cells[group].append(f"{value} `{Path(pack).stem}`")
        lines.append(f"| [{spec.title}]({name}/README.md) | `{name}` | {'学习类' if spec.learned else '非学习类'} | "
                     f"{spec.weights or '无'} | {'<br>'.join(cells['synchrotron'])} | {'<br>'.join(cells['lab'])} |")
    lines += ["", "数值来自 `scripts/run_toy_benchmarks.py`（推荐默认参数、种子 0），原始记录见 "
              "[results.json](results.json)。toy 数据仅用于 smoke 测试，不同数据包之间的数值不可直接比较。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "algorithms/results.json"
    data = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    results, environment = data.get("runs", {}), data.get("environment", {})
    outputs = {ROOT / "algorithms/README.md": index(results)}
    for name, spec in registry.ALGORITHMS.items():
        outputs[ROOT / f"algorithms/{name}/README.md"] = card(spec, results, environment)
    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text() != text]
    if args.check:
        for p in stale:
            print(f"stale: {p.relative_to(ROOT)}")
        return 1 if stale else 0
    for p, text in outputs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    print(f"wrote {len(outputs)} files ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
