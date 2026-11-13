# 工业 CT 与 X 射线数据资源及下载说明

本页汇总工业数据的内容、公开入口、下载方法及获取限制，供本项目实验选用。
这里登记的是外部数据资源，不代表数据已随项目分发或已完成训练接口适配。

核对日期：**2026-10-28**。已核对公开页面、文件元数据或论文的数据声明；
本次文档整理没有下载、解压和逐文件验证完整数据包。文件大小采用十进制 GB/MB。

## 目录

- [按几何分类的重建数据（算法接入用）](#按几何分类的重建数据算法接入用)
- [有公开下载入口的数据](#有公开下载入口的数据)
- [工业教程样例](#工业教程样例)
- [需向作者索取的数据](#需向作者索取的数据)
- [存放、校验与引用](#存放校验与引用)

## 按几何分类的重建数据（算法接入用）

核对日期：**2026-11-13**。下表列出带投影（或可直接模拟投影）的 CT 数据，以及它们在本框架中
对应的算子。BIL 条目来自 [Battery Imaging Library 官方目录](https://github.com/antonyvam/BatteryImagingLibrary)
（commit `6455d03`，2026-10-10 核对）；目录尺寸取自目录本身，未逐个校验图像头。

| 数据 | 几何 | 规模 | 本框架算子 | 许可 | 获取 |
| --- | --- | --- | --- | --- | --- |
| BIL 同步辐射 micro-CT（22 个扫描，#45–#55、#103–#114） | 平行束，180°，1800 或 3600 个投影；体素 3.24 / 7.91 / 18.53 µm | 每个扫描为 Zenodo 上的原始投影；#45/#46 为 operando 4D（每个记录不是单一静态体） | 二维逐层：`ParallelBeamRadon2D`；整体：`ParallelBeamASTRA3D` | CC BY 4.0 | 各扫描的 Zenodo DOI 见官方目录，例如 #103 [10.5281/zenodo.17930623](https://doi.org/10.5281/zenodo.17930623) |
| BIL 同步辐射 nano-CT（#64） | 平行束，360°，6200 个投影，0.55 µm | 6914²×2048 | `ParallelBeamASTRA3D` | CC BY 4.0 | 原始 [10.5281/zenodo.17102561](https://doi.org/10.5281/zenodo.17102561)，重建 [10.5281/zenodo.17106400](https://doi.org/10.5281/zenodo.17106400) |
| BIL 实验室 micro-CT（20 个扫描，#65–#84，Diondo D5） | 圆轨道锥束，360°，1000–2200 个投影；体素 9–40 µm；XML 给出 SOD/SDD 与探测器中心偏移 | 针式、AAA/AAAA、18650、21700、4680 电芯 | `ConeBeamASTRA`（`det_offset_u` 对应 XML `ProjectionCenterOffsetX`，符号为 −1） | CC BY 4.0 | 例如 #77 原始投影 [10.5281/zenodo.17011146](https://doi.org/10.5281/zenodo.17011146)；20 个扫描的 DOI 见官方目录 |
| RWTH Samsung 35E（4 颗 18650，初始/老化） | 仅重建体（实验室锥束 VG `.REK`），无投影 | 每个归档约 6.6 GB（压缩） | 用 `ConeBeamASTRA` / `ParallelBeamASTRA3D` 模拟投影 | CC BY 4.0 | Hugging Face [`bsebench-org/rwth-jellyroll-deformation-2021-raw`](https://huggingface.co/datasets/bsebench-org/rwth-jellyroll-deformation-2021-raw) |
| CWI Walnuts（42 个核桃） | 实验室锥束，3 个光源高度，每个 1201 个投影 | 每个样本约 6 GB | `ConeBeamASTRA`（作者提供 ASTRA 几何向量） | CC BY 4.0 | 样本 1–8：Zenodo [2686726](https://zenodo.org/records/2686726)；其余样本见同一记录页与论文 [Der Sarkissian et al., Sci. Data 2019](https://doi.org/10.1038/s41597-019-0235-y) |
| DM4CT 同步辐射岩石（F3_1、F3_2） | 平行束 | 投影 1.9 GB×2，重建 6.9 GB×2 | `ParallelBeamRadon2D` / `ParallelBeamASTRA3D` | CC BY 4.0 | Zenodo [15420527](https://zenodo.org/records/15420527) |
| LoDoInd | 重建切片（平行束模拟） | Noise1–5 与 Ref 各约 22.5 GB | 模拟：`ParallelBeamRadon2D`、`FanBeamASTRA2D` | CC BY-NC 4.0 | Zenodo [10356955](https://zenodo.org/records/10356955)、[10391277](https://zenodo.org/records/10391277)、[10391412](https://zenodo.org/records/10391412) |
| NaSn 截断视场 CT | 同步辐射平行束，内部断层 | 目标 4 个扫描约 37 GiB，FullCT 4 个约 28 GiB | `TomoPyCUDARadon2D` | CC BY 4.0（发布时） | 私有 Hugging Face Dataset，需向仓库所有者申请；见 [NaSn 说明](nasn_ct.md) |

下载示例：

```sh
# BIL：单个扫描（以 #77 为例），先在记录页查看文件列表
curl -s https://zenodo.org/api/records/17011146 | python -c "import sys,json; [print(f['key'], f['size']) for f in json.load(sys.stdin)['files']]"

# RWTH 35E：从中国大陆访问时，直连镜像通常比代理快
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download bsebench-org/rwth-jellyroll-deformation-2021-raw --repo-type dataset \
  --local-dir "$INDUSTRIAL_DATA_DIR/rwth35e"

# CWI Walnut 1
curl --fail --location --retry 3 --continue-at - -o Walnut1.zip \
  'https://zenodo.org/records/2686726/files/Walnut1.zip?download=1'
```

使用 BIL 原始投影时，需要按扫描 XML 构建几何。几何与预处理（I0、坏点、按能谱组归一化）
的已验证做法记录在电池多电芯 benchmark 项目（`indus_CT/battery_multicell_20261113`，
`bmc/geometry.py`、`preprocess_protocol.json`）中，本框架的 `circular_cone_geometry` 与其
`cone_vec` 约定相同。Mayo 等医学数据不属于工业 CT，未列入本页。

## 有公开下载入口的数据

| 数据 | 内容 | 公开文件规模 | 下载入口 |
|---|---|---|---|
| 锂电池 CT 分割数据 | 6 个基础电池样本的合成 CT，以及分割标签；包含不同扫描条件的变化 | 6 个 CT ZIP，每个约 4.47 GB；标签约 177.6 MB；合计约 27.0 GB | [Zenodo 20506526](https://zenodo.org/records/20506526) |
| 层间角联锁编织复合材料 CT | 1 个真实重建体及对应人工语义分割标签 | 重建体约 503.9 MB，标签约 268.5 MB | [Zenodo 14891845](https://zenodo.org/records/14891845) |
| 正交编织复合材料 CT | 1 个真实重建体；公开文件列表未列分割标签 | 约 537.0 MB | [Zenodo 15389427](https://zenodo.org/records/15389427) |
| 工业 X 射线运动模糊数据 | 射线图像、参考图、运动信息与模拟代码；**部分公开，完整数据需索取** | 已登记的公开 ZIP 约 572.9 MB；以仓库当前文件为准 | [GitLab 仓库](https://gitlab.tue.nl/tue-waari/xray-motion-blur) |
| 强伪影微 CT／CAD 配准数据 | 33 keV、80 keV 投影／正弦图，附配准代码、教程和结果 | 按所选文件下载 | [Mendeley Data v1](https://data.mendeley.com/datasets/r8b5mcrj6j/1) |

以下命令在 Linux/macOS 的终端使用。先把 `/path/to/industrial-data` 替换为实际数据存放位置：

```sh
export INDUSTRIAL_DATA_DIR=/path/to/industrial-data
mkdir -p "$INDUSTRIAL_DATA_DIR"
```

这些命令由使用者按需运行；无需一次下载所有资源。

### 1. 锂电池 CT 分割数据

[数据页面](https://zenodo.org/records/20506526)说明：使用 gVXR 模拟投影并由 CIL 重建，
面向机器学习分割的数据增强。6 个基础样本的电压、曝光、源物距等扫描条件有变化。

**浏览器下载：**打开数据页面，在 Files 区域选择 `CT_data-sample1.zip` 至
`CT_data-sample6.zip`；需要标签时另下载 `Labels.zip`。

**命令行示例：只下载第 1 个样本和标签。**

```sh
mkdir -p "$INDUSTRIAL_DATA_DIR/battery"
curl --fail --location --retry 3 --continue-at - \
  --output "$INDUSTRIAL_DATA_DIR/battery/CT_data-sample1.zip" \
  'https://zenodo.org/records/20506526/files/CT_data-sample1.zip?download=1'
curl --fail --location --retry 3 --continue-at - \
  --output "$INDUSTRIAL_DATA_DIR/battery/Labels.zip" \
  'https://zenodo.org/records/20506526/files/Labels.zip?download=1'
```

需要其余样本时，将文件名中的 `sample1` 改为 `sample2` 至 `sample6`，输出文件名也同步修改。
第 1 个 ZIP 约 4.47 GB；解压还需要额外磁盘空间。可先查看归档内容：

```sh
unzip -l "$INDUSTRIAL_DATA_DIR/battery/CT_data-sample1.zip"
unzip -l "$INDUSTRIAL_DATA_DIR/battery/Labels.zip"
```

校验值见下表。标签不是灰度 CT；扫描条件变化也不等于新增独立电池样本。

### 2. 层间角联锁编织复合材料 CT

公开文件为同一个物体的重建体和人工标签，两者不是两个独立 CT 样本。
来源：[Zenodo 数据页](https://zenodo.org/records/14891845)。

```sh
mkdir -p "$INDUSTRIAL_DATA_DIR/composites/layer_interlock"
curl --fail --location --retry 3 --continue-at - \
  --output "$INDUSTRIAL_DATA_DIR/composites/layer_interlock/real_layer2layer_sample_reconstruction.tiff" \
  'https://zenodo.org/records/14891845/files/real_layer2layer_sample_reconstruction.tiff?download=1'
curl --fail --location --retry 3 --continue-at - \
  --output "$INDUSTRIAL_DATA_DIR/composites/layer_interlock/real_layer2layer_sample_segmentation.tiff" \
  'https://zenodo.org/records/14891845/files/real_layer2layer_sample_segmentation.tiff?download=1'
```

下载后得到 TIFF 文件，不需要按 ZIP 解压。此记录公开的是重建体和标签，文件列表未列原始投影序列。

### 3. 正交编织复合材料 CT

来源：[Zenodo 数据页](https://zenodo.org/records/15389427)。公开文件为一个重建体：

```sh
mkdir -p "$INDUSTRIAL_DATA_DIR/composites/orthogonal"
curl --fail --location --retry 3 --continue-at - \
  --output "$INDUSTRIAL_DATA_DIR/composites/orthogonal/real_orthogonal_noobed_sample_reconstruction.tiff" \
  'https://zenodo.org/records/15389427/files/real_orthogonal_noobed_sample_reconstruction.tiff?download=1'
```

公开文件列表未列分割标签或原始投影序列。两项复合材料数据的研究背景见
[论文](https://research.chalmers.se/publication/547380/file/547380_Fulltext.pdf)。

### 4. 工业 X 射线运动模糊数据

[论文](https://link.springer.com/article/10.1007/s10921-025-01279-6)报告了 63 张运动模糊图和
6 张静态参考图；[仓库 README](https://gitlab.tue.nl/tue-waari/xray-motion-blur/-/blob/main/README.md)
同时明确完整数据可向作者索取。因此，不应仅根据论文数量认定公开 ZIP 包含全部实验数据。

下载公开部分：

1. 打开 [GitLab 仓库](https://gitlab.tue.nl/tue-waari/xray-motion-blur)。
2. 进入 [XrayMotionBlurDataSet.zip](https://gitlab.tue.nl/tue-waari/xray-motion-blur/-/blob/main/XrayMotionBlurDataSet.zip)，使用文件页的下载按钮。
3. 需要模拟代码、配置和模型时，可克隆仓库：

```sh
git clone --depth 1 https://gitlab.tue.nl/tue-waari/xray-motion-blur.git \
  "$INDUSTRIAL_DATA_DIR/xray-motion-blur"
```

下载后先查看 ZIP 文件列表，确认实际提供的图像、参考图与运动文件。
完整数据的索取联系人由 README 给出：`b.g.l.meere@tue.nl`。
这是一套二维射线运动模糊资源，不能直接视为多角度 CT 采集序列。

### 5. 强伪影微 CT／CAD 配准数据

打开 [Mendeley Data v1](https://data.mendeley.com/datasets/r8b5mcrj6j/1)，在 Files 区域
选择所需文件，或使用 **Download All** 获取完整补充材料。数据在页面描述的 `data` 目录中，
与代码、教程、结果目录分别存放。建议优先查看以下文件：

| 文件 | 页面列出的采集信息 |
|---|---|
| `proj-33keV.tif` | 平场校正后的投影；1024 个探测器像素，1.9 μm 像素间距，180° 内 900 个角度 |
| `proj-0000-80keV.tif` | 第一层投影；2702 个探测器像素，1.22 μm 像素间距，360° 内 1750 个角度 |
| `proj-1023-80keV.tif` | 最后一层投影 |

页面另外提供 gVXR 源码、配准脚本、Jupyter 教程、配准输出和可视化。
上述 80 keV 文件描述的是首、末层，不能据文件编号认定所有中间层都公开。

Mendeley 的下载按钮可能产生临时链接，本文使用固定数据页作为入口，不写死临时地址。
使用前应根据教程确认 TIFF 存储的是透射强度还是对数投影，避免重复取对数。

## 工业教程样例

以下 4 项是公开教程资源，未发现完整的实测 CT 投影序列或完整实测三维重建体。
有模型和扫描配置的样例可以用于生成模拟投影；模拟角度数量不等于公开实测投影数量。

| 样例 | 公开内容 | 入口 |
|---|---|---|
| 铜／PVC 电缆 | `img_00000.tif` 投影、`slice_idx_0800.tiff` 单层重建；铜线和绝缘层 STL／SCAD；照片、`geom.csv`、扫描配置 | [Wire-Cu-2mm-17.54umvx](https://github.com/TomographicImaging/gVXR-Tutorials/tree/main/data/Wire-Cu-2mm-17.54umvx) |
| 铸铁茶壶 | `Proj/` 中的投影样例；原始及简化 STL、照片、扫描与几何配置 | [theiere-Fe-10mm](https://github.com/TomographicImaging/gVXR-Tutorials/tree/main/data/theiere-Fe-10mm) |
| Al-part 铝零件 | `Proj/` 中的投影样例、照片、`geom.csv`、扫描配置和能谱配置；未确认公开完整三维体或 STL | [Al-part](https://github.com/TomographicImaging/gVXR-Tutorials/tree/main/data/Al-part) |
| 三相复合材料标签图 | `labels.tif`／`labels.png` 二维标签图，描述碳芯、SiC 纤维和钛合金基体；可作为仿真输入 | [TIFF](https://github.com/TomographicImaging/gVXR-Tutorials/blob/main/data/labels.tif)、[PNG](https://github.com/TomographicImaging/gVXR-Tutorials/blob/main/data/labels.png) |

**获取上述全部样例：**只检出官方仓库的 `data` 目录。

```sh
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/TomographicImaging/gVXR-Tutorials.git \
  "$INDUSTRIAL_DATA_DIR/gvxr-tutorials"
git -C "$INDUSTRIAL_DATA_DIR/gvxr-tutorials" sparse-checkout set data
```

下载后的对应位置：

```text
gvxr-tutorials/data/
├── Wire-Cu-2mm-17.54umvx/
├── theiere-Fe-10mm/
├── Al-part/
├── labels.tif
└── labels.png
```

如目录已克隆，直接执行 `git ... sparse-checkout set data`，不必再次克隆。
只需要标签图时，也可直接下载：

```sh
mkdir -p "$INDUSTRIAL_DATA_DIR/gvxr-labels"
curl --fail --location --retry 3 \
  --output "$INDUSTRIAL_DATA_DIR/gvxr-labels/labels.tif" \
  'https://raw.githubusercontent.com/TomographicImaging/gVXR-Tutorials/main/data/labels.tif'
curl --fail --location --retry 3 \
  --output "$INDUSTRIAL_DATA_DIR/gvxr-labels/labels.png" \
  'https://raw.githubusercontent.com/TomographicImaging/gVXR-Tutorials/main/data/labels.png'
```

二维标签图不是灰度 CT 体；教程中的挤出或沿深度复制属于人工生成。
相关操作见 [标签到 CT 仿真教程](https://github.com/TomographicImaging/gVXR-Tutorials/blob/main/notebooks/segmentation-to-CT_scan-simulation.ipynb)。

## 需向作者索取的数据

以下 3 项没有在本次核对中确认完整数据的公开直接下载地址。
它们与上文“部分公开”的运动模糊数据分开列出。

| 数据 | 内容与规模 | 获取方法 |
|---|---|---|
| Textomos 合成复合材料 CT | 论文实验共 1000 个 512³ 合成体，含自动标签；700／150／150 用于训练／验证／测试；约 700 GB | [论文](https://research.chalmers.se/publication/547380/file/547380_Fulltext.pdf)说明整套数据未托管，可向作者索取；联系 Johan Friemann，`johan.friemann@chalmers.se` |
| 铝压铸件气孔数据 | 真实射线图、CT 参考信息及合成孔隙图像／标签；完整文件规模待确认 | [论文](https://doi.org/10.3390/s24092933)或 [PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086142/)的 Data Availability 声明可向通讯作者索取；联系方式见论文通讯作者栏 |
| FML 冲击损伤 CT | 不同冲击条件下的纤维金属层板损伤数据；完整体数据数量和格式待确认 | [论文](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1599345/full)的 Data availability statement 表示作者可提供原始数据；通过论文通讯作者栏联系 |

索取时建议明确需要：原始投影还是重建体、对应标签、扫描几何、体素尺寸／单位、
可用的下载方式和使用条件。本文仅提供索取入口，没有代为联系作者。

Textomos 也提供[公开生成代码](https://github.com/Johan-Friemann/textomos)：

```sh
git clone --depth 1 https://github.com/Johan-Friemann/textomos.git \
  "$INDUSTRIAL_DATA_DIR/textomos"
```

克隆得到的是生成器代码，不会自动获得论文的 700 GB 数据集。
自行生成时按该仓库 README 安装与运行；新生成的数据不应标为论文原始数据。

## 存放、校验与引用

建议原始文件放在仓库外，并按来源分目录保存：

```text
industrial-data/
├── battery/
├── composites/
│   ├── layer_interlock/
│   └── orthogonal/
├── xray-motion-blur/
├── microct-cad/
├── gvxr-tutorials/
└── requested/           # 后续从作者处取得的数据
```

Zenodo 页面提供了 MD5，以下是本次登记的值；核验时也应对照数据页的当前版本。

| 文件 | MD5 |
|---|---|
| `CT_data-sample1.zip` | `e35c429b02a21ace65969a5c77b908f3` |
| `CT_data-sample2.zip` | `45fec7611ad8a8bbbcb59b3dd827b0be` |
| `CT_data-sample3.zip` | `4828974178072f61968ae9eb0ef597d7` |
| `CT_data-sample4.zip` | `b9c34b84bbd03f428fc8c95d2c66c9c6` |
| `CT_data-sample5.zip` | `22c9d02fce73ded921dd503eec97356d` |
| `CT_data-sample6.zip` | `b04d8ab8e744cab7cae60100e04aaf91` |
| `Labels.zip` | `9f3572f6c2863dce628b8115bd6c3a57` |
| `real_layer2layer_sample_reconstruction.tiff` | `40401f14af7c914fe8203fdd365b70d7` |
| `real_layer2layer_sample_segmentation.tiff` | `bf6b318f4ac4a5ae9e248c682c0b2d12` |
| `real_orthogonal_noobed_sample_reconstruction.tiff` | `6c36483b5fe160cc046b09cb3540bb61` |

例如校验第一个电池数据包：

```sh
printf '%s  %s\n' 'e35c429b02a21ace65969a5c77b908f3' \
  "$INDUSTRIAL_DATA_DIR/battery/CT_data-sample1.zip" | md5sum --check -
unzip -t "$INDUSTRIAL_DATA_DIR/battery/CT_data-sample1.zip"
```

`md5sum` 是 Linux/GNU 工具；macOS 可使用 `md5 文件路径` 并对照上表。
ZIP 校验通过后可解压到独立目录；TIFF 文件直接保留原格式：

```sh
unzip "$INDUSTRIAL_DATA_DIR/battery/CT_data-sample1.zip" \
  -d "$INDUSTRIAL_DATA_DIR/battery/sample1"
unzip "$INDUSTRIAL_DATA_DIR/battery/Labels.zip" \
  -d "$INDUSTRIAL_DATA_DIR/battery/labels"
```

保留原始文件名、数据 DOI／版本、仓库 commit 和下载日期，便于复现实验。
引用和分发以各数据页的许可为准；其中层间角联锁数据登记为 CC BY-SA 4.0，
正交复合材料和 Mendeley 配准数据登记为 CC BY 4.0。
代码仓库的许可证不自动代表其链接的所有数据都采用相同许可。
