# Interior CT 文献分类与阅读索引

本目录整理内部 CT（Interior CT）与截断投影 ROI 重建相关的 **30 项文献**：
**传统方法 A01–A15** 和 **深度学习方法 B01–B15**，各 15 项。A01 是数学专著，其余为论文条目。
分类按主要重建策略组织，用于查找文献与比较方法；交叉使用多种技术的文章只归入一个主要子类。

当前收录 **21 篇 PDF、1 篇文本全文（A09）**，另有 **8 项未收录本地全文**。
下方每项提供本地文件（如有）和 DOI 入口；“未收录全文”仅表示本目录的收录状态。

## 目录结构

```text
interior_CT_30papers/
├── README.md          # 分类说明与逐篇阅读索引
├── classical/         # 传统方法，文件名前缀 Axx
└── deep_learning/     # 深度学习方法，文件名前缀 Bxx
```

## 分类总览

### 传统方法

| 子类 | 论文编号 | 主要思路 |
|---|---|---|
| 数学基础与局部特征重建 | A01–A02 | 介绍 CT 反问题的数学基础，以及局部边界、奇异结构等特征的恢复。 |
| DBP/BPF 与 Hilbert 解析重建 | A03–A05 | 通过微分反投影、反投影滤波和 Hilbert 反演，研究 ROI 重建的解析形式与数据条件。 |
| 已知子区域先验重建 | A06–A09 | 利用 ROI 内小块区域的已知函数值，约束截断数据下的内部重建。 |
| SVD 反演与稳定性分析 | A10–A12 | 分析截断 Hilbert 算子的奇异值与稳定性，并构造正则化反演。 |
| TV／高阶 TV 稀疏正则化 | A13–A15 | 利用分片常数、梯度稀疏或分片多项式先验，通过 TV／高阶 TV 约束重建。 |

### 深度学习方法

| 子类 | 论文编号 | 主要思路 |
|---|---|---|
| 深度学习可解性理论 | B05 | 分析图像先验、低维解模型与训练数据结构对欠定问题可学习性的影响。 |
| 图像及解析变换域学习校正 | B01、B03、B04、B09–B11 | 在 FBP 图像、DBP、滤波投影或加权反投影等中间表示上学习校正。 |
| 投影处理与双域重建 | B06–B08、B13、B15 | 通过投影外推、ROI 投影提取或投影域与图像域交互处理截断数据。 |
| 直接投影到图像映射 | B02 | 直接学习从正弦图（sinogram）到重建图像的映射。 |
| 模型驱动深度展开 | B12 | 把含物理数据一致性和正则项的迭代重建过程展开为可训练网络。 |
| 隐式神经表示 INR | B14 | 用坐标网络表示衰减场，并根据当前扫描的投影逐病例优化网络。 |

### 分类口径

- **局部特征与定量重建**：A02 侧重局部边界和奇异结构，不能仅凭“局部重建”名称将其等同于恢复 ROI 内全部衰减值。
- **解析方法与先验条件**：A03–A05 关注解析重建的数据条件；A06–A09 强调已知子区域先验；A10–A12 则聚焦反演与稳定性。它们在 Hilbert 变换等技术上存在交叉。
- **学习校正与双域处理**：B04、B09 虽涉及投影，但重点是解析重建流程中的滤波投影校正，因此归入“图像及解析变换域学习校正”；B06–B08、B13、B15 侧重投影补全、分离或跨域交互。
- **端到端与直接映射**：“端到端训练”并非 B02 独有；B02 的分类依据是直接学习正弦图到图像的变换。
- **监督方式**：深度学习类并不全部依赖成对训练数据。B11 采用无配对学习，B14 根据当前扫描投影自监督优化；B05 是可解性理论分析。

## 传统方法：逐篇索引

### 数学基础与局部特征重建（A01–A02）

介绍 CT 反问题的数学基础，以及局部边界、奇异结构等特征的恢复。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **A01** | The Mathematics of Computerized Tomography（1986） | CT 数学基础教材，涉及不完全数据、唯一性、稳定性与重建算法。 | 未收录全文 · [DOI](https://doi.org/10.1007/978-3-663-01409-6) |
| **A02** | Local Tomography（1992） | 局部/λ tomography，恢复边界和奇异结构等局部特征。 | [PDF](classical/A02_Local_Tomography.pdf) · [DOI](https://doi.org/10.1137/0152026) |

### DBP/BPF 与 Hilbert 解析重建（A03–A05）

通过微分反投影、反投影滤波和 Hilbert 反演，研究 ROI 重建的解析形式与数据条件。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **A03** | A two-step Hilbert transform method for 2D image reconstruction（2004） | 两步法：DBP 后进行一维 Hilbert 反演，研究可重建 ROI 的数据条件。 | 未收录全文 · [DOI](https://doi.org/10.1088/0031-9155/49/17/006) |
| **A04** | Image reconstruction in peripheral and central regions-of-interest and data redundancy（2005） | 扇束 BPF 与数据冗余，研究中心和周边 ROI 的解析重建。 | 未收录全文 · [DOI](https://doi.org/10.1118/1.1844171) |
| **A05** | Truncated Hilbert transform and image reconstruction from limited tomographic data（2006） | 将有限数据 ROI 重建联系到截断 Hilbert 反演，分析数据充分性。 | 未收录全文 · [DOI](https://doi.org/10.1088/0266-5611/22/3/019) |

### 已知子区域先验重建（A06–A09）

利用 ROI 内小块区域的已知函数值，约束截断数据下的内部重建。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **A06** | A General Local Reconstruction Approach Based on a Truncated Hilbert Transform（2007） | 利用已知区间函数值等先验，通过截断 Hilbert 与解析延拓进行局部重建。 | [PDF](classical/A06_A_General_Local_Reconstruction_Approach_Based_on_a_Truncated_Hilbert_Transform.pdf) · [DOI](https://doi.org/10.1155/2007/63634) |
| **A07** | Exact Interior Reconstruction with Cone-Beam CT（2007） | 已知 VOI 内小子区域，结合广义 PI 弦与 BPF/FBP 推导锥束精确重建。 | [PDF](classical/A07_Exact_Interior_Reconstruction_with_Cone_Beam_CT.pdf) · [DOI](https://doi.org/10.1155/2007/10693) |
| **A08** | Tiny a priori knowledge solves the interior problem in computed tomography（2008） | 以微小已知子区域保证可解性，并采用 DBP-POCS 实现重建。 | 未收录全文 · [DOI](https://doi.org/10.1088/0031-9155/53/9/001) |
| **A09** | Solving the interior problem of computed tomography using a priori knowledge（2008） | 分析已知子区域条件下的 interior 唯一性及截断 Hilbert 反演。 | [文本全文](classical/A09_Solving_the_interior_problem_of_computed_tomography_using_a_priori_knowledge.txt) · [DOI](https://doi.org/10.1088/0266-5611/24/6/065001) |

### SVD 反演与稳定性分析（A10–A12）

分析截断 Hilbert 算子的奇异值与稳定性，并构造正则化反演。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **A10** | Interior Reconstruction Using the Truncated Hilbert Transform via Singular Value Decomposition（2008） | 使用离散 SVD、截断 SVD/Tikhonov 反演截断 Hilbert，替代较慢的 POCS。 | [PDF](classical/A10_Interior_Reconstruction_Using_the_Truncated_Hilbert_Transform_via_Singular_Value_Decomposition.pdf) · [DOI](https://doi.org/10.3233/XST-2008-00207) |
| **A11** | Singular value decomposition for the truncated Hilbert transform（2010） | 研究截断 Hilbert 算子的奇异值分解，并讨论解析延拓算法。 | 未收录全文 · [DOI](https://doi.org/10.1088/0266-5611/26/11/115011) |
| **A12** | Singular value decomposition for the truncated Hilbert transform: Part II（2011） | 继续分析不同区间构型下截断 Hilbert 算子的奇异值分解。 | 未收录全文 · [DOI](https://doi.org/10.1088/0266-5611/27/7/075006) |

### TV／高阶 TV 稀疏正则化（A13–A15）

利用分片常数、梯度稀疏或分片多项式先验，通过 TV／高阶 TV 约束重建。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **A13** | Compressed sensing based interior tomography（2009） | 以 ROI 分片常数和梯度稀疏为先验，通过 TV 最小化重建。 | [PDF](classical/A13_Compressed_sensing_based_interior_tomography.pdf) · [DOI](https://doi.org/10.1088/0031-9155/54/9/014) |
| **A14** | A General Total Variation Minimization Theorem for Compressed Sensing Based Interior Tomography（2009） | 给出压缩感知 interior 重建中 TV 最小化的理论论证。 | [PDF](classical/A14_A_General_Total_Variation_Minimization_Theorem_for_Compressed_Sensing_Based_Interior_Tomography.pdf) · [DOI](https://doi.org/10.1155/2009/125871) |
| **A15** | High-order total variation minimization for interior tomography（2010） | 用高阶 TV 将分片常数先验推广到分片多项式 ROI。 | [PDF](classical/A15_High_order_total_variation_minimization_for_interior_tomography.pdf) · [DOI](https://doi.org/10.1088/0266-5611/26/3/035013) |

## 深度学习方法：逐篇索引

### 深度学习可解性理论（B05）

分析图像先验、低维解模型与训练数据结构对欠定问题可学习性的影响。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **B05** | Deep learning-based solvability of underdetermined inverse problems in medical imaging（2021） | 利用低维解模型及 M-RIP 等条件，分析包括 interior CT 在内的可学习性。 | [PDF](deep_learning/B05_Deep_learning_based_solvability_of_underdetermined_inverse_problems_in_medical_imaging.pdf) · [DOI](https://doi.org/10.1016/j.media.2021.101967) |

### 图像及解析变换域学习校正（B01、B03、B04、B09–B11）

在 FBP 图像、DBP、滤波投影或加权反投影等中间表示上学习校正。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **B01** | Deep Learning Interior Tomography for Region-of-Interest Reconstruction（2018） | 对截断投影的 FBP 图像做 CNN 校正，抑制杯状伪影。 | [PDF](deep_learning/B01_Deep_Learning_Interior_Tomography_for_Region_of_Interest_Reconstruction.pdf) · [DOI](https://doi.org/10.48550/arXiv.1712.10248) |
| **B03** | One network to solve all ROIs: Deep learning CT for any ROI using differentiated backprojection（2019） | 在 DBP 域学习逆截断 Hilbert 映射，研究不同 ROI 的泛化。 | [PDF](deep_learning/B03_One_network_to_solve_all_ROIs_Deep_learning_CT_for_any_ROI_using_differentiated_backprojection.pdf) · [DOI](https://doi.org/10.1002/mp.13631) |
| **B04** | Reconstructing interior transmission tomographic images with an offset-detector using a deep-neural-network（2019） | 在偏置探测器几何下学习有效区域内的 ramp-filtered 投影，再用 FBP。 | [PDF](deep_learning/B04_Reconstructing_interior_transmission_tomographic_images_with_an_offset_detector_using_a_deep_neural_network.pdf) · [DOI](https://doi.org/10.1117/12.2534888) |
| **B09** | CNN-based Region-of-Interest Image Reconstruction from Truncated Data in Cone-Beam CT（2022） | 用 U-Net 校正截断区域的滤波投影，再通过 FDK 重建 CBCT ROI。 | [PDF](deep_learning/B09_CNN_based_Region_of_Interest_Image_Reconstruction_from_Truncated_Data_in_Cone_Beam_CT.pdf) · [DOI](https://doi.org/10.1117/12.2611310) |
| **B10** | Deep-Interior: A new pathway to interior tomographic image reconstruction via a weighted backprojection and deep learning（2024） | 加权反投影生成中间图像，再用网络学习反卷积。 | [PDF](deep_learning/B10_Deep_Interior_A_new_pathway_to_interior_tomographic_image_reconstruction_via_a_weighted_backprojection_and_deep_learning.pdf) · [DOI](https://doi.org/10.1002/mp.16880) |
| **B11** | PIDNET: Polar Transformation Based Implicit Disentanglement Network for Truncation Artifacts（2024） | 极坐标变换结合 CycleGAN 隐式解耦，在图像域分离内容与截断伪影。 | [PDF](deep_learning/B11_PIDNET_Polar_Transformation_Based_Implicit_Disentanglement_Network_for_Truncation_Artifacts.pdf) · [DOI](https://doi.org/10.3390/e26020101) |

### 投影处理与双域重建（B06–B08、B13、B15）

通过投影外推、ROI 投影提取或投影域与图像域交互处理截断数据。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **B06** | Data Extrapolation From Learned Prior Images for Truncation Correction in Computed Tomography（2021） | 从学习得到的先验图像前向投影，补偿未测投影并保持已测数据一致性。 | [PDF](deep_learning/B06_Data_Extrapolation_From_Learned_Prior_Images_for_Truncation_Correction_in_Computed_Tomography.pdf) · [DOI](https://doi.org/10.1109/TMI.2021.3072568) |
| **B07** | Generative adversarial networks improve interior computed tomography angiography reconstruction（2021） | 双 Pix2Pix GAN：先外推截断 sinogram，再精修重建图像。 | [PDF](deep_learning/B07_Generative_adversarial_networks_improve_interior_computed_tomography_angiography_reconstruction.pdf) · [DOI](https://doi.org/10.1088/2057-1976/ac31cb) |
| **B08** | End-to-end deep learning for interior tomography with low-dose x-ray CT（2022） | 双域 CNN 分解处理低剂量噪声与截断投影外推。 | [PDF](deep_learning/B08_End_to_end_deep_learning_for_interior_tomography_with_low_dose_x_ray_CT.pdf) · [DOI](https://doi.org/10.1088/1361-6560/ac6560) |
| **B13** | Disentangled deep learning method for interior tomographic reconstruction of low-dose x-ray CT（2025） | DPER 分离 ROI 外背景及噪声贡献、提取 ROI 投影；DPER-Pro 加入渐进补偿。 | [PDF](deep_learning/B13_Disentangled_deep_learning_method_for_interior_tomographic_reconstruction_of_low_dose_x_ray_CT.pdf) · [DOI](https://doi.org/10.1088/1361-6560/ae02dc) |
| **B15** | A Two-Module Parallel Dual-Domain Network for interior tomography reconstruction（2026） | TPDDN 先做初始恢复，再通过投影域与图像域的并行交互分支融合。 | 未收录全文 · [DOI](https://doi.org/10.1177/08953996261433954) |

### 直接投影到图像映射（B02）

直接学习从正弦图（sinogram）到重建图像的映射。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **B02** | Learning to Reconstruct Computed Tomography Images Directly From Sinogram Data Under A Variety of Data Acquisition Conditions（2019） | iCT-Net 直接学习 sinogram 到图像的变换，验证稀疏视角与截断耦合情形。 | [PDF](deep_learning/B02_Learning_to_Reconstruct_Computed_Tomography_Images_Directly_From_Sinogram_Data_Under_A_Variety_of_Data_Acquisition_Conditions.pdf) · [DOI](https://doi.org/10.1109/TMI.2019.2910760) |

### 模型驱动深度展开（B12）

把含物理数据一致性和正则项的迭代重建过程展开为可训练网络。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **B12** | Fourier-enhanced high-order total variation (FeHOT) iterative network for interior tomography（2025） | 把 HOT 与 primal-dual 迭代展开为网络，再用 Fourier-enhanced U-Net 精修。 | [PDF](deep_learning/B12_Fourier_enhanced_high_order_total_variation_FeHOT_iterative_network_for_interior_tomography.pdf) · [DOI](https://doi.org/10.1088/1361-6560/adc8f6) |

### 隐式神经表示 INR（B14）

用坐标网络表示衰减场，并根据当前扫描的投影逐病例优化网络。

| 编号 | 文献（年份） | 方法要点 | 阅读入口 |
|---|---|---|---|
| **B14** | Adaptive multi-resolution hash-encoding framework for INR-based dental CBCT reconstruction with truncated FOV（2026） | 哈希编码 INR 表示衰减场，在扩展重建域逐病例拟合本次扫描投影。 | [PDF](deep_learning/B14_Adaptive_multi_resolution_hash_encoding_framework_for_INR_based_dental_CBCT_reconstruction_with_truncated_FOV.pdf) · [DOI](https://doi.org/10.1002/mp.70226) |

## 收录状态与版本说明

- **传统方法**：7 篇 PDF；A09 有 TXT／HTML／XML 正文文件，未收录原版 PDF。
- **深度学习方法**：B01–B14 均有 PDF；B15 未收录全文。
- **未收录全文的条目**：A01、A03–A05、A08、A11–A12、B15。可通过各条目的 DOI 查看出版页面或获取渠道。
- **版本与年份**：本地 PDF 包括正式版、作者稿和预印本。B01 的预印本为 2017 年，会议论文为 2018 年；B08 按 2022 年期刊论文列出，其 arXiv 版本于 2025 年上传；B10、B14 分别按正式卷期年 2024、2026 年列出。

题录、方法摘要和分类依据来自[原始文献压缩包](../interior_CT_30papers.zip)内的
`download_manifest.csv` 与 `method_classification.json`；本地阅读链接按当前目录中的实际文件生成。
未收录全文的论文分类沿用包内记录的摘要与书目信息，A01 依据目录与书目信息。

## 常用缩写

| 缩写 | 含义 |
|---|---|
| ROI／VOI | 感兴趣区域／感兴趣体积 |
| DBP | 微分反投影（Differentiated Backprojection） |
| BPF／FBP | 反投影滤波／滤波反投影 |
| SVD | 奇异值分解 |
| TV／HOT | 全变分／高阶全变分 |
| INR | 隐式神经表示（Implicit Neural Representation） |
