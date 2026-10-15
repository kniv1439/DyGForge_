# DyGForge_

## 一、项目背景与研究目标

**论文名称：** DyGForge: A Modular Benchmark for Dissecting Transformer Architectures on Dynamic Graphs

本工作主要研究连续时间动态图（Continuous-Time Dynamic Graph, CTDG）中，Transformer 不同架构模块对下游任务预测性能和计算效率的影响。

现有基于 Transformer 的动态图模型通常会采用不同的输入编码、注意力机制与位置编码方法，但较少在统一框架下系统比较这些模块的独立贡献及组合效果。

因此，本项目主要围绕三个设计维度展开：

1. **Attention Mechanism**：比较不同注意力机制对动态图历史交互信息建模能力的影响。
2. **Positional Encoding**：分析时间位置编码、图结构位置编码以及时空联合位置编码的效果。
3. **Patching**：研究历史交互序列patch对预测精度、显存消耗和计算效率的影响。

### 主要研究问题

- **RQ1 — Module-wise Comparison：** 不同 Attention、PE 模块在动态链路预测和节点分类任务中表现如何？是否存在任务偏好？
- **RQ2 — Model-wise Comparison：** 将表现较好的 Attention 和 PE 组合后，能否取得优于现有 Baseline 的性能？
- **RQ3 — Performance-Efficiency Trade-off：** 不同模块的计算成本和性能收益之间有什么关系？
- **RQ4 — Patching and Higher-order Neighbors：** Patching 与二阶邻居数量如何共同影响模型性能和效率？

整体研究定位是动态图 Transformer 的模块化 Benchmark 与实验分析，而不只是提出单个新模型。

## 二、整体模型框架

本工作采用统一输入编码器，并允许替换不同 Transformer 模块，以保证实验对比尽可能公平。

模型主要流程为：

历史邻居采样 → Input Encoder → Positional Encoding → Patching（可选）→ Transformer Blocks → Pooling → Prediction Head。

### 1. Input Encoder

采用 DyGFormer 风格的输入编码，主要包括四个信息通道：

- Node Features
- Edge Features
- Time Interval Encoding
- Neighbor Co-occurrence Encoding

输入编码器原则上保持统一。

### 2. Attention Modules

比较以下五种注意力设计：

| 模型 | Attention |
|---|---|
| SimpleDyG | Multi-Head Self-Attention |
| DADyG | Differential Attention |
| NPADyG | Non-Parametric Attention |
| CARDyG | CARD Attention |
| SPSEDyG | Simple Path Structure Biased Attention |

### 3. Positional Encoding Modules

在 MHSA （Multi-Head Self-Attention）基础上比较以下五种位置编码方法：

| 模型 | Positional Encoding |
|---|---|
| SPEDyG | Sinusoidal PE |
| RoPEDyG | Rotary PE |
| SDPEDyG | Shortest Distance PE |
| RWPEDyG | Random Walk PE |
| STPEDyG | Spatial-Temporal PE |

### 4. Combined Models

动态链路预测重点比较四种组合：

| 模型 | Attention | PE |
|---|---|---|
| DASTDyG | Differential | Spatial-Temporal |
| DARoDyG | Differential | RoPE |
| CASTDyG | CARD | Spatial-Temporal |
| CARoDyG | CARD | RoPE |

动态节点分类重点比较：

- SDPEDyG
- STPEDyG
- NPASDDyG
- NPASTDyG

其中前两个模型已在 PE 模块比较中出现，不需要重复实现。

## 三、数据集与实验协议

### 1. 动态链路预测

共涉及九个数据集：

Wikipedia、Reddit、MOOC、Enron、UCI、Can. Parl.、US Legis.、UN Trade、UN Vote。

每个模型需要区分：

- Transductive：AP、AUC-ROC
- Inductive：AP、AUC-ROC

论文中的主要链路预测结果采用 Random Negative Sampling。

### 2. 动态节点分类

主要在 Wikipedia 和 Reddit 上进行实验。

评价指标为 AUC-ROC，包含两种训练设置：

- Backbone & Classifier：联合训练模型骨干网络和分类器。
- Only Classifier：只训练分类器。

### 模型名称映射表（Model Name Mapping）

| 论文模型名称 | 代码实际名称 | Attention | Positional Encoding | Patch Size |
|---|---|---|---|---|
| SimpleDyG | DyGFormer | MHSA | None | 4 |
| SimpleDyG (w/o Patch) | DyGFormerC | MHSA | None | 1 |
| DADyG | DyGFormerF | Differential Attention | None | 4 |
| NPADyG | NPTDyG | Non-Parametric Attention | None | 4 |
| CARDyG | 待填写 | CARD-inspired Attention | None | 4 |
| SPSEDyG | DyGFormerSPSE | SPSE Attention Bias | None | 4 |
| SPEDyG | DyGFormerP | MHSA | Sinusoidal PE | 4 |
| RoPEDyG | DyGFormerR | MHSA | Rotary PE | 4 |
| SDPEDyG | DyGFormerST | MHSA | Shortest Distance PE | 4 |
| RWPEDyG | DyGFormerD | MHSA | Random Walk PE | 4 |
| STPEDyG | DyGFormerWSDMP | MHSA | Spatial-Temporal PE | 4 |
| DASTDyG | DyGFormerWSDMPF | Differential Attention | Spatial-Temporal PE | 4 |
| DARoDyG | DyGFormerFF | Differential Attention | Rotary PE | 4 |
| CASTDyG | 待填写 | CARD-inspired Attention | Spatial-Temporal PE | 4 |
| CARoDyG | 待填写 | CARD-inspired Attention | Rotary PE | 4 |
| NPASDDyG | 待填写 | Non-Parametric Attention | Shortest Distance PE | 4 |
| NPASTDyG | 待填写 | Non-Parametric Attention | Spatial-Temporal PE | 4 |



### 3. 默认超参数

实验设置如下：

| 参数 | 配置 |
|---|---|
| Transformer Layers | 2 |
| Attention Heads | 4 |
| Dropout | 0.2 |
| Sequence Length | 512 |
| Encoding Dimension | 32 |
| Patch Size | 4 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 200 |
| Maximum Epochs | 100 |
| Early Stopping Patience | 20 |
| 数据划分 | 时间顺序 70% / 15% / 15% |



## 四、需要完成或核验的实验

## 数据集下载及预处理
Nine datasets are used in DyGForge, including Wikipedia, Reddit, MOOC, Enron, UCI, Can. Parl., US Legis., UN Trade, UN Vote.

All of the used original dynamic graph datasets come from Towards Better Evaluation for Dynamic Link Prediction, which can be downloaded [here [https://github.com/erfanloghmani/myket-android-application-market-dataset](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Please download them and put them in DG_data folder. 


We can run `preprocess_data/preprocess_data.py` for pre-processing the datasets. For example, to preprocess the *Wikipedia* dataset, we can run the following commands:

```bash
cd preprocess_data/
python preprocess_data.py --dataset_name wikipedia
```

We can also run the following commands to preprocess all the original datasets at once:

```bash
cd preprocess_data/
python preprocess_all_data.py
```



### E1. Attention 模块对比实验

**对应：§6.2.1、Figure 2、Table 5**

比较 SimpleDyG、DADyG、CARDyG、NPADyG 和 SPSEDyG。

实验范围：九个动态链路预测数据集，包含 Transductive / Inductive 的 AP 与 AUC-ROC。

目的：分析不同 Attention 机制对于历史交互信息建模的影响，以及在不同数据集上的表现差异。

### E2. Positional Encoding 模块对比实验

**对应：§6.2.1、Table 4**

比较 SPEDyG、RoPEDyG、SDPEDyG、RWPEDyG 和 STPEDyG。

实验范围：九个链路预测数据集，包含 Transductive / Inductive AP 和 AUC-ROC。

目的：研究序列位置编码、图结构编码和时空联合编码的性能差异。

### E3. Patching 消融实验

**对应：§6.2.1、Figure 3、Table 8**

比较：

- SimpleDyG（默认 Patch Size = 4）
- SimpleDyG（Without Patching）

实验范围：九个链路预测数据集。

目的：分析 Patching 对细粒度时序信息与预测性能的影响，并核对不同指标上的实际变化。

### E4. 动态节点分类模块对比

**对应：§6.2.2、Figure 4**

在 Wikipedia、Reddit 上比较 11 个基础模型及单模块变体：

SimpleDyG、DADyG、CARDyG、NPADyG、SPSEDyG、SPEDyG、SDPEDyG、RWPEDyG、RoPEDyG、STPEDyG、SimpleDyG (w/o Patch)。

评价指标：AUC-ROC。

需要分别整理 Backbone & Classifier 和 Only Classifier 两种训练设置的实验结果。

目的：研究动态节点分类与动态链路预测是否具有不同的模块偏好。

### E5. 链路预测组合模型与 SOTA 对比

**对应：§6.3.1、Figure 5、Table 6–7**

比较四种组合模型：

DASTDyG、DARoDyG、CASTDyG、CARoDyG。

论文中的对照模型包括：

RepeatMixer、TIDFormer、DyGFormer、CNEN、TGAT、TGN、CAWN、EdgeBank、GraphMixer、TCL。

实验范围：九个链路预测数据集。

需要整理 Transductive / Inductive AP、AUC-ROC，以及模型平均排名。

目的：验证表现较好的模块组合能否取得有竞争力的结果。

### E6. 节点分类组合模型与 Baseline 对比

**对应：§6.3.2、Figure 6**

主要比较：

SDPEDyG、STPEDyG、NPASDDyG、NPASTDyG。

实验范围：Wikipedia、Reddit。

需要在 Backbone & Classifier、Only Classifier 两种训练设置下比较 AUC-ROC，并与 Figure 6 中的 Baseline 对应。

目的：检验不同 Attention 与 PE 的组合是否能够进一步提高节点分类性能。

### E7. 计算效率实验

**对应：§6.4、Figure 7–8、Table 9–10**

动态链路预测主要涉及：

Wikipedia、Reddit、MOOC、Can. Parl.。

节点分类主要涉及：

Wikipedia、Reddit。

需要统计以下指标：

- GPU Memory Usage（MB）
- Trainable Parameter Size（MB）
- Training Time of the First Step（s）

需要区分模块级模型和组合模型、Baseline 的效率对比。



### E8. Patching 与二阶邻居数量实验

**对应：§6.5、Figure 9**

采用 DARoDyG 作为基础模型。

数据集：Wikipedia、Can. Parl.。

二阶邻居数量：

`0、16、32、64、128`

Patch Size：

`1、4、8、16、32`

两个维度组成 25 种配置，每个数据集分别进行实验。

需要统计：

- Transductive AP
- GPU Memory Usage
- Training Time per Epoch

对于出现 OOM 的设置，也需要保留完整记录。

目的：研究二阶邻居数量与 Patch Size 对模型预测效果和效率的共同影响。

## 五、历史实验结果整理说明

目前项目已经开展过部分模型训练和实验，但相关结果仍需依据实际文件进一步核实。

前期实验曾在不同计算设备和运行环境下进行，部分实验记录、输出文件和统计结果没有完全统一整理。此外，部分实验尚未完成完整测试。

接手后建议首先整理现存实验结果，而不是立即重新运行全部实验。

### 实验状态分类

| 状态 | 含义 | 后续工作 |
|---|---|---|
| Verified | 结果、配置和计算方式已核验 | 整理为最终统计表 |
| Pending Aggregation | 各 Seed 结果文件完整，但未汇总 | 计算 Mean ± Std |
| Pending Verification | 结果文件存在，但对应关系或指标尚未核实 | 核对配置、输出和评估代码 |
| Needs Rerun | 实验不完整、文件缺失或已知结果有误 | 补跑或重新运行 |

### 多 Seed 结果整理

对于已经完成多个 Seed 的实验：

1. 找到各个 Seed 对应的原始结果文件。
2. 核对模型、数据集、配置、测试设置和 Seed。
3. 如果已保存各 Seed 的指标，确认评估方式一致后统一汇总。
4. 如果只保存 Prediction Scores 和 Labels，需要先使用统一评估脚本重新计算 AP、AUC。
5. 计算 Mean ± Std，并记录实际参与统计的 Seed 数量及标准差定义。
6. 对原始记录缺失或者无法确认有效性的实验进行标注，必要时重新运行。

历史统计结果如果缺少原始文件支持，暂时只作为参考，不直接视为最终论文结果。



## 六、建议的后续工作顺序

**第一阶段：熟悉代码与现有实验**

了解模块实现、模型入口、训练脚本、数据集和基本运行流程。

**第二阶段：整理已有结果**

汇总不同服务器上的现存结果，识别各实验对应的 Seed、参数配置和评价指标。

**第三阶段：核验并重新统计指标**

统一评估脚本，对可核实的历史结果重新计算各 Seed 指标及 Mean ± Std，优先完成已有结果的整理。

**第四阶段：补充缺失实验**

根据 E1–E8 的实验清单，补充尚未完成或需要重新运行的实验。

**第五阶段：更新论文**

根据经核实的实验结果更新表格、图像和分析结论，保证文本、代码及结果一致。

---

**交接说明：**

本次交接主要提供项目已有的模型代码、实验脚本、现存结果文件及已知问题说明。具体实验完成状态仍需根据代码和原始文件确认。后续实验安排、模型优化及论文修改可在此基础上继续开展。
