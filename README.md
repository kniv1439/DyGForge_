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
- Only Classifier：加载模型在链路预测任务上训练好的backbone，只训练分类器。

### 模型名称映射表（Model Name Mapping）

| 论文模型名称 | 代码实际名称 | Attention | Positional Encoding | Patch Size |
|---|---|---|---|---|
| SimpleDyG | DyGFormer | MHSA | None | 4 |
| SimpleDyG (w/o Patch) | DyGFormerC | MHSA | None | 1 |
| DADyG | DyGFormerF | Differential Attention | None | 4 |
| NPADyG | NPTDyG | Non-Parametric Attention | None | 4 |
| CARDyG | 待搭建 | CARD Attention | None | 4 |
| SPSEDyG | DyGFormerSPSE | SPSE Attention Bias | None | 4 |
| SPEDyG | DyGFormerP | MHSA | Sinusoidal PE | 4 |
| RoPEDyG | DyGFormerR | MHSA | Rotary PE | 4 |
| SDPEDyG | DyGFormerST | MHSA | Shortest Distance PE | 4 |
| RWPEDyG | DyGFormerD | MHSA | Random Walk PE | 4 |
| STPEDyG | DyGFormerWSDMP | MHSA | Spatial-Temporal PE | 4 |
| DASTDyG | DyGFormerWSDMPF | Differential Attention | Spatial-Temporal PE | 4 |
| DARoDyG | DyGFormerFF | Differential Attention | Rotary PE | 4 |
| CASTDyG | 待搭建 | CARD Attention | Spatial-Temporal PE | 4 |
| CARoDyG | 待搭建 | CARD Attention | Rotary PE | 4 |
| NPASDDyG | NPTSTDyG | Non-Parametric Attention | Shortest Distance PE | 4 |
| NPASTDyG | NPTWSDMP | Non-Parametric Attention | Spatial-Temporal PE | 4 |

（注：组合模型均需重新跑完全部模块级实验之后对比选出Top-2的优势模块进行组合，注意力2*位置编码2，四个组合模型，各自分别在链路预测和节点分类上与十个baseline进行对比）


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

##已经跑完的实验结果放在saved_results目录下。

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

实验进展：目前已跑完SimpleDyG（DyGFormer）在enron，uci，mooc的链路预测三个runs的实验、DADyG（DyGFormerF）在除了USLegis以外的八个数据集上的三个runs的实验；

需要补充的实验：SimpleDyG、DADyG在各自剩余数据集上的链路预测的三个runs的实验、NPADyG在九个数据集上的链路预测的三个runs的实验、CARDyG需要结合CARD: Channel Aligned Robust Blend Transformer for Time Series
Forecasting.原始论文，将SimpleDyG里的标准多头注意力替换成原始论文里的CARD注意力作为CARDDyG，搭建好模型后跑完九个数据集上的链路预测的三个runs的实验、 SPSEDyG因为采取了SPSE基于图拓扑结构信息作为注意力分数的偏置，为了提高效率，需要对所有数据集的SPSE特征进行预计算并存储到磁盘里，需要同时考虑正负样本在计算SPSE时计算口径和方式一致，避免训练时出现正负样本SPSE不对称，导致模型分数更容易区分正负样本而拉高指标，再跑完九个数据集上的链路预测的三个runs的实验。目前SimpleDyG、DADyG、NPADyG 和 SPSEDyG模型代码都已搭建好，可直接训练；SPSEDyG需要再完善修改下正负样本的SPSE的预计算、存储方式再进行训练，CARDyG需要在SimpleDyG代码基础上替换成和原始论文一致的CARD注意力，再进行训练。

### E2. Positional Encoding 模块对比实验

**对应：§6.2.1、Table 4**

比较 SPEDyG、RoPEDyG、SDPEDyG、RWPEDyG 和 STPEDyG。

实验范围：九个链路预测数据集，包含 Transductive / Inductive AP 和 AUC-ROC。

目的：研究序列位置编码、图结构编码和时空联合编码的性能差异。

实验进展：目前已跑完RoPEDyG（DyGFormerR）在uci上的一个runs的实验、SDPEDyG（DyGFormerST）在UNvote上的一个runs的实验、RWPEDyG（DyGFormerD）在wikipedia上预计算rwpe dim=8的三个runs的实验，在mooc上的预计算rwpe dim=24的三个runs的实验。

需要补充的实验：SPEDyG（DyGFormerP）、RoPEDyG（DyGFormerR）、SDPEDyG（DyGFormerST）、RWPEDyG（DyGFormerD） 和 STPEDyG（DyGFormerWSDMP）在九个数据集上的链路预测的三个runs的实验。SDPEDyG（DyGFormerST）、RWPEDyG（DyGFormerD）同样因为是基于图拓扑结构的位置编码，可以预先提前对数据集进行预计算，存储来提高训练效率，也需要注意正负样本在计算特征时计算口径和方式一致。五个模型代码都已搭建完成，SDPEDyG（DyGFormerST）、RWPEDyG（DyGFormerD）的预计算代码我已完成好，分别名为precompute_spd_modelfn.py；precompute_rwpe_modelfn.py；因为RWPE里的random walk位置编码的特征维度K，代表它是一个K维的向量，最多包含经过K步游走回到它自身的概率，原始论文里在不同的数据集用了不同的K值，原始论文里是在2,4,8,12,16,20,24,28,32这些2的倍数里选择，我们也需要在这个范围里调一下K值，如果k的选择影响不大，可以不必花费太多时间在挑这个上面，并且运行RWPEDyG（DyGFormerD）的预计算代码时，可以通过更改文件里面预计算时的特征维度值（k）来改变这个预计算和存储的特征维度。


### E3. Patching 消融实验

**对应：§6.2.1、Figure 3、Table 8**

比较：

- SimpleDyG（默认 Patch Size = 4）
- SimpleDyG（Without Patching）

实验范围：九个链路预测数据集。

目的：分析 Patching 对细粒度时序信息与预测性能的影响，并核对不同指标上的实际变化。

实验进展：目前已跑完 SimpleDyG（Without Patching）（DyGFormerC）在mooc上的一个runs的实验；

需要补充的实验：SimpleDyG（Without Patching）（DyGFormerC）在剩余八个数据集上的链路预测的三个runs的实验。


### E4. 动态节点分类模块对比

**对应：§6.2.2、Figure 4**

在 Wikipedia、Reddit 上比较 11 个基础模型及单模块变体：

SimpleDyG、DADyG、CARDyG、NPADyG、SPSEDyG、SPEDyG、SDPEDyG、RWPEDyG、RoPEDyG、STPEDyG、SimpleDyG (w/o Patch)。

评价指标：AUC-ROC。

需要分别整理 Backbone & Classifier 和 Only Classifier 两种训练设置的实验结果。

目的：研究动态节点分类与动态链路预测是否具有不同的模块偏好。

需要补充的实验：需要重新对SimpleDyG、DADyG、CARDyG、NPADyG、SPSEDyG、SPEDyG、SDPEDyG、RWPEDyG、RoPEDyG、STPEDyG、SimpleDyG (w/o Patch) 以上所有模型跑完在wikipedia和reddit数据集上的Backbone & Classifier 和 Only Classifier 两种训练设置跑完3个runs，并针对注意力、位置编码挑选不同的Top-2优势模块进行组合，专门在E6. 节点分类组合模型与 Baseline 对比。

### E5. 链路预测组合模型与 SOTA 对比

**对应：§6.3.1、Figure 5、Table 6–7**

比较四种组合模型：

论文中的对照模型包括：

RepeatMixer、TIDFormer、DyGFormer、CNEN、TGAT、TGN、CAWN、EdgeBank、GraphMixer、TCL。

实验范围：九个链路预测数据集。

需要整理 Transductive / Inductive AP、AUC-ROC，以及模型平均排名。

目的：验证表现较好的模块组合能否取得有竞争力的结果。

需要补充的实验：需要根据跑完的所有Attention 模块对比实验和Positional Encoding 模块对比实验来挑选各自的Top-2优势模块分别进行组合，得到四个模型与十个baseline在链路预测的九个数据集上进行比较。

已跑完的实验：DASTDyG（DyGFormerWSDMPF）在CanParl, UNtrade, UNvote, USLegis,  uci, wikipedia上的三个runs的实验、在mooc，enron上的两个runs的实验，在reddit上的一个run的实验。

### E6. 节点分类组合模型与 Baseline 对比

**对应：§6.3.2、Figure 6**

主要比较：

待定

实验范围：Wikipedia、Reddit。

需要在 Backbone & Classifier、Only Classifier 两种训练设置下比较 AUC-ROC，并与 Figure 6 中的 Baseline 对应。

目的：检验不同 Attention 与 PE 的组合是否能够进一步提高节点分类性能。

需要补充的实验：需要重新在E4. 动态节点分类模块挑选出Top-2的优势模块进行组合，与RepeatMixer、TIDFormer、DyGFormer、CNEN、TGAT、TGN、CAWN、EdgeBank、GraphMixer、TCL十个baseline模型均重新对比跑完在wikipedia和reddit数据集上的Backbone & Classifier 和 Only Classifier 两种训练设置的3个runs的综合结果，均值±标准差。

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

需要补充的实验：需要重新对工作中所有模块级与组合模型级的实验以及十个baseline在链路预测Wikipedia、Reddit、MOOC、Can. Parl.、节点分类Wikipedia、Reddit上的GPU Memory Usage（MB）、Trainable Parameter Size（MB）、Training Time of the First Step（s）三种指标的统计。

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

需要补充的实验：需要重新对DARoDyG在只改变二阶邻居数量与Patch Size，而其他超参数与默认实验参数一致不变的情况下，跑完在Wikipedia、Can. Parl.上的链路预测的3个runs的实验，进行后续分析，同时统计Transductive AP、GPU Memory Usage、Training Time per Epoch三个指标。

##画图代码
论文里用到的雷达图，柱状图，气泡图的画图代码位于figure/目录下，热力图的代码在Latex文档中可以找到。

## 五、训练命令示例

## Scripts for Dynamic Link Prediction

## Model Training
  
- Example of training *DyGFormer* on *Wikipedia* dataset:

  ```bash
  python train_link_prediction.py --dataset_name uci --model_name DyGFormer --num_layers 2 --num_heads 4 --dropout 0.2 --max_input_sequence_length 512 --channel_embedding_dim 32 --patch_size 4 --learning_rate 0.0001 --weight_decay 0.0001 --batch_size 200 --num_epochs 100 --patience 20 --gpu 0
  ```


- If you want to use the best model configurations to train *DyGFormer* on *Wikipedia* dataset, run

```bash
python train_link_prediction.py --dataset_name uci --model_name DyGFormer --load_best_configs --gpu 0
```
## Scripts for Dynamic Node Classification

- Example of training *DyGFormer* on *Wikipedia* dataset:

- If you want to use the best model configurations to train *DyGFormer* on *Wikipedia* dataset, run
- Backbone & Classifier：联合训练模型骨干网络和分类器。
```bash
python train_node_classification.py --dataset_name wikipedia --model_name DyGFormer --node_cls_train_mode random_finetune --load_best_configs --gpu 0
```
  
- Only Classifier：只训练分类器。
```bash
python train_node_classification.py --dataset_name wikipedia --model_name DyGFormer --node_cls_train_mode classifier_only --load_best_configs --gpu 0
```
---


