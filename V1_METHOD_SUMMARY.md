# V1：噪声感知病灶保持 Patch Graph 方法总结

## 1. 方法定位

V1 建立在 AA-CLIP 的异常感知文本—图像对齐框架上，面向医学图像中噪声、
低对比度病灶和跨模态域偏移问题，引入一个**模态噪声感知、病灶保持的
Patch Graph**，并使用 CLIP 最终 CLS 输出作为图像级全局语义锚点。

V1 的目标不是重建或去噪输入图像，而是利用原始视图和空间对齐的辅助噪声视图，
估计每个 patch 对噪声的敏感程度，并在特征图上进行可靠性加权的图传播：

- 可靠的正常 patch 可以提供上下文；
- 高噪声 patch 的外传信息受到抑制；
- 正常区域与疑似病灶之间的跨边界传播受到抑制；
- 疑似病灶 patch 减少图更新，避免被正常组织过度平滑。

V1 **不使用扩散模型、生成式去噪器或知识蒸馏**。辅助噪声视图是一个不改变空间
位置的强度扰动视图，仅用于估计特征不确定性。

## 2. 总体框架

V1 保留 AA-CLIP 的两阶段适配流程：

1. **文本适配阶段**：学习正常与异常文本锚点；
2. **图像适配阶段**：固定文本锚点，训练 Image Adapter 和 V1 Patch Graph。

总体数据流为：

```text
正常/异常提示
      │
      ▼
冻结的 CLIP Text Encoder + Text Adapter
      │
      └──────────────> 正常/异常文本锚点 (t_n, t_a)

原始医学图像 I ─────────> 共享图像编码器 ──> 主视图特征 P
      │
      └─> 模态相关噪声 T_m ─> 辅助视图 Ĩ ─> 辅助视图特征 R（无梯度）

P、R、文本锚点
      │
      ▼
噪声感知病灶保持 Patch Graph
      │
      ▼
多尺度精炼特征 ─> 文本相似度 ─> Pixel Anomaly Map ─> max ─┐
                                                          ├─> Medical Image Score
CLIP 最终 CLS ─> 与图检测特征融合 ─> Global Score ─────────┘
```

完整 Mermaid 方法图见 [V1_METHOD_DIAGRAM.md](V1_METHOD_DIAGRAM.md)。

## 3. 异常感知文本锚点

给定正常文本提示和异常文本提示，Text Adapter 在冻结 CLIP 文本编码器的基础上
生成正常与异常文本锚点：

\[
T=[t_n,t_a], \qquad t_n,t_a\in\mathbb{R}^{D}.
\]

文本适配阶段利用图像 patch 特征与训练 mask，使正常和异常文本表示能够为医学
病灶定位提供更清晰的语义方向，同时使用正交约束减少两个锚点之间的混淆。

完成文本适配后，文本锚点在图像适配阶段保持固定，用于：

- 计算 patch 正常/异常相似度；
- 产生图传播所需的异常先验；
- 计算双视图预测一致性；
- 生成最终异常热图。

## 4. 模态相关辅助噪声视图

对原始图像 \(I\) 施加保持空间对齐的强度扰动：

\[
\widetilde I=T_m(I;\sigma),
\]

其中 \(m\) 表示医学模态，\(\sigma\) 表示噪声强度。当前实现的默认映射为：

| 数据集 | 模态近似 | 辅助扰动 |
|---|---|---|
| Brain | MRI | Rician 幅值噪声 + 低频 bias field |
| Liver | CT | 信号相关量子噪声近似 |
| DDTI | Ultrasound | 乘性 speckle + 弱电子噪声 |
| Retina | Fundus | shot noise + 弱加性噪声 |
| Colon | Endoscopy | 光照变化 + 弱加性噪声 |

这些扰动是用于构造噪声敏感性探针的近似机制，并不宣称完整模拟真实医学设备的
采集物理过程。

辅助视图不进行旋转、裁剪或空间形变，因此与原始图像及训练 mask 保持像素对齐。
当前默认强度为：

```text
noise_severity = 0.06
```

## 5. 共享双视图特征编码

主视图和辅助视图通过共享参数的 CLIP 图像编码器与 Image Adapter：

\[
P^{(l)}=E_l(I), \qquad R^{(l)}=E_l(\widetilde I),
\]

其中 \(l\in\{6,12,18,24\}\) 表示四个视觉层级。

CLIP 主干被冻结，前若干 Transformer 层中的 Image Adapter、特征投影层和 Patch
Graph 可训练。辅助视图分支在 `torch.no_grad()` 下运行：

- 主分支承担梯度更新；
- 辅助分支作为稳定的噪声参考；
- 避免双分支同时漂移以投机性降低一致性损失；
- 降低训练显存消耗。

主视图还保留第 24 层的 CLS token，并执行与 CLIP 原始视觉输出一致的全局池化、
`ln_post` 和视觉投影，得到最终全局特征 (C)。该特征不参与 Pixel Anomaly Map，
仅用于图像级语义锚定。

## 6. Patch 噪声不确定性

对每个 patch，使用主视图和辅助视图特征的余弦差异估计噪声不确定性：

\[
u_i=\frac{1-\cos(P_i,R_i)}{2}, \qquad u_i\in[0,1].
\]

- \(u_i\) 较小：该 patch 对扰动稳定，信息更可靠；
- \(u_i\) 较大：该 patch 对噪声敏感，不应大量向邻居传播信息。

V1 不把该不确定性解释为严格校准的概率，而是将其作为图传播中的相对可靠性
指标。

## 7. 图输入与异常先验

### 7.1 双视图图输入

V1 将主视图和辅助视图特征平均后作为图节点特征：

\[
G_i=\frac{P_i+R_i}{2}.
\]

这种融合可以降低单次强度扰动产生的随机波动，但也意味着噪声辅助特征会直接参与
V1 的图输入。这是 V1 与后续 primary-only 设计的主要区别之一。

### 7.2 文本引导异常先验

使用正常/异常文本锚点计算 patch 异常概率：

\[
a_i=operatorname{Softmax}
\left(10G_i^\top[t_n,t_a]\right)_{\mathrm{abnormal}}.
\]

该异常先验来自模型预测，而不是测试 mask，因此目标域推理时不需要标注。

## 8. 混合 Patch Graph

### 8.1 语义邻接

对归一化 patch 特征计算两两余弦相似度，并为每个节点选择 Top-K 语义邻居：

\[
A_{\mathrm{KNN},ij}=1
\quad\text{if}\quad
j\in\mathcal N_K(i).
\]

当前默认：

```text
patch_graph_k = 8
```

KNN 邻接经过对称化，使任一方向被选中的节点对都能够建立连接。

### 8.2 空间邻接

如果 patch 数量可以构成规则二维网格，则为每个 patch 加入周围八邻域空间边，得到
\(A_{\mathrm{spatial}}\)。

### 8.3 混合拓扑

将语义图和空间图组合：

\[
A=\alpha A_{\mathrm{KNN}}+(1-\alpha)A_{\mathrm{spatial}}.
\]

当前默认：

```text
patch_graph_alpha = 0.7
```

即语义邻接占主要作用，同时保留局部空间连续性。

## 9. 噪声感知与病灶保持的边权

V1 对基础图边加入三种连续权重。

### 9.1 特征亲和度

\[
F_{ij}=\exp\left(
\frac{\cos(G_i,G_j)-1}{\tau_f}
\right).
\]

特征差异越大的 patch，消息传播权重越低。

### 9.2 噪声源可靠性

\[
R_j=1-u_j.
\]

这里 \(j\) 是消息的来源节点。高不确定性源节点向其他节点传播的信息受到抑制，
最低可靠性在实现中被截断为 0.05，避免节点完全失去连接。

### 9.3 病灶边界亲和度

\[
B_{ij}=\exp\left(
-\frac{|a_i-a_j|}{\tau_a}
\right).
\]

如果两个相邻 patch 的异常概率差异较大，则它们可能位于病灶边界两侧，相应边权
被减小。

### 9.4 最终加权邻接

\[
\widetilde A_{ij}
=A_{ij}\,F_{ij}\,B_{ij}\,(1-u_j).
\]

加入自环并执行行归一化后得到转移矩阵：

\[
\Pi_{ij}
=\frac{\widetilde A_{ij}+\delta_{ij}}
{\sum_k(\widetilde A_{ik}+\delta_{ik})}.
\]

## 10. 图消息传播与病灶保护门控

邻居信息首先进行加权聚合和线性投影：

\[
Z_i=\operatorname{LN}\left(
W\sum_j\Pi_{ij}G_j
\right).
\]

然后计算节点级更新门：

\[
q_i=\sigma\left(
s_u u_i-s_a a_i+b
\right),
\]

其中 \(s_u\) 和 \(s_a\) 由可训练参数经过 Softplus 得到。最终输出为：

\[
F'_i=(1-q_i)G_i+q_iZ_i.
\]

该门控具有直接的可解释性：

- **高不确定性** \(u_i\) 增大更新门，使受噪声影响的 patch 更多请求可靠邻居上下文；
- **高异常概率** \(a_i\) 减小更新门，使疑似病灶更多保留自身特征；
- 结合边界亲和度，减少正常组织特征跨越病灶边界的传播。

## 11. 多尺度预测

V1 对四个层级的精炼 patch 特征分别与文本锚点计算相似度，得到多尺度异常图。
所有异常图经过高斯平滑、双线性上采样并求和，得到最终 Pixel Anomaly Map：

\[
M(x,y)=\sum_l M^{(l)}(x,y).
\]

检测图分支首先对精炼 patch 特征聚合，得到 $Z_g$，再与 CLIP 最终全局特征融合：

\[
Z_{\mathrm{img}}
=\operatorname{Norm}\left[
(1-\lambda_{\mathrm{cls}})Z_g+\lambda_{\mathrm{cls}}C
\right].
\]

当前默认 `clip_global_weight=0.2`，即保留 80% 图检测表示并加入 20% CLIP
全局语义锚点。随后由 $Z_{\mathrm{img}}$ 与文本锚点得到全局异常分数
$S_{\mathrm{global}}$。医学图像最终得分为：

\[
S_{\mathrm{image}}
=(1-\mu)\max_{x,y}M(x,y)+\mu S_{\mathrm{global}}.
\]

融合前，局部最大响应和全局分数按当前评估协议分别做数据集级
Min–Max 归一化。当前默认 `medical_image_score_global_weight=0.2`。
Pixel AUC/AP 仍完全由局部热图
计算，因此加入 CLS 不会直接修改像素定位结果。两个权重必须通过源域验证协议固定，
不能根据 Liver、DDTI 等目标测试集单独调整。

## 12. 训练目标

### 12.1 图像分类损失

\[
\mathcal L_{\mathrm{cls}}
=\operatorname{CE}(S_{\mathrm{det}},y).
\]

### 12.2 多尺度分割损失

每个层级使用 Focal Loss 与正常/异常双通道 Dice Loss：

\[
\mathcal L_{\mathrm{seg}}
=\sum_l
\left(
\mathcal L_{\mathrm{focal}}^{(l)}
+\mathcal L_{\mathrm{dice}}^{(l)}
\right).
\]

### 12.3 双视图预测一致性

主视图与辅助视图特征分别投影到正常/异常文本空间，在所有层级计算对称 KL：

\[
\mathcal L_{\mathrm{cons}}
=\frac{1}{2}
\left[
D_{\mathrm{KL}}(p\|r)
+D_{\mathrm{KL}}(r\|p)
\right].
\]

该损失约束语义预测对强度噪声保持稳定，而不是要求两个视图的原始特征完全相同。

### 12.4 病灶特征保持损失

在训练 mask 覆盖的病灶 patch 上，约束精炼特征不要偏离图传播前的主视图特征：

\[
\mathcal L_{\mathrm{pres}}
=\frac{1}{|\Omega_L|}
\sum_{i\in\Omega_L}
\left[1-\cos(F'_i,\operatorname{sg}(P_i))\right].
\]

其中 `sg` 表示停止梯度，防止目标特征与输出特征共同漂移来投机性降低损失。

### 12.5 病灶边界对比损失

对训练 mask 中跨越正常/病灶边界的水平与垂直 patch 对，要求其特征距离不低于
边界间隔 \(m\)：

\[
\mathcal L_{\mathrm{boundary}}
=\frac{1}{|\mathcal E_B|}
\sum_{(i,j)\in\mathcal E_B}
\max\left(0,m-d_{\cos}(F'_i,F'_j)\right).
\]

### 12.6 总损失

\[
\mathcal L
=\mathcal L_{\mathrm{cls}}
+\mathcal L_{\mathrm{seg}}
+\lambda_c\mathcal L_{\mathrm{cons}}
+\lambda_p\mathcal L_{\mathrm{pres}}
+\lambda_b\mathcal L_{\mathrm{boundary}}.
\]

当前 `train.bat` 使用：

```text
λ_c = 0.10
λ_p = 0.10
λ_b = 0.05
boundary_margin = 0.20
```

## 13. 训练与推理的区别

### 训练阶段

- 使用源域图像、图像标签和病灶 mask；
- 生成源域模态对应的辅助噪声视图；
- mask 只进入分割、病灶保持和边界损失；
- CLIP 主干冻结；
- Text Adapter 在第一阶段训练；
- Image Adapter 与 Patch Graph 在第二阶段训练。
- 图像分类损失作用于图检测特征与最终 CLIP CLS 的融合表示。

### 推理阶段

- 输入目标数据集图像；
- 根据目标数据集生成模态对应的辅助噪声视图；
- 不使用目标图像标签或病灶 mask；
- 通过双视图估计不确定性并完成图传播；
- 输出 Pixel Anomaly Map；
- 使用热图最大响应和 CLS—图融合全局分数共同计算医学 Image Score。

当前 `test.py` 中的 `--noise_severity` 只控制**内部辅助噪声视图**。当前版本没有给
主测试输入额外加噪，因此默认测试仍然是“干净主输入 + 噪声辅助探针”。

另外，当前 `test.py` 将图像 checkpoint 固定为：

```text
image_adapter_2.pth
```

因此它不会自动测试第一轮或全部轮次。如果论文实验采用第一轮 V1，必须显式修改
checkpoint，或者使用独立的批量评估脚本；同时应通过验证协议预先确定轮次，不能按
目标测试集结果逐数据集选择不同 checkpoint。

## 14. V1 的主要创新点

### 14.1 模态相关噪声敏感性估计

不依赖额外去噪网络，而是利用空间对齐的辅助强度视图估计 patch 对噪声的相对
敏感性。

### 14.2 噪声可靠性引导的图传播

将 patch 不确定性显式引入图的消息来源权重，避免高噪声 patch 污染周围区域。

### 14.3 文本引导的病灶保护

使用正常/异常文本锚点产生异常先验，在不依赖测试 mask 的情况下抑制疑似病灶的
图更新和跨边界传播。

### 14.4 训练期病灶与边界约束

利用源域 mask 限制图传播造成的病灶特征漂移和边界过平滑，同时保持目标域无标注
推理能力。

### 14.5 图像级与像素级协同改善

通过可靠 patch 传播与病灶响应保持改善局部异常图，同时使用 CLIP 最终 CLS 约束
图像级语义，避免图像判断完全依赖单个最大 patch。全局分数只参与 Image Score，
不直接广播到局部 patch，避免损害病灶定位。

## 15. 鲁棒性的理论解释

V1 的鲁棒性不是严格的最坏情况鲁棒性证明，而是具有以下结构性依据。

### 15.1 噪声源抑制

如果某个源 patch 的特征在辅助扰动下变化明显，则 \(u_j\) 增大，消息权重
\((1-u_j)\) 减小，因此其噪声扰动不容易通过图传播扩散到其他节点。

### 15.2 相似邻居聚合

语义 KNN 和连续特征亲和度优先聚合相似 patch，可在局部噪声破坏单个 patch 时
利用上下文恢复更稳定的语义证据。

### 15.3 病灶区域低更新

异常先验通过 \(-s_a a_i\) 减小疑似病灶接收图更新的比例，降低正常组织上下文覆盖
病灶特征的风险。

### 15.4 边界传播抑制

异常概率差异会降低边界两侧的连接权重，训练期边界损失进一步保持正常与病灶
特征的分离。

这些机制支持“对训练和测试所覆盖噪声分布具有经验鲁棒性”的论述，但不能单凭网络
结构宣称对所有未知噪声具有理论保证。鲁棒性结论仍需通过多数据集、多噪声强度和
多随机种子实验验证。

## 16. 当前优势与实验现象

加入 CLS 之前的原 V1 实验观察表明：

- V1 第一轮在 Liver 和 DDTI 上优于后续轮次；
- Liver 上 Pixel AUC 保持的同时，Image AUC 相比 AA-CLIP 提高约 13 个百分点
  （最终论文需使用完全一致的测试设置复核）；
- 这说明原 V1 可能改善了图像间最大异常响应的排序，同时没有明显破坏全局像素排序。

上述结果不能直接作为“V1 + CLS Score”的结果。加入 CLS 后必须重新训练或至少明确
标注为旧 checkpoint 的后验融合实验，并重新报告所有指标。

第一轮最优很可能与以下因素有关：

- Brain full-shot 每轮包含大量优化步骤；
- Image Adapter 约有千万级可训练参数；
- 默认图像学习率 `5e-4` 对预训练模型微调可能偏大；
- 继续训练会进一步适应 Brain 的纹理、病灶形态和增强分布，损害跨域泛化；
- 图传播和一致性约束训练过久可能造成小病灶过平滑。

因此，第一轮并非“训练不足”，而可能已经是通用预训练特征与源域医学适配之间的
最佳平衡点。

## 17. 当前局限性

### 17.1 V1 仅使用一个辅助噪声视图

单次不确定性估计可能受到随机噪声实现影响，需要固定随机种子并通过多次运行验证
稳定性。

### 17.2 辅助特征直接参与图输入

V1 使用 \((P+R)/2\) 作为图输入。当辅助噪声过强时，噪声特征可能直接进入预测
分支。

### 17.3 Hard KNN 拓扑不连续

输入特征发生轻微变化时，Top-K 邻居集合可能突然变化，因此 Hard KNN 本身不提供
严格的连续稳定性保证。

### 17.4 噪声模型是近似机制

当前扰动没有模拟完整扫描协议、重建算法、设备差异和真实临床伪影，论文中应表述
为“模态启发的噪声近似”，避免过度物理化解释。

### 17.5 源域 mask 依赖

目标域推理不需要 mask，但病灶保持和边界约束需要源域像素级标注。

### 17.6 当前 checkpoint 选择风险

不能根据 Liver、DDTI 等目标测试集选择最优 epoch，再报告同一测试集结果。正式
实验应通过独立验证集或预先固定的源域协议选择 checkpoint。

## 18. 推荐消融实验

为证明每个模块的贡献，建议使用相同训练数据、随机种子和 checkpoint 选择策略进行：

| 实验 | 噪声视图 | Patch Graph | 不确定性 | 异常门控 | 病灶约束 | CLS锚点 | 全局评分融合 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AA-CLIP | × | × | × | × | × | × | × |
| Graph only | × | ✓ | × | × | × | × | × |
| Noise-aware Graph | ✓ | ✓ | ✓ | × | × | × | × |
| Noise + Anomaly Gate | ✓ | ✓ | ✓ | ✓ | × | × | × |
| 原 V1 Full | ✓ | ✓ | ✓ | ✓ | ✓ | × | × |
| V1 + CLS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| V1 + CLS Score | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

还应报告：

- 干净输入性能；
- 不同噪声强度下的性能；
- Brain、Liver、DDTI、Retina、Colon 跨域结果；
- Image AUC/AP 与 Pixel AUC/AP；
- 多随机种子的均值和标准差；
- V1 相对 AA-CLIP 的参数量、显存和推理时间。

## 19. 具体病例分析

[case_analysis.py](case_analysis.py) 用于在同一医学测试输入上比较原始 AA-CLIP 和
V1。脚本输出：

```text
case_predictions.csv
image_summary.csv
selected_cases.csv
analysis_info.txt
cases/*.png
```

自动选择的病例包括：

- 异常图像排名改善；
- 正常图像误报抑制；
- 小病灶保持；
- V1 失败案例。

病例按照组内中位数而不是最大提升选择，以减少 cherry-picking。案例图展示原图、
测试输入、GT、AA-CLIP热图、V1热图和V1不确定性图。

## 20. 论文中建议使用的名称

中文：

> 噪声感知病灶保持 Patch 图适配网络

英文：

> Noise-Aware Lesion-Preserving Patch Graph Adaptation

推荐缩写：

> **NLPG-Adapter**

如果希望名称更强调医学场景，也可以使用：

> **Med-NLPG**: Medical Noise-Aware Lesion-Preserving Graph

## 21. 建议论文表述

可以将 V1 的核心思想概括为：

> We construct a spatially aligned, modality-aware auxiliary intensity view
> to estimate patch-wise noise sensitivity without image reconstruction. The
> proposed lesion-preserving graph suppresses messages from unreliable source
> patches and inhibits propagation across predicted lesion boundaries, while
> an anomaly-aware update gate prevents normal-context over-smoothing at
> suspicious regions. Source masks are used only to regularize lesion features
> and boundaries during adaptation and are not required at target-domain
> inference.

论文中应避免以下过强表述：

- “理论上对任意噪声鲁棒”；
- “精确模拟真实MRI/CT/超声噪声”；
- “不需要任何像素级标注”；
- 在没有完成原 V1、V1+CLS、V1+CLS Score 消融前，直接声称“Image AUC 提升来自
  全局分类分支”。

更准确的结论是：

> V1 通过模态启发的辅助扰动、patch可靠性图传播和训练期病灶保持约束，提高了医学
> 异常检测在所评估噪声与跨域设置下的经验稳定性，同时尽量保留病灶定位能力。
