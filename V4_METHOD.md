# V4.1：模态条件化多层图拉普拉斯残差融合

## 1. 方法定位

V4.1 是与 V1、V3 独立的简化实验分支。它保留冻结 CLIP 第 6、12、18、24 层特征，
但删除 V3 的 Haar 小波、复杂图残差 Head、病灶门、频带干预和多分支图像池化。

V4.1 只有一个学习模块：根据固定人工模态模板生成四层权重，以及每层一阶、二阶
图拉普拉斯的有界有符号残差系数。CLIP Image Encoder 和 Text Encoder 始终完全冻结。

当前默认使用仓库原有的正常/异常模板文本（`--prompt_source template`），训练和测试
过程中不调用 LLM。LLM Prompt Bank 只作为后续独立文本消融，不与模板 checkpoint
混用。

## 2. 总体结构

```text
医学图像
   ↓
冻结 CLIP Image Encoder
   ↓
第 6/12/18/24 层 Patch 特征
   ↓
各层正常—异常文本 Margin
   ↓
固定四邻域图拉普拉斯：m、Lm、L²m
   ↓
模态模板条件化的层权重与 L/L² 有界残差系数
   ↓
Pixel Anomaly Map
   ├── 有效 Mask 的 Pixel Loss
   └── 固定局部读出 ─┐
冻结最终 CLS Margin ─┴→ 固定平均 → Image BCE
```

V4.1 当前只使用上下左右四个邻居，不使用 V3 的八邻域传播，也没有可训练图卷积。

## 3. 四层冻结文本异常响应

给定第 \(l\) 层归一化 Patch 特征 \(F_i^{(l)}\) 和固定正常、异常文本锚点
\(t_n,t_a\)，基础异常 Margin 为：

\[
m_i^{(l)}=\tau_t\left[
\langle F_i^{(l)},t_a\rangle-
\langle F_i^{(l)},t_n\rangle
\right],
\qquad l\in\{6,12,18,24\}.
\]

四层分别提供纹理、局部结构和深层语义证据，但不再分别运行四个复杂 Head。

## 4. 固定四邻域图拉普拉斯

每层 Patch 网格只连接上下左右四个位置。相邻 Patch 的固定语义亲和度为：

\[
A_{ij}^{(l)}=
\exp\left(
\frac{\cos(F_i^{(l)},F_j^{(l)})-1}{\tau_g}
\right).
\]

令 \(P=D^{-1}A\) 为随机游走归一化邻接矩阵，则图拉普拉斯为：

\[
L=I-P.
\]

对每层 Margin 构造三个谱阶次：

\[
B_0^{(l)}=m^{(l)},\qquad
B_1^{(l)}=Lm^{(l)},\qquad
B_2^{(l)}=L^2m^{(l)}.
\]

- \(B_0\) 保留原始文本异常语义和弥漫响应；
- \(B_1\) 描述一次局部图差异；
- \(B_2\) 描述更高阶的局部谱变化。

它是同一个图算子的不同阶次，不再额外叠加小波频率分支。

## 5. 模态条件化有界残差融合

正常/异常锚点继续使用仓库原模板。条件生成器则独立使用以下固定人工模态模板：

| 数据集 | 模态模板 |
|---|---|
| Brain | `a brain MRI scan` |
| Liver | `a liver CT scan` |
| Retina | `a retinal OCT scan` |
| Chest | `a chest X-ray` |
| Retina_OCT2017 | `a retinal OCT scan` |
| Histopathology | `a histopathology microscopy image` |

这些模板只描述已知成像模态，不由 LLM 生成，也不包含目标图像、诊断、Mask 或标签。
冻结 Text Encoder 将对应模板编码为模态条件向量 \(t_m\)。

条件生成器首先产生四层非负权重：

\[
\alpha=\operatorname{Softmax}(W_\alpha t_m+b_\alpha),
\qquad \sum_l\alpha_l=1.
\]

默认保留 20% 均匀层权重质量，因此四层中每层至少占 5%。一阶、二阶谱系数使用
有界 `tanh`：

\[
\beta_{l,k}=\beta_{max}\tanh
\left((W_\beta t_m+b_\beta)_{l,k}\right),
\qquad k\in\{1,2\}.
\]

最终异常图为四层原始语义主干加图拉普拉斯残差：

\[
S=\sum_l\alpha_l\left[
m^{(l)}+\beta_{l,1}Lm^{(l)}+\beta_{l,2}L^2m^{(l)}
\right].
\]

所有条件器参数初始化为零，因此训练开始时 \(\alpha_l=1/4\)、\(\beta_{l,k}=0\)，
模型严格等于四层原始 CLIP Margin 均值。图拉普拉斯只能在训练后作为可正可负、幅度
不超过 `max_spectral_coefficient` 的残差修正，不能在初始化时稀释原始语义。对于
768 维 CLIP 特征，两组线性条件器合计仍为 9,228 个可训练参数。

严格 LODO 时，目标数据集不参加训练；测试系数只由目标数据集的固定模态模板生成，
不读取目标图像、Mask 或标签进行选择。

## 6. 固定图像读出与 CLS 基线

V4.1 不训练 Top-k、GeM 或 CLS 混合权重。局部图像 Logit 使用固定温度的无参数注意力
读出：

\[
q_i=\frac{\exp(S_i/\tau_r)}{\sum_j\exp(S_j/\tau_r)},
\qquad
z_{local}=\sum_iq_iS_i.
\]

最终 CLS 使用同一组正常—异常文本锚点得到 \(z_{CLS}\)，图像 Logit 固定为：

\[
z_{image}=\frac{z_{CLS}+z_{local}}{2}.
\]

因此 CLS 提供全局语义基线，局部谱融合器提供可训练的 Patch 证据，但不存在额外
可学习池化模块。

## 7. 训练目标

所有样本都使用图像标签：

\[
\mathcal L_{image}=\operatorname{BCE}(z_{image},y).
\]

正常样本具有可信全零 Mask；异常样本只有在真实 Mask 存在时才参加 Pixel
Focal/Dice 损失。异常无 Mask 的 HIS、Chest 样本不会被伪造为全零异常图。

\[
\mathcal L=\lambda_I\mathcal L_{image}
+\lambda_P\mathcal L_{pixel}.
\]

`train_v4.bat` 和 `train_v4_lodo.bat` 显式提供：

```bat
--image_loss_weight 1.0 ^
--pixel_loss_weight 1.0 ^
```

可以手动把 Pixel Loss 权重改为 `0.5`、`0.25` 或其他非负值，但正式 LODO 实验必须
根据源域验证协议统一选择，不能查看目标测试 AUC 后为不同目标分别调整。

## 8. 运行

单数据集训练和跨数据集测试：

```bat
train_v4.bat
test_v4.bat
```

严格 BMAD LODO，例如留出 Chest：

```bat
train_v4_lodo.bat Chest
test_v4_lodo.bat Chest
```

默认保存目录为：

```text
ckpt/v4_1_graph_spectral
ckpt/v4_1_bmad_lodo/<TARGET>
```

V4.1 checkpoint 使用独立方法标识 `modality_graph_spectral_v4_1`、`version=4` 和
`revision=1`，并记录固定模态模板哈希。旧 V4、V3 checkpoint 均不能混用。
