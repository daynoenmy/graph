# V4：模态条件化多层图拉普拉斯谱融合

## 1. 方法定位

V4 是与 V1、V3 独立的简化实验分支。它保留冻结 CLIP 第 6、12、18、24 层特征，
但删除 V3 的 Haar 小波、复杂图残差 Head、病灶门、频带干预和多分支图像池化。

V4 只有一个学习模块：根据冻结医学文本生成四个特征层、三个图拉普拉斯阶次的联合
融合权重。CLIP Image Encoder 和 Text Encoder 始终完全冻结。

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
模态文本条件化的 4×3 谱权重
   ↓
Pixel Anomaly Map
   ├── 有效 Mask 的 Pixel Loss
   └── 固定局部读出 ─┐
冻结最终 CLS Margin ─┴→ 固定平均 → Image BCE
```

V4 当前只使用上下左右四个邻居，不使用 V3 的八邻域传播，也没有可训练图卷积。

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

## 5. 模态条件化联合融合

正常和异常锚点的共同部分用于形成模态—解剖条件向量：

\[
t_m=\operatorname{Norm}\left(\frac{t_n+t_a}{2}\right).
\]

一个线性条件生成器输出十二个联合权重：

\[
\pi=\operatorname{Softmax}(W t_m+b),
\qquad \pi\in\mathbb R^{4\times3}.
\]

最终异常图为：

\[
S=\sum_{l\in\{6,12,18,24\}}
\sum_{k=0}^{2}\pi_{l,k}B_k^{(l)}.
\]

默认保留 20% 均匀权重质量，避免源域训练完全关闭某个层级—谱阶组合。对于
768 维 CLIP 特征，条件生成器只有 \(768\times12+12=9,228\) 个可训练参数。

严格 LODO 时，目标数据集不参加训练；测试权重只由目标数据集的固定文本锚点生成，
不读取目标图像、Mask 或标签进行选择。

## 6. 固定图像读出与 CLS 基线

V4 不训练 Top-k、GeM 或 CLS 混合权重。局部图像 Logit 使用固定温度的无参数注意力
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
\mathcal L=lambda_I\mathcal L_{image}
+\lambda_P\mathcal L_{pixel}.
\]

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
ckpt/v4_graph_spectral
ckpt/v4_bmad_lodo/<TARGET>
```

V4 checkpoint 使用独立方法标识 `modality_graph_spectral_v4` 和 `version=4`，不能与
V3 checkpoint 混用。
