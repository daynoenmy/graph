# V4.2-Lite：轻量文本增强的模态条件图拉普拉斯异常检测

> 状态：方法设计稿，尚未实现。
> 基础版本：V4.1 `FrozenModalityGraphSpectralModel`。
> 设计目标：保持冻结 CLIP 和轻量图残差主干，只增强当前偏弱的文本表达，避免引入复杂网络。

## 1. 方法定位

V4.2-Lite 是 V4.1 的轻量文本增强版本。方法继续完全冻结 CLIP Image Encoder 和
Text Encoder，保留四层 Patch 特征、固定四邻域图拉普拉斯和有界残差融合，不增加
大型 Text Adapter、Prompt Transformer、可训练图卷积或额外视觉分支。

V4.1 当前将同一状态下的多条 Prompt 特征均匀平均，并使用较通用的正常/异常描述。
Brain 和 Liver 等数据集的基础文本都可能退化为通用的 `scan`，难以充分表达器官、
成像模态及病变形态。V4.2-Lite 通过以下两级增强改善文本先验：

1. **默认模块：解剖—模态特异 Prompt + 共享可学习 Prompt 加权**；
2. **可选模块：共享 Rank-4 文本残差适配器**。

默认方案只增加约 16 个可训练参数。启用可选文本适配器后，整个新增文本模块仍只有
约 6K 参数。

## 2. 总体结构

```text
固定医学 Prompt 集
   ↓
冻结 CLIP Text Encoder
   ↓
逐条归一化 Prompt 特征
   ↓
共享可学习 Prompt 加权
   ↓
正常/异常文本锚点
   ├──────────────────────────────┐
   │                              │
医学图像                         固定模态描述
   ↓                              ↓
冻结 CLIP Image Encoder          冻结 CLIP Text Encoder
   ↓                              ↓
第 6/12/18/24 层 Patch 特征      模态条件向量
   ↓                              │
正常—异常文本 Margin             │
   ↓                              │
固定四邻域图拉普拉斯：m、Lm、L²m │
   └───────────┬──────────────────┘
               ↓
      模态条件化有界残差融合
               ↓
         Pixel Anomaly Map
               ├── Pixel Focal/Dice Loss
               └── 固定局部读出 ─┐
冻结最终 CLS Margin ──────────────┴→ Image BCE Loss
```

V4.2-Lite 不改变 V4.1 的视觉主干和图融合主线。主要变化发生在正常/异常文本锚点的
构造阶段。

## 3. 冻结多层视觉特征

给定医学图像 \(I\)，冻结的 CLIP Image Encoder 提取第 6、12、18、24 层 Patch
特征：

\[
F^{(l)}=E_{img}^{(l)}(I),
\qquad l\in\{6,12,18,24\}.
\]

所有 Patch 特征经过 CLIP 视觉投影和 L2 归一化。CLIP 参数始终不参与反向传播，
因此新增文本模块不会显著增加视觉侧显存或训练参数量。

## 4. 解剖—模态特异 Prompt

每个数据集不再使用过于通用的名称，而是显式包含器官和成像方式：

| 数据集 | 推荐基础名称 |
|---|---|
| Brain | `brain MRI scan` |
| Liver | `liver CT scan` |
| Retina | `retinal OCT scan` |
| Chest | `chest X-ray` |
| Retina_OCT2017 | `retinal OCT scan` |
| Histopathology | `histopathology microscopy image` |

正常状态使用少量互补描述，例如：

```text
{}
a normal {}
a healthy {}
```

异常状态使用覆盖局灶、弥漫和结构异常的描述，例如：

```text
an abnormal {}
a pathological {}
a {} with a focal lesion
a {} with a diffuse abnormality
a {} with structural distortion
```

每个状态描述继续与固定句式模板组合。Prompt 集必须在实验前固定，并记录内容哈希；
训练和测试过程中不在线调用 LLM，也不能根据目标域测试结果修改 Prompt。

## 5. 共享可学习 Prompt 加权

### 5.1 当前均匀平均

设状态 \(s\in\{normal,abnormal\}\) 共有 \(K_s\) 条 Prompt，冻结 Text Encoder
产生特征：

\[
e_{s,k}=\operatorname{Norm}(E_{text}(p_{s,k})).
\]

V4.1 对这些特征进行均匀平均：

\[
t_s^{V4.1}=\operatorname{Norm}
\left(\frac{1}{K_s}\sum_{k=1}^{K_s}e_{s,k}\right).
\]

这种方式稳定，但默认所有描述具有相同价值。

### 5.2 V4.2-Lite 加权锚点

V4.2-Lite 为正常和异常 Prompt 分别引入一组可学习 Logit：

\[
a_{s,k}=\operatorname{Softmax}(w_s)_k,
\qquad
t_s=\operatorname{Norm}
\left(\sum_{k=1}^{K_s}a_{s,k}e_{s,k}\right).
\]

其中 \(w_s\) 在训练开始时全部初始化为零，因此：

\[
a_{s,k}=\frac{1}{K_s}.
\]

初始文本锚点严格等于 V4.1 的均匀平均基线，训练只能在此基础上重新分配不同 Prompt
的重要性。

Prompt 权重在所有源数据集之间共享，而不是为每个数据集单独学习。这样可以：

- 避免将 Prompt 权重退化为数据集标识；
- 保持严格 LODO，不需要目标域训练数据；
- 直接解释不同正常/异常描述的贡献；
- 将新增参数控制在约 16 个。

## 6. 可选共享 Rank-4 文本残差

只有当 Prompt 加权在源域验证中仍不足时，才启用共享低秩文本适配器。正常和异常锚点
使用同一个残差模块：

\[
\widehat t_s=\operatorname{Norm}
\left[t_s+\gamma U\,\sigma(Vt_s)\right],
\]

其中：

\[
V\in\mathbb R^{4\times768},
\qquad
U\in\mathbb R^{768\times4}.
\]

残差强度 \(\gamma\) 使用有界参数化并初始化为 0，使模型初始状态严格等于未经适配的
冻结文本锚点。Rank-4 适配器约增加 6K 参数，仍远小于 CLIP 主干。

该模块是可选消融，不属于默认 V4.2-Lite。若它不能在多随机种子和严格源域验证下产生
稳定提升，应保持关闭。

## 7. Patch 异常语义 Margin

给定归一化 Patch 特征 \(f_i^{(l)}\)、正常锚点 \(t_n\) 和异常锚点 \(t_a\)，基础
异常 Margin 为：

\[
m_i^{(l)}=\tau_t\left[
\langle f_i^{(l)},t_a\rangle-
\langle f_i^{(l)},t_n\rangle
\right].
\]

文本增强只改变正常/异常语义方向，不直接读取图像标签、Mask 或目标域样本。

## 8. 固定图拉普拉斯残差

每层 Patch 网格继续只连接上下左右四个邻居。相邻特征亲和度为：

\[
A_{ij}^{(l)}=
\exp\left(
\frac{\cos(F_i^{(l)},F_j^{(l)})-1}{\tau_g}
\right).
\]

V4.2-Lite 默认沿用 V4.1 的随机游走图拉普拉斯：

\[
L=I-D^{-1}A.
\]

每层构造：

\[
B_0^{(l)}=m^{(l)},\qquad
B_1^{(l)}=Lm^{(l)},\qquad
B_2^{(l)}=L^2m^{(l)}.
\]

为保持方法简单，正式实验必须比较仅使用 \(L\) 与同时使用 \(L+L^2\) 的效果。如果
二阶残差没有稳定收益，应删除 \(L^2\)，将每层残差简化为一个系数。

## 9. 模态条件化有界融合

固定模态描述经过冻结 Text Encoder 得到模态向量 \(t_m\)。模态条件器生成四层权重：

\[
\alpha=\operatorname{Softmax}(W_\alpha t_m+b_\alpha),
\qquad \sum_l\alpha_l=1.
\]

同时生成有界有符号图残差系数：

\[
\beta_{l,k}=\beta_{max}\tanh
\left((W_\beta t_m+b_\beta)_{l,k}\right).
\]

最终 Patch Logit 为：

\[
S=\sum_l\alpha_l\left[
m^{(l)}+\beta_{l,1}Lm^{(l)}+\beta_{l,2}L^2m^{(l)}
\right].
\]

V4.2-Lite 不增加局部门、图卷积、KNN 分支、小波分支或额外视觉 Adapter。

## 10. 固定图像级读出

局部图像 Logit 继续使用固定温度注意力：

\[
q_i=\frac{\exp(S_i/\tau_r)}{\sum_j\exp(S_j/\tau_r)},
\qquad
z_{local}=\sum_iq_iS_i.
\]

最终 CLIP CLS 特征使用同一组文本锚点得到 \(z_{CLS}\)，图像级 Logit 为：

\[
z_{image}=\frac{z_{CLS}+z_{local}}{2}.
\]

不引入可学习 Top-k、GeM 或图像池化网络。读出温度 \(\tau_r\) 必须在源域验证协议下
从固定候选值中统一选择，不能根据不同目标域分别调整。

## 11. 训练目标

所有样本参加图像级二分类损失：

\[
\mathcal L_{image}=\operatorname{BCEWithLogits}(z_{image},y).
\]

正常样本的全零 Mask 是有效监督；异常样本只有在真实 Mask 存在时才参加 Pixel
Focal/Dice Loss：

\[
\mathcal L=
\lambda_I\mathcal L_{image}
+\lambda_P\mathcal L_{pixel}.
\]

没有病灶 Mask 的异常样本不能作为全零异常图参加 Pixel Loss。

可训练参数包括：

- V4.1 模态条件图融合器：9,228；
- 默认 Prompt 权重：约 16；
- 可选共享 Rank-4 文本适配器：约 6K。

默认 V4.2-Lite 总可训练参数约 9.2K；启用可选适配器后约 15K。冻结 ViT-L/14 仍是
主要推理开销，因此该方法属于参数高效模型，而不是小型视觉骨干。

## 12. 严格 LODO 协议

留出目标数据集时：

1. 目标数据集不参加参数训练；
2. Prompt 内容和顺序在训练前固定；
3. Prompt 权重只由源数据集学习；
4. 模态条件只读取公开的成像模态描述；
5. 不读取目标图像、Mask 或标签选择 Prompt、温度或 checkpoint；
6. checkpoint 必须根据源域验证结果选择；
7. 每个目标域使用同一套超参数选择规则。

## 13. 必需消融

| 编号 | 文本锚点 | Prompt 融合 | 图残差 | 目的 |
|---|---|---|---|---|
| A0 | 原始通用文本 | 均匀平均 | 无 | 冻结 CLIP 基线 |
| A1 | 解剖—模态特异文本 | 均匀平均 | 无 | 验证 Prompt 内容 |
| A2 | 解剖—模态特异文本 | 可学习加权 | 无 | 验证 Prompt 加权 |
| A3 | 解剖—模态特异文本 | 可学习加权 | 仅 \(L\) | 验证一阶图残差 |
| A4 | 解剖—模态特异文本 | 可学习加权 | \(L+L^2\) | 验证二阶残差 |
| A5 | A4 + Rank-4 Adapter | 可学习加权 | \(L+L^2\) | 验证可选文本适配 |

还应加入一个不使用模态文本、只学习全局层权重和残差系数的对照。如果全局融合与模态
条件融合性能接近，应优先保留更简单的全局版本。

所有正式结果至少运行 3 个随机种子，并报告均值和标准差。除 Image AUC/AP 和 Pixel
AUC/AP 外，还应报告可用异常 Mask 覆盖率，避免不同数据集监督完整度造成误导。

## 14. 实现边界

V4.2-Lite 的默认实现只应包含以下新增内容：

1. 解剖—模态特异基础名称；
2. 返回每条 Prompt 的冻结文本特征；
3. 正常和异常各一组共享 Prompt 权重；
4. checkpoint 中保存 Prompt 权重、Prompt 顺序和 Prompt Bank 哈希；
5. 训练/测试时验证文本配置与 checkpoint 完全一致。

以下内容不属于默认实现：

- 解冻完整 CLIP Text Encoder；
- 大型 Text Adapter；
- 在线 LLM Prompt 生成；
- 每个目标数据集单独学习 Prompt；
- 新增图神经网络、局部门或额外视觉分支；
- 根据目标测试结果选择 Prompt、温度或 checkpoint。

## 15. 方法总结

V4.2-Lite 保留 V4.1 冻结 CLIP 和轻量图拉普拉斯残差主干，通过解剖—模态特异
Prompt 与共享可学习 Prompt 加权增强正常/异常文本锚点。默认文本增强只增加约 16 个
参数，并以均匀平均零偏置初始化，保持与 V4.1 的严格初始等价。可选 Rank-4 文本残差
仅在充分消融后启用。

一句话概括：

> V4.2-Lite 用极少量共享 Prompt 权重选择更可靠的医学文本语义，再以模态条件化的
> 有界图拉普拉斯残差修正冻结 CLIP 多层 Patch 响应，在不显著增加网络复杂度的前提下
> 提升医学异常描述能力。
