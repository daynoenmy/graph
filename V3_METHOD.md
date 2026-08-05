# V3.1：冻结编码器病灶保持空间—频率图

## 1. 方法定位

V3 是与 V1 独立的实验分支，当前 checkpoint version 3 对应多层增强后的 V3.1。V1 通过
Text/Image Adapter、双噪声视图和病灶保持 Patch Graph 适配 CLIP；V3.1 则完全冻结
CLIP 图像与文本 Encoder，只在 Encoder 输出之后训练一个约 2.6 万参数的病灶保持
空间—频率图残差头。

V3.1 的中心假设不是“高频等于病灶”，而是：真实病灶更可能同时获得文本异常语义、
邻域异常共识和结构化频率残差的支持；孤立噪声即使频率响应较高，也缺少稳定的邻域
异常共识。

当前数据是二维静态医学图像，因此论文中应称为**空间—频率变换**，不能称为
时间—频率变换。只有使用连续超声帧、时间序列或有序体数据切片时，才存在真实的
时间或切片维度。

## 2. 与近期工作的关系

- [Q-Former Autoencoder](https://arxiv.org/abs/2507.18481) 表明冻结视觉基础模型可用于
  医学异常检测；V3 不使用其 Q-Former 与重建 Decoder。
- [VisualAD](https://arxiv.org/abs/2603.07952) 使用冻结 ViT 和视觉异常 Token；V3 不向
  Encoder 内插入 Token，也不使用 SCA/SAF。
- [FE-CLIP](https://openaccess.thecvf.com/content/ICCV2025/html/Gong_FE-CLIP_Frequency_Enhanced_CLIP_Model_for_Zero-Shot_Anomaly_Detection_and_ICCV_2025_paper.html)
  使用 DCT 频率 Adapter；V3 不修改 Encoder，而是在冻结特征之后使用固定平稳小波。
- [DLVP-CLIP](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_DLVP-CLIP_Enhancing_Fine-Grained_Zero-Shot_Anomaly_Detection_via_Dynamic_Local_Visual_CVPR_2026_paper.html)
  已使用小波高低频分解和动态视觉 Prompt；V3.1 不使用 Prompt Token 或 IDWT 重建，
  核心区别是无自环邻域共识、病灶门控图、孤立伪异常证据与有界残差。
- [Wave-MambaAD](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Wave-MambaAD_Wavelet-driven_State_Space_Model_for_Multi-class_Unsupervised_Anomaly_Detection_ICCV_2025_paper.html)
  分别建模高低频；V3 不使用 Mamba 或重建，而是建模医学 patch 的空间—频率一致性。
- [CLAP](https://arxiv.org/abs/2411.07546) 说明正常/异常对比有助于减少医学误报；V3
  使用固定 LLM 描述集和冻结 CLIP Text Encoder，不训练 Prompt 或 Text Adapter。

这些论文只构成设计动机。V3 的具体组合是：**冻结 CLIP 特征、固定平稳小波、局部
空间—频率图、正常/异常语义先验和频带干预一致性**。

## 3. 总体流程

```mermaid
flowchart LR
    IMG["医学图像"] --> FROZEN["完全冻结的 CLIP Image Encoder"]
    PROMPT["固定 LLM Prompt Bank"] --> TEXT["完全冻结的 CLIP Text Encoder"]
    TEXT --> ANCHOR["正常与异常文本锚点"]
    FROZEN --> PATCH["第 6/12/18/24 层冻结 Patch 特征"]
    PATCH --> SWT["固定平稳 Haar 小波"]
    SWT --> BANDS["低频结构与三个方向高频"]
    PATCH --> GRAPH["固定八邻域空间—频率图"]
    BANDS --> GRAPH
    ANCHOR --> PRIOR["Patch 正常异常语义竞争"]
    PATCH --> PRIOR
    PRIOR --> GRAPH
    GRAPH --> GATE["病灶共识门与孤立伪异常证据"]
    GATE --> HEAD["有界轻量图残差评分头"]
    HEAD --> PROB["分层 Patch 异常概率"]
    PROB --> FUSE["非负受约束分层融合"]
    FUSE --> MAP["Pixel Anomaly Map"]
    PROB --> LOCAL["Top-k 局灶证据与 GeM 弥漫证据"]
    FROZEN --> CLS["最终 CLS 全局特征"]
    CLS --> GLOBAL["正常异常全局文本竞争"]
    ANCHOR --> GLOBAL
    LOCAL --> MIX["受约束全局局部融合"]
    GLOBAL --> MIX
    MIX --> SCORE["Image Anomaly Score"]
```

## 4. 冻结特征与文本先验

给定图像 \(I\)，从冻结 CLIP 的四个层级提取 patch token，并使用冻结的
`ln_post` 与视觉投影得到：

\[
F^{(l)}=E_{\mathrm{frozen}}^{(l)}(I),
\qquad l\in\{6,12,18,24\}.
\]

四个层级分别经过同一个共享参数的空间—频率图 Head。融合发生在概率空间，而不是
直接平均正负可互相抵消的 Logit：

\[
p_i^{(l)}=\sigma(s_i^{(l)}),\qquad
w_l=\frac{\rho}{L}+(1-\rho)\operatorname{Softmax}(\omega)_l,
\]

\[
p_i=\sum_l w_l p_i^{(l)},\qquad s_i=\operatorname{logit}(p_i).
\]

默认 \(L=4,\rho=0.2\)，因此每层权重至少为 0.05，且总和严格为 1。模型可在五个
训练源上学习各层可靠性，又不能完全关闭任何一个尺度。

浅层补充纹理与边缘，中层描述局部结构，深层提供高层语义。共享 Head 避免参数量随
层数扩大，并使不同层级在同一异常评分规则下融合。

LLM Prompt Bank 中同一状态的多条文本经冻结 Text Encoder 编码、平均并归一化，
得到 \(t_n,t_a\)。Patch 异常概率和语义 margin 为：

\[
[z_{i,n},z_{i,a}]
=\tau F_i^\top[t_n,t_a],
\]

\[
a_i=\operatorname{Softmax}([z_{i,n},z_{i,a}])_a,
\qquad m_i=z_{i,a}-z_{i,n}.
\]

所有 CLIP 参数的 `requires_grad=False`，模型处于 `eval()` 状态。checkpoint 只保存
图 Head、分层融合权重和图像池化权重，不复制 CLIP 权重。测试时会强制核对特征层、
Head 维度、图边温度、修正上界以及池化结构，避免训练和测试配置不一致却未被发现。

## 5. 固定空间—频率变换

将 patch token 恢复成二维网格，在每个特征通道执行不降采样的固定 Haar 分析：

\[
(F_{LL},F_{LH},F_{HL},F_{HH})=\mathcal W_{\mathrm{SWT}}(F).
\]

- `LL` 描述低频和局部结构；
- `LH/HL/HH` 描述三个方向的细节变化；
- 固定滤波器没有可训练参数；
- 不降低 patch 网格分辨率，避免普通 DWT 下采样损伤小病灶定位。

三个高频能量使用同一图像级尺度归一化，保留方向频带之间的相对能量，不能分别
标准化后把频带差异抵消。

## 6. 病灶保持空间—频率图

每个 patch 是一个节点，只连接固定八邻域。边权同时考虑低频结构相似性和三个高频
方向的相对能量：

\[
w_{ij}\propto A_{ij}^{\mathrm{spatial}}
\exp\left(-\frac{1-\cos(F_{LL,i},F_{LL,j})}{\tau_L}
-\frac{\lVert q_i-q_j\rVert_2^2}{\tau_H}\right),
\]

其中 \(q_i=[E_{LH,i},E_{HL,i},E_{HH,i}]\)。第一阶段只使用空间—频率边权，并且
计算邻域异常共识时排除中心节点：

\[
c_i=\frac{\sum_{j\in\mathcal N(i)}w_{ij}a_j}
{\sum_{j\in\mathcal N(i)}w_{ij}},
\qquad
g_i=a_i c_i,
\qquad
u_i=a_i(1-c_i).
\]

- \(g_i\) 是病灶保持门：节点和邻域都支持异常时才会升高；
- \(u_i\) 是孤立异常证据：节点异常概率高、邻域却不支持时升高；
- 排除自环可避免单个伪异常用自身异常概率制造虚假的邻域共识。

第二阶段构造有向病灶保持权重：

\[
\widehat w_{ij}=w_{ij}
\left[(1-g_i)+g_i
\exp\left(-\frac{(a_i-a_j)^2}{\tau_A}\right)\right].
\]

只有得到邻域支持的病灶节点才抑制跨语义消息；孤立异常保持较低的 \(g_i\)，继续从
结构相容的正常邻居获得修正。这比直接依据节点异常概率切断边更不容易保留噪声
伪阳性。归一化后的图邻域均值为：

\[
\bar F_i=\sum_j\widetilde{\widehat w}_{ij}F_j,
\qquad
\bar q_i=\sum_j\widetilde{\widehat w}_{ij}q_j.
\]

语义和频率残差为：

\[
r_i^s=1-\cos(F_i,\bar F_i),
\qquad
r_i^f=\lVert q_i-\bar q_i\rVert_2.
\]

随机高频噪声可能有较高 \(r_i^f\) 和 \(u_i\)，但通常缺乏较高 \(g_i\)；真实病灶
更可能同时具有较高 \(a_i\)、\(c_i\)、\(g_i\) 和结构化空间—频率残差。这是待实验
验证的医学成像假设，不是对任意噪声的无条件保证。

## 7. 有界轻量残差评分头

评分头只学习对冻结 CLIP margin 的修正：

\[
s_i=m_i+\Delta_i,
\qquad
\Delta_i=C\tanh\left(\frac{H_\theta(z_i)}{C}\right),
\]

其中 \(z_i\) 包含语义概率、邻域共识、病灶门、孤立异常证据以及语义/频率残差。
最后一层初始化为零，因此训练开始时 V3.1 严格等于冻结 CLIP 的正常/异常文本
margin。默认 `C=4`，始终满足 \(|s_i-m_i|\leq4\)；当 \(|m_i|>4\) 时，残差 Head
不能翻转冻结语义判断的符号。这是一条可验证的有界修正性质，不是准确率保证。
默认 `hidden_dim=32` 时，Head 约有 2.6 万参数。

图像级预测不再对所有 Patch 做平均或 LogSumExp 近似最大池化。对每层 Patch 概率，
Top-k 分支保留小病灶的局部高响应，GeM 分支保留大面积或弥漫异常：

\[
P_{\mathrm{topk}}^{(l)}=\frac{1}{K}\sum_{i\in\operatorname{TopK}}p_i^{(l)},
\qquad
P_{\mathrm{gem}}^{(l)}=
\left(\frac{1}{N}\sum_i(p_i^{(l)})^q\right)^{1/q}.
\]

默认 \(K=\lceil0.05N\rceil,q=3\)。局部证据为：

\[
P_{\mathrm{local}}=\sum_l w_l
\left[\beta P_{\mathrm{topk}}^{(l)}+(1-\beta)P_{\mathrm{gem}}^{(l)}\right].
\]

同时恢复冻结 CLIP 的最终 CLS 输出 \(c\)，并让它与同一组正常/异常文本锚点竞争：

\[
P_{\mathrm{cls}}=
\operatorname{Softmax}\left(\tau c^\top[t_n,t_a]\right)_a,
\qquad
P_{\mathrm{image}}=\alpha P_{\mathrm{cls}}+(1-\alpha)P_{\mathrm{local}}.
\]

\(\alpha\) 和 \(\beta\) 是源域训练得到的两个标量，初始化为 0.5，并被限制在
`[0.1, 0.9]`，保证全局/局部、局灶/弥漫证据都不能被完全关闭。CLS 只参与图像级
判断，不进入 Pixel 异常图，因此不会直接平滑定位结果。

## 8. 训练目标

\[
\mathcal L
=\mathcal L_{\mathrm{focal+dice}}
+\lambda_I\mathcal L_{\mathrm{image}}
+\lambda_N\mathcal L_{\mathrm{normal-band}}
+\lambda_L\mathcal L_{\mathrm{lesion-preserve}}.
\]

频带干预在冻结 Encoder 只运行一次的前提下，把随机高频方向缩放到默认 `[0.5,1.5]`
范围：小于 1 模拟细节衰减/模糊，大于 1 模拟高频噪声增强。正常 Patch 使用双向一致
性约束；病灶 Patch 使用单边保持损失，只惩罚干预后病灶概率下降，避免通过压低病灶
响应获得表面稳定性。两条分支只重新运行约 2.6 万参数的 Head，不生成第二张图，也不
再次运行 CLIP Encoder。

V3.1 checkpoint `version=3`。由于特征输入升级为第 6、12、18、24 层概率融合，并
加入可学习的 CLS/Top-k/GeM 图像池化，旧 `version=1/2` checkpoint 不能直接加载，
必须重新训练。

## 9. 与 V1 的代码隔离

V3 新增：

- `model/frozen_sfgraph.py`
- `v3_utils.py`
- `train_v3.py`
- `test_v3.py`
- `train_v3.bat`
- `test_v3.bat`

V1 的 `train.py`、`test.py` 和 `model/adapter.py` 保持为 V1 对照入口。

## 10. 运行方法

Windows 训练：

```bat
train_v3.bat
```

默认使用 Brain full-shot，保存到：

```text
ckpt/v3_multilayer_sfgraph
```

测试所有 epoch：

```bat
test_v3.bat
```

干净测试使用 `--test_noise_severity 0.0`。例如测试目标模态噪声：

```bat
python test_v3.py ^
  --dataset Liver ^
  --save_path ./ckpt/v3_multilayer_sfgraph ^
  --checkpoint v3_head_epoch_*.pth ^
  --test_noise_severity 0.06
```

该参数只扰动测试主输入，不参与模型内部计算，并按文件名和随机种子生成稳定的噪声
实现。

## 11. 必须完成的消融

1. Frozen CLIP + Text Margin；
2. + Spatial Graph；
3. + SWT Frequency Descriptor；
4. + Spatial–Frequency Edge；
5. + Lesion-Preservation Gate；
6. + Bounded Correction；
7. + 双向 Band Consistency；
8. + Lesion Preservation Loss；
9. Template Prompt 与 LLM Prompt；
10. 单层、第 12/18/24 层与第 6/12/18/24 层特征；
11. Logit 平均、概率融合与受约束分层融合；
12. Top-k、GeM、Top-k+GeM、仅 CLS 与完整全局—局部池化；
13. V1、原始 V3 与 V3.1 的参数量、显存、稳定性和跨域性能。

checkpoint 必须由 Brain 源域验证协议统一选择，不能分别查看 Liver、DDTI、Retina
和 Colon 测试结果后选择不同 epoch。

## 12. BMAD 混合监督 LODO

V3.1 支持 BMAD 六数据集 leave-one-dataset-out。每个 Fold 完全排除一个目标数据集，
其余五个源数据集通过 dataset-balanced sampler 联合训练。设 `mask_valid` 表示 Pixel
标注是否可信，则 Pixel 损失与正常区域一致性只在可信样本上计算；病灶保持损失只在
确实具有异常 Mask 的样本上计算。

这是有标签源域的 domain-generalization 协议，不等同于 BMAD 原论文的无监督训练
设置；实验表和论文正文必须单独标明 `BMAD-LODO (ours)` 协议。

正常图即使没有单独 Mask 文件，也具有已知的全零异常 Mask；异常无 Mask 图的空间
位置未知，只参加 Image BCE，不能使用全零伪 Mask。测试时所有数据集报告 Image
AUC/AP，仅 Mask 覆盖的定位数据集报告 Pixel AUC/AP，另外输出异常 Mask 覆盖率。

详细 Metadata 格式、路径配置、六折命令和防止目标域 epoch 选择泄漏的规则见
`BMAD_LODO.md`。
