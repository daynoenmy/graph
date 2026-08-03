# BMAD 混合监督 Leave-One-Dataset-Out

> 本实现是“有标签源域训练、目标域完全留出”的医学域泛化协议，不是 BMAD 论文原始的
> 无监督训练协议。论文中应明确写成自定义 BMAD LODO/domain-generalization setting，
> 不应把结果表述为 BMAD 官方无监督排行榜结果。

## 数据集命名

代码使用六个 BMAD 数据集：

| 代码名称 | BMAD 子集 | 异常 Mask |
|---|---|---|
| `Brain` | Brain MRI | 有 |
| `Liver` | Liver CT | 有 |
| `Retina` | Retina RESC OCT | 有 |
| `Chest` | Chest X-ray | 无 |
| `Retina_OCT2017` | Retina OCT2017 | 无 |
| `Histopathology` | Digital histopathology | 无 |

仓库保留原有的 `Retina` 名称，它在 BMAD LODO 中代表 Retina RESC，而不是眼底照片。

## 数据路径

六个 BMAD 数据集的根目录固定写在 `dataset/constants.py`：

```python
BMAD_BASE_PATH = r"E:\datasets\bmad\BMAD"
```

代码自动拼接以下目录：

```text
E:\datasets\bmad\BMAD\Brain
E:\datasets\bmad\BMAD\Liver
E:\datasets\bmad\BMAD\Retina_RESC
E:\datasets\bmad\BMAD\Chest
E:\datasets\bmad\BMAD\Retina_OCT2017
E:\datasets\bmad\BMAD\Histopathology
```

不再需要设置 `BMAD_DATA_PATH`、`BRAIN_DATA_PATH` 等环境变量。以后更换机器时，只需
修改 `BMAD_BASE_PATH` 一处。

## Metadata 格式

每个数据集需要：

```text
dataset/metadata/<数据集名称>/full-shot.jsonl
```

正常样本不需要 `mask_path`：

```json
{"image_path":"normal/001.png","label":0,"class_name":"Chest"}
```

有 Mask 的异常样本：

```json
{"image_path":"abnormal/001.png","label":1,"class_name":"Brain","mask_path":"mask/001.png"}
```

无 Mask 的异常样本：

```json
{"image_path":"abnormal/001.png","label":1,"class_name":"Chest"}
```

数据加载器会返回两个显式标志：

- `mask_valid`：该样本是否可以参加 Pixel 监督。正常图始终为真；异常图仅在存在
  `mask_path` 时为真。
- `has_anomaly_mask`：异常病灶 Mask 是否真实存在，只控制病灶保持损失和覆盖率统计。

不能给无 Mask 异常图填一个全零假 Mask，否则会把真实异常区域错误监督成正常。

## 混合监督损失

| 样本 | Image BCE | Pixel Focal/Dice | 正常频带一致性 | 病灶保持 |
|---|---:|---:|---:|---:|
| 正常，无 Mask 文件 | 是 | 全零 Mask | 全图 | 否 |
| 异常，有 Mask | 是 | 是 | Mask 外 | Mask 内 |
| 异常，无 Mask | 是 | 否 | 否 | 否 |

异常无 Mask 样本仍然通过 Image BCE 学习图像级异常，并通过同一张 Patch 异常图的
LogSumExp 池化把弱监督传递到 Head；代码不会为它伪造 Pixel 标签。

## 训练一个 LODO Fold

例如把 Chest 完全留作目标域：

```bat
train_v3_lodo.bat Chest
```

训练源自动解析为：

```text
Brain + Liver + Retina + Retina_OCT2017 + Histopathology
```

五个源数据集使用 dataset-balanced `WeightedRandomSampler`。每个数据集的单样本权重
为其样本数的倒数，因此各源数据集在一个 epoch 中具有相同的期望采样概率质量。

checkpoint 保存到：

```text
ckpt/v3_bmad_lodo/Chest
```

## 测试

```bat
test_v3_lodo.bat Chest
```

LODO 脚本只测试固定训练轮次的 `v3_head_latest.pth`，不能查看 Chest 的多个 epoch 后
选择最优模型。代码也会拒绝使用 LODO checkpoint 一次扫描多个 epoch。checkpoint 中
记录的 `lodo_target` 必须与 `--dataset` 相同，否则测试直接报错。

- 六个数据集均报告 Image AUC/AP；
- Brain、Liver、Retina 报告 Pixel AUC/AP；
- Chest、Retina_OCT2017、Histopathology 的 Pixel 指标为 `NaN`；
- `masked anomaly coverage` 会显示异常 Mask 覆盖率，分类数据集应为 `0%`。

## 六折命令

```bat
train_v3_lodo.bat Brain
train_v3_lodo.bat Liver
train_v3_lodo.bat Retina
train_v3_lodo.bat Chest
train_v3_lodo.bat Retina_OCT2017
train_v3_lodo.bat Histopathology
```

完成训练后，将 `train` 替换为 `test` 依次评估。固定文本来自
`dataset/bmad_llm_prompts.json`，该文件在训练前冻结，不使用目标图像、Mask 或测试
标签生成文本。论文中仍需报告 Template Prompt 消融，以区分文本先验与网络贡献。
