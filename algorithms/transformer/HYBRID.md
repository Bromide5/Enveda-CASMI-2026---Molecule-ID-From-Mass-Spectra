# 检索优先、生成补充：实验流程

这是可运行的实验原型，还不是训练完成的鉴定模型。现有 Encoder 只做了 4 次示例更新，Decoder 只做了 2 次示例更新；这些权重不能用于实际鉴定。所有步骤从项目根目录执行，输出路径必须是新路径。

## 架构

查询谱由共享 Encoder 编码。池化后的单位向量用于谱库检索；池化前的 token 特征供 Decoder 交叉注意力读取。第一版冻结 Encoder（包括关闭其 Dropout），只训练 Decoder，避免改变已训练的检索空间。Decoder 使用因果注意力和右移 SMILES 标签，通过 teacher forcing 预测下一个字符，交叉熵忽略 padding。

字符词表仅来自训练集的 RDKit 规范 SMILES，包含 PAD/BOS/EOS。未知字符不能编码，超长训练标签报错，不静默截断。生成采用束搜索，按平均 token 对数概率排名，只接收 EOS 完成且 RDKit 可解析的非空结构，再按规范 SMILES 去重。未完成序列和非法结构丢弃；无合法结果会写空候选并记录 `generation_failed`。

合法结构不代表质量匹配或结构正确：当前生成未加入分子式/前体质量约束，也没有谱一致性重排。首版按规范 SMILES 去重，未统一互变异构体。模型还未使用前体质量、加合物和碰撞能量作为生成条件；这些应作为后续改进。

## 安装和训练

```text
conda activate casmi-course
python -m pip install -r tools/environments/hybrid.txt
python -m algorithms.transformer.train_decoder --train data/processed/transformer_split/train.parquet --encoder-checkpoint outputs/T1_smoke/last.pt --epochs 1 --steps-per-epoch 2 --max-length 512 --out outputs/H0_decoder_smoke
```

这是 Decoder 的两批训练流程检查。正式实验先充分训练 Encoder，再使用对应检查点训练 Decoder；省略 `--steps-per-epoch` 会遍历所有训练谱。保存 `config.json`、`history.json` 和含 Encoder/Decoder、词表、优化器状态的 `last.pt`，目前不支持 CLI 断点续训，也不自动选最佳轮次。

## 库内/库外验证

```text
python -m evaluation.build_hybrid_validation --source data/processed/transformer_split/validation --out data/processed/hybrid_validation
```

保留原查询与标签，按结构移除约一半查询结构的全部参考谱；标签新增 `in_library` 字段。该字段只供校准和评估使用，不传入预测。原始划分已隔离训练结构；这是 inchikey14 级别隔离，不保证分子骨架隔离。先前的多谱过滤只保留至少两张谱的结构，属于偏小的流程样本，不能代表全部库外分子。

## 检索和门控校准

```text
python -m algorithms.transformer.hybrid --checkpoint outputs/H0_decoder_smoke/last.pt --reference data/processed/hybrid_validation/reference.parquet --queries data/processed/hybrid_validation/queries.parquet --retrieval-only --out outputs/H0_retrieval
python -m evaluation.calibrate_hybrid --run outputs/H0_retrieval --labels data/processed/hybrid_validation/labels.csv --out outputs/H0_gate.json
```

检索先严格筛选同加合物、同离子模式和 20 ppm 前体窗口，不在无候选时放宽条件。按结构去重、多查询谱取最高分。门控同时要求第一名分数和前两名不同结构的分差达到阈值；不足两个候选时直接生成。

校准在验证集搜索阈值，默认要求接受样本的经验 Top-1 精度至少 95%、库外误接受率至多 5%、至少接受 5 个样本。条件不能满足时输出 `reject_all=true`。这些是可调整的实验目标，不是已保证的精度或置信概率；同一校准集上的结果不能作为独立测试性能。

门控记录模型、参考库摘要和 ppm。更换模型或 ppm 会拒绝加载；更换参考库会记录 `reference_changed_since_calibration`，在独立 holdout 库上可以测试迁移，但已有校准性能不保证适用。当前小样本验证只有 10 张保留参考谱，筛选后候选稀少，实际校准结果为全部转交生成。

## 混合预测与评估

```text
python -m algorithms.transformer.hybrid --checkpoint outputs/H0_decoder_smoke/last.pt --reference data/processed/hybrid_validation/reference.parquet --queries data/processed/hybrid_validation/queries.parquet --gate outputs/H0_gate.json --beam-size 2 --max-new-tokens 16 --out outputs/H0_hybrid
python -m evaluation.evaluate_hybrid --run outputs/H0_hybrid --labels data/processed/hybrid_validation/labels.csv
```

示例仅用 16 个生成步验证流程，会排除长 SMILES，不能用其结果衡量正式生成能力。实际评估应省略该参数使用模型长度上限，或明确报告生成预算。

`predictions.csv` 沿用 molecule_id/分号分隔 SMILES 格式，每分子最多 25 个候选。`routing.csv` 记录检索分数、候选数及 `retrieval`/`generation`/`generation_failed` 分支；`--retrieval-only` 仅用于校准诊断。未提供门控文件时默认全部生成，不采用未经验证的固定相似度阈值。

`hybrid_metrics.json` 分别报告库内、库外的 Top-1/5/25、MRR 和路由计数，以及库外样本被检索分支接受的比例。匹配使用 InChIKey14，不区分立体异构且未统一互变异构体，不等同官方评分。

固定训练和门控方案后，对 `transformer_split/holdout` 运行同样的库内/库外构造，再使用已固定门控预测、评分；不能用 holdout 重新选择阈值。所有推理和参考编码当前都面向小样本，原始峰与向量会加载到内存，不适合直接传入全量数 GB 谱库。Kaggle 打包入口尚未接入混合模型。

## 检查

```text
python -m unittest discover -s tests -p "test_*.py" -v
```

测试涵盖因果屏蔽、谱条件影响、冻结 Encoder、Decoder 更新、EOS 与非法 SMILES、两条门控路径和校准拒绝行为。
