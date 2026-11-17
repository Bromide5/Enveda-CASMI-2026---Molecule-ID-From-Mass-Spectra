# 质谱分子识别课程大作业

《人工智能数学原理与算法 B》· 2026 年秋季

第一次接触质谱？先读 [赛题与数据入门](赛题与数据入门.md)，再按本文开始实验。

## 做什么

参加 [Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，根据质谱识别小分子结构。可以把质谱理解为分子的“碎片线索”：峰的位置和强度帮助我们寻找原来的分子。

输入是一个分子的一张或多张质谱；输出是按可能性排序的候选结构，最多 25 个。结构用 SMILES 字符串表示，例如乙醇是 `CCO`。每个分子提交一行，候选用分号分隔；正确答案排得越靠前，得分越高。

项目已提供“谱分箱 → 余弦相似度检索 → 合并候选”的基线。重点是跑通流程、设计改动并解释结果，不要求化学背景或大型深度模型。

## 怎样组队和完成作业

**三人一组，每组最终汇报共 5 分钟。** 三人分别注册 Kaggle 账号，在竞赛页报名并接受规则，由一人在 Team 页面发起邀请或合并，另外两人接受，组成同一支队伍。Kaggle 是运行代码、提交预测并查看成绩的平台。

每组完成以下工作：

1. 跑通基线，完成至少 **两项有明确目的的改动**，保留至少 **三组可比较实验**。可从分箱宽度、强度变换、参考库规模入手。
2. 在 Kaggle 完成至少 **一次成功评分的正式提交**，保存版本、分数和截图。
3. 提交一份小组材料，汇报方法、对比结果、失败分析和三位成员的贡献。允许参考公开代码和使用 AI，说明来源及自己的工作。

## 什么时候交

| 事项 | 北京时间 |
| --- | --- |
| 课程材料截止 | **待公布** |
| 小组汇报 | **待公布**，每组 5 分钟 |
| Kaggle 报名及组队截止 | 2026 年 12 月 8 日 07:59 |
| Kaggle 最终提交截止 | 2026 年 12 月 15 日 07:59 |

竞赛日期见[官方日程](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)，如有调整以官方为准。

## 交什么、交到哪里

**Kaggle：** 提交运行代码生成 `submission.csv` 的 Notebook 版本，等待评分完成。具体操作见下方“提交到 Kaggle”。

**课程：** 每组提交 `组号_队伍名_final_project.zip`，**提交渠道待公布**。材料包括：

- 代码、依赖和复现步骤，以及使用的额外模型或数据的获取方式。
- 小组报告（建议 4–6 页）：成员姓名、学号、Kaggle 用户名与队伍链接，方法、三组实验对比、结果分析、成员贡献及代码和 AI 使用说明。
- 汇报幻灯片（建议 5–6 页，附可编辑文件）、实验记录、选定版本的 `submission.csv`，以及提交版本、状态、分数、时间和运行日志。

原始数据、环境和账号凭据无需打包。实验记录用一张表即可：**版本｜改动｜数据与参数｜指标｜耗时｜结论｜负责人**。

## 开始实验

下载[课程数据](https://pan.ustc.edu.cn/share/index/5a8f5ad972224a97b401?p=1)（密码 **C8F3**），将 `train.parquet`、`test.parquet`、`sample_submission.csv` 放到 `data/raw/`。从项目根目录打开终端，按下面步骤运行。先用小谱库完成本地实验，再按第三步用 B2 全量谱库完成正式提交。

<details>
<summary><strong>第一步：配置环境，跑通本地基线</strong></summary>

使用实验课已有的 conda，新建环境；已有同名环境时直接激活。

```text
conda create -n casmi-course --override-channels -c conda-forge python=3.11 pip
conda activate casmi-course
python -m pip install -r tools/environments/baseline.txt
python -m submission.check_environment
```

下面四条命令依次完成抽样、划分、预测和评分。`python -m` 表示运行项目里的指定 Python 模块；`--` 开头的是参数，后面跟文件路径或数值。

**① 抽取小样本。** 从训练文件中随机抽取 5000 张谱，保存到 `reference_sample.parquet`，供后续实验使用。`--seed 0` 固定随机种子，方便重复得到相同的抽样结果。

```text
python -m data.scripts.inspect_data --file data/raw/train.parquet --sample-out data/processed/reference_sample.parquet --sample-rows 5000 --seed 0
```

**② 划分验证数据。** 从小样本中选取最多 200 个符合条件的分子，生成参考谱库、待预测的查询谱和正确答案，保存到 `data/processed/validation/`。`--mode known` 保留同一分子的其他谱用于检索，移除查询谱及其重复项；查询分为选参数用的 `tune` 和最终验证用的 `holdout`。

```text
python -m evaluation.build_validation --reference data/processed/reference_sample.parquet --out data/processed/validation --mode known --molecules 200 --seed 0
```

**③ 运行基线预测。** 用 `--reference` 指定的参考谱库，为 `--queries` 中每个分子检索候选结构，预测结果和运行参数保存到 `outputs/B0/`。`B0` 是这次基线实验的编号。

```text
python -m algorithms.baseline.predict --reference data/processed/validation/reference.parquet --queries data/processed/validation/queries_tune.parquet --out outputs/B0
```

**④ 给预测评分。** 将 `--predictions` 中的预测与 `--labels` 中的正确答案比较，计算 MRR 和命中率，保存到 `outputs/B0/metrics.json`；`--reference` 还用于统计参考库覆盖了多少正确结构。

```text
python -m evaluation.evaluate --predictions outputs/B0/predictions.csv --labels data/processed/validation/labels_tune.csv --reference data/processed/validation/reference.parquet --out outputs/B0
```

结果在 `outputs/B0/`：`predictions.csv` 是预测，`config.json` 是参数，`metrics.json` 是指标，`run_summary.json` 含耗时。MRR@25 是正确候选排名倒数的平均值，例如排第 2 得 0.5，未命中得 0；越高越好。本地默认按字符串匹配，是课堂近似指标，官方按结构评分。

这里验证的是“参考库已有该结构的其他谱，能否找回来”。`tune` 用于选参数，固定方案后将上述预测和评分命令中的 `tune` 换成 `holdout`，输出换到新目录做最终验证。可见 `test.parquet` 用于检查运行流程。若提示可用结构不足，增大抽样量。

</details>

<details>
<summary><strong>第二步：改进方法并比较结果</strong></summary>

复用同一验证划分。下面两项改动可以作为起点，也可以自行设计：

```text
python -m algorithms.baseline.predict --reference data/processed/validation/reference.parquet --queries data/processed/validation/queries_tune.parquet --out outputs/A1 --power 1
python -m algorithms.baseline.predict --reference data/processed/validation/reference.parquet --queries data/processed/validation/queries_tune.parquet --out outputs/A2 --bin-width 0.5
```

`A1` 取消强度平方根变换，`A2` 将分箱宽度从 1 改成 0.5。将第一步评分命令中的两处 `B0` 分别换成 `A1`、`A2`，给两组结果评分，再比较：

```text
python -m evaluation.compare_runs --a outputs/B0 --b outputs/A1
python -m evaluation.compare_runs --a outputs/B0 --b outputs/A2
```

记录改动的理由和实际结果，没有提升的尝试也值得分析。更换参考库规模、候选筛选或学习排序都可以继续探索。纯谱库检索只能返回参考库中已有的结构。

代码位置：`data/scripts/` 读取与画谱图，`algorithms/baseline/` 表示与检索，`evaluation/` 划分与评分，`submission/` 打包与提交。各脚本加 `--help` 查看参数。

</details>

<details>
<summary><strong>可选：为 Transformer 准备训练数据（第一步）</strong></summary>

数据管道位于 `evaluation/build_training_split.py` 和 `algorithms/transformer/dataset.py`，实现数据划分和批次采样。本步骤复用 `casmi-course` 环境，不需要安装 PyTorch；后续的编码器检查和训练需要 PyTorch。

先完成前面的 5000 张谱抽样，再按 `inchikey14` 将结构划为模型训练、验证、最终检查三组。每个结构至少需要两张不同的谱；完全重复谱会去重，完全相同输入却有冲突结构标签的谱会剔除。少于两张谱的结构暂不参加此项实验。默认约 80%/10%/10% 按**结构数**划分，各验证组至少保留两个结构。

```text
python -m evaluation.build_training_split --reference data/processed/reference_sample.parquet --out data/processed/transformer_split --seed 0
python -m algorithms.transformer.dataset --train data/processed/transformer_split/train.parquet --structures-per-batch 8 --spectra-per-structure 2 --batches 3 --out outputs/T0/data_check.json
```

输出结构如下；划分目录和检查报告已存在时会拒绝覆盖，复现实验请使用新的路径。

```text
data/processed/transformer_split/
├── train.parquet                  # 只用这一份更新模型参数
├── validation/
│   ├── reference.parquet          # 验证参考谱库
│   ├── queries.parquet            # 不含结构标签的验证查询
│   └── labels.csv                 # 验证答案
├── holdout/                       # 同样的三个文件，固定方案后才评价
├── manifest.csv                   # 每张谱的输入行号、结构、摘要和归属
└── split.json                     # 清洗统计、划分参数和规模
```

验证与 holdout 的结构不参与模型参数训练，但各自的参考库保留查询结构的其他谱。这测量的是“模型未训练过的结构，在有对应参考谱时能否找回”，不等于从库外生成新结构。结构覆盖按 `inchikey14` 计算；质量窗口、加合物过滤后仍可能没有正确候选。此划分没有做骨架隔离或互变异构体统一。

默认每个验证结构抽一张查询谱。如果要测试多谱融合，可在划分命令中设置 `--queries-per-structure 2`；这会要求每个结构至少有三张谱，以便留出参考谱。训练每结构抽样数量 K 若大于 2，也应同步设置划分脚本的 `--spectra-per-structure K`。

批次检查命令每批抽取 8 个不同结构，每个结构抽 2 张不同谱，预期输入维度为 `[16, 1500]`、标签维度为 `[16]`。标签只用来确定正负样本，不能作为编码器的输入。检查报告验证非有限数值、向量范数和批次结构计数。随机采样允许跨批重复，不承诺每轮遍历所有谱。

安装 PyTorch 后，数据集和采样器可直接接入其 `DataLoader`：

```python
from torch.utils.data import DataLoader
from algorithms.transformer.dataset import SpectrumDataset, PKBatchSampler

dataset = SpectrumDataset("data/processed/transformer_split/train.parquet")
sampler = PKBatchSampler(dataset.labels, structures_per_batch=8, spectra_per_structure=2)
loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

sampler.set_epoch(0)  # 每个 epoch 调用一次，传入当前轮次
batch = next(iter(loader))
spectra, labels = batch["spectrum"], batch["label"]
```

训练时应把每个同结构样本都视为正样本，不要把同一结构的其他谱当作负样本。分箱、强度变换和质量范围必须在训练与推理时保持一致。

此版本会把抽样后的峰数组加载到内存；建议先用小样本检查流程，再扩大抽样。不要直接把约 3 GB 的原始训练文件传给该划分脚本或数据集。小样本中可组成正样本的结构通常远少于谱数，检查 `split.json` 后再决定是否扩大样本。

验证数据隔离和采样逻辑的测试命令：

```text
python -m unittest discover -s tests -p test_training_data.py -v
```

</details>

<details>
<summary><strong>可选：检查 Transformer Encoder（第二步）</strong></summary>

`algorithms/transformer/model.py` 实现共享权重的谱编码器，参数由 `configs/transformer.json` 配置。查询谱和参考谱调用同一个 `SpectrumEncoder` 实例。结构标签只用于后续损失计算。

```text
分箱强度 [B,1500]
→ Linear(1,128) + 质量分箱中心的正余弦编码 [B,1500,128]
→ 2 层双向 Transformer Encoder（4 个头，前馈维度 512）
→ 非空分箱均值池化 [B,128]
→ Linear(128,128) + L2 归一化 [B,128]
```

零强度分箱不能作为注意力的 key/value，也不参与池化；位置编码保留峰的质量位置。没有因果遮罩、Decoder 或自回归生成。各编码层独立初始化，使用 PyTorch 的 [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)。这里仍然计算完整的稠密序列，屏蔽空分箱并不会消除长序列的计算开销。

先在项目根目录安装依赖（以下为 CPU 版本；GPU 版本按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择）：

```text
conda activate casmi-course
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tools/environments/transformer.txt
python -m algorithms.transformer.check_model --train data/processed/transformer_split/train.parquet --backward --out outputs/T0/model_check.json
python -m unittest discover -s tests -p "test_*.py" -v
```

检查脚本用 DataLoader 取出一批 16 张真实训练谱，默认每次处理 2 张，拼回 `[16,128]` 输出，以降低 CPU 内存占用。`--backward` 对一个小批次使用随机目标检查梯度，**不更新参数，不代表完成训练，也不能用该损失判断检索性能**。报告已存在时需选择新的输出路径。

正式训练见下一节。实验性的检索与生成混合入口见 [混合模型流程](algorithms/transformer/HYBRID.md)，现有 baseline 和 Kaggle 提交入口尚未接入这个模型。

</details>

<details>
<summary><strong>可选：训练 Transformer Encoder（第三步）</strong></summary>

`algorithms/transformer/losses.py` 实现监督式多正样本对比损失：每张谱作为 anchor，同结构的其他谱都是正样本，不同结构为负样本。分母包含批次内除自身以外的所有谱，对每个正样本的负对数概率求平均。默认温度为 0.1。

`algorithms/transformer/train.py` 使用 AdamW，按 `zero_grad → forward → loss → backward → 梯度裁剪 → optimizer.step` 更新共享编码器。默认学习率为 1e-4，权重衰减为 0.01，梯度裁剪阈值为 1。标签不输入网络。

先进行 4 次真实权重更新的流程检查（已有输出目录时请换新名称）：

```text
python -m algorithms.transformer.train --train data/processed/transformer_split/train.parquet --epochs 2 --batches-per-epoch 2 --structures-per-batch 2 --spectra-per-structure 2 --out outputs/T1_smoke
```

此命令每批使用 2 个结构、每结构 2 张谱，共 4 张。训练需要完整批次共同计算对比损失，不使用检查脚本的独立小批次诊断方式。1500 个 token 的稠密注意力仍较耗内存，先用小批次跑通再扩大 P。小批次的负样本较少。

输出文件：

- `config.json`：模型、预处理和训练参数、数据规模。
- `history.json`：各轮训练平均损失、裁剪前梯度范数与累计更新次数。
- `last.pt`：最后完成轮次的模型权重、优化器状态及配置，每轮保存一次。
- `summary.json`：总更新次数、相对初始化的参数变化量。

读取训练权重进行推理：

```python
import torch
from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig

checkpoint = torch.load("outputs/T1_smoke/last.pt", map_location="cpu", weights_only=True)
model = SpectrumEncoder(SpectrumEncoderConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
# spectra 必须使用检查点中的 bin_width、max_mz、power 进行预处理。
with torch.no_grad():
    vectors = model(spectra)
```

省略 `--batches-per-epoch` 时，采样器按可用结构数估算每轮批次数；允许跨批重复，不保证遍历全部谱。训练入口当前从头训练，尚未提供断点续训参数。

训练只读取指定的训练文件，不使用验证集或 holdout。当前保存的是最后一轮，而非验证选出的最佳模型。四次更新只验证流程；训练损失下降也不能代替检索准确率评估。后续可使用 [混合模型流程](algorithms/transformer/HYBRID.md) 训练 Decoder、构造库内/库外验证、校准检索门控并运行混合预测。

### 在 Kaggle GPU 上训练 Encoder

`submission/kaggle/train_encoder.py` 是本地训练实现，`submission/prepare_encoder_training.py` 会生成一个自包含的单文件 Kaggle 训练脚本，把 Encoder、数据管道和配置嵌入其中。这样 Kaggle 的 script Kernel 不需要导入旁边的 `algorithms/` 目录。脚本默认从竞赛挂载的 `/kaggle/input/**/train.parquet` 读取数据，输出检查点到 `/kaggle/working/encoder_run`。

```text
python -m submission.prepare_encoder_training --out submission/build/K0_encoder --username YOUR_USERNAME
kaggle kernels push -p submission/build/K0_encoder/notebook
```

Kaggle 页面或 metadata 中应启用 GPU；竞赛数据通过 `competition_sources` 挂载，Internet 保持关闭。首次先用较小步数确认显存和路径：

```text
python train_encoder.py --epochs 1 --batches-per-epoch 20 --structures-per-batch 4 --spectra-per-structure 2 --amp --out /kaggle/working/encoder_smoke
```

正式运行可增加 `--batches-per-epoch` 和 `--epochs`。`--amp` 只在 CUDA 上启用混合精度；每轮保存 `last.pt`、`history.json` 和 `summary.json`。训练完成后从 Kaggle 输出下载 `last.pt`，在本地验证集上评估和校准门控，再训练 Decoder。当前脚本把整张 parquet 读入内存，直接处理数 GB 全量文件前应先确认 Kaggle 内存；`num_workers` 默认 2，Windows 本地验证使用 `--num-workers 0`。

</details>

<details>
<summary><strong>第三步：使用 B2 全量谱库提交到 Kaggle</strong></summary>

B0 用 5000 张谱练习流程，B2 将参考库扩大为全部训练谱。B2 保持分箱宽度 1、强度平方根变换、20 ppm 筛选和最高分合并，分批读取数据。全量参考库始终保留，程序只对符合当前查询筛选条件的谱计算相似度。

**1. 准备 B2 代码。** 在安装了课程依赖的环境中执行，将 `YOUR_USERNAME` 换成自己的用户名。每个构建目录只生成一次。

```text
python -m submission.prepare_assets --full-training --out submission/build/B2 --username YOUR_USERNAME
```

生成 `submission/build/B2/notebook/`，其中 `main.py` 是独立运行程序。它直接使用官方竞赛输入中的 `train.parquet` 和 `test.parquet`，无需上传谱库或更新 B0 的 Dataset。

**2. 登录、上传并等待运行。** 首次使用 Kaggle 命令行时，在课程 Python 环境中安装并登录：

```text
python -m pip install -r tools/environments/cli.txt
kaggle auth login
```

按提示完成浏览器授权。然后上传 B2：

```text
kaggle kernels push -p .\submission\build\B2\notebook
kaggle kernels status YOUR_USERNAME/casmi-b2-full-library
```

命令中的 `YOUR_USERNAME` 换成自己的 Kaggle 用户名。上传成功后终端会给出程序链接；后续查询以这个链接中的实际名称为准。

日志先显示检查全部参考谱，再显示 `Scanned ...` 扫描进度。运行完成后，打开个人 Code 中的 **CASMI B2 Full Library**，在 Output 中确认 `submission.csv`；`run_summary.json` 的 `reference_spectra` 应为完整训练文件中有结构标签的谱数（当前数据为 2,539,608）。

**3. 网页提交评分。** 在竞赛 Submissions 页面点击 **Submit Prediction**，选择 **CASMI B2 Full Library**、成功运行的版本和 `submission.csv`，备注写 `B2 full library`。等待公开分数出现。

`B2` 是实验名称；Kaggle 的 `Version 1/2/...` 是这份代码的保存次数，提交时选择页面实际显示的成功版本。

如需在本地试跑，使用新的输出目录：

```text
python submission/build/B2/notebook/main.py --train data/raw/train.parquet --test data/raw/test.parquet --out outputs/B2_deployed
```

这个命令检查公开测试数据上的运行流程。全量训练库包含原本用于本地验证的谱，比较验证成绩时，应先从库中移除验证查询及其重复项。

</details>

<details>
<summary><strong>可选：上传自己准备的谱库（以 B0 为例）</strong></summary>

以下以 `B0` 为例；选用其他实验时替换对应目录。`YOUR_USERNAME` 换成负责提交成员的 Kaggle 用户名。

**1. 打包并在本地运行提交入口。** 在 `casmi-course` 环境执行：

```text
python -m submission.prepare_assets --reference data/processed/reference_sample.parquet --config outputs/B0/config.json --out submission/build/B0 --username YOUR_USERNAME
python submission/build/B0/notebook/main.py --assets submission/build/B0/assets --test data/raw/test.parquet --out outputs/B0_deployed
```

成功后得到 `outputs/B0_deployed/submission.csv`。上传包位于 `submission/build/B0/`，包括 `assets/`（参考谱库、算法和参数）与 `notebook/`（运行入口）。在 `assets/README.md` 补充数据来源和小组改动，在 `assets/dataset-metadata.json` 将许可证占位符换成符合数据使用条款的 Kaggle 许可证标识。

**2. 配置 Kaggle 命令行并上传。** 首次使用时执行：

```text
conda create -n kaggle-cli --override-channels -c conda-forge python=3.11 pip
conda activate kaggle-cli
python -m pip install -r tools/environments/cli.txt
kaggle auth login
kaggle datasets create -p .\submission\build\B0\assets --dir-mode zip
kaggle datasets status YOUR_USERNAME/casmi-course-assets
```

按提示在浏览器登录；无法使用浏览器授权时，在 [API 设置](https://www.kaggle.com/settings/api) 生成令牌，运行 `python tools/kaggle/configure_token.py` 输入。数据集就绪后推送代码：

```text
kaggle kernels push -p submission/build/B0/notebook
kaggle kernels status YOUR_USERNAME/casmi-course-submit
```

**3. 等待运行完成，提交该版本。** 在 Kaggle 的个人 Code 页面打开 `casmi-course-submit`，确认输出含 `submission.csv`，记下版本号。下例的 `3` 换成实际版本：

```text
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k YOUR_USERNAME/casmi-course-submit -v 3 -f submission.csv -m "B0"
kaggle competitions submissions enveda-CASMI26-molecule-id-mass-spectra
```

看到成功评分后保存分数、版本和截图。由一位成员统一提交；返回超时先查看提交记录。后续改进使用新的构建目录，并以 `kaggle datasets version -p .\submission\build\A1\assets --dir-mode zip -m "A1"` 更新已有附件，再推送和提交新代码版本。

正式评分会替换为隐藏测试数据并重新运行代码；保留入口中的官方数据读取逻辑和断网设置。外部资源遵守[竞赛规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)，其中 [NIST 质谱及其训练模型禁用](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470)。命令参考 [Kaggle 官方说明](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)。

</details>
