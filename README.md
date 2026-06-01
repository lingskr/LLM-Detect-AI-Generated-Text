# LLM-Detect AI Generated Text

本项目 **LLM - Detect AI Generated Text**，目标是判断一篇学生作文是否由大语言模型生成。核心方案参考 `code/llm-detect-code.ipynb`，采用 **TF-IDF 线性模型 + DistilRoBERTa + DeBERTa-v3-small** 的多模型融合，并在第二阶段使用高置信度样本做伪标签训练，最终生成 `submission.csv`。

## 项目思路

整体方案包含三条预测线：

1. **基于 TF-IDF 的线性模型**
   - 使用测试集文本训练 BPE tokenizer。
   - 对训练集和测试集分词后构造 word n-gram TF-IDF 特征，`ngram_range=(3, 5)`。
   - 使用 `MultinomialNB` 与 `SGDClassifier` 组成 soft voting 分类器。
   - 根据预测分数的 rank 选择高置信度样本加入训练集，进行二阶段伪标签训练。

2. **基于 DistilRoBERTa 的文本二分类模型**
   - 参考开源 Kaggle notebook `train-detectai-distilroberta-0-927.ipynb`。
   - 使用微调后的 DistilRoBERTa checkpoint 对测试集推理，输出 `sub_nn.csv`。

3. **基于 DeBERTa-v3-small 的大规模数据二分类模型**
   - 使用 Pile、UltraFeedback、UltraChat、LMSYS 等大规模人类/AI文本数据进行训练。
   - 训练脚本为 `code/deberta_train_exp5.py`。
   - 推理输出 `sub_nn2.csv`。

最终融合流程：

1. 第一阶段融合：
   - `sub_linear.csv`
   - `sub_nn.csv`
   - `sub_nn2.csv`
   - 对三个结果分别做 rank 归一化。
   - 融合权重为 `0.7 * linear + 0.2 * distilroberta + 0.1 * deberta`。
   - 输出 `sub_stage1.csv`。

2. 第二阶段伪标签线性模型：
   - 使用第一阶段融合结果作为伪标签筛选依据。
   - 选取预测排名前 15% 和后 15% 的测试样本加入训练。
   - 重新训练线性模型，输出 `sub_linear_stage2.csv`。

3. 最终融合：
   - 再次对 `sub_linear_stage2.csv`、`sub_nn.csv`、`sub_nn2.csv` 做 rank 归一化。
   - 继续使用 `0.7 / 0.2 / 0.1` 权重融合。
   - 输出最终 `submission.csv`。

## 目录结构

```text
.
├── code/
│   ├── llm-detect-code.ipynb              # 主方案 notebook，包含完整推理和融合流程
│   ├── deberta_train_exp5.py              # DeBERTa-v3-small 训练脚本
│   ├── train-detectai-distilroberta-0-927.ipynb
│   ├── ai-or-not-ai-delving-into-essays-with-eda.ipynb
│   ├── embed_vis.py                       # 文本嵌入 t-SNE 可视化脚本
│   ├── robustness_hf_eval.py              # Hugging Face 模型鲁棒性评估脚本
│   ├── train1.csv                         # 线性模型使用的训练数据
│   └── nonTargetText_llm_slightly_modified_gen.csv
├── data/
│   ├── llm-detect-ai-generated-text/      # Kaggle 官方竞赛数据
│   └── train_v2_drcat_02.csv              # DAIGT v2 外部训练数据
├── plies-and-ultra/                       # DeBERTa 训练用大规模 parquet 数据
├── models/                                # 本地模型权重目录
├── notebooks/                             # 补充实验、训练、EDA 与 CSEE 外部验证 notebook
├── outputs/                               # EDA 图表、外部验证结果与预测结果
├── output/                                # DistilRoBERTa 训练实验输出
├── requirements.txt                       # Python 依赖
└── LLM2 讲解.pdf                          # 项目讲解材料
```

## 环境安装

建议使用 Python 3.10 或以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要训练 Transformer 模型，建议使用带 CUDA 的 PyTorch 环境。`requirements.txt` 中没有固定 CUDA 版本，实际安装时请根据本机或 Kaggle GPU 环境选择匹配的 PyTorch 版本。

## 数据说明

项目主要使用以下数据：

- Kaggle 官方数据：`data/llm-detect-ai-generated-text/`
  - `train_essays.csv`
  - `test_essays.csv`
  - `sample_submission.csv`
  - `train_prompts.csv`
- DAIGT v2 外部训练集：`data/train_v2_drcat_02.csv`
- 线性模型训练集：`code/train1.csv`
- 外部验证集：`code/nonTargetText_llm_slightly_modified_gen.csv`
- 大规模 DeBERTa 训练数据：`plies-and-ultra/`
  - `pile2.parquet`
  - `plies3.parquet`
  - `plies4.parquet`
  - `Ultra.parquet`
  - `lmsys.parquet`
- CSEE 外部验证数据：`CESS/`

`code/llm-detect-code.ipynb` 中的推理路径使用 Kaggle 环境路径，例如 `/kaggle/input/...`。如果在本地运行，需要把 notebook 或生成的脚本中的路径替换为本仓库对应目录。

## 主 notebook 复现流程

主流程位于：

```text
code/llm-detect-code.ipynb
```

该 notebook 会依次生成并执行以下推理脚本：

```text
lin_infer.py
distilroberta_infer.py
deberta_infer.py
lin_infer_stage2.py
```

执行顺序为：

1. 运行 `lin_infer.py`，得到 `sub_linear.csv`。
2. 运行 `distilroberta_infer.py`，得到 `sub_nn.csv`。
3. 运行 `deberta_infer.py`，得到 `sub_nn2.csv`。
4. 对三路结果做第一阶段 rank 融合，得到 `sub_stage1.csv`。
5. 运行 `lin_infer_stage2.py`，根据 `sub_stage1.csv` 做伪标签训练，得到 `sub_linear_stage2.csv`。
6. 对第二阶段线性模型和两个 Transformer 模型再次融合，得到最终 `submission.csv`。

在 Kaggle Notebook 中可直接按单元格顺序运行；在本地运行时需要先准备模型 checkpoint，并修改输入数据路径。

## DeBERTa-v3-small 训练

DeBERTa 训练脚本为：

```bash
python code/deberta_train_exp5.py \
  --model-checkpoint microsoft/deberta-v3-small \
  --pile-dir plies-and-ultra \
  --valid-csv code/nonTargetText_llm_slightly_modified_gen.csv \
  --output-dir models/deberta/deberta-v3-small-finetuned_v5 \
  --max-length 384 \
  --batch-size 16 \
  --num-train-epochs 6
```

脚本会从 `--pile-dir` 中读取 parquet 数据，构造 `text,label` 二分类训练集，并使用外部验证集按 ROC-AUC 保存最优模型。

## 辅助实验

项目还包含若干辅助脚本和 notebook：

- `code/embed_vis.py`：使用 SentenceTransformer 抽取文本嵌入，并通过 t-SNE 可视化 LLM 文本和学生作文的分布。
- `code/robustness_hf_eval.py`：对 Hugging Face 二分类模型进行鲁棒性测试，包括字符替换、截断、空白归一化等扰动。
- `notebooks/01_linear_stage1_stage2_training.ipynb`：线性模型一阶段和二阶段训练实验。
- `notebooks/02_distilroberta_training.ipynb`：DistilRoBERTa 训练实验。
- `notebooks/03_deberta_v3_small_training.ipynb`：DeBERTa-v3-small 训练实验。
- `notebooks/04_csee_deepseek_generate_llm.ipynb`：CSEE 数据的 LLM 生成流程。
- `notebooks/05_three_model_inference_csee.ipynb`：三模型在 CSEE 外部验证集上的推理。
- `notebooks/06_csee_human_llm_eda.ipynb`：CSEE 数据分析。

## 输出结果

主要输出文件：

- `sub_linear.csv`：一阶段 TF-IDF 线性模型预测。
- `sub_nn.csv`：DistilRoBERTa 预测。
- `sub_nn2.csv`：DeBERTa-v3-small 预测。
- `sub_stage1.csv`：第一阶段三模型融合结果。
- `sub_linear_stage2.csv`：二阶段伪标签线性模型预测。
- `submission.csv`：最终提交文件。

外部验证和分析结果保存在 `outputs/`：

- `outputs/csee_three_line_metrics.csv`
- `outputs/table15_csee_external_validation_results.csv`
- `outputs/csee_three_line_predictions.csv`
- `outputs/csee_eda/`
- `outputs/csee_eda_only/`

其中 CSEE 外部验证结果显示，融合模型在该验证集上取得较高的 AUC 和 F1，具体指标可查看 `outputs/table15_csee_external_validation_results.csv`。

## 注意事项

- 主 notebook 主要面向 Kaggle 环境，路径与模型 checkpoint 需要按运行环境调整。
- Transformer 推理依赖本地或 Kaggle 输入目录中的模型权重。
- `train1.csv`、`plies-and-ultra/`、`resources/` 等数据文件较大，迁移项目时需要确认是否完整拷贝。
- 二阶段伪标签依赖第一阶段预测结果，不能跳过 `sub_stage1.csv`。
- 最终融合使用 rank 归一化而不是直接平均原始概率，复现时不要省略该步骤。
