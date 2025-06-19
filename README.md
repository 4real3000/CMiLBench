# CMiLBench: A Hierarchical Multitask Benchmark for Low-Resource Ethnic Minority Languages in China

<p align="center">
  <!-- Language Switch Buttons -->
  <a href="#chinese">
    <img src="https://img.shields.io/badge/lang-中文-red.svg?style=flat-square" alt="Chinese">
  </a>
  <a href="#english">
    <img src="https://img.shields.io/badge/lang-English-blue.svg?style=flat-square" alt="English">
  </a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-green.svg?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Tasks-17-orange.svg?style=flat-square" alt="Tasks">
  <img src="https://img.shields.io/badge/Languages-3-purple.svg?style=flat-square" alt="Languages">
</p>

<a id="english"></a>🇺🇸 English

## 基准简介

**CMiLBench** 是一个层次化的多任务评测基准，专为中国少数民族语言（藏语 `bo`、蒙古语 `mn`、维吾尔语 `ug`）设计。该基准旨在系统评估大语言模型在低资源语言环境下的理解、生成与安全对齐能力。

CMiLBench 包含以下三大任务类别，共计 17 个子任务，覆盖语言基础能力、文化知识能力与多语言安全性：

- 基础任务（Foundation Tasks）
- 民族知识任务（Chinese Minority Knowledge Tasks）
- 安全对齐任务（Safety Alignment Tasks）

### 任务分布总览图

![task_categories_overview](./assets/category.png)

---

## 文件结构说明

```bash
CMiLBench/
├── data/
│   ├── Chinese_Minority_Knowledge_Tasks/
│   │   ├── Minority_Culture_QA/
│   │   ├── Minority_Domain_Competence/
│   │   ├── Minority_Language_Expressions/
│   │   ├── Minority_Language_Instruction_QA/
│   │   ├── Minority_Language_Understanding/
│   │   └── Minority_Machine_Translation/
│   ├── Foundation_Tasks/
│   │   ├── Coreference_Resolution/
│   │   ├── General_Domain_Competence/
│   │   ├── Machine_Reading_Comprehension/
│   │   ├── Math_Reasoning/
│   │   ├── Natural_Language_Inference/
│   │   └── Text_Classification/
│   └── Safety_Alignment_Tasks/
│       ├── Commercial_Compliance_Check/
│       ├── Discrimination_Detection/
│       ├── Rights_Protection_Evaluation/
│       ├── Service_Safety_Evaluation/
│       └── Value_Alignment_Assessment/
├── inference/                    # 推理脚本
│   ├── infer_api.py
│   ├── infer_api.sh
│   ├── infer_vllm.py
│   └── infer_vllm.sh
├── evaluation/                  # 评估脚本
│   ├── answer_extraction.py
│   ├── comprehensive_evaluation.py
│   └── llm_evaluation.py.py
└── README.md

```

## 🚀 快速开始

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/CMiLBench.git
cd CMiLBench

# 2. 创建conda环境
conda create -n cmilbench python=3.11
conda activate cmilbench

# 3. 安装依赖
pip install -r requirements.txt
```

### 开始评测

#### 1. API模型推理（API-based Inference）

**第一步：配置API信息**

在执行脚本中配置您的API密钥和地址：

```bash
# 编辑推理脚本
nano inference/infer_api.sh

# 修改以下配置项（必需）：
model_name="gpt-4o"                   # 您要使用的模型名称
api_key="your_api_key_here"           # 替换为您的实际API密钥
api_base="https://api.openai.com/v1"  # 替换为您的API地址
BASE_PATH="/path/to/CMiLBench"        # 修改为实际的数据集路径
INFER_SCRIPT="/path/to/infer_vllm.py" # 推理脚本路径
```

**第二步：执行推理**

```bash
# 运行完整推理（所有任务、所有语言）
cd inference
bash infer_api.sh
```

#### 2. 本地模型推理（Local Model Inference）

**第一步：配置模型和数据集路径**

在执行脚本中配置您的本地模型和数据集路径：

```bash
# 编辑推理脚本
nano inference/infer_vllm.sh

# 修改以下配置项（必需）：
model_type="qwen"                      # 模型类型: qwen, aya, llama, mistral, gemma
model_path="/path/to/your/model"       # 替换为您的本地模型路径
model_name="gpt-4o"                    # 您要使用的模型名称
BASE_PATH="/path/to/CMiLBench"         # 修改为实际的数据集路径
INFER_SCRIPT="/path/to/infer_vllm.py"  # 推理脚本路径

# GPU配置（可选）：
export CUDA_VISIBLE_DEVICES=0          # 指定使用的GPU
gpu_memory_utilization=0.9             # GPU内存使用率
tensor_parallel_size=1                 # 张量并行大小
```

**第二步：执行推理**

```bash
# 运行完整推理（所有任务、所有语言）
cd inference
bash infer_vllm.sh
```

推理完成后，结果将保存在以下目录结构中：

```
output/
├── {model_name}/
│   ├── Foundation_Tasks/
│   │   ├── Text_Classification/{lang}/
│   │   ├── Natural_Language_Inference/{lang}/
│   │   └── ...
│   ├── Chinese_Minority_Knowledge_Tasks/
│   │   ├── Minority_Culture_QA/{lang}/
│   │   ├── Minority_Machine_Translation/{lang}/
│   │   └── ...
│   └── Safety_Alignment_Tasks/
│       ├── Commercial_Compliance_Check/{lang}/
│       ├── Discrimination_Detection/{lang}/
│       └── ...
```

每个任务的结果文件包含：
- `id`: 样本ID
- `pred`: 模型预测结果
- `gold`: 标准答案
  
#### 3. 结果评估（Evaluation）

##### 第一步：答案提取

对推理结果进行标准化答案提取，为后续评估做准备：

```bash
# 编辑答案提取脚本
nano evaluation/answer_extraction.py

# 执行答案提取
python evaluation/answer_extraction.py \
    --base_path "/path/to/output"              # 📁 推理结果的基础路径
    --output_dir "/path/to/extracted_answers"  # 📁 提取答案的输出目录
    --model "gpt-4o"                          # 🎯 指定处理的模型（可选，不指定则处理所有模型）
    --task "Text_Classification"              # 📋 指定处理的任务（可选，不指定则处理所有任务）
    --language "bo"                           # 🌐 指定处理的语言（可选，不指定则处理所有语言）
```

提取完成后，结果将保存在以下目录结构中：

output_dir/
├── {model_name}/
│   ├── {task_name}/
│   │   └── {language}/
│   │       └── *.json                    # 处理后的数据文件
├── processed_files_map.json              # 处理文件映射表
├── extraction_failed_ids.json            # 提取失败ID记录
├── extraction_statistics.json            # 提取统计数据
└── extraction_report.txt                 # 可读统计报告

##### 第二步：生成式任务评估

使用LLM对生成式任务（传统文化问答和文本生成）进行多维度质量评估：

```bash
# 编辑生成式任务评估脚本
nano evaluation/generative_evaluation.py

# 执行生成式任务评估
python evaluation/generative_evaluation.py \
    --test_data_path "/path/to/CMiLBench"                    # 📁 测试数据集基础路径
    --models_predictions_path "/path/to/extracted_answers"   # 📁 模型预测结果路径（第一步的输出）
    --output_path "/path/to/evaluation_results"             # 📁 评估结果输出路径
    --api_key "your_api_key_here"                           # 🔑 OpenAI API密钥
    --api_base "https://api.openai.com/v1"                  # 🌐 API基础URL（可选）
    --model "gpt-4o"                                        # 🤖 用于评估的LLM模型
    --max_workers 5                                         # ⚡ 并行处理线程数
    --models_to_evaluate "Qwen2.5-7B-Instruct"             # 🎯 指定要评估的模型（可选）
    --task "text_generation"                                # 📋 指定任务类型（可选）
    --language "bo"                                         # 🌐 指定语言（可选）
    --sample_size 100                                       # 📊 评估样本数（可选，默认全部）
    --resume                                                # 🔄 从断点继续（可选）
```

评估完成后，结果将保存在以下目录结构中：
```
evaluation_results/
├── {model_name}/
│   ├── Minority_Culture_QA/
│   │   ├── bo_evaluation.json
│   │   ├── mn_evaluation.json
│   │   ├── ug_evaluation.json
│   │   ├── bo_checkpoint.json (临时文件)
│   │   ├── bo_errors.log
│   │   └── bo_error_ids.json
│   └── Minority_Language_Instruction_QA/
│       ├── bo_evaluation.json
│       ├── mn_evaluation.json
│       ├── ug_evaluation.json
│       ├── bo_checkpoint.json (临时文件)
│       ├── bo_errors.log
│       └── bo_error_ids.json
```
#### 第三步：综合评估（Comprehensive Evaluation）

使用综合评估脚本对所有任务进行多维度评估，计算准确率、ROUGE-L、BLEU、chrF++等指标，并生成详细的评估报告和模型排名：

```bash
# 编辑综合评估脚本
nano evaluation/comprehensive_evaluation.py

# 执行综合评估
python evaluation/comprehensive_evaluation.py \
    --input_dir "/path/to/extracted_answers"           # 📁 答案提取结果的目录（第一步的输出）
    --output_dir "/path/to/comprehensive_results"      # 📁 综合评估结果的输出目录
    --llm_eval_dir "/path/to/llm_evaluation_results"   # 📁 LLM评价结果目录（第二步的输出，用于生成式任务）
    --model "gpt-4o"                                   # 🎯 指定要评估的模型（可选，不指定则评估所有模型）
    --task "Text_Classification"                       # 📋 指定要评估的任务目录名（可选，不指定则评估所有任务）
    --language "bo"                                    # 🌐 指定要评估的语言（可选，不指定则评估所有语言）
```
###### 输出结果

综合评估完成后，将在输出目录中生成以下文件：

```
comprehensive_results/
├── evaluation_summary.csv          # 📊 详细评估汇总表
├── task_ranking.csv               # 🏆 任务级别排名表
├── model_overall_ranking.csv      # 🥇 模型综合排名表
└── ranking_report.txt             # 📄 可读排名报告
```

**📊 `evaluation_summary.csv` - 详细评估汇总表**
包含每个模型在每个任务上的详细评估结果：

| 字段 | 说明 |
|------|------|
| Model | 模型名称 |
| Task | 任务名称 |
| Language | 评估语言 |
| File | 结果文件名 |
| Metric | 评估指标 |
| Score_Type | 得分类型（all/success） |
| Score | 评估得分 |
| Sample_Count | 总样本数 |
| Success_Count | 成功处理样本数 |
| Success_Rate | 成功处理率 |

**🏆 `task_ranking.csv` - 任务级别排名表**
每个任务的模型排名情况：

| 字段 | 说明 |
|------|------|
| Task_Key | 任务标识键 |
| Rank | 排名 |
| Model | 模型名称 |
| Metric | 主要评估指标 |
| Score | 评估得分 |

**🥇 `model_overall_ranking.csv` - 模型综合排名表**
基于所有任务表现的模型综合排名：

| 字段 | 说明 |
|------|------|
| Model | 模型名称 |
| Overall_Rank | 综合排名 |
| Average_Rank | 平均排名 |
| Total_Score | 总得分 |
| Tasks_Evaluated | 评估任务数 |

**📄 `ranking_report.txt` - 可读排名报告**
人类可读的排名报告文本，包含：
- 综合排名概览
- 各任务详细排名
- 模型表现分析
