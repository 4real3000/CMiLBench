# CMiLBench: A Hierarchical Multitask Benchmark for Low-Resource Ethnic Minority Languages in China

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
#### 输出结果 ####

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
- 
#### 3. 结果评估（Evaluation）

#### 第一步：答案提取

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
