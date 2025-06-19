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

---

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

#### 方法二：使用现有环境

```bash
# 如果您已有配置好的环境（如eval）
conda activate eval  # 或您的环境名
cd CMiLBench

# 验证环境
python --version      # 应显示 Python 3.11.11
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # 应显示 2.6.0+cu124
```

### 开始评测

#### 1. API模型推理（API-based Inference）

```bash
# 使用OpenAI API进行推理
cd inference
python infer_api.py \
    --model_name gpt-4o \
    --api_key your_api_key_here \
    --task_list tasks_bo.json \
    --output_dir ./results \
    --batch_size 1
```

#### 2. 本地模型推理（Local Model Inference）

```bash
# 使用vLLM进行本地模型推理
python infer_vllm.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --task_list tasks_mn.json \
    --output_dir ./results \
    --tensor_parallel_size 1
```

#### 3. 批量推理（Batch Inference）

```bash
# API模型批量推理
bash infer_api.sh

# 本地模型批量推理
bash infer_vllm.sh
```

#### 4. 结果评估（Evaluation）

```bash
# 综合评估
cd evaluation
python comprehensive_evaluation.py \
    --result_dir ../inference/results \
    --output_dir ./evaluation_results

# LLM-as-a-Judge评估
python llm_evaluation.py \
    --result_dir ../inference/results \
    --judge_model gpt-4 \
    --api_key your_api_key_here
```

### 数据格式说明（Data Format）

每个任务目录包含三种语言的数据文件：
- `bo.json` - 藏语数据
- `mn.json` - 蒙古语数据
- `ug.json` - 维吾尔语数据

数据格式示例：
```json
[
    {
        "id": "sample_001",
        "question": "问题文本",
        "answer": "参考答案",
        "metadata": {
            "language": "bo",
            "task_type": "classification"
        }
    }
]
```


