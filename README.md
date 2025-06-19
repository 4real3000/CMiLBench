# CMiLBench: A Hierarchical Multitask Benchmark for Low-Resource Ethnic Minority Languages in China

## 基准简介

**CMiLBench** 是一个层次化的多任务评测基准，专为中国少数民族语言（藏语 `bo`、蒙古语 `mn`、维吾尔语 `ug`）设计。该基准旨在系统评估大语言模型在低资源语言环境下的理解、生成与安全对齐能力。

CMiLBench 包含以下三大任务类别，共计 17 个子任务，覆盖语言基础能力、文化知识能力与多语言安全性：

- 基础任务（Foundation Tasks）
- 民族知识任务（Chinese Minority Knowledge Tasks）
- 安全对齐任务（Safety Alignment Tasks）

### 任务分布总览图

![task_categories_overview](./assets/task_categories_overview.png)

> 图示展示了 CMiLBench 的三大任务类别与 17 个子任务之间的映射关系（如图3所示，示意图需放置在 `assets/` 文件夹中）。

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
