#!/bin/bash

# 设置基础路径 - 更新为新的数据集路径
BASE_PATH="/home/u2007087/liyijie/多民族语言benchmark/0617CMiLBench/CMiLBench_test"

# 设置模型信息
model_type="qwen"  # 模型类型: qwen, aya, llama, mistral, gemma
model_path="/home/u2007087/liyijie/models/Qwen2.5-3B-Instruct"  # 模型路径
model_name="Qwen2.5-3B-Instruct"  # 模型名称，用作保存推理结果的目录名
prompt_lang="zh"  # 提示语言: en, zh
langs=("bo" "mn" "ug")  # 评估语言列表: bo, ug, mn

# 设置推理脚本路径
INFER_SCRIPT="/home/u2007087/liyijie/多民族语言benchmark/0617CMiLBench/infer/infer_vllm.py"

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 设置VLLM相关参数
gpu_memory_utilization=0.9  # GPU内存使用率
tensor_parallel_size=1      # 张量并行大小

# 创建输出目录
output_dir="${BASE_PATH}/output/${model_name}"
mkdir -p ${output_dir}

# 对每种语言生成任务列表
for eval_lang in "${langs[@]}"; do
    echo "========== 生成 ${eval_lang} 语言的任务列表 =========="
    
    # 创建任务配置JSON文件
    tasks_file="tasks_${eval_lang}.json"
    echo "[" > ${tasks_file}
    
    # Foundation Tasks
    
    # 文本分类 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"text_classification\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Text_Classification/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Text_Classification/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_passage_len\": 512," >> ${tasks_file}
    echo "    \"max_new_tokens\": 100" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 自然语言推理 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"entailment\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Natural_Language_Inference/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Natural_Language_Inference/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 50" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 代词指代消解 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"coref_resolution\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Coreference_Resolution/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Coreference_Resolution/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 50" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 机器阅读理解 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"reading_comprehension\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Machine_Reading_Comprehension/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Machine_Reading_Comprehension/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 1024" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 数学推理 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"math_reasoning\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Math_Reasoning/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Math_Reasoning/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 200" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 通用领域能力 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"professional_skills\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/General_Domain_Competence/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/General_Domain_Competence/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # Chinese Minority Knowledge Tasks
    
    # 少数民族文化问答 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"traditional_culture\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Culture_QA/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Culture_QA/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 200" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 少数民族领域能力 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"ethnic_vocabulary\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Domain_Competence/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Domain_Competence/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 少数民族语言表达 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"ethnic_vocabulary\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Language_Expressions/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Language_Expressions/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 少数民族语言指令问答 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"text_generation\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Language_Instruction_QA/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Language_Instruction_QA/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 1000" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 少数民族语言理解 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"ethnic_language_understanding\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Language_Understanding/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Language_Understanding/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 少数民族机器翻译 (中文->评估语言) - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"translation\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Machine_Translation/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Machine_Translation/${eval_lang}/${prompt_lang}-prompt_zh2${eval_lang}_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"src_lang\": \"zh\"," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"tgt_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"max_new_tokens\": 300" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 少数民族机器翻译 (评估语言->中文) - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"translation\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Machine_Translation/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Machine_Translation/${eval_lang}/${prompt_lang}-prompt_${eval_lang}2zh_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"src_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"tgt_lang\": \"zh\"," >> ${tasks_file}
    echo "    \"max_new_tokens\": 300" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # Safety Alignment Tasks - 使用新的输出路径
    
    # 商业合规检查 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"safety\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Commercial_Compliance_Check/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Commercial_Compliance_Check/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 歧视检测 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"safety\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Discrimination_Detection/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Discrimination_Detection/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 权利保护评估 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"safety\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Rights_Protection_Evaluation/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Rights_Protection_Evaluation/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 服务安全评估 - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"safety\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Service_Safety_Evaluation/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Service_Safety_Evaluation/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 价值观对齐评估 (最后一个任务，不加逗号) - 使用新的输出路径
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"safety\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Value_Alignment_Assessment/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Value_Alignment_Assessment/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }" >> ${tasks_file}
    
    # 关闭JSON数组
    echo "]" >> ${tasks_file}

    # 创建必要的输出目录 - 使用新的目录名称
    mkdir -p ${output_dir}/Text_Classification/${eval_lang}
    mkdir -p ${output_dir}/Natural_Language_Inference/${eval_lang}
    mkdir -p ${output_dir}/Coreference_Resolution/${eval_lang}
    mkdir -p ${output_dir}/Machine_Reading_Comprehension/${eval_lang}
    mkdir -p ${output_dir}/Math_Reasoning/${eval_lang}
    mkdir -p ${output_dir}/General_Domain_Competence/${eval_lang}
    mkdir -p ${output_dir}/Minority_Culture_QA/${eval_lang}
    mkdir -p ${output_dir}/Minority_Domain_Competence/${eval_lang}
    mkdir -p ${output_dir}/Minority_Language_Expressions/${eval_lang}
    mkdir -p ${output_dir}/Minority_Language_Instruction_QA/${eval_lang}
    mkdir -p ${output_dir}/Minority_Language_Understanding/${eval_lang}
    mkdir -p ${output_dir}/Minority_Machine_Translation/${eval_lang}
    # 创建安全任务目录 - 使用新的目录名称
    mkdir -p ${output_dir}/Commercial_Compliance_Check/${eval_lang}
    mkdir -p ${output_dir}/Discrimination_Detection/${eval_lang}
    mkdir -p ${output_dir}/Rights_Protection_Evaluation/${eval_lang}
    mkdir -p ${output_dir}/Service_Safety_Evaluation/${eval_lang}
    mkdir -p ${output_dir}/Value_Alignment_Assessment/${eval_lang}
    
    echo "任务列表已生成: ${tasks_file}"
    echo "开始处理 ${eval_lang} 语言的所有任务..."
    
    # 运行批处理脚本
    python ${INFER_SCRIPT} \
        --model_type ${model_type} \
        --model_path ${model_path} \
        --dataset_path ${BASE_PATH} \
        --task_list ${tasks_file} \
        --batch_size 4 \
        --save_frequency 5 \
        --max_test_example_num -1 \
        --gpu_memory_utilization ${gpu_memory_utilization} \
        --tensor_parallel_size ${tensor_parallel_size}
    
    echo "完成 ${eval_lang} 语言的所有任务"
done

echo "所有任务处理完成！"
echo "结果保存在: ${output_dir}"

# 显示输出目录结构
echo ""
echo "========== 输出目录结构 =========="
if command -v tree &> /dev/null; then
    tree ${output_dir} -L 2
else
    find ${output_dir} -type d | head -30
fi