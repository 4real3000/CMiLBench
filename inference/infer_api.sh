#!/bin/bash

# =================================================================
# API 调用设置
# =================================================================
# 设置API模型信息
model_name="gpt-4o"  
api_key="sk-Kt5Y22tHX24GbnxMrQIs2JP9DqVIYsQCpwRy44iIDr1d2HEz"
api_base="https://xiaoai.plus/v1" 
api_delay="0.05"  # API调用之间的延迟(秒)，以避免频率限制

# =================================================================
# Benchmark 配置 (与VLLM脚本对齐)
# =================================================================
# 设置基础路径 - 与VLLM脚本保持一致
BASE_PATH="/home/u2007087/liyijie/多民族语言benchmark/0617CMiLBench/CMiLBench_test"

# 设置模型与语言信息
prompt_lang="zh"  # 提示语言: en, zh
langs=("bo" "mn" "ug")  # 评估语言列表: bo, ug, mn

# 设置推理脚本路径 - 修改为API推理脚本
INFER_SCRIPT="/home/u2007087/liyijie/多民族语言benchmark/0617CMiLBench/infer/infer_api.py"

# =================================================================
# 脚本执行部分
# =================================================================

# 创建顶层输出目录
output_dir="${BASE_PATH}/output/${model_name}"
mkdir -p ${output_dir}

# 对每种语言生成并执行任务列表
for eval_lang in "${langs[@]}"; do
    echo "========== 正在为语言 '${eval_lang}' 生成任务配置文件 =========="
    
    # 为当前语言创建任务配置JSON文件
    tasks_file="tasks_${eval_lang}_${model_name}.json"
    echo "[" > ${tasks_file}
    
    # -----------------------------------------------------------------
    # Foundation Tasks - 使用新的路径名称
    # -----------------------------------------------------------------
    
    # 1. 文本分类 - 使用新路径名称 Text_Classification
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Text_Classification\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Text_Classification/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Text_Classification/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_passage_len\": 512," >> ${tasks_file}
    echo "    \"max_new_tokens\": 100" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 2. 自然语言推理 - 使用新路径名称 Natural_Language_Inference
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Natural_Language_Inference\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Natural_Language_Inference/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Natural_Language_Inference/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 50" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 3. 代词指代消解 - 使用新路径名称 Coreference_Resolution
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Coreference_Resolution\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Coreference_Resolution/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Coreference_Resolution/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 50" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 4. 机器阅读理解 - 使用新路径名称 Machine_Reading_Comprehension
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Machine_Reading_Comprehension\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Machine_Reading_Comprehension/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Machine_Reading_Comprehension/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 1024" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 5. 数学推理 - 使用新路径名称 Math_Reasoning
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Math_Reasoning\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/Math_Reasoning/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Math_Reasoning/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 200" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 6. 通用领域能力 - 使用新路径名称 General_Domain_Competence
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"General_Domain_Competence\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Foundation_Tasks/General_Domain_Competence/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/General_Domain_Competence/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # -----------------------------------------------------------------
    # Chinese Minority Knowledge Tasks - 使用新的路径名称
    # -----------------------------------------------------------------
    
    # 7. 少数民族文化问答 - 使用新路径名称 Minority_Culture_QA
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Culture_QA\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Culture_QA/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Culture_QA/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 200" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 8. 少数民族领域能力 - 使用新路径名称 Minority_Domain_Competence
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Domain_Competence\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Domain_Competence/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Domain_Competence/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 9. 少数民族语言表达 - 使用新路径名称 Minority_Language_Expressions
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Language_Expressions\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Language_Expressions/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Language_Expressions/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 10. 少数民族语言指令问答 - 使用新路径名称 Minority_Language_Instruction_QA
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Language_Instruction_QA\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Language_Instruction_QA/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Language_Instruction_QA/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 1000" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 11. 少数民族语言理解 - 使用新路径名称 Minority_Language_Understanding
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Language_Understanding\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Chinese_Minority_Knowledge_Tasks/Minority_Language_Understanding/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Minority_Language_Understanding/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 12. 少数民族机器翻译 (中文->评估语言) - 使用新路径名称 Minority_Machine_Translation
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Machine_Translation\"," >> ${tasks_file}
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
    
    # 13. 少数民族机器翻译 (评估语言->中文) - 使用新路径名称 Minority_Machine_Translation
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Minority_Machine_Translation\"," >> ${tasks_file}
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
    
    # -----------------------------------------------------------------
    # Safety Alignment Tasks - 使用新的路径名称
    # -----------------------------------------------------------------
    
    # 14. 商业合规检查 - 使用新路径名称 Commercial_Compliance_Check
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Commercial_Compliance_Check\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Commercial_Compliance_Check/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Commercial_Compliance_Check/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 15. 歧视检测 - 使用新路径名称 Discrimination_Detection
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Discrimination_Detection\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Discrimination_Detection/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Discrimination_Detection/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 16. 权利保护评估 - 使用新路径名称 Rights_Protection_Evaluation
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Rights_Protection_Evaluation\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Rights_Protection_Evaluation/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Rights_Protection_Evaluation/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 17. 服务安全评估 - 使用新路径名称 Service_Safety_Evaluation
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Service_Safety_Evaluation\"," >> ${tasks_file}
    echo "    \"eval_lang\": \"${eval_lang}\"," >> ${tasks_file}
    echo "    \"prompt_lang\": \"${prompt_lang}\"," >> ${tasks_file}
    echo "    \"input_file\": \"${BASE_PATH}/Safety_Alignment_Tasks/Service_Safety_Evaluation/${eval_lang}.json\"," >> ${tasks_file}
    echo "    \"output_file\": \"${output_dir}/Service_Safety_Evaluation/${eval_lang}/${prompt_lang}-prompt_test.json\"," >> ${tasks_file}
    echo "    \"exemplar_file\": null," >> ${tasks_file}
    echo "    \"num_exemplar\": 3," >> ${tasks_file}
    echo "    \"max_new_tokens\": 20" >> ${tasks_file}
    echo "  }," >> ${tasks_file}
    
    # 18. 价值观对齐评估 (最后一个任务，末尾无逗号) - 使用新路径名称 Value_Alignment_Assessment
    echo "  {" >> ${tasks_file}
    echo "    \"task\": \"Value_Alignment_Assessment\"," >> ${tasks_file}
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

    # 创建所有必需的输出子目录 - 使用新的路径名称
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
    mkdir -p ${output_dir}/Commercial_Compliance_Check/${eval_lang}
    mkdir -p ${output_dir}/Discrimination_Detection/${eval_lang}
    mkdir -p ${output_dir}/Rights_Protection_Evaluation/${eval_lang}
    mkdir -p ${output_dir}/Service_Safety_Evaluation/${eval_lang}
    mkdir -p ${output_dir}/Value_Alignment_Assessment/${eval_lang}
    
    echo "任务配置文件已生成: ${tasks_file}"
    echo "========== 开始通过API处理 '${eval_lang}' 语言的所有任务... =========="
    
    # 运行API批处理脚本
    python ${INFER_SCRIPT} \
        --model_name "${model_name}" \
        --api_key "${api_key}" \
        --api_base "${api_base}" \
        --api_delay ${api_delay} \
        --dataset_path ${BASE_PATH} \
        --task_list ${tasks_file} \
        --batch_size 1 \
        --save_frequency 5 \
        --max_test_example_num -1
    
    echo "========== 完成 '${eval_lang}' 语言的所有任务 =========="
    echo ""
done

echo "********************************************"
echo "所有语言的所有任务均已处理完成！"
echo "结果保存在: ${output_dir}"
echo "********************************************"

# 显示输出目录结构以便检查
echo ""
echo "========== 输出目录结构预览 (顶层) =========="
if command -v tree &> /dev/null; then
    tree ${output_dir} -L 2
else
    echo "tree命令未找到，将使用find命令显示目录结构："
    find ${output_dir} -maxdepth 2
fi