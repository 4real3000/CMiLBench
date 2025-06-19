import json
import time
import argparse
import os
import signal
import sys
from tqdm import tqdm
import torch
import traceback
from datetime import datetime
from vllm import LLM, SamplingParams

# 创建一个记录错误的函数
def log_error(error_file, error_id_file, message, error, item_ids=None):
    """
    记录错误到日志文件并保存相关ID
    
    参数:
        error_file: 错误日志文件路径
        error_id_file: 错误ID文件路径
        message: 错误信息
        error: 异常对象
        item_ids: 出错的数据项ID列表
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_traceback = traceback.format_exc()
    
    # 记录详细错误日志
    with open(error_file, 'a', encoding='utf-8') as f:
        f.write(f"=== {timestamp} ===\n")
        f.write(f"错误信息: {message}\n")
        f.write(f"异常: {str(error)}\n")
        f.write(f"堆栈跟踪:\n{error_traceback}\n\n")
    
    # 保存出错的ID
    if item_ids:
        try:
            # 读取现有ID（如果文件存在）
            existing_ids = []
            if os.path.exists(error_id_file):
                with open(error_id_file, 'r', encoding='utf-8') as f:
                    existing_ids = json.load(f)
            
            # 添加新的ID
            for item_id in item_ids:
                if item_id not in existing_ids:
                    existing_ids.append(item_id)
            
            # 写回文件
            with open(error_id_file, 'w', encoding='utf-8') as f:
                json.dump(existing_ids, f, indent=4, ensure_ascii=False)
            
            print(f"已将{len(item_ids)}个错误ID保存到 {error_id_file}")
        except Exception as e:
            print(f"保存错误ID时发生异常: {str(e)}")

def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n捕获到中断信号，正在退出...")
    sys.exit(0)

def save_results(results, output_file):
    """安全地保存结果到文件"""
    # 保存到临时文件，然后重命名，避免保存过程中断导致文件损坏
    temp_file = output_file + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # 替换原文件
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)
        print(f"已保存{len(results)}条结果到{output_file}")
        return True
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
        return False
        
abbr_to_lang_en = {
    "zh": "Chinese",
    "en": "English",
    "mn": "Mongolian",
    "ug": "Uyghur",
    "bo": "Tibetan",
}

abbr_to_lang_zh = {
    "zh": "汉语",
    "en": "英语",
    "mn": "蒙古语",
    "ug": "维吾尔语",
    "bo": "藏语",
}

# 任务名称映射：新的目录结构 -> 代码中的任务名称
TASK_MAPPING = {
    # Foundation_Tasks
    'Coreference_Resolution': 'coref_resolution',
    'General_Domain_Competence': 'professional_skills',
    'Machine_Reading_Comprehension': 'reading_comprehension',
    'Math_Reasoning': 'math_reasoning',
    'Natural_Language_Inference': 'entailment',
    'Text_Classification': 'text_classification',
    
    # Chinese_Minority_Knowledge_Tasks
    'Minority_Culture_QA': 'traditional_culture',
    'Minority_Domain_Competence': 'ethnic_domain_knowledge',
    'Minority_Language_Expressions': 'ethnic_vocabulary',
    'Minority_Language_Instruction_QA': 'text_generation',
    'Minority_Language_Understanding': 'ethnic_language_understanding',
    'Minority_Machine_Translation': 'translation',
    
    # Safety_Alignment_Tasks - 统一映射到 'safety'
    'Commercial_Compliance_Check': 'safety',
    'Discrimination_Detection': 'safety',
    'Rights_Protection_Evaluation': 'safety',
    'Service_Safety_Evaluation': 'safety',
    'Value_Alignment_Assessment': 'safety',
}

# 反向映射：代码中的任务名称 -> 新的目录结构
REVERSE_TASK_MAPPING = {v: k for k, v in TASK_MAPPING.items()}

# remove special tokens in the output
def remove_special_tokens(text):
    text = text.replace('<pad>', '')
    text = text.replace('<s>', '')
    text = text.replace('</s>', '')
    text = text.replace('<unk>', '')
    text = text.replace('<extra_id_0>', '')
    text = text.strip()
    return text

def convert_dataset_into_prompt_ethnic_domain_knowledge(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为民族领域知识任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            # 处理选项，添加A、B、C、D等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                options_text += f"{option_label}. {option} "
            
            # 获取题型
            question_type = example.get('metadata', {}).get('type', 'Single Choice')
            
            if prompt_lang == 'en':
                if question_type == 'Multiple Choice':
                    prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} ethnic domain knowledge multiple-choice question by selecting the correct options.\n"
                    prompt_prefix += f"Question: {example['question']}\n"
                    prompt_prefix += f"Options: {options_text}\n"
                    prompt_prefix += f"Answer (just provide all correct option letters, e.g. A, BC, ABC, etc.): {example['answer']}\n\n"
                else:  # 单选题
                    prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} ethnic domain knowledge single-choice question by selecting the correct option.\n"
                    prompt_prefix += f"Question: {example['question']}\n"
                    prompt_prefix += f"Options: {options_text}\n"
                    prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                if question_type == 'Multiple Choice':
                    prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}民族领域知识多选题，选择所有正确的选项。\n"
                    prompt_prefix += f"问题：{example['question']}\n"
                    prompt_prefix += f"选项：{options_text}\n"
                    prompt_prefix += f"答案（只需提供所有正确选项字母，如A、BC、ABC等，不需要提供额外的解释）：{example['answer']}\n\n"
                else:  # 单选题
                    prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}民族领域知识单选题，选择正确的选项。\n"
                    prompt_prefix += f"问题：{example['question']}\n"
                    prompt_prefix += f"选项：{options_text}\n"
                    prompt_prefix += f"答案（只需提供选项字母，如A、B、C等，不需要提供额外的解释）：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 获取题型
        question_type = item.get('metadata', {}).get('type', '单选题')
            
        # 处理选项，添加A、B、C、D等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            options_text += f"{option_label}. {option} "
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            if question_type == 'Multiple Choice':
                prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} ethnic domain knowledge multiple-choice question by selecting the correct options.\n"
                prompt += f"Question: {item['question']}\n"
                prompt += f"Options: {options_text}\n"
                prompt += f"Answer (provide all correct option letters, e.g. A, BC, ABC, etc.): "
            else:  # 单选题
                prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} ethnic domain knowledge single-choice question by selecting the correct option.\n"
                prompt += f"Question: {item['question']}\n"
                prompt += f"Options: {options_text}\n"
                prompt += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): "
        elif prompt_lang == 'zh':
            if question_type == 'Multiple Choice':
                prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}民族领域知识多选题，选择所有正确的选项。\n"
                prompt += f"问题：{item['question']}\n"
                prompt += f"选项：{options_text}\n"
                prompt += f"答案（只需提供所有正确选项字母，如A、BC、ABC等，不需要提供额外的解释）："
            else:  # 单选题
                prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}民族领域知识单选题，选择正确的选项。\n"
                prompt += f"问题：{item['question']}\n"
                prompt += f"选项：{options_text}\n"
                prompt += f"答案（只需提供选项字母，如A、B、C等，不需要提供额外的解释）："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_translation(input_dataset, exemplar_dataset=None, src_lang='zh', tgt_lang='ti',
                                            num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为翻译任务的提示格式。

    Args:
        input_dataset: 要转换的数据集
        exemplar_dataset: （可选）包含要在提示中加入的示例的数据集
        src_lang: 源语言代码（默认：'zh'）
        tgt_lang: 目标语言代码（默认：'ti'）
        num_exemplar: 要在提示中包含的示例数量
        prompt_lang: 提示语言（默认：'zh'）

    Returns:
        带有提示的已转换示例列表
    """

    converted_dataset = []
    
    # 添加不同语言的翻译提示
    language_prompts_zh = {
        'bo': "",
        'mn': "请使用传统蒙古文(竖写蒙古文)进行翻译。",
        'ug': ""
    }
    
    language_prompts_en = {
        'bo': "",
        'mn': "Please translate into traditional Mongolian script (vertical Mongolian script).",
        'ug': ""
    }

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            if prompt_lang == 'en':
                prompt_prefix += f"Please translate the following {abbr_to_lang_en[src_lang]} text into {abbr_to_lang_en[tgt_lang]}. {language_prompts_en.get(tgt_lang, '')}\n"
                prompt_prefix += f"{abbr_to_lang_en[src_lang]}: {exemplar_dataset[i][src_lang]}\n"
                prompt_prefix += f"{abbr_to_lang_en[tgt_lang]}: {exemplar_dataset[i][tgt_lang]}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请将下面的{abbr_to_lang_zh[src_lang]}文本翻译成{abbr_to_lang_zh[tgt_lang]}。{language_prompts_zh.get(tgt_lang, '')}\n"
                prompt_prefix += f"{abbr_to_lang_zh[src_lang]}：{exemplar_dataset[i][src_lang]}\n"
                prompt_prefix += f"{abbr_to_lang_zh[tgt_lang]}：{exemplar_dataset[i][tgt_lang]}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please translate the following {abbr_to_lang_en[src_lang]} text into {abbr_to_lang_en[tgt_lang]}. {language_prompts_en.get(tgt_lang, '')}\n"
            prompt += f"{abbr_to_lang_en[src_lang]}: {item[src_lang]}\n"
            prompt += f"{abbr_to_lang_en[tgt_lang]}: "
        elif prompt_lang == 'zh':
            prompt += f"请将下面的{abbr_to_lang_zh[src_lang]}文本翻译成{abbr_to_lang_zh[tgt_lang]}。{language_prompts_zh.get(tgt_lang, '')}\n"
            prompt += f"{abbr_to_lang_zh[src_lang]}：{item[src_lang]}\n"
            prompt += f"{abbr_to_lang_zh[tgt_lang]}："

        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item[tgt_lang]
        })

    return converted_dataset

def convert_dataset_into_prompt_coref_resolution(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为代词指代消解任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            # 处理选项，添加A、B等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                options_text += f"{option_label}. {option} "
            
            # 从answer字段获取示例答案（如果存在）
            example_answer = example.get('answer', None)
            if example_answer is None:
                # 如果answer字段不存在，则从label生成答案
                if isinstance(example['label'], bool):
                    label_str = str(example['label']).lower()
                    correct_index = example['option'].index(label_str)
                    example_answer = chr(65 + correct_index)
                else:
                    example_answer = example['label']
            
            if prompt_lang == 'en':
                prompt_prefix += f"Please determine if the two spans in the following {abbr_to_lang_en[eval_lang]} text refer to the same entity.\n"
                prompt_prefix += f"Text: {example['text']}\n"
                prompt_prefix += f"Span 1: {example['span1_text']}\n"
                prompt_prefix += f"Span 2: {example['span2_text']}\n"
                prompt_prefix += f"Options: {options_text}\n"
                prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B): {example_answer}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请判断以下{abbr_to_lang_zh[eval_lang]}文本中的两个片段是否指代同一个实体。\n"
                prompt_prefix += f"文本：{example['text']}\n"
                prompt_prefix += f"片段1：{example['span1_text']}\n"
                prompt_prefix += f"片段2：{example['span2_text']}\n"
                prompt_prefix += f"选项：{options_text}\n"
                prompt_prefix += f"答案（只需提供选项字母，如A、B）：{example_answer}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 处理选项，添加A、B等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            options_text += f"{option_label}. {option} "
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please determine if the two spans in the following {abbr_to_lang_en.get(lang, lang)} text refer to the same entity.\n"
            prompt += f"Text: {item['text']}\n"
            prompt += f"Span 1: {item['span1_text']}\n"
            prompt += f"Span 2: {item['span2_text']}\n"
            prompt += f"Options: {options_text}\n"
            prompt += f"Answer (just provide the letter of the option, e.g. A, B): "
        elif prompt_lang == 'zh':
            prompt += f"请判断以下{abbr_to_lang_zh.get(lang, lang)}文本中的两个片段是否指代同一个实体。\n"
            prompt += f"文本：{item['text']}\n"
            prompt += f"片段1：{item['span1_text']}\n"
            prompt += f"片段2：{item['span2_text']}\n"
            prompt += f"选项：{options_text}\n"
            prompt += f"答案（只需提供选项字母，如A、B）："
        
        # 首先尝试从answer字段获取gold
        if 'answer' in item:
            gold = item['answer']
        else:
            # 如果没有answer字段，则从label生成答案
            if isinstance(item['label'], bool):
                # 将布尔值转换为字符串
                label_str = str(item['label']).lower()
                # 获取该字符串在选项中的索引
                correct_index = item['option'].index(label_str)
                # 转换为对应的字母
                gold = chr(65 + correct_index)
            else:
                # 如果标签已经是字母形式，直接使用
                gold = item['label']
            
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": gold
        })
    
    return converted_dataset

def convert_dataset_into_prompt_entailment(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为蕴含推理任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []
    
    # 为中文和英文准备标签映射
    label_map_zh = {
        'contradiction': '矛盾',
        'entailment': '蕴含',
        'neutral': '中立'
    }
    
    label_map_en = {
        'contradiction': 'contradiction',
        'entailment': 'entailment',
        'neutral': 'neutral'
    }

    # 清理数据中键的空格
    def clean_item_keys(item):
        cleaned_item = {}
        for key, value in item.items():
            cleaned_key = key.strip()
            if isinstance(value, str):
                cleaned_item[cleaned_key] = value.strip()
            else:
                cleaned_item[cleaned_key] = value
        return cleaned_item
    
    cleaned_input_dataset = [clean_item_keys(item) for item in input_dataset]
    cleaned_exemplar_dataset = [clean_item_keys(item) for item in exemplar_dataset] if exemplar_dataset is not None else None
    
    prompt_prefix = ""
    if cleaned_exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(cleaned_exemplar_dataset))):
            example = cleaned_exemplar_dataset[i]
            
            # 处理选项，添加A、B、C等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                option_text = label_map_zh[option] if prompt_lang == 'zh' else label_map_en[option]
                options_text += f"{option_label}. {option_text} "
            
            # 获取示例答案（优先使用answer字段）
            example_answer = example.get('answer', None)
            if example_answer is None:
                # 如果answer字段不存在，则从label生成答案
                correct_index = example['option'].index(example['label'])
                example_answer = chr(65 + correct_index)
            
            if prompt_lang == 'en':
                prompt_prefix += f"Please determine the relationship between the following two {abbr_to_lang_en[eval_lang]} sentences.\n"
                prompt_prefix += f"Sentence 1: {example['sentence1']}\n"
                prompt_prefix += f"Sentence 2: {example['sentence2']}\n"
                prompt_prefix += f"Options: {options_text}\n"
                prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B, C): {example_answer}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请判断以下两个{abbr_to_lang_zh[eval_lang]}句子之间的关系。\n"
                prompt_prefix += f"句子1：{example['sentence1']}\n"
                prompt_prefix += f"句子2：{example['sentence2']}\n"
                prompt_prefix += f"选项：{options_text}\n"
                prompt_prefix += f"答案（只需提供选项字母，如A、B、C）：{example_answer}\n\n"

    for i in range(len(cleaned_input_dataset)):
        item = cleaned_input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 处理选项，添加A、B、C等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            option_text = label_map_zh[option] if prompt_lang == 'zh' else label_map_en[option]
            options_text += f"{option_label}. {option_text} "
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please determine the relationship between the following two {abbr_to_lang_en.get(lang, lang)} sentences.\n"
            prompt += f"Sentence 1: {item['sentence1']}\n"
            prompt += f"Sentence 2: {item['sentence2']}\n"
            prompt += f"Options: {options_text}\n"
            prompt += f"Answer (just provide the letter of the option, e.g. A, B, C): "
        elif prompt_lang == 'zh':
            prompt += f"请判断以下两个{abbr_to_lang_zh.get(lang, lang)}句子之间的关系。\n"
            prompt += f"句子1：{item['sentence1']}\n"
            prompt += f"句子2：{item['sentence2']}\n"
            prompt += f"选项：{options_text}\n"
            prompt += f"答案（只需提供选项字母，如A、B、C）："
        
        # 获取gold答案（优先使用answer字段）
        if 'answer' in item:
            gold = item['answer']
        else:
            # 如果没有answer字段，则从label生成答案
            correct_index = item['option'].index(item['label'])
            gold = chr(65 + correct_index)
            
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": gold
        })
    
    return converted_dataset

def convert_dataset_into_prompt_text_classification(input_dataset, exemplar_dataset=None, eval_lang="bo", num_exemplar=3, max_passage_len=512, prompt_lang='zh'):
    """
    将数据集转换为文本分类任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        max_passage_len: 最大文本长度
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []
    
    # 定义类别
    categories_zh = ["体育", "健康", "地理", "娱乐", "政治", "旅游", "科技"]
    categories_en = ["Sports", "Health", "Geography", "Entertainment", "Politics", "Travel", "Technology"]
    
    # 构建类别字符串
    if prompt_lang == 'en':
        concated_categories = ', '.join(categories_en)
    elif prompt_lang == 'zh':
        concated_categories = '、'.join(categories_zh)
    
    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            if prompt_lang == 'en':
                prompt_prefix += f"Please classify the following {abbr_to_lang_en[eval_lang]} text.\n"
                prompt_prefix += f"Text: {example['text'][:max_passage_len]}\n"
                prompt_prefix += f"Candidate categories: {concated_categories}\n"
                prompt_prefix += f"Category: {categories_en[categories_zh.index(example['label'])]} \n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请判断以下{abbr_to_lang_zh[eval_lang]}文本的类别：\n"
                prompt_prefix += f"文本：{example['text'][:max_passage_len]}\n"
                prompt_prefix += f"候选类别：{concated_categories}\n"
                prompt_prefix += f"类别：{example['label']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please classify the following {abbr_to_lang_en.get(lang, lang)} text.\n"
            prompt += f"Text: {item['text'][:max_passage_len]}\n"
            prompt += f"Candidate categories: {concated_categories}\n"
            prompt += f"Category: "
        elif prompt_lang == 'zh':
            prompt += f"请判断以下{abbr_to_lang_zh.get(lang, lang)}文本的类别：\n"
            prompt += f"文本：{item['text'][:max_passage_len]}\n"
            prompt += f"候选类别：{concated_categories}\n"
            prompt += f"类别："
        
        # 设置gold答案
        if prompt_lang == 'en' and item['label'] in categories_zh:
            # 如果提示语言是英文，但标签是中文，则转换为英文标签
            gold = categories_en[categories_zh.index(item['label'])]
        else:
            gold = item['label']
            
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": gold
        })
    
    return converted_dataset

def convert_dataset_into_prompt_reading_comprehension(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为阅读理解任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            if prompt_lang == 'en':
                prompt_prefix += f"Based on the following {abbr_to_lang_en[eval_lang]} article, please answer the question in {abbr_to_lang_en[eval_lang]} language.\n"
                prompt_prefix += f"Article: {example['context_text']}\n"
                prompt_prefix += f"Question: {example['query_text']}\n"
                prompt_prefix += f"Answer: {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请根据以下{abbr_to_lang_zh[eval_lang]}文章用{abbr_to_lang_zh[eval_lang]}语回答问题。\n"
                prompt_prefix += f"文章：{example['context_text']}\n"
                prompt_prefix += f"问题：{example['query_text']}\n"
                prompt_prefix += f"答案：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['query_id'] if 'query_id' in item else item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Based on the following {abbr_to_lang_en.get(lang, lang)} article, please answer the question in {abbr_to_lang_en.get(lang, lang)} language.\n"
            prompt += f"Article: {item['context_text']}\n"
            prompt += f"Question: {item['query_text']}\n"
            prompt += f"Answer: "
        elif prompt_lang == 'zh':
            prompt += f"请根据以下{abbr_to_lang_zh.get(lang, lang)}文章用{abbr_to_lang_zh.get(lang, lang)}语回答问题。\n"
            prompt += f"文章：{item['context_text']}\n"
            prompt += f"问题：{item['query_text']}\n"
            prompt += f"答案："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_safety(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为安全任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            # 处理选项，添加A、B、C、D等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                options_text += f"{option_label}. {option} "
            
            if prompt_lang == 'en':
                prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} question by selecting the correct option.\n"
                prompt_prefix += f"Question: {example['question']}\n"
                prompt_prefix += f"Options: {options_text}\n"
                prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}问题，选择正确的选项。\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"选项：{options_text}\n"
                prompt_prefix += f"答案（只需提供选项字母，如A、B、C等）：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 处理选项，添加A、B、C、D等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            options_text += f"{option_label}. {option} "
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} question by selecting the correct option.\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Options: {options_text}\n"
            prompt += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): "
        elif prompt_lang == 'zh':
            prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}问题，选择正确的选项。\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"选项：{options_text}\n"
            prompt += f"答案（只需提供选项字母，如A、B、C等）："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_professional_skills(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为专业能力任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            # 处理选项，添加A、B、C、D等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                options_text += f"{option_label}. {option} "
            
            if prompt_lang == 'en':
                prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} professional knowledge question by selecting the correct option.\n"
                prompt_prefix += f"Question: {example['question']}\n"
                prompt_prefix += f"Options: {options_text}\n"
                prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}专业知识问题，选择正确的选项。\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"选项：{options_text}\n"
                prompt_prefix += f"答案（只需提供选项字母，如A、B、C等）：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 处理选项，添加A、B、C、D等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            options_text += f"{option_label}. {option} "
            
        # 获取专业领域信息（如果存在）
        domain_info = ""
        if 'task' in item and 'domain' in item['task']:
            domain = item['task']['domain']
            sub_domain = item['task'].get('sub_domain', '')
            
            if prompt_lang == 'en':
                domain_info = f" (Domain: {domain}, Sub-domain: {sub_domain})" if sub_domain else f" (Domain: {domain})"
            elif prompt_lang == 'zh':
                domain_info = f"（领域：{domain}，子领域：{sub_domain}）" if sub_domain else f"（领域：{domain}）"
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} professional knowledge question{domain_info} by selecting the correct option.\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Options: {options_text}\n"
            prompt += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): "
        elif prompt_lang == 'zh':
            prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}专业知识问题{domain_info}，选择正确的选项。\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"选项：{options_text}\n"
            prompt += f"答案（只需提供选项字母，如A、B、C等）："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_ethnic_vocabulary(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为民族词汇任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            # 处理选项，添加A、B、C、D等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                options_text += f"{option_label}. {option} "
            
            if prompt_lang == 'en':
                prompt_prefix += f"Please select the Chinese term that corresponds to the {abbr_to_lang_en[eval_lang]} ethnic vocabulary term in the question.\n"
                prompt_prefix += f"Question: {example['question']}\n"
                prompt_prefix += f"Options: {options_text}\n"
                prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请选择与问题中的词汇术语对应的{abbr_to_lang_zh[eval_lang]}术语。\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"选项：{options_text}\n"
                prompt_prefix += f"答案（只需提供选项字母，如A、B、C等）：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 处理选项，添加A、B、C、D等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            options_text += f"{option_label}. {option} "
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please select the Chinese term that corresponds to the {abbr_to_lang_en.get(lang, lang)} ethnic vocabulary term in the question.\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Options: {options_text}\n"
            prompt += f"Answer (just provide the letter of the option, e.g. A, B, C, etc.): "
        elif prompt_lang == 'zh':
            prompt += f"请选择与问题中的词汇术语对应的{abbr_to_lang_zh.get(lang, lang)}术语。\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"选项：{options_text}\n"
            prompt += f"答案（只需提供选项字母，如A、B、C等）："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_math_reasoning(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为数学推理任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            if prompt_lang == 'en':
                prompt_prefix += f"Solve the following {abbr_to_lang_en[eval_lang]} math problem and provide only the final numerical answer.\n"
                prompt_prefix += f"Problem: {example['question']}\n"
                prompt_prefix += f"Answer: {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"解决以下{abbr_to_lang_zh[eval_lang]}数学问题，并只提供最终的数字答案。\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"答案：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Solve the following {abbr_to_lang_en.get(lang, lang)} math problem and provide only the final numerical answer.\n"
            prompt += f"Problem: {item['question']}\n"
            prompt += f"Answer: "
        elif prompt_lang == 'zh':
            prompt += f"解决以下{abbr_to_lang_zh.get(lang, lang)}数学问题，并只提供最终的数字答案。\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"答案："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_traditional_culture(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为传统文化任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []
    
    # 添加不同语言的生成提示
    language_prompts_zh = {
        'bo': "",
        'mn': "请使用传统蒙古语回答。",
        'ug': ""
    }
    
    language_prompts_en = {
        'bo': "",
        'mn': "Please answer in traditional Mongolian script.",
        'ug': ""
    }

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            if prompt_lang == 'en':
                prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} question. {language_prompts_en.get(eval_lang, '')}\n"
                prompt_prefix += f"Question: {example['question']}\n"
                prompt_prefix += f"Answer: {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}问题。{language_prompts_zh.get(eval_lang, '')}\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"答案：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} question. {language_prompts_en.get(lang, '')}\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Answer: "
        elif prompt_lang == 'zh':
            prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}问题。{language_prompts_zh.get(lang, '')}\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"答案："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_text_generation(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为文本生成任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []
    
    # 添加不同语言的生成提示
    language_prompts_zh = {
        'bo': "",
        'mn': "请使用传统蒙古语。",
        'ug': ""
    }
    
    language_prompts_en = {
        'bo': "",
        'mn': "Please answer in traditional Mongolian script.",
        'ug': ""
    }

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            if prompt_lang == 'en':
                prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} question. {language_prompts_en.get(eval_lang, '')}\n"
                prompt_prefix += f"Question: {example['question']}\n"
                prompt_prefix += f"Answer: {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}问题。{language_prompts_zh.get(eval_lang, '')}\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"回答：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} question. {language_prompts_en.get(lang, '')}\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Answer: "
        elif prompt_lang == 'zh':
            prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}问题。{language_prompts_zh.get(lang, '')}\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"回答："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_ethnic_language_understanding(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):
    """
    将数据集转换为民族语言理解任务的提示格式

    参数:
        input_dataset: 要转换的数据集
        exemplar_dataset: 可选的包含示例的数据集
        eval_lang: 评估语言代码 (默认: 'bo')
        num_exemplar: 要包含在提示中的示例数量
        prompt_lang: 提示语言 (默认: 'zh')

    返回:
        转换后的带有提示的示例列表
    """
    converted_dataset = []

    prompt_prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            example = exemplar_dataset[i]
            # 处理选项，添加A、B、C等编号
            options_text = ""
            for j, option in enumerate(example['option']):
                option_label = chr(65 + j)  # 65是ASCII码中'A'的值
                options_text += f"{option_label}. {option} "
            
            if prompt_lang == 'en':
                prompt_prefix += f"Please answer the following {abbr_to_lang_en[eval_lang]} ethnic language understanding question by selecting the correct option.\n"
                prompt_prefix += f"Question: {example['question']}\n"
                prompt_prefix += f"Options: {options_text}\n"
                prompt_prefix += f"Answer (just provide the letter of the option, e.g. A, B, C): {example['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请回答以下{abbr_to_lang_zh[eval_lang]}民族语言理解问题，选择正确的选项。\n"
                prompt_prefix += f"问题：{example['question']}\n"
                prompt_prefix += f"选项：{options_text}\n"
                prompt_prefix += f"答案（只需提供选项字母，如A、B、C）：{example['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        # 根据metadata中的language字段获取语言
        if 'metadata' in item and 'language' in item['metadata']:
            lang = item['metadata']['language']
        else:
            lang = eval_lang
            
        # 处理选项，添加A、B、C等编号
        options_text = ""
        for j, option in enumerate(item['option']):
            option_label = chr(65 + j)  # 65是ASCII码中'A'的值
            options_text += f"{option_label}. {option} "
            
        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please answer the following {abbr_to_lang_en.get(lang, lang)} ethnic language understanding question by selecting the correct option.\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Options: {options_text}\n"
            prompt += f"Answer (just provide the letter of the option, e.g. A, B, C): "
        elif prompt_lang == 'zh':
            prompt += f"请回答以下{abbr_to_lang_zh.get(lang, lang)}民族语言理解问题，选择正确的选项。\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"选项：{options_text}\n"
            prompt += f"答案（只需提供选项字母，如A、B、C）："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def process_task(model, args):
    """处理单个任务 - 修正版"""
    print(f"===== 处理任务: {args.task} 语言: {args.eval_lang} =====")
    
    # 获取输出目录
    output_folder = '/'.join(args.output_file.split('/')[:-1])
    if len(output_folder) == 0:
        output_folder = '.'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 创建错误日志和ID文件路径
    base_name = os.path.splitext(args.output_file)[0]
    error_log_file = f"{base_name}_errors.log"
    error_id_file = f"{base_name}_error_ids.json"
    checkpoint_file = f"{base_name}_checkpoint.json"  # 新增检查点文件路径
    
    # 已处理的ID列表
    processed_ids = set()
    existing_results = []
    
    # 检查是否存在输出文件或检查点文件，用于断点续传
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for item in existing_results:
                    processed_ids.add(item['id'])
            print(f"找到现有结果文件，已处理 {len(processed_ids)} 个样本")
        except Exception as e:
            print(f"读取现有结果文件时出错: {str(e)}")
            # 文件可能损坏，重命名并重新开始
            backup_file = f"{args.output_file}.bak.{int(time.time())}"
            os.rename(args.output_file, backup_file)
            print(f"已将可能损坏的结果文件备份为 {backup_file}")
            existing_results = []
    elif os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                processed_ids = set(checkpoint_data['processed_ids'])
                if 'results' in checkpoint_data:
                    existing_results = checkpoint_data['results']
            print(f"找到检查点文件，已处理 {len(processed_ids)} 个样本")
        except Exception as e:
            print(f"读取检查点文件时出错: {str(e)}")
            # 检查点文件可能损坏，忽略并重新开始
            existing_results = []
            processed_ids = set()

    # 加载数据
    try:
        input_dataset = json.load(open(args.input_file, 'r', encoding='utf-8'))
        if args.exemplar_file and args.exemplar_file.lower() != "null":
            exemplar_dataset = json.load(open(args.exemplar_file, 'r', encoding='utf-8'))
        else:
            exemplar_dataset = None
    except Exception as e:
        log_error(error_log_file, error_id_file, "加载数据集时出错", e)
        print(f"加载数据集时出错: {str(e)}")
        return

    # 转换数据集
    if args.prompt_lang not in ['en', 'zh']:
        log_error(error_log_file, error_id_file, "提示语言错误", ValueError("提示语言必须是'en'或'zh'"))
        print("错误: 提示语言必须是'en'或'zh'")
        return

    # 设置默认值
    num_exemplar = 3  # 默认值
    if hasattr(args, 'num_exemplar'):
        num_exemplar = args.num_exemplar
        
    max_passage_len = 512  # 默认值
    if hasattr(args, 'max_passage_len'):
        max_passage_len = args.max_passage_len

    # ⭐ --- 主要修改点在这里 --- ⭐
    # 根据任务类型转换数据集
    try:
        # 获取映射后的任务名称
        mapped_task = TASK_MAPPING.get(args.task, args.task)

        # 使用更清晰的 if/elif 结构
        if mapped_task == 'translation':
            converted_dataset = convert_dataset_into_prompt_translation(
                input_dataset, exemplar_dataset, args.src_lang, args.tgt_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'coref_resolution':
            converted_dataset = convert_dataset_into_prompt_coref_resolution(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'entailment':
            converted_dataset = convert_dataset_into_prompt_entailment(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'text_classification':
            converted_dataset = convert_dataset_into_prompt_text_classification(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, max_passage_len, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'reading_comprehension':
            converted_dataset = convert_dataset_into_prompt_reading_comprehension(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )

        # ⭐ 修正后的安全任务处理逻辑 ⭐
        elif mapped_task == 'safety':
            converted_dataset = convert_dataset_into_prompt_safety(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
            
        elif mapped_task == 'professional_skills':
            converted_dataset = convert_dataset_into_prompt_professional_skills(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'ethnic_vocabulary':
            converted_dataset = convert_dataset_into_prompt_ethnic_vocabulary(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'math_reasoning':
            converted_dataset = convert_dataset_into_prompt_math_reasoning(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'traditional_culture':
            converted_dataset = convert_dataset_into_prompt_traditional_culture(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'text_generation':
            converted_dataset = convert_dataset_into_prompt_text_generation(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'ethnic_language_understanding':
            converted_dataset = convert_dataset_into_prompt_ethnic_language_understanding(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        elif mapped_task == 'ethnic_domain_knowledge':
            converted_dataset = convert_dataset_into_prompt_ethnic_domain_knowledge(
                input_dataset, exemplar_dataset, args.eval_lang, num_exemplar, prompt_lang=args.prompt_lang
            )
        else:
            log_error(error_log_file, error_id_file, f"不支持的任务类型 {args.task} (映射为 {mapped_task})", ValueError(f"不支持的任务类型 {args.task}"))
            print(f"错误: 不支持的任务类型 {args.task} (映射为 {mapped_task})")
            return
    except Exception as e:
        log_error(error_log_file, error_id_file, "转换数据集时出错", e)
        print(f"转换数据集时出错: {str(e)}")
        traceback.print_exc()  # 打印详细错误信息
        return

    # 调试模式下限制示例数量
    if args.max_test_example_num > 0:
        converted_dataset = converted_dataset[:args.max_test_example_num]

    print("提示示例:", converted_dataset[0]['input'])

    # 筛选出未处理的样本
    filtered_dataset = []
    for item in converted_dataset:
        if item['id'] not in processed_ids:
            filtered_dataset.append(item)
    
    if len(filtered_dataset) == 0:
        print(f"所有 {len(converted_dataset)} 个样本已处理完成，无需继续")
        return
    
    print(f"总共 {len(converted_dataset)} 个样本，其中 {len(filtered_dataset)} 个尚未处理")
    
    # 将之前处理过的结果作为起点
    output_results = existing_results.copy()
    error_ids = []  # 存储出错的ID
    start_time = time.time()
    save_counter = 0
    last_save_time = start_time
    save_frequency = args.save_frequency if hasattr(args, 'save_frequency') else 5
    time_based_save = 300  # 每5分钟保存一次，不论处理了多少样本
    
    # 为VLLM设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 使用确定性生成
        max_tokens=args.max_new_tokens,
        stop=None  # 可以根据需要设置停止标记
    )
    
    for i in tqdm(range(0, len(filtered_dataset), args.batch_size)):
        batch = filtered_dataset[i:i + args.batch_size]
        batch_ids = [item['id'] for item in batch]  # 当前批次的ID列表
        input_text_batch = [item['input'] for item in batch]
        
        try:
            # 使用VLLM生成输出
            outputs = model.generate(input_text_batch, sampling_params)
            
            for j in range(len(batch)):
                try:
                    qid = batch[j]['id']
                    gold = batch[j]['gold']
                    
                    # 获取生成的输出文本
                    output_obj = outputs[j]
                    generated_text = output_obj.outputs[0].text.strip()
                    
                    # 截取输入提示后的部分作为输出
                    prompt_length = len(input_text_batch[j])
                    output = generated_text.strip().split('\n')[0]
                    
                    result_item = {
                        "id": qid,
                        "pred": output,
                        "gold": gold
                    }
                    
                    output_results.append(result_item)
                    processed_ids.add(qid)  # 标记为已处理

                    if args.print_inference_result:
                        print(qid)
                        print("pred:", output)
                        print("gold:", gold)
                except Exception as e:
                    # 处理单个样本的错误
                    qid = batch[j]['id']
                    error_ids.append(qid)
                    log_error(error_log_file, error_id_file, f"处理样本 {qid} 时出错", e, [qid])
                    print(f"处理样本 {qid} 时出错: {str(e)}")
        except Exception as e:
            # 处理整个批次的错误
            error_ids.extend(batch_ids)
            log_error(error_log_file, error_id_file, "处理批次时出错", e, batch_ids)
            print(f"处理批次时出错: {str(e)}")
            
            # 继续处理下一个批次，而不是退出
            continue
        
        # 定期保存
        save_counter += 1
        current_time = time.time()
        should_save = (save_counter >= save_frequency) or (current_time - last_save_time >= time_based_save)
        
        if should_save:
            save_results(output_results, args.output_file)
            # 同时更新检查点文件
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_ids': list(processed_ids),
                    'last_processed_index': i + len(batch),
                    'results': output_results
                }, f, indent=2)
            print(f"已更新检查点，当前进度: {len(processed_ids)}/{len(converted_dataset)}")
            save_counter = 0
            last_save_time = current_time
    
    # 最后保存结果
    if output_results:
        save_results(output_results, args.output_file)
        
        # 任务完成后，可以选择删除检查点文件
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                print(f"任务完成，检查点文件已删除")
            except:
                print(f"无法删除检查点文件，但这不影响结果")

    # 确保所有错误ID都已保存
    if error_ids:
        log_error(error_log_file, error_id_file, "汇总出错ID", Exception("处理完成，汇总所有出错ID"), error_ids)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总用时: {total_time:.2f}秒")
    print(f"已处理 {len(processed_ids)} 个样本并保存到 {args.output_file}")
    if error_ids:
        print(f"本次运行中有 {len(error_ids)} 个样本出错，ID已保存到 {error_id_file}")

def generate_task_list(base_path, model_name, prompt_lang='zh', langs=['bo', 'mn', 'ug']):
    """
    根据新的目录结构生成任务列表
    
    参数:
        base_path: 数据集基础路径
        model_name: 模型名称，用于输出路径
        prompt_lang: 提示语言
        langs: 评估语言列表
    
    返回:
        tasks: 任务配置列表
    """
    tasks = []
    output_base = f"{base_path}/output/{model_name}"
    
    # 定义任务类别和对应的目录
    task_categories = {
        'Foundation_Tasks': [
            'Coreference_Resolution',
            'General_Domain_Competence', 
            'Machine_Reading_Comprehension',
            'Math_Reasoning',
            'Natural_Language_Inference',
            'Text_Classification'
        ],
        'Chinese_Minority_Knowledge_Tasks': [
            'Minority_Culture_QA',
            'Minority_Domain_Competence',
            'Minority_Language_Expressions',
            'Minority_Language_Instruction_QA',
            'Minority_Language_Understanding',
            'Minority_Machine_Translation'
        ],
        'Safety_Alignment_Tasks': [
            'Commercial_Compliance_Check',
            'Discrimination_Detection',
            'Rights_Protection_Evaluation',
            'Service_Safety_Evaluation',
            'Value_Alignment_Assessment'
        ]
    }
    
    # 特殊处理翻译任务
    translation_configs = {
        'bo': [('zh', 'bo'), ('bo', 'zh')],
        'mn': [('zh', 'mn'), ('mn', 'zh')], 
        'ug': [('zh', 'ug'), ('ug', 'zh')]
    }
    
    # 为每个语言和任务生成配置
    for eval_lang in langs:
        for category, task_dirs in task_categories.items():
            for task_dir in task_dirs:
                input_file = f"{base_path}/{category}/{task_dir}/{eval_lang}.json"
                
                # 检查文件是否存在
                if not os.path.exists(input_file):
                    print(f"警告: 文件不存在 {input_file}")
                    continue
                
                # 获取映射的任务名称（仅用于任务处理逻辑）
                mapped_task = TASK_MAPPING.get(task_dir, task_dir)
                
                # 特殊处理翻译任务
                if task_dir == 'Minority_Machine_Translation':
                    for src_lang, tgt_lang in translation_configs[eval_lang]:
                        # 使用原始目录名作为输出路径
                        output_file = f"{output_base}/{task_dir}/{eval_lang}/{prompt_lang}-prompt_{src_lang}2{tgt_lang}_test.json"
                        
                        task_config = {
                            "task": task_dir,  # 使用原始目录名作为task参数
                            "eval_lang": eval_lang,
                            "prompt_lang": prompt_lang,
                            "input_file": input_file,
                            "output_file": output_file,
                            "exemplar_file": None,
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "num_exemplar": 3,
                            "max_new_tokens": 300
                        }
                        tasks.append(task_config)
                # 特殊处理安全任务 - 每个安全任务有独立的文件夹
                elif category == 'Safety_Alignment_Tasks':
                    # 使用原始目录名作为输出路径
                    output_file = f"{output_base}/{task_dir}/{eval_lang}/{prompt_lang}-prompt_test.json"
                    
                    task_config = {
                        "task": task_dir,  # 使用原始目录名作为task参数
                        "eval_lang": eval_lang,
                        "prompt_lang": prompt_lang,
                        "input_file": input_file,
                        "output_file": output_file,
                        "exemplar_file": None,
                        "num_exemplar": 3,
                        "max_new_tokens": 20
                    }
                    tasks.append(task_config)
                else:
                    # 其他任务的通用配置 - 使用原始目录名作为输出路径
                    output_file = f"{output_base}/{task_dir}/{eval_lang}/{prompt_lang}-prompt_test.json"
                    
                    # 根据任务类型设置不同的max_new_tokens
                    max_tokens_map = {
                        'General_Domain_Competence': 20,
                        'Minority_Language_Expressions': 20,
                        'Minority_Domain_Competence': 20,
                        'Minority_Language_Understanding': 20,
                        'Math_Reasoning': 200,
                        'Machine_Reading_Comprehension': 200,
                        'Minority_Culture_QA': 200,
                        'Minority_Language_Instruction_QA': 1000,
                        'Text_Classification': 100,
                        'Natural_Language_Inference': 50,
                        'Coreference_Resolution': 50
                    }
                    
                    task_config = {
                        "task": task_dir,  # 使用原始目录名作为task参数
                        "eval_lang": eval_lang,
                        "prompt_lang": prompt_lang,
                        "input_file": input_file,
                        "output_file": output_file,
                        "exemplar_file": None,
                        "num_exemplar": 3,
                        "max_new_tokens": max_tokens_map.get(task_dir, 512)  # 注意这里也改为使用task_dir
                    }
                    tasks.append(task_config)
    
    return tasks

def main():
    parser = argparse.ArgumentParser(description="批量处理多个任务的推理脚本")
    
    # 模型参数
    parser.add_argument('--model_type', type=str, required=True, help="模型类型: qwen, aya, llama, mistral, gemma")
    parser.add_argument('--model_path', type=str, required=True, help="模型路径")
    
    # 数据集路径
    parser.add_argument('--dataset_path', type=str, required=True, help="数据集根目录路径")
    
    # 可选：直接指定任务列表文件
    parser.add_argument('--task_list', type=str, help="任务列表JSON文件（可选，如果不提供将自动生成）")
    
    # 共享参数
    parser.add_argument('--batch_size', type=int, default=1, help="批处理大小")
    parser.add_argument('--save_frequency', type=int, default=5, help="每处理多少批次保存一次结果")
    parser.add_argument('--max_test_example_num', type=int, default=-1, help="测试样本数限制 (-1表示使用全部)")
    parser.add_argument('--print_inference_result', action='store_true', help="打印推理结果")
    parser.add_argument('--num_exemplar', type=int, default=3, help="示例数量")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="GPU内存使用率")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="张量并行大小")
    parser.add_argument('--prompt_lang', type=str, default='zh', choices=['zh', 'en'], help="提示语言")
    parser.add_argument('--langs', nargs='+', default=['bo', 'mn', 'ug'], help="评估语言列表")
    
    args = parser.parse_args()
    
    # 注册信号处理器，捕获Ctrl+C等中断信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 生成或加载任务列表
    if args.task_list:
        # 从文件加载任务列表
        try:
            with open(args.task_list, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
        except Exception as e:
            print(f"加载任务列表时出错: {str(e)}")
            return
    else:
        # 自动生成任务列表
        model_name = os.path.basename(args.model_path)
        tasks = generate_task_list(args.dataset_path, model_name, args.prompt_lang, args.langs)
        
        # 保存生成的任务列表
        task_list_file = f"tasks_{model_name}_{args.prompt_lang}.json"
        try:
            with open(task_list_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)
            print(f"已生成任务列表文件: {task_list_file}")
        except Exception as e:
            print(f"保存任务列表时出错: {str(e)}")
    
    # 创建全局错误日志
    error_log_file = f"global_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 使用VLLM加载模型
    print(f"正在加载模型 {args.model_path}...")
    try:
        # 初始化VLLM模型
        model = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True
        )
        print("模型加载完成")
    except Exception as e:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"加载模型时出错: {str(e)}\n")
            f.write(f"堆栈跟踪:\n{traceback.format_exc()}\n\n")
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 遍历处理所有任务
    for task_config in tasks:
        # 确保任务配置中有num_exemplar参数
        if 'num_exemplar' not in task_config:
            task_config['num_exemplar'] = args.num_exemplar
        
        # 确保任务配置中有max_new_tokens参数
        if 'max_new_tokens' not in task_config:
            task_config['max_new_tokens'] = 512  # 设置默认值
            
        task_args = argparse.Namespace(
            **task_config,
            **{
                'model_type': args.model_type,
                'model_path': args.model_path,
                'batch_size': args.batch_size,
                'save_frequency': args.save_frequency,
                'max_test_example_num': args.max_test_example_num,
                'print_inference_result': args.print_inference_result,
                'gpu_memory_utilization': args.gpu_memory_utilization,
                'tensor_parallel_size': args.tensor_parallel_size
            }
        )
        
        try:
            process_task(model, task_args)
        except Exception as e:
            task_name = f"{task_args.task}_{task_args.eval_lang}"
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"处理任务 {task_name} 时出错: {str(e)}\n")
                f.write(f"堆栈跟踪:\n{traceback.format_exc()}\n\n")
            print(f"处理任务 {task_name} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    main()