import os
import json
import re
from collections import defaultdict
import argparse

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
    
    # Safety_Alignment_Tasks - 每个安全任务都有独立的映射
    'Commercial_Compliance_Check': 'commercial_compliance',
    'Discrimination_Detection': 'discrimination_detection',
    'Rights_Protection_Evaluation': 'rights_protection',
    'Service_Safety_Evaluation': 'service_safety',
    'Value_Alignment_Assessment': 'value_alignment',
}

# 类别定义
categories_zh = ["体育", "健康", "地理", "娱乐", "政治", "旅游", "科技"]

def extract_answer_for_choice_question(item):
    """
    提取选择题的答案（单选和多选，A/B/C/D，支持大小写和多种格式）
    """
    pred = item.get('pred', '')
    
    # 首先尝试从pred字段提取
    result = extract_choice_from_text(pred)
    if result:
        return result, True
    
    # 如果pred字段没有匹配到，检查api_response
    api_response = item.get('api_response', {})
    choices = api_response.get('choices', [{}])
    if choices and len(choices) > 0:
        message = choices[0].get('message', {})
        content = message.get('content', '')
        
        if content:
            result = extract_choice_from_text(content)
            if result:
                return result, True
    
    # 没有找到任何匹配项
    return "none", False

def extract_choice_from_text(text):
    """
    从文本中提取选择题答案（支持单选和多选）
    
    返回:
        标准化的答案字符串，如 "A", "AB", "ACD" 等
    """
    if not text:
        return None
    
    # 多选题匹配模式
    multi_choice_patterns = [
        # 空格分隔: "A B C", "a b c"
        r'(?:答案[是：]?\s*)?([A-Da-d](?:\s+[A-Da-d])+)',
        # 逗号分隔: "A,B,C", "A，B，C"
        r'(?:答案[是：]?\s*)?([A-Da-d](?:[,，]\s*[A-Da-d])+)',
        # 顿号分隔: "A、B、C"
        r'(?:答案[是：]?\s*)?([A-Da-d](?:、\s*[A-Da-d])+)',
        # 连续字母: "ABC", "abc"
        r'(?:答案[是：]?\s*)?([A-Da-d]{2,4})',
        # 和/与连接: "A和B和C", "A与B与C"
        r'(?:答案[是：]?\s*)?([A-Da-d](?:[和与]\s*[A-Da-d])+)',
        # 中文表达: "选择A和B", "选项A、B、C"
        r'选择?\s*([A-Da-d](?:[、，,和与]\s*[A-Da-d])+)',
        r'选项\s*([A-Da-d](?:[、，,和与]\s*[A-Da-d])+)',
    ]
    
    # 单选题匹配模式
    single_choice_patterns = [
        r'\b([A-Da-d])\b',                    # 独立的字母（大小写）
        r'答案[是：]?\s*([A-Da-d])',          # 答案是A, 答案：A, 答案a
        r'选择\s*([A-Da-d])',                 # 选择A, 选择a
        r'选项\s*([A-Da-d])',                 # 选项A, 选项a
        r'([A-Da-d])\s*[是为]正确',           # A是正确, a是正确
        r'正确答案[是：]?\s*([A-Da-d])',      # 正确答案是A, 正确答案是a
        r'我选择\s*([A-Da-d])',               # 我选择A, 我选择a
        r'应该选\s*([A-Da-d])',               # 应该选A, 应该选a
        r'答案应该是\s*([A-Da-d])',           # 答案应该是A, 答案应该是a
        r'([A-Da-d])\s*选项',                 # A选项, a选项
        r'选\s*([A-Da-d])',                   # 选A, 选a
    ]
    
    # 首先尝试多选题匹配
    for pattern in multi_choice_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 取最后一个匹配项
            choice_text = matches[-1]
            choices = extract_letters_from_choice_text(choice_text)
            if len(choices) > 1:  # 确保是多选
                return normalize_choice_answer(choices)
    
    # 然后尝试单选题匹配
    for pattern in single_choice_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 取最后一个匹配项
            choice = matches[-1].upper()
            return choice
    
    # 最后尝试提取所有可能的字母并判断
    all_letters = re.findall(r'[A-Da-d]', text)
    if all_letters:
        # 转换为大写并去重
        unique_letters = list(dict.fromkeys([letter.upper() for letter in all_letters]))
        
        # 如果有多个不同的字母，可能是多选题
        if len(unique_letters) > 1 and len(unique_letters) <= 4:
            return normalize_choice_answer(unique_letters)
        elif len(unique_letters) == 1:
            return unique_letters[0]
    
    return None

def extract_letters_from_choice_text(choice_text):
    """
    从选择文本中提取所有字母
    
    参数:
        choice_text: 包含选择的文本，如 "A B C", "A,B,C", "ABC" 等
    
    返回:
        字母列表
    """
    # 提取所有A-D字母（大小写）
    letters = re.findall(r'[A-Da-d]', choice_text)
    
    # 转换为大写并去重，保持顺序
    unique_letters = []
    seen = set()
    for letter in letters:
        upper_letter = letter.upper()
        if upper_letter not in seen:
            unique_letters.append(upper_letter)
            seen.add(upper_letter)
    
    return unique_letters

def normalize_choice_answer(letters):
    """
    标准化选择题答案
    
    参数:
        letters: 字母列表，如 ['A', 'C', 'B']
    
    返回:
        标准化的答案字符串，如 "ABC"
    """
    if not letters:
        return ""
    
    # 去重并排序
    unique_letters = sorted(set(letter.upper() for letter in letters))
    
    # 返回连接的字符串
    return ''.join(unique_letters)

def extract_answer_for_question_answering(item):
    """
    提取问答题的答案内容
    """
    # 首先尝试使用api_response中的content
    api_response = item.get('api_response', {})
    choices = api_response.get('choices', [{}])
    if choices and len(choices) > 0:
        message = choices[0].get('message', {})
        content = message.get('content', '')
        
        if content:
            return content.strip(), True
    
    # 如果没有api_response或content，使用pred字段
    pred = item.get('pred', '')
    if pred.strip():
        return pred.strip(), True
    
    return "", False

def extract_answer_for_text_classification(item):
    """
    提取文本分类任务的答案（从类别列表中）
    """
    pred = item.get('pred', '')
    
    # 查找第一个出现的类别
    for category in categories_zh:
        if category in pred:
            return category, True
            
    # 如果pred字段没有匹配到，检查api_response
    api_response = item.get('api_response', {})
    choices = api_response.get('choices', [{}])
    if choices and len(choices) > 0:
        message = choices[0].get('message', {})
        content = message.get('content', '')
        
        if content:
            # 查找最后一个出现的类别
            found_categories = []
            for category in categories_zh:
                if category in content:
                    found_categories.append(category)
            
            if found_categories:
                # 返回内容中最后一个类别
                for category in reversed(categories_zh):
                    if category in found_categories:
                        return category, True
    
    # 没有找到任何匹配项
    return "none", False

def extract_answer_for_math_reasoning(item):
    """
    提取数学推理任务的答案（数字）
    """
    pred = item.get('pred', '')
    
    # 查找第一个出现的数字
    number_match = re.search(r'\d+(?:\.\d+)?', pred)
    if number_match:
        return number_match.group(0), True
        
    # 如果pred字段没有匹配到，检查api_response
    api_response = item.get('api_response', {})
    choices = api_response.get('choices', [{}])
    if choices and len(choices) > 0:
        message = choices[0].get('message', {})
        content = message.get('content', '')
        
        if content:
            # 查找所有数字，返回最后一个
            number_matches = re.findall(r'\d+(?:\.\d+)?', content)
            if number_matches:
                return number_matches[-1], True
    
    # 没有找到任何匹配项
    return "none", False

def process_result_file(file_path, task_name):
    """
    处理结果文件，提取标准答案和预测答案，同时统计失败数和记录失败ID
    
    参数:
        file_path: 结果文件路径
        task_name: 任务名称（新的目录结构名称，如 'Coreference_Resolution'）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        invalid_items = 0  # 无效数据项计数
        extraction_failed_items = 0  # 提取失败计数
        
        # 记录失败的项目ID
        invalid_item_ids = []
        extraction_failed_ids = []
        
        # 将新的目录名映射为旧的任务名，用于判断任务类型
        mapped_task_name = TASK_MAPPING.get(task_name, task_name.lower())
        
        for item in data:
            # 确保是有效的数据项
            if 'id' not in item or 'gold' not in item:
                invalid_items += 1
                if 'id' in item:
                    invalid_item_ids.append(item['id'])
                continue
                
            # 处理不同任务类型的答案提取
            extraction_success = True
            
            # 使用映射后的任务名进行判断
            if mapped_task_name in ['coref_resolution', 'entailment', 'ethnic_language_understanding', 
                                   'ethnic_vocabulary', 'professional_skills', 'ethnic_domain_knowledge',
                                   'commercial_compliance', 'discrimination_detection', 'rights_protection', 
                                   'service_safety', 'value_alignment']:
                # 选择题（单选和多选）
                answer, extraction_success = extract_answer_for_choice_question(item)
            elif mapped_task_name in ['reading_comprehension', 'text_generation', 'traditional_culture']:
                # 问答题
                answer, extraction_success = extract_answer_for_question_answering(item)
            elif mapped_task_name == 'text_classification':
                # 文本分类
                answer, extraction_success = extract_answer_for_text_classification(item)
            elif mapped_task_name == 'math_reasoning':
                # 数学推理
                answer, extraction_success = extract_answer_for_math_reasoning(item)
            elif mapped_task_name == 'translation':
                # 翻译
                answer, extraction_success = extract_answer_for_question_answering(item)
            else:
                # 其他任务，直接使用pred
                pred = item.get('pred', '').strip()
                answer = pred
                extraction_success = bool(pred)
            
            if not extraction_success:
                extraction_failed_items += 1
                extraction_failed_ids.append(item['id'])
            
            # 组装处理后的数据
            processed_item = {
                'id': item['id'],
                'gold': item['gold'],
                'pred': item.get('pred', ''),
                'answer': answer,
                'extraction_success': extraction_success
            }
            
            processed_data.append(processed_item)
        
        stats = {
            'total_items': len(data),
            'invalid_items': invalid_items,
            'extraction_failed_items': extraction_failed_items,
            'processed_items': len(processed_data),
            'success_rate': (len(processed_data) - extraction_failed_items) / len(processed_data) if processed_data else 0,
            'invalid_item_ids': invalid_item_ids,
            'extraction_failed_ids': extraction_failed_ids
        }
            
        return processed_data, stats
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return [], {
            'error': str(e), 
            'total_items': 0, 
            'invalid_items': 0, 
            'extraction_failed_items': 0, 
            'processed_items': 0, 
            'success_rate': 0,
            'invalid_item_ids': [],
            'extraction_failed_ids': []
        }

def main():
    parser = argparse.ArgumentParser(description="处理评估数据文件，提取精确的待评估答案")
    parser.add_argument("--base_path", required=True, help="结果存储的基础路径")
    parser.add_argument("--output_dir", required=True, help="处理后数据的输出目录")
    parser.add_argument("--model", help="指定要处理的模型，不指定则处理所有模型")
    parser.add_argument("--task", help="指定要处理的任务，不指定则处理所有任务")
    parser.add_argument("--language", help="指定要处理的语言，不指定则处理所有语言")
    args = parser.parse_args()
    
    BASE_PATH = args.base_path
    OUTPUT_DIR = args.output_dir
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 统计数据
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    # 统计提取失败的数据
    extraction_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # 记录处理的文件
    processed_file_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 记录所有失败的ID (模型 -> 任务 -> 语言 -> 文件名 -> 失败ID列表)
    all_failed_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # 遍历模型目录
    models = [args.model] if args.model else os.listdir(BASE_PATH)
    for model_name in models:
        model_path = os.path.join(BASE_PATH, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"处理模型: {model_name}")
        
        # 创建模型输出目录
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 遍历任务目录
        tasks = [args.task] if args.task else os.listdir(model_path)
        for task_name in tasks:
            task_path = os.path.join(model_path, task_name)
            if not os.path.isdir(task_path):
                continue
                
            print(f"  处理任务: {task_name}")
            
            # 创建任务输出目录
            task_output_dir = os.path.join(model_output_dir, task_name)
            os.makedirs(task_output_dir, exist_ok=True)
            
            # 遍历语言目录
            languages = [args.language] if args.language else os.listdir(task_path)
            for language in languages:
                lang_path = os.path.join(task_path, language)
                if not os.path.isdir(lang_path):
                    continue
                    
                print(f"    处理语言: {language}")
                
                # 创建语言输出目录
                lang_output_dir = os.path.join(task_output_dir, language)
                os.makedirs(lang_output_dir, exist_ok=True)
                
                # 遍历结果文件
                for file_name in os.listdir(lang_path):
                    if not file_name.endswith('.json'):
                        continue
                        
                    total_files += 1
                    file_path = os.path.join(lang_path, file_name)
                    print(f"      处理文件: {file_name}")
                    
                    # 处理结果文件（传入新的目录结构名称）
                    processed_data, stats = process_result_file(file_path, task_name)
                    
                    # 记录提取统计信息
                    extraction_stats[model_name][task_name][language][file_name] = stats
                    
                    # 记录失败的ID
                    if stats['extraction_failed_ids']:
                        all_failed_ids[model_name][task_name][language][file_name] = stats['extraction_failed_ids']
                    
                    if not processed_data:
                        print(f"      警告: 文件 {file_name} 处理失败或无数据")
                        failed_files += 1
                        continue
                    
                    # 输出处理后的数据
                    output_file = os.path.join(lang_output_dir, file_name)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=2)
                    
                    # 记录处理成功
                    processed_files += 1
                    processed_file_map[model_name][task_name][language].append({
                        'file_name': file_name,
                        'item_count': len(processed_data),
                        'extraction_failed_count': stats['extraction_failed_items'],
                        'success_rate': stats['success_rate']
                    })
                    
                    print(f"      成功处理: {file_name} ({len(processed_data)} 项数据，提取失败: {stats['extraction_failed_items']}，成功率: {stats['success_rate']*100:.2f}%)")
    
    # 输出处理统计
    print("\n处理完成!")
    print(f"总共文件数: {total_files}")
    print(f"成功处理: {processed_files} 文件")
    print(f"处理失败: {failed_files} 文件")
    
    # 保存处理文件映射表
    map_file = os.path.join(OUTPUT_DIR, 'processed_files_map.json')
    with open(map_file, 'w', encoding='utf-8') as f:
        json.dump(processed_file_map, f, ensure_ascii=False, indent=2)
    
    # 保存失败ID记录
    failed_ids_file = os.path.join(OUTPUT_DIR, 'extraction_failed_ids.json')
    with open(failed_ids_file, 'w', encoding='utf-8') as f:
        json.dump(all_failed_ids, f, ensure_ascii=False, indent=2)
    
    # 生成提取统计报告
    # 1. 按模型汇总
    model_summary = {}
    for model_name, task_data in extraction_stats.items():
        model_total = 0
        model_failed = 0
        task_stats = {}
        
        for task_name, lang_data in task_data.items():
            task_total = 0
            task_failed = 0
            language_stats = {}
            
            for language, file_data in lang_data.items():
                lang_total = 0
                lang_failed = 0
                
                for file_name, stats in file_data.items():
                    lang_total += stats['processed_items']
                    lang_failed += stats['extraction_failed_items']
                
                language_stats[language] = {
                    'total_items': lang_total,
                    'failed_items': lang_failed,
                    'success_rate': (lang_total - lang_failed) / lang_total if lang_total else 0
                }
                
                task_total += lang_total
                task_failed += lang_failed
            
            task_stats[task_name] = {
                'total_items': task_total,
                'failed_items': task_failed,
                'success_rate': (task_total - task_failed) / task_total if task_total else 0,
                'languages': language_stats
            }
            
            model_total += task_total
            model_failed += task_failed
        
        model_summary[model_name] = {
            'total_items': model_total,
            'failed_items': model_failed,
            'success_rate': (model_total - model_failed) / model_total if model_total else 0,
            'tasks': task_stats
        }
    
    # 保存统计报告
    stats_file = os.path.join(OUTPUT_DIR, 'extraction_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'by_model': model_summary,
            'raw_stats': extraction_stats
        }, f, ensure_ascii=False, indent=2)
    
    # 生成简洁的可读报告
    report_file = os.path.join(OUTPUT_DIR, 'extraction_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("提取统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("模型级别汇总:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型名称':<20} {'总项目数':<10} {'提取失败数':<10} {'成功率':<10}\n")
        for model_name, stats in model_summary.items():
            success_rate = stats['success_rate'] * 100
            f.write(f"{model_name:<20} {stats['total_items']:<10} {stats['failed_items']:<10} {success_rate:.2f}%\n")
        
        f.write("\n\n详细统计:\n")
        f.write("=" * 80 + "\n")
        
        for model_name, model_stats in model_summary.items():
            f.write(f"\n模型: {model_name}\n")
            f.write("-" * 80 + "\n")
            
            for task_name, task_stats in model_stats['tasks'].items():
                task_success_rate = task_stats['success_rate'] * 100
                f.write(f"  任务: {task_name} (总计: {task_stats['total_items']}, 失败: {task_stats['failed_items']}, 成功率: {task_success_rate:.2f}%)\n")
                
                for language, lang_stats in task_stats['languages'].items():
                    lang_success_rate = lang_stats['success_rate'] * 100
                    f.write(f"    语言: {language} (总计: {lang_stats['total_items']}, 失败: {lang_stats['failed_items']}, 成功率: {lang_success_rate:.2f}%)\n")
            
            f.write("\n")
        
        # 添加失败ID的统计
        f.write("\n\n提取失败ID统计:\n")
        f.write("=" * 80 + "\n")
        
        total_failed_models = 0
        for model_name, task_data in all_failed_ids.items():
            model_failed_count = 0
            for task_name, lang_data in task_data.items():
                for language, file_data in lang_data.items():
                    for file_name, failed_ids in file_data.items():
                        model_failed_count += len(failed_ids)
            
            if model_failed_count > 0:
                total_failed_models += 1
                f.write(f"\n模型: {model_name} (总失败ID数: {model_failed_count})\n")
                
                for task_name, lang_data in task_data.items():
                    task_failed_count = 0
                    for language, file_data in lang_data.items():
                        for file_name, failed_ids in file_data.items():
                            task_failed_count += len(failed_ids)
                    
                    if task_failed_count > 0:
                        f.write(f"  任务: {task_name} (失败ID数: {task_failed_count})\n")
                        
                        for language, file_data in lang_data.items():
                            lang_failed_count = 0
                            for file_name, failed_ids in file_data.items():
                                lang_failed_count += len(failed_ids)
                            
                            if lang_failed_count > 0:
                                f.write(f"    语言: {language} (失败ID数: {lang_failed_count})\n")
                                
                                for file_name, failed_ids in file_data.items():
                                    if failed_ids:
                                        f.write(f"      文件: {file_name} (失败ID数: {len(failed_ids)})\n")
                                        # 如果失败ID太多，只显示前10个
                                        if len(failed_ids) > 10:
                                            f.write(f"        失败ID示例(前10个): {', '.join(failed_ids[:10])}...\n")
                                        else:
                                            f.write(f"        失败ID: {', '.join(failed_ids)}\n")
        
        if total_failed_models == 0:
            f.write("\n没有发现提取失败的ID！\n")
    
    print(f"处理文件映射表已保存到: {map_file}")
    print(f"提取失败的ID已保存到: {failed_ids_file}")
    print(f"提取统计报告已保存到: {stats_file}")
    print(f"可读统计报告已保存到: {report_file}")

if __name__ == "__main__":
    main()