import os
import json
import time
import re
import argparse
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import sys
import signal
import traceback
from datetime import datetime
import random

LANGUAGE_NAMES = {"zh": "汉语", "bo": "藏语", "mn": "蒙古语", "ug": "维吾尔语"}

# 只处理两个生成式任务（不包含翻译）
TASK_TYPES = {
    "traditional_culture": "传统文化", 
    "text_generation": "文本生成"
}

# 新旧任务名称映射 (只包含需要的两个任务)
TASK_MAPPING = {
    'Minority_Culture_QA': 'traditional_culture',
    'Minority_Language_Instruction_QA': 'text_generation',
}

# 添加反向映射：从旧名称到新名称
REVERSE_TASK_MAPPING = {v: k for k, v in TASK_MAPPING.items()}

GRACEFUL_EXIT_REQUESTED = False

# --- Signal Handler ---
def signal_handler(sig, frame):
    global GRACEFUL_EXIT_REQUESTED
    if not GRACEFUL_EXIT_REQUESTED:
        print(f"\n捕获到信号 {sig}, 请求优雅退出... 后续新的API调用将中止，当前批次完成后将保存。")
        print("如果程序无法正常退出，请再次按 Ctrl+C 强制退出。")
        GRACEFUL_EXIT_REQUESTED = True
    else:
        print(f"\n再次捕获到信号 {sig}, 强制退出程序...")
        print("正在尝试保存当前进度...")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Utility Functions ---
def get_client(api_key, api_base=None):
    return OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)

def save_json_safely(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        return True
    except Exception as e:
        print(f"保存文件 {filepath} 时出错: {str(e)}")
        if os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass
        return False

def log_error(error_log_file, error_id_file, message, error=None, item_ids=None, details=""):
    """记录错误到日志文件并保存相关ID"""
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_id_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"=== {timestamp} ===\n"
    log_entry += f"错误信息: {message}\n"
    if error:
        log_entry += f"异常类型: {type(error).__name__}\n"
        log_entry += f"异常详情: {str(error)}\n"
        log_entry += f"堆栈跟踪:\n{traceback.format_exc()}\n"
    if details:
        log_entry += f"补充信息: {details}\n"
    log_entry += "\n"
    
    try:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e_log:
        print(f"写入错误日志 {error_log_file} 失败: {e_log}")
    
    # 保存出错的ID
    if item_ids:
        try:
            existing_ids = []
            if os.path.exists(error_id_file):
                try:
                    with open(error_id_file, 'r', encoding='utf-8') as f:
                        existing_ids = json.load(f)
                    if not isinstance(existing_ids, list):
                        existing_ids = []
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_ids = []
            
            for item_id in item_ids:
                if item_id and item_id not in existing_ids:
                    existing_ids.append(item_id)
            
            save_json_safely(existing_ids, error_id_file)
            print(f"已将{len(item_ids)}个错误ID保存到 {error_id_file}")
        except Exception as e:
            print(f"保存错误ID时发生异常: {str(e)}")

# --- Core Logic Functions ---
def load_test_data(task_type, language, base_path):
    """根据新的目录结构加载测试数据"""
    
    # 根据任务类型确定数据文件路径
    if task_type == "traditional_culture":
        file_path = os.path.join(base_path, f"Chinese_Minority_Knowledge_Tasks/Minority_Culture_QA/{language}.json")
    elif task_type == "text_generation":
        file_path = os.path.join(base_path, f"Chinese_Minority_Knowledge_Tasks/Minority_Language_Instruction_QA/{language}.json")
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    if not os.path.exists(file_path):
        print(f"警告: 测试数据文件不存在: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
        id_to_data = {item.get('id', item.get('query_id')): item for item in data if item.get('id') or item.get('query_id')}
        return id_to_data
    except Exception as e:
        print(f"加载测试数据 {file_path} 出错: {str(e)}")
        return {}

def load_model_predictions(model_name, task_type, language_param, models_base_path):
    """加载模型预测结果 - 修改为使用新的文件夹结构"""
    try:
        if not language_param:
            raise ValueError(f"{task_type} 任务需要指定语言参数。")
        
        # 🔥 关键修改：将旧任务名称转换为新的文件夹名称
        new_task_name = REVERSE_TASK_MAPPING.get(task_type, task_type)
        file_path = os.path.join(models_base_path, model_name, new_task_name, language_param, "zh-prompt_test.json")
        
        if not os.path.exists(file_path):
            print(f"模型预测文件不存在: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f: 
            predictions = json.load(f)
        return predictions
    except Exception as e:
        print(f"加载模型 {model_name} ({task_type}, {language_param}) 的预测结果出错: {str(e)}")
        return None

def construct_prompt(task_type, language, question, reference, prediction, subcategory=None):
    """构建评估提示"""
    # 通用评分标准
    scoring_criteria = """
评分标准（1-5分）：
- 5分：优秀 - 完全符合要求，表现卓越
- 4分：良好 - 基本符合要求，表现较好，有轻微不足
- 3分：一般 - 部分符合要求，表现中等，有明显不足
- 2分：较差 - 少部分符合要求，表现不佳，问题较多
- 1分：很差 - 不符合要求，表现很差，问题严重
"""

    # 评估维度定义
    dimension_definitions = {
        "事实正确性": "回答中提供的信息是否准确无误，是否基于可信的事实和数据",
        "满足用户需求": "回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应",
        "安全无害": "回答是否不会引发危险或有害行为，是否不会误导用户或传播有害信息",
        "清晰度": "回答是否表达清晰、易懂，是否使用了简洁的语言和结构，以便用户可以轻松理解",
        "逻辑性": "回答是否在逻辑或者推理上连贯且合理",
        "完备性": "回答是否提供了足够的信息和细节，以满足用户的需求，是否遗漏了重要的方面",
        "创造性": "回答是否具有创新性或独特性，是否提供了新颖的见解或解决方法",
        "可负责程度": "回答中提供的建议或信息是否可行，是否负有一定的责任，是否考虑了潜在风险和后果",
        "逻辑连贯性": "回答是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾",
        "公平与可负责程度": "回答是否考虑了不同观点和立场，是否提供了公正的信息或建议，不携带私人的观点或者偏见，不偏袒某一方，提供的建议或信息是否可行，是否负有一定的责任，是否考虑了潜在风险和后果",
        "丰富度": "回答包含丰富的信息、深度、上下文考虑、多样性、详细解释和实例，以满足用户需求并提供全面理解"
    }

    # 回答类型到评估维度的映射
    answer_type_dimensions = {
        "事实与解释型回答": ["事实正确性", "满足用户需求", "清晰度", "完备性"],
        "逻辑推理型回答": ["事实正确性", "满足用户需求", "逻辑连贯性", "完备性"],
        "生成型回答": ["事实正确性", "满足用户需求", "逻辑连贯性", "创造性", "丰富度"],
        "建议型回答": ["事实正确性", "满足用户需求", "公平与可负责程度", "创造性"]
    }

    # 子类别到回答类型的映射
    subcategory_to_answer_type = {
        "常识知识": "事实与解释型回答",
        "阅读理解": "事实与解释型回答",
        "翻译": "生成型回答",
        "文本分类": "事实与解释型回答",
        "信息抽取": "事实与解释型回答",
        "字词理解": "事实与解释型回答",
        "文化理解": "事实与解释型回答",
        "观点表达": "建议型回答",
        "寻求建议": "建议型回答",
        "实用文体写作": "生成型回答",
        "创意文体写作": "生成型回答",
        "专业文体写作": "生成型回答",
        "其他写作类": "生成型回答",
        "证明": "逻辑推理型回答",
        "推理": "逻辑推理型回答",
        "初等数学": "逻辑推理型回答",
        "高等数学": "逻辑推理型回答",
        "应用数学": "逻辑推理型回答",
        "现实生活类": "生成型回答",
        "游戏娱乐类": "生成型回答",
        "功能类": "生成型回答",
        "现实名人类": "生成型回答",
        "（虚拟）恋爱类": "生成型回答",
        "物理": "事实与解释型回答",
        "化学": "事实与解释型回答",
        "计算机": "事实与解释型回答",
        "生物医学": "事实与解释型回答",
        "经济": "事实与解释型回答",
        "天文": "事实与解释型回答",
        "历史": "事实与解释型回答",
        "音乐": "事实与解释型回答",
        "法律": "事实与解释型回答",
        "体育": "事实与解释型回答",
        "地理": "事实与解释型回答",
        "文学": "事实与解释型回答",
        "其他": "事实与解释型回答"
    }

    if task_type == "text_generation":
        # 根据subcategory确定评估维度
        if subcategory and subcategory in subcategory_to_answer_type:
            answer_type = subcategory_to_answer_type[subcategory]
            dimensions = answer_type_dimensions[answer_type].copy()
        else:
            # 默认使用生成型回答的维度
            answer_type = "生成型回答"
            dimensions = answer_type_dimensions[answer_type].copy()
        
        # 添加语言使用准确性维度
        dimensions.append("语言使用准确性")
        
        # 构建评估维度说明
        dimension_descriptions = []
        for i, dim in enumerate(dimensions, 1):
            if dim == "语言使用准确性":
                dimension_descriptions.append(f"{i}. {dim}：评估回答是否使用了正确的目标语言（{LANGUAGE_NAMES.get(language, language)}），是否存在语言混用或使用错误语言的情况")
            else:
                dimension_descriptions.append(f"{i}. {dim}：{dimension_definitions[dim]}")
        
        # 构建输出格式
        output_format = []
        for dim in dimensions:
            output_format.append(f"{dim}：[1-5的数字]")
        
        prompt = f"""请你作为一个专业的{LANGUAGE_NAMES.get(language, language)}文本生成质量评估专家，对以下模型生成的文本进行全面评分。

任务语言：{LANGUAGE_NAMES.get(language, language)}
子类别：{subcategory if subcategory else '未指定'}
评估类型：{answer_type}

问题/提示：
{question}

参考答案：
{reference}

模型生成文本：
{prediction}

{scoring_criteria}

请从以下{len(dimensions)}个维度对生成文本质量进行评分（每个维度1-5分）：

{chr(10).join(dimension_descriptions)}

请严格按照以下格式输出评分结果：
{chr(10).join(output_format)}
分析总结：[简要分析生成文本的优缺点和质量特点，100-200字]
最终分数：[{len(dimensions)}个维度的平均分，保留两位小数]"""

    elif task_type == "traditional_culture":
        prompt = f"""请你作为一个专业的{LANGUAGE_NAMES.get(language, language)}传统文化专家，对以下模型回答进行全面评分。

任务语言：{LANGUAGE_NAMES.get(language, language)}

问题：
{question}

参考答案：
{reference}

模型回答：
{prediction}

{scoring_criteria}

请从以下6个维度对回答质量进行评分（每个维度1-5分）：

1. 知识准确性（文化知识是否准确）：评估回答中文化知识点的准确性和可靠性
2. 文化理解深度（对文化内涵的理解程度）：评估对文化背景、内涵和意义的深入理解
3. 语言表达适切性（语言使用是否得体）：评估语言表达是否符合文化语境和表达习惯
4. 内容完整性（回答是否全面完整）：评估回答的完整性和全面性
5. 内部视角真实性（是否体现该文化的内部视角）：评估是否真实反映了该文化群体的内部观点和认知
6. 语言使用准确性（是否使用正确的目标语言）：评估回答是否使用了正确的目标语言（{LANGUAGE_NAMES.get(language, language)}），是否存在语言混用或使用错误语言的情况

请严格按照以下格式输出评分结果：
知识准确性：[1-5的数字]
文化理解深度：[1-5的数字]
语言表达适切性：[1-5的数字]
内容完整性：[1-5的数字]
内部视角真实性：[1-5的数字]
语言使用准确性：[1-5的数字]
分析总结：[简要分析回答在文化理解和知识表达方面的优缺点，100-200字]
最终分数：[六个维度的平均分，保留两位小数]"""
    
    else:
        print(f"警告: 未知任务类型 '{task_type}' 无法构建提示。")
        return "错误：未知任务类型。"
    
    return prompt

def evaluate_sample(client, model, task_type, language, question, reference, prediction, subcategory=None):
    global GRACEFUL_EXIT_REQUESTED
    
    if GRACEFUL_EXIT_REQUESTED:
        return "评估API调用因脚本退出而中止"
    
    # 添加调试信息
    if task_type == "text_generation" and subcategory:
        print(f"Debug: 正在构建{task_type}任务的prompt，subcategory: {subcategory}")
    
    prompt = construct_prompt(task_type, language, question, reference, prediction, subcategory)
    
    # 检查prompt是否构建成功
    if prompt.startswith("错误："):
        return f"构建prompt失败: {prompt}"
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        if GRACEFUL_EXIT_REQUESTED:
            return "评估API调用因脚本退出而中止"
        
        try:
            # 添加超时机制
            response = client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.2,
                timeout=30  # 30秒超时
            )
            result = response.choices[0].message.content.strip()
            
            # 添加调试信息，检查API响应
            if task_type == "text_generation" and subcategory:
                print(f"Debug: API响应成功，响应长度: {len(result)}")
            
            return result
        except Exception as e:
            if GRACEFUL_EXIT_REQUESTED:
                return f"评估API调用因脚本退出而中止: {str(e)}"
            
            error_msg = f"{type(e).__name__}: {str(e)}"
            if attempt < max_retries - 1:
                print(f"API调用失败 (attempt {attempt+1}/{max_retries}): {error_msg}")
                # 检查是否需要退出
                for i in range(retry_delay * (attempt + 1)):
                    if GRACEFUL_EXIT_REQUESTED:
                        return "评估API调用因脚本退出而中止"
                    time.sleep(1)
            else:
                return f"评估API调用失败: {error_msg}"
    
    return "评估API调用失败: 已达最大重试次数"

def parse_scores(evaluation_text, task_type, subcategory=None):
    """解析评估结果中的分数"""
    scores = {}
    
    # 添加调试信息
    if task_type == "text_generation" and subcategory:
        print(f"Debug: 正在解析{task_type}任务的评分，subcategory: {subcategory}")
    
    # 定义维度映射（与construct_prompt中保持一致）
    subcategory_to_answer_type = {
        "常识知识": "事实与解释型回答", "阅读理解": "事实与解释型回答", "翻译": "生成型回答",
        "文本分类": "事实与解释型回答", "信息抽取": "事实与解释型回答", "字词理解": "事实与解释型回答",
        "文化理解": "事实与解释型回答", "观点表达": "建议型回答", "寻求建议": "建议型回答",
        "实用文体写作": "生成型回答", "创意文体写作": "生成型回答", "专业文体写作": "生成型回答",
        "其他写作类": "生成型回答", "证明": "逻辑推理型回答", "推理": "逻辑推理型回答",
        "初等数学": "逻辑推理型回答", "高等数学": "逻辑推理型回答", "应用数学": "逻辑推理型回答",
        "现实生活类": "生成型回答", "游戏娱乐类": "生成型回答", "功能类": "生成型回答",
        "现实名人类": "生成型回答", "（虚拟）恋爱类": "生成型回答", "物理": "事实与解释型回答",
        "化学": "事实与解释型回答", "计算机": "事实与解释型回答", "生物医学": "事实与解释型回答",
        "经济": "事实与解释型回答", "天文": "事实与解释型回答", "历史": "事实与解释型回答",
        "音乐": "事实与解释型回答", "法律": "事实与解释型回答", "体育": "事实与解释型回答",
        "地理": "事实与解释型回答", "文学": "事实与解释型回答", "其他": "事实与解释型回答"
    }
    
    answer_type_dimensions = {
        "事实与解释型回答": ["事实正确性", "满足用户需求", "清晰度", "完备性"],
        "逻辑推理型回答": ["事实正确性", "满足用户需求", "逻辑连贯性", "完备性"],
        "生成型回答": ["事实正确性", "满足用户需求", "逻辑连贯性", "创造性", "丰富度"],
        "建议型回答": ["事实正确性", "满足用户需求", "公平与可负责程度", "创造性"]
    }
    
    if task_type == "text_generation":
        # 动态构建评估维度（与construct_prompt保持一致）
        if subcategory and subcategory in subcategory_to_answer_type:
            answer_type = subcategory_to_answer_type[subcategory]
            dimensions = answer_type_dimensions[answer_type].copy()
            print(f"Debug: 根据subcategory '{subcategory}' 确定answer_type为 '{answer_type}'")
        else:
            dimensions = answer_type_dimensions["生成型回答"].copy()
            print(f"Debug: 使用默认生成型回答维度 (subcategory: {subcategory})")
        
        # 添加语言使用准确性维度
        dimensions.append("语言使用准确性")
        print(f"Debug: 最终使用的维度: {dimensions}")
        
        # 动态构建正则表达式模式，使用标准化的键名
        score_patterns = {}
        dimension_key_mapping = {
            "事实正确性": "factual_accuracy",
            "满足用户需求": "user_needs_satisfaction", 
            "清晰度": "clarity",
            "完备性": "completeness",
            "逻辑连贯性": "logical_coherence",
            "创造性": "creativity",
            "丰富度": "richness",
            "公平与可负责程度": "fairness_responsibility",
            "语言使用准确性": "language_usage_accuracy"
        }
        
        for dim in dimensions:
            escaped_dim = dim.replace("(", r"\(").replace(")", r"\)").replace("（", r"\（").replace("）", r"\）")
            key = dimension_key_mapping.get(dim, dim.replace(" ", "_").lower())
            # ✅ 修改点: 让正则表达式兼容方括号
            score_patterns[key] = f"{escaped_dim}[：:]\s*\[?([1-5](?:\.\d+)?)\]?"
        
        # 添加分析总结和最终分数的模式
        score_patterns["analysis_summary"] = r"分析总结[：:]\s*(.+?)(?=最终分数[：:])"
        # ✅ 修改点: 让正则表达式兼容方括号
        score_patterns["final_score"] = r"最终分数[：:]\s*\[?([1-5](?:\.\d+)?)\]?"
        
        print(f"Debug: 构建的score_patterns keys: {list(score_patterns.keys())}")
        
    elif task_type == "traditional_culture":
        # ✅ 修改点: 让所有分数的正则表达式都兼容方括号
        score_patterns = {
            "knowledge_accuracy": r"知识准确性[：:]\s*\[?([1-5](?:\.\d+)?)\]?",
            "cultural_depth": r"文化理解深度[：:]\s*\[?([1-5](?:\.\d+)?)\]?",
            "expression": r"语言表达适切性[：:]\s*\[?([1-5](?:\.\d+)?)\]?",
            "completeness": r"内容完整性[：:]\s*\[?([1-5](?:\.\d+)?)\]?",
            "insider_perspective": r"内部视角真实性[：:]\s*\[?([1-5](?:\.\d+)?)\]?",
            "language_usage_accuracy": r"语言使用准确性[：:]\s*\[?([1-5](?:\.\d+)?)\]?",
            "analysis_summary": r"分析总结[：:]\s*(.+?)(?=最终分数[：:])",
            "final_score": r"最终分数[：:]\s*\[?([1-5](?:\.\d+)?)\]?"
        }
    else:
        print(f"Debug: 不支持的任务类型: {task_type}")
        return {}

    # 检查评估文本是否有效
    if evaluation_text is None or \
       evaluation_text.startswith("评估API调用失败:") or \
       evaluation_text.startswith("评估API调用因脚本退出而中止") or \
       evaluation_text.startswith("构建prompt失败:"):
        print(f"Debug: 评估文本无效: {evaluation_text[:200] if evaluation_text else 'None'}")
        for key_pattern in score_patterns: 
            scores[key_pattern] = None
        return scores

    # 解析评分
    parsed_count = 0
    failed_patterns = []
    
    for key, pattern in score_patterns.items():
        match = re.search(pattern, evaluation_text, re.DOTALL)
        if match:
            if key == "analysis_summary":
                # 处理分析总结，清理换行和多余空格
                analysis = match.group(1).strip()
                analysis = re.sub(r'\s+', ' ', analysis)
                scores[key] = analysis
                parsed_count += 1
            else:
                try:
                    score_str = match.group(1)
                    if score_str is None: # 处理匹配到但捕获组为空的情况（不太可能但为了健壮性）
                        raise ValueError("Captured group is None")
                    score = float(score_str)
                    # 验证分数在1-5范围内
                    if 1 <= score <= 5:
                        scores[key] = score
                        parsed_count += 1
                    else:
                        print(f"Debug: 分数超出范围 {key}: {score}")
                        scores[key] = None
                        failed_patterns.append(f"{key}(超出范围:{score})")
                except (ValueError, TypeError):
                    print(f"Debug: 无法解析分数 {key}: {match.group(1)}")
                    scores[key] = None
                    failed_patterns.append(f"{key}(解析错误:{match.group(1)})")
        else:
            scores[key] = None
            failed_patterns.append(f"{key}(未找到)")
    
    print(f"Debug: 成功解析 {parsed_count}/{len(score_patterns)} 个评分")
    
    # 如果解析失败，输出详细的调试信息
    if parsed_count < len(score_patterns) and parsed_count > 0:
        print("="*80)  
        print("⚠️  DEBUG: 部分解析失败，以下是详细信息:")
        print(f"📋 Task: {task_type}, Subcategory: {subcategory}")
        print(f"✅ 成功解析: {parsed_count}/{len(score_patterns)}")
        print(f"❌ 失败的模式: {failed_patterns}")
        print("\n📝 评估模型的完整原始回复:")
        print("-"*50)
        print(evaluation_text)
        print("-"*50)
        print("="*80)
    elif parsed_count == 0:
        print("="*80)
        print("🔍 DEBUG: 完全无法解析评分，以下是详细信息:")
        print(f"📋 Task: {task_type}, Subcategory: {subcategory}")  
        print(f"🎯 期望的评分模式: {list(score_patterns.keys())}")
        print(f"❌ 失败的模式: {failed_patterns}")
        print("\n📝 评估模型的完整原始回复:")
        print("-"*50)
        print(evaluation_text)
        print("-"*50)
        print("\n🔍 期望的正则表达式模式:")
        for key, pattern in score_patterns.items():
            print(f"  {key}: {pattern}")
        print("="*80)

    # 如果LLM没有提供最终分数或未能解析，则计算平均值
    if scores.get("final_score") is None:
        valid_sub_scores = [v for k, v in scores.items() if k not in ["final_score", "analysis_summary"] and v is not None]
        if valid_sub_scores: 
            scores["final_score"] = round(sum(valid_sub_scores) / len(valid_sub_scores), 2)
            print(f"Debug: 计算平均最终分数: {scores['final_score']}")
        else:
            print("Debug: 没有有效的子分数，无法计算最终分数")
    
    return scores

def is_evaluation_successful(evaluation_result_dict):
    """检查评估是否成功"""
    if not isinstance(evaluation_result_dict, dict): 
        return False
    
    eval_text = evaluation_result_dict.get("evaluation", "")
    if eval_text is None or \
       eval_text.startswith("评估API调用失败:") or \
       eval_text.startswith("评估API调用因脚本退出而中止"):
        return False
    
    return evaluation_result_dict.get("final_score") is not None

def validate_evaluation_result(result_dict, task_type, subcategory=None):
    """验证评估结果的完整性和有效性"""
    if not isinstance(result_dict, dict):
        return False, "结果不是字典格式"
    
    # 检查必需的字段
    required_fields = ["id", "model", "task_type", "evaluation", "final_score"]
    missing_fields = [field for field in required_fields if field not in result_dict]
    if missing_fields:
        return False, f"缺少必需字段: {missing_fields}"
    
    # 检查final_score是否有效
    final_score = result_dict.get("final_score")
    if final_score is None or not isinstance(final_score, (int, float)) or not (1 <= final_score <= 5):
        return False, f"final_score无效: {final_score}"
    
    # 根据任务类型检查特定字段
    if task_type == "text_generation":
        expected_scores = ["factual_accuracy", "user_needs_satisfaction", "language_usage_accuracy"]
    elif task_type == "traditional_culture":
        expected_scores = ["knowledge_accuracy", "cultural_depth", "expression", "completeness", "insider_perspective", "language_usage_accuracy"]
    else:
        return False, f"未知任务类型: {task_type}"
    
    # 检查至少有一些维度分数
    score_fields = [field for field in expected_scores if field in result_dict and result_dict[field] is not None]
    if len(score_fields) == 0:
        return False, "没有找到任何有效的维度分数"
    
    return True, "验证通过"

# --- Command Line Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="使用LLM评估模型输出质量")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据集基础路径")
    parser.add_argument("--models_predictions_path", type=str, required=True, help="模型预测结果基础路径")
    parser.add_argument("--output_path", type=str, required=True, help="评估结果输出路径")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API密钥")
    parser.add_argument("--api_base", type=str, default=None, help="API基础URL")
    parser.add_argument("--model", type=str, default="gpt-4o", help="用于评估的LLM模型名称")
    parser.add_argument("--max_workers", type=int, default=5, help="并行处理线程数")
    parser.add_argument("--sample_size", type=int, default=None, help="每个任务评估的样本数 (None for all)")
    parser.add_argument("--models_to_evaluate", nargs='+', default=None, help="要评估的模型列表 (None for all in models_predictions_path)")
    parser.add_argument("--resume", action="store_true", help="从上次中断的地方继续执行")
    parser.add_argument("--task", type=str, help="特定任务类型 (traditional_culture, text_generation)")
    parser.add_argument("--language", type=str, help="特定语言 (e.g., bo, mn, ug)")
    parser.add_argument("--checkpoint_interval_items", type=int, default=10, help="每处理N个项目后，若有成功项则保存检查点")
    parser.add_argument("--checkpoint_interval_time", type=int, default=300, help="每N秒后，若有成功项则保存检查点")
    return parser.parse_args()

# --- Main Processing Function ---
def process_task(args, client, eval_model_name, task_type, language_param_for_file):
    global GRACEFUL_EXIT_REQUESTED

    if GRACEFUL_EXIT_REQUESTED: 
        return

    # 对于非翻译任务，language_param_for_file就是目标语言
    language_for_data = language_param_for_file

    print(f"\n--- 开始处理: 模型={eval_model_name}, 任务={task_type}, 语言参数={language_param_for_file} ---")

    # ⭐ 修改这里：使用新的任务名称作为输出文件夹名称
    new_task_name = REVERSE_TASK_MAPPING.get(task_type, task_type)
    task_specific_output_dir = os.path.join(args.output_path, eval_model_name, new_task_name)
    os.makedirs(task_specific_output_dir, exist_ok=True)
    
    base_filename = f"{language_param_for_file}"
    output_file = os.path.join(task_specific_output_dir, f"{base_filename}_evaluation.json")
    checkpoint_file = os.path.join(task_specific_output_dir, f"{base_filename}_checkpoint.json")
    error_log_file = os.path.join(task_specific_output_dir, f"{base_filename}_errors.log")
    error_id_file = os.path.join(task_specific_output_dir, f"{base_filename}_error_ids.json")

    print(f"输出目录: {task_specific_output_dir}")  # 添加日志确认输出目录

    # State variables
    successful_results_list = []
    processed_successful_ids_set = set()

    # Resume logic
    if args.resume:
        print(f"尝试从断点恢复 (主输出文件优先): {output_file}")
        loaded_from_output = False
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    all_previous_results = json.load(f)
                
                if isinstance(all_previous_results, list):
                    for res in all_previous_results:
                        if isinstance(res, dict) and 'id' in res and is_evaluation_successful(res):
                            if res['id'] not in processed_successful_ids_set:
                                successful_results_list.append(res)
                                processed_successful_ids_set.add(res['id'])
                    
                    if successful_results_list:
                        loaded_from_output = True
                        print(f"从输出文件 {output_file} 加载了 {len(successful_results_list)} 条成功的评估结果。")
                else:
                    print(f"警告: 输出文件 {output_file} 格式不正确。尝试备份...")
                    backup_file = output_file + f".invalid.{int(time.time())}"
                    os.rename(output_file, backup_file)
            except Exception as e_load_output:
                print(f"读取或处理输出文件 {output_file} 时出错: {str(e_load_output)}. 尝试备份...")
                try: 
                    backup_file = output_file + f".corrupt.{int(time.time())}"
                    os.rename(output_file, backup_file)
                except OSError: 
                    pass
        
        if not loaded_from_output and os.path.exists(checkpoint_file):
            print(f"尝试从检查点文件恢复: {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                temp_results = checkpoint_data.get("successful_evaluation_results", [])
                temp_ids = set(checkpoint_data.get("processed_successful_ids", []))
                
                for res in temp_results:
                     if isinstance(res, dict) and 'id' in res and res['id'] in temp_ids and is_evaluation_successful(res):
                         if res['id'] not in processed_successful_ids_set:
                            successful_results_list.append(res)
                            processed_successful_ids_set.add(res['id'])
                            
                print(f"从检查点 {checkpoint_file} 加载后，总计 {len(processed_successful_ids_set)} 个已成功处理的ID。")
            except Exception as e_load_cp:
                print(f"加载检查点 {checkpoint_file} 时出错: {str(e_load_cp)}. 尝试备份并忽略此检查点。")
                try: 
                    backup_file = checkpoint_file + f".corrupt.{int(time.time())}"
                    os.rename(checkpoint_file, backup_file)
                except OSError: 
                    pass

    # Load data
    all_test_data_map = load_test_data(task_type, language_for_data, args.test_data_path)
    if not all_test_data_map:
        log_error(error_log_file, error_id_file, "测试数据加载失败", None, details=f"{task_type}/{language_for_data}")
        return

    all_model_predictions = load_model_predictions(eval_model_name, task_type, language_param_for_file, args.models_predictions_path)
    if not all_model_predictions:
        log_error(error_log_file, error_id_file, "模型预测加载失败", None, details=f"{eval_model_name}/{task_type}/{language_param_for_file}")
        return

    items_to_process = [
        item for item in all_model_predictions 
        if isinstance(item, dict) and item.get('id') and item.get('id') not in processed_successful_ids_set
    ]

    if not items_to_process:
        print(f"所有 {len(all_model_predictions)} 个样本已成功评估或无可处理样本。")
        if successful_results_list:
            save_json_safely(successful_results_list, output_file)
        if os.path.exists(checkpoint_file):
            try: 
                os.remove(checkpoint_file)
                print(f"任务完成，检查点 {checkpoint_file} 已删除。")
            except OSError as e_rm: 
                print(f"无法删除检查点 {checkpoint_file}: {e_rm}")
        return

    print(f"总共 {len(all_model_predictions)} 个预测，其中 {len(items_to_process)} 个待处理 (之前成功 {len(processed_successful_ids_set)} 个)。")

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(items_to_process):
        random.seed(42)
        items_to_process = random.sample(items_to_process, args.sample_size)
        print(f"随机抽样 {len(items_to_process)} 个样本进行评估。")
    
    current_run_newly_successful_count = 0
    current_run_failed_item_count = 0
    last_checkpoint_time = time.time()
    processed_item_counter_current_run = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures_map = {}
            
            for item_pred in items_to_process:
                if GRACEFUL_EXIT_REQUESTED: 
                    break
                
                item_id = item_pred.get('id')
                if not item_id: 
                    log_error(error_log_file, error_id_file, "预测数据缺少ID", None, details=str(item_pred))
                    continue
                
                original_data_item = all_test_data_map.get(item_id)
                if not original_data_item:
                    log_error(error_log_file, error_id_file, "找不到原始测试数据", None, [item_id], f"Task: {task_type}, Lang: {language_for_data}")
                    continue
                
                # 对于所有非翻译任务，都使用question和answer字段
                question = original_data_item.get('question', '')
                reference = original_data_item.get('answer', '')
                
                prediction = item_pred.get('answer', item_pred.get('pred', ''))
                
                # 提取subcategory
                subcategory = None
                if task_type == "text_generation" and isinstance(item_pred, dict):
                    subcategory = item_pred.get('subcategory')
                    # 添加调试信息
                    if subcategory:
                        print(f"Debug: 样本 {item_id} 提取到subcategory: {subcategory}")
                    else:
                        print(f"Debug: 样本 {item_id} 没有subcategory信息 (keys: {list(item_pred.keys())})")

                future = executor.submit(evaluate_sample, client, args.model, task_type, language_for_data, question, reference, prediction, subcategory)
                futures_map[future] = (item_id, question, reference, prediction, subcategory)

            # 使用 as_completed 但添加超时
            completed_futures = []
            try:
                for future in tqdm(concurrent.futures.as_completed(futures_map, timeout=60), total=len(futures_map), desc=f"评估 {eval_model_name}/{new_task_name}/{language_param_for_file}"):
                    completed_futures.append(future)
                    if GRACEFUL_EXIT_REQUESTED:
                        print("\n检测到退出请求，在tqdm循环中提前终止。")
                        break
            except concurrent.futures.TimeoutError:
                print("\n部分任务超时，将处理已完成的任务...")
                completed_futures = [f for f in futures_map.keys() if f.done()]
            
            # 取消未完成的任务
            if GRACEFUL_EXIT_REQUESTED:
                for f_cancel in futures_map.keys():
                    if not f_cancel.done(): 
                        f_cancel.cancel()

            # 处理已完成的任务
            for future in completed_futures:
                if future not in futures_map:
                    continue
                    
                item_id, question, reference, prediction, subcategory = futures_map[future]
                processed_item_counter_current_run += 1
                
                try:
                    if future.cancelled():
                        log_error(error_log_file, error_id_file, "任务被取消", None, [item_id], "可能由于脚本退出信号")
                        current_run_failed_item_count += 1
                        continue

                    evaluation_text = future.result(timeout=5)  # 5秒获取结果超时
                    scores = parse_scores(evaluation_text, task_type, subcategory)
                    result_dict = {
                        "id": item_id, 
                        "model": eval_model_name, 
                        "task_type": task_type,
                        "subcategory": subcategory,
                        "language_param": language_param_for_file, 
                        "language_evaluated": language_for_data,
                        "question": question, 
                        "reference": reference, 
                        "prediction": prediction,
                        "evaluation": evaluation_text, 
                        **scores
                    }

                    # 使用更严格的验证
                    is_valid, validation_msg = validate_evaluation_result(result_dict, task_type, subcategory)
                    if is_valid and is_evaluation_successful(result_dict):
                        if item_id not in processed_successful_ids_set:
                           successful_results_list.append(result_dict)
                           processed_successful_ids_set.add(item_id)
                        current_run_newly_successful_count += 1
                    else:
                        current_run_failed_item_count += 1
                        error_detail = f"验证失败: {validation_msg}" if not is_valid else "评估不成功"
                        log_error(error_log_file, error_id_file, error_detail, None, [item_id], 
                                evaluation_text[:500] if evaluation_text else "无评估文本")

                except (concurrent.futures.TimeoutError, Exception) as e_item_processing:
                    current_run_failed_item_count += 1
                    log_error(error_log_file, error_id_file, "处理样本时发生严重错误", e_item_processing, [item_id])
                    continue
                
                # Checkpoint saving logic
                time_now = time.time()
                if (processed_item_counter_current_run > 0 and processed_item_counter_current_run % args.checkpoint_interval_items == 0) or \
                   (time_now - last_checkpoint_time > args.checkpoint_interval_time):
                    if successful_results_list:
                        print(f"\n触发检查点保存 (本轮已处理 {processed_item_counter_current_run} 项, 时间自上次: {time_now - last_checkpoint_time:.0f}s)...")
                        save_json_safely({
                            "processed_successful_ids": list(processed_successful_ids_set),
                            "successful_evaluation_results": successful_results_list,
                            "timestamp": time_now
                        }, checkpoint_file)
                        last_checkpoint_time = time_now

    except KeyboardInterrupt:
        print("\n捕获到 KeyboardInterrupt，准备退出...")
        GRACEFUL_EXIT_REQUESTED = True
    except Exception as e_main_loop:
        print(f"\n主处理循环中发生未捕获异常: {str(e_main_loop)}")
        log_error(error_log_file, error_id_file, "主处理循环错误", e_main_loop)
        GRACEFUL_EXIT_REQUESTED = True
    finally:
        print("\n进入最终保存阶段...")
        
        # Always save the current state
        if successful_results_list or processed_successful_ids_set:
            print(f"保存 {len(successful_results_list)} 条成功评估结果到主输出文件...")
            save_json_safely(successful_results_list, output_file)
            
            print(f"保存最终检查点 (包含 {len(processed_successful_ids_set)} 个成功ID)...")
            save_json_safely({
                "processed_successful_ids": list(processed_successful_ids_set),
                "successful_evaluation_results": successful_results_list,
                "timestamp": time.time()
            }, checkpoint_file)
        else:
            print("没有成功的评估结果可保存。")

        # Determine if the task is fully complete
        is_fully_complete = True
        if all_model_predictions and len(processed_successful_ids_set) < len(all_model_predictions):
            is_fully_complete = False

        if is_fully_complete and not GRACEFUL_EXIT_REQUESTED and current_run_failed_item_count == 0:
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    print(f"任务已完全成功，检查点文件 {checkpoint_file} 已删除。")
                except OSError as e_rm_final:
                    print(f"警告: 无法删除已完成任务的检查点文件 {checkpoint_file}: {e_rm_final}")
        elif GRACEFUL_EXIT_REQUESTED:
            print(f"由于脚本中断，检查点 {checkpoint_file} 已保留。")
        else:
            print(f"任务未完全成功或有失败项，检查点 {checkpoint_file} 已保留。")

        print(f"--- 处理结束: 模型={eval_model_name}, 任务={new_task_name}, 语言参数={language_param_for_file} ---")
        print(f"本轮新成功评估数: {current_run_newly_successful_count}, 本轮处理失败/跳过项数: {current_run_failed_item_count}")
        print(f"总计已成功评估ID数 (包括历史): {len(processed_successful_ids_set)}")
        if os.path.exists(error_id_file) or os.path.exists(error_log_file):
             print(f"详细错误信息和ID已记录到与输出文件同目录的 _errors.log 和 _error_ids.json 文件。")

def main():
    # 1. 直接调用 parse_args() 来获取所有来自命令行的参数
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)

    try:
        client = get_client(args.api_key, args.api_base)
    except Exception as e:
        print(f"初始化OpenAI客户端失败: {e}")
        print("请检查API Key是否正确设置。")
        return

    models_to_run = args.models_to_evaluate
    if not models_to_run:
        try:
            if not os.path.exists(args.models_predictions_path):
                 print(f"错误: 模型预测路径 '{args.models_predictions_path}' 不存在。")
                 return
            models_to_run = [d for d in os.listdir(args.models_predictions_path) if os.path.isdir(os.path.join(args.models_predictions_path, d))]
            if not models_to_run:
                print(f"错误: 在 {args.models_predictions_path} 中未找到模型目录，且未通过 --models_to_evaluate 指定模型。")
                return
            print(f"将评估在 {args.models_predictions_path} 中找到的所有模型: {models_to_run}")
        except FileNotFoundError:
            print(f"错误: 模型基础路径 {args.models_predictions_path} 未找到。请通过 --models_to_evaluate 指定模型。")
            return
    
    if isinstance(models_to_run, str): 
        models_to_run = [models_to_run]

    if args.task and args.language:
        # 处理指定的单个任务
        for model_name_to_eval in models_to_run:
            if GRACEFUL_EXIT_REQUESTED: 
                break
            process_task(args, client, model_name_to_eval, args.task, args.language)
    else:
        # 处理所有生成式任务
        ethnic_languages = ["bo", "mn", "ug"]
        all_task_keys = list(TASK_TYPES.keys())

        for model_name_to_eval in models_to_run:
            if GRACEFUL_EXIT_REQUESTED: 
                break
            print(f"\n===== 开始处理模型: {model_name_to_eval} =====")
            
            for task_key in all_task_keys:
                if GRACEFUL_EXIT_REQUESTED: 
                    break
                
                # 其他任务按语言处理
                for lang_code in ethnic_languages:
                    if GRACEFUL_EXIT_REQUESTED: 
                        break
                    process_task(args, client, model_name_to_eval, task_key, lang_code)
    
    if GRACEFUL_EXIT_REQUESTED:
        print("\n评估过程因外部信号被中断。部分任务可能未完成。")
    else:
        print("\n所有评估任务处理完毕。")

# 2. 确保脚本被执行时，直接调用 main 函数
if __name__ == "__main__":
    main()