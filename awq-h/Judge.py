import re
import torch
from transformers import AutoTokenizer
from model_utils import load_model
from logger_config import logger

def judge(model_path, history_out, out):
    """
    评判函数 - 为每个量化后的回答打分
    """
    n_grid = len(history_out)
    model, tokenizer = load_model(model_path)

    scores = []
    logger.info("开始逐个评估答案，共{}个答案需要评估".format(n_grid))

    # 逐个评估每个答案
    for idx, answer in enumerate(history_out):
        # 构建针对单个答案的评估提示
        prompt = (
            "尊敬的评判员：\n"
            "请依据以下标准，对量化模型的回答进行客观评分。\n\n"
            "原始模型回答：{}\n\n"
            "量化模型回答（第{}项）：{}\n\n"
            "评分标准（满分50分）：\n"
            "1. 正确性（20分）：回答内容是否与原始回答一致，且准确无误。正确性评分仅接受0分或20分。\n"
            "2. 完整性（8分）：回答是否完整地保留了原始回答中的关键信息，无重要遗漏。\n"
            "3. 逻辑结构（7分）：回答的论述逻辑和结构是否连贯一致，思路是否清晰。\n"
            "4. 语言流畅性（5分）：回答的语句是否自然流畅，是否存在不自然的断句或表达。\n"
            "5. 风格一致性（5分）：回答的语气、措辞和表达方式是否与原始回答保持一致。\n"
            "6. 专业性（5分）：回答中专业术语的使用是否恰当，领域知识是否准确传达。\n\n"
            "请对上述各项进行评分，并给出总分。同时，提供简短的评分理由。\n"
            "评分格式要求如下：\n"
            "总分=XX\n"
            "- 正确性: XX/20\n"
            "- 完整性: XX/8\n"
            "- 逻辑结构: XX/7\n"
            "- 语言流畅性: XX/5\n"
            "- 风格一致性: XX/5\n"
            "- 专业性: XX/5\n"
        ).format(out, idx + 1, answer)

        # 对提示进行编码并生成模型输出
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_length=500)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 解析评分结果
        pattern = re.compile(r'总分=(\d+)')
        match = pattern.search(response_text)

        # 如果没有找到匹配的格式，尝试其他可能的格式
        if not match:
            pattern = re.compile(r'总分[:：]\s*(\d+)')
            match = pattern.search(response_text)

        if not match:
            pattern = re.compile(r'(\d+)\s*/\s*50')
            match = pattern.search(response_text)

        if not match:
            # 更宽松的模式：查找任何看起来像是50分制评分的部分
            pattern = re.compile(r'总分.*?(\d+).*?(分|/50)')
            match = pattern.search(response_text)

        if match:
            try:
                score = int(match.group(1))
                # 确保分数在合理范围内
                score = max(0, min(score, 50))
                scores.append(score)
                logger.info("评估完成：答案{}的分数为{}".format(idx + 1, score))
            except (ValueError, IndexError):
                logger.warning("无法解析分数，赋予默认分数0")
                scores.append(0)
        else:
            logger.warning("无法从回复中提取分数: {}".format(response_text))
            # 如果无法解析，赋予0分
            scores.append(0)

        # 记录详细评分结果(可选)
        logger.debug("答案{}的评分详情: {}".format(idx + 1, response_text))

    logger.info("所有答案评估完成，评分结果: {}".format(scores))
    return scores