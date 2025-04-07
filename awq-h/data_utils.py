import os
import json
import random
from logger_config import logger

def load_data(data_path="data/oral_datas.json", num_samples=10):
    """
    加载用户提供的数据文件
    
    参数:
    data_path - 数据文件路径，默认为"data/oral_datas.json"
    num_samples - 要加载的样本数量
    
    返回:
    samples - 文本样本列表
    """
    samples = []
    
    try:
        if os.path.exists(data_path):
            logger.info("从 {} 加载数据".format(data_path))
            with open(data_path, 'r', encoding='utf-8') as f:
                # 根据文件扩展名处理不同格式
                if data_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        # 列表格式
                        if len(data) > 0:
                            if isinstance(data[0], str):
                                # 字符串列表
                                samples = data[:num_samples]
                            elif isinstance(data[0], dict):
                                # 字典列表，尝试提取文本
                                for item in data[:num_samples]:
                                    if 'text' in item:
                                        samples.append(item['text'])
                                    elif 'content' in item:
                                        samples.append(item['content'])
                                    else:
                                        # 取第一个值
                                        try:
                                            samples.append(next(iter(item.values())))
                                        except:
                                            pass
                    elif isinstance(data, dict):
                        # 字典格式
                        if 'data' in data:
                            samples = data['data'][:num_samples]
                        elif 'samples' in data:
                            samples = data['samples'][:num_samples]
                        elif 'texts' in data:
                            samples = data['texts'][:num_samples]
                        else:
                            # 取所有值
                            samples = list(data.values())[:num_samples]
                else:
                    # 文本文件，每行一个样本
                    samples = [line.strip() for line in f.readlines() if line.strip()][:num_samples]
        else:
            logger.error("数据文件 {} 不存在".format(data_path))
            raise FileNotFoundError("数据文件 {} 不存在".format(data_path))
    except Exception as e:
        logger.error("加载数据失败: {}".format(e))
        raise RuntimeError("无法加载数据: {}".format(e))
    
    # 确保样本为字符串类型并且非空
    samples = [str(s) for s in samples if s]
    
    # 限制样本数量
    if len(samples) > num_samples:
        samples = samples[:num_samples]
    
    logger.info("成功加载 {} 个样本".format(len(samples)))
    return samples