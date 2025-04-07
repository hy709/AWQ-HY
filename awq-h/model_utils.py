import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from logger_config import logger

def load_model(model_path, dtype="float16"):
    """
    加载模型
    """
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # 直接加载模型配置和权重
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto"
    )

    model.eval()  # 设置为评估模式

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def leaf2layer(model):
    """
    将叶子层转换为 layer 层
    """
    pattern = re.compile(r'\d+')

    layer_count = {}

    for name, param in model.named_parameters():
        match = pattern.findall(name)
        if match:
            layer_number = int(match[0])  # 将匹配的数字字符串转换为整数
            if layer_number in layer_count:
                layer_count[layer_number] += 1
            else:
                layer_count[layer_number] = 1

    # 将结果转换为列表，按顺序排列
    max_layer = max(layer_count.keys()) if layer_count else -1
    result = [layer_count.get(i, 0) for i in range(max_layer + 1)]

    return result

@torch.no_grad()
def get_act_scale(x):
    """
    获取激活分布
    """
    return x.abs().view(-1, x.shape[-1]).mean(0)