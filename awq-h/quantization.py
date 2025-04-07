import torch
from tqdm import tqdm
from model_utils import leaf2layer, get_act_scale
from quant_utils import pseudo_quantize_tensor, w_quantize_func
from data_utils import load_calibration_data
from Judge import judge
from logger_config import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def awq_h(model, input_text, divide_num=1):
    """
    具体算法-awq_h
    """
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 获取原始输出
    with torch.no_grad():
        org_out = model(**inputs).logits

    # 先计算分开的份数
    layer_list = leaf2layer(model)
    grouped_layers = []
    n = len(layer_list) // divide_num  # 每组的大小
    for i in range(0, len(layer_list), n):
        # 计算每组的和
        group_sum = sum(layer_list[i:i + n])
        grouped_layers.append(group_sum)

    logger.info("模型共有{}层，分为{}组".format(len(layer_list), len(grouped_layers)))

    # 保存原始模型状态
    org_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    best_scales = []

    # 获取所有有参数的模块列表
    layer_modules = []
    for name, module in model.named_modules():
        if any(isinstance(p, torch.nn.Parameter) for p in module.parameters(recurse=False)):
            layer_modules.append((name, module))

    # 遍历每个组（跳过第一组）
    for group_idx in range(1, len(grouped_layers)):
        # 计算当前组的层范围
        start_layer = sum(grouped_layers[:group_idx])
        end_layer = sum(grouped_layers[:group_idx + 1])

        logger.info("处理第{}组，层范围：{}-{}".format(group_idx, start_layer, end_layer - 1))

        # 前向传播以获取激活值
        with torch.no_grad():
            outputs = model(**inputs)
            x = outputs.logits  # 使用模型输出作为激活值

        # 获取激活值的分布
        x_max = get_act_scale(x)

        n_grid = 20
        history_out = []
        history_ratio_scales = {}

        # 遍历不同的缩放比例
        for ratio_idx in range(n_grid):
            ratio = ratio_idx * 1.0 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            history_ratio_scales[ratio] = scales

            # 对当前组的层进行量化
            for layer_idx in range(start_layer, end_layer):
                if layer_idx >= len(layer_modules):
                    continue

                module_name, module = layer_modules[layer_idx]

                # 应用缩放因子到模块的所有权重参数
                for name, param in module.named_parameters():
                    if "weight" in name and len(param.shape) > 1:
                        scale_view = scales.view(1, -1).to(param.device)
                        param.data = param.data * scale_view
                        param.data = w_quantize_func(param.data)
                        param.data = param.data / scale_view

            # 获取量化后的输出
            with torch.no_grad():
                out = model(**inputs).logits
            history_out.append(out)

            # 恢复原始状态
            model.load_state_dict(org_state)

        # 选择最佳缩放因子
        best_idx = judge(model.config._name_or_path, history_out, org_out)
        if isinstance(best_idx, list):
            # 如果judge返回的是列表，取最高分的索引
            best_idx = best_idx.index(max(best_idx))
        else:
            # judge可能直接返回索引，确保它是整数类型
            best_idx = int(best_idx) if isinstance(best_idx, (int, float)) else 0

        ratio = best_idx * 1.0 / n_grid
        best_scale = history_ratio_scales[ratio]

        best_scales.append(best_scale.view(-1))
        logger.info("第{}组最佳比例: {:.2f}".format(group_idx, ratio))

    return best_scales

def apply_awq_h_quantization(model, best_scales, divide_num=1, n_bit=8, zero_point=True):
    """
    应用自定义量化算法到模型的所有参数
    """
    # 需要重新计算layer_list和grouped_layers
    layer_list = leaf2layer(model)
    grouped_layers = []
    n = len(layer_list) // divide_num
    for i in range(0, len(layer_list), n):
        group_sum = sum(layer_list[i:i + n])
        grouped_layers.append(group_sum)

    # 获取所有有参数的模块列表
    layer_modules = []
    for name, module in model.named_modules():
        if any(isinstance(p, torch.nn.Parameter) for p in module.parameters(recurse=False)):
            layer_modules.append((name, module))

    # 应用最佳缩放因子
    for group_idx, scale in enumerate(best_scales):
        if scale is None:
            continue

        # 计算正确的层范围，避免索引错误
        start_layer = sum(grouped_layers[:group_idx + 1]) if group_idx + 1 < len(grouped_layers) else 0
        end_layer = sum(grouped_layers[:group_idx + 2]) if group_idx + 2 < len(grouped_layers) else len(layer_modules)

        logger.info("应用量化到第{}组，层范围：{}-{}".format(group_idx + 1, start_layer, end_layer - 1))

        for layer_idx in range(start_layer, end_layer):
            if layer_idx >= len(layer_modules):
                continue

            module_name, module = layer_modules[layer_idx]

            # 应用缩放因子到模块的所有权重参数
            for name, param in module.named_parameters():
                if "weight" in name and len(param.shape) > 1:
                    scale_view = scale.view(1, -1).to(param.device)
                    param.data = param.data * scale_view
                    # 应用量化函数
                    param.data = w_quantize_func(param.data, n_bit=n_bit, zero_point=zero_point)
                    param.data = param.data / scale_view

    return model