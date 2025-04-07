import re
import time
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, dtype="float16"):
    """加载模型并自动分配设备"""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def model_out(model, tokenizer, input_text, return_activations=False, max_length=5000):
    """
    获取模型响应（确保设备一致性）
    """
    # 确保输入数据与模型在同一设备
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    if return_activations:
        layer_activations = []

        def hook_fn(_, __, output):
            layer_activations.append(output[0].detach().cpu())

        hooks = [layer.register_forward_hook(hook_fn)
                 for layer in model.model.layers]

        with torch.no_grad():
            model(**inputs)

        for hook in hooks:
            hook.remove()

        return layer_activations
    else:
        with torch.no_grad():
            outputs = model.generate(
                max_length=max_length,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                **inputs
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

@torch.no_grad()
def get_act_scale(x):
    """获取激活分布"""
    return x.abs().view(-1, x.shape[-1]).mean(0)

def pseudo_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1):
    """伪量化张量（保持设备一致性）"""
    org_device = w.device
    org_shape = w.shape

    if q_group_size > 0:
        w = w.view(-1, q_group_size)

    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        scales = (max_val - min_val).clamp(min=1e-5) / (2 ** n_bit - 1)
        zeros = (-torch.round(min_val / scales)).clamp_(0, 2 ** n_bit - 1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scales = max_val / (2 ** (n_bit - 1) - 1)
        zeros = torch.zeros_like(scales)

    w = (torch.clamp(torch.round(w / scales) + zeros, 0, 2 ** n_bit - 1) - zeros) * scales
    return w.view(org_shape).to(org_device)

def w_quantize_func(weight, scales, n_bit=8, zero_point=True):
    """权重量化函数（设备感知）"""
    # 确保scales与weight在同一设备
    scales = scales.view(1, -1).to(weight.device)
    weight.data = pseudo_quantize_tensor(
        weight.data * scales,
        n_bit=n_bit,
        zero_point=zero_point
    )
    return weight

def judge(model, tokenizer, history_out, out):
    """评判量化结果"""
    scores = []
    logger.info(f"评估{len(history_out)}个答案")

    for idx, answer in enumerate(history_out):
        prompt = f"""
        请严格按以下要求评估（回答只需包含分数）：
        1. 评分标准（总分30分）：
           - 正确性（10分） 
           - 完整性（8分）
           - 逻辑结构（7分） 
           - 专业性（5分）
        2. 输出格式（仅一行）：
        总分=XX

        原始回答：{out}
        量化回答：{answer}
        """

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                max_length=10000,
                generation_config=GenerationConfig(max_time=30),
                **inputs
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            match = re.search(r'总分=(\d+)', response)
            score = int(match.group(1)) if match else 0
            scores.append(min(max(score, 0), 30))

        except Exception as e:
            logger.warning(f"评估失败: {e}")
            scores.append(0)

    return scores

def awq_h(model_path, judge_model, judge_tokenizer, input_text, divide_num=4, n_bit=4):
    """AWQ量化主函数"""
    model, tokenizer = load_model(model_path, dtype="float16")
    layers = model.model.layers
    # but 其实如果考虑在每过完一次 n_grid, 就考虑一下激活值然后再做下一轮的 n_grid 会不会更好一点
    layer_activations = model_out(model, tokenizer, input_text, return_activations=True)

    logger.info(f"模型共{len(layers)}层，分{divide_num}组")
    org_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_scales = []
    group_size = len(layers) // divide_num
    org_out = model_out(model, tokenizer, input_text)

    # check 一下原始输出
    logger.info("原始模型输出: {}".format(org_out))

    for group_idx in tqdm(range(divide_num), desc="量化进度"):
        start = group_idx * group_size
        end = start + group_size - 1

        x = layer_activations[end]
        x_max = get_act_scale(x)
        history_out = []
        history_scales = {}

        # 考虑渐进量化
        group_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        n_grid = 20
        check_point = 5

        for ratio_idx in range(n_grid):
            if (ratio_idx + 1) % check_point == 0:
                logger.info(f"  - 组 {group_idx + 1} 比例 {ratio_idx / n_grid:.2f}")
            model.load_state_dict(group_state)

            # if group_idx == divide_num - 1 and ratio_idx == n_grid - 1:
            #     quant_out = model_out(model, tokenizer, input_text)
            #     logger.info("最后一层的量化模型输出: {}".format(quant_out))

            ratio = ratio_idx * 1.0 / n_grid
            scales = (x_max.pow(ratio).clamp(min=1e-4) / (x_max.max() * x_max.min()).sqrt()).view(-1)
            history_scales[ratio] = scales

            for layer_idx in range(start, end + 1):
                layer = model.model.layers[layer_idx]
                for name, param in layer.named_parameters():
                    if "weight" in name and len(param.shape) > 1 and param.shape[-1] == len(scales):
                        w_quantize_func(param.data, scales, n_bit)

            history_out.append(model_out(model, tokenizer, input_text))
            torch.cuda.empty_cache()

        answer = True
        for i in range(len(history_out)-1):
            if history_out[i] !=history_out[i+1]:
                answer = False
        logger.info("量化模型的输出是否相等: {}".format(answer))

        scores = judge(judge_model, judge_tokenizer, history_out, org_out)
        best_ratio = list(history_scales.keys())[scores.index(max(scores))]
        best_scales.append(history_scales[best_ratio])

        logger.info(f"第{group_idx + 1}组评分结果:")
        for ratio, score in zip(history_scales.keys(), scores):
            logger.info(f"  - 比例 {ratio:.2f}: {score}/30")
        logger.info(f"  => 选择最佳比例: {best_ratio:.3f}")

    # 自由选择是否恢复模型状态，不需要就给它注释掉
    # model.load_state_dict(org_state)

    # 优化best_scales的输出显示
    formatted_scales = []
    for i, scale in enumerate(best_scales):
        if isinstance(scale, torch.Tensor):
            scale = scale.cpu().numpy()
        formatted_scales.append({
            "group": i + 1,
            "shape": scale.shape if hasattr(scale, 'shape') else len(scale),
            "min": float(scale.min()),
            "max": float(scale.max()),
            "mean": float(scale.mean()),
            "dtype": str(scale.dtype) if hasattr(scale, 'dtype') else type(scale).__name__
        })

    logger.info("量化结果统计:")
    for s in formatted_scales:
        logger.info(
            f"组 {s['group']}: 形状={s['shape']}, 范围=[{s['min']:.4f}, {s['max']:.4f}], 均值={s['mean']:.4f}, 类型={s['dtype']}")

    return [s.cpu().tolist() for s in best_scales]


if __name__ == "__main__":
    try:
        logger.info("🚀 开始量化流程")
        start_time = time.time()

        model_path = '/root/autodl-tmp/Qwen2.5-0.5B-Instruct'
        judge_model_path = '/root/autodl-fs/Qwen2.5-7B-Instruct'
        input_text = '什么是哈希表？它的工作原理和常见应用是什么？'

        judge_model, judge_tokenizer = load_model(judge_model_path, dtype="float16")
        result = awq_h(model_path, judge_model, judge_tokenizer, input_text, divide_num=4, n_bit=2)

        logger.info(f"✅ 量化完成！耗时: {time.time() - start_time:.1f}s")
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        raise