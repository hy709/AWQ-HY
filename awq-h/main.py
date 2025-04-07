import os
import argparse
import json
import torch
from model_utils import load_model
from quantization import awq_h, apply_awq_h_quantization
from data_utils import load_calibration_data
from logger_config import logger

def main():
    """
    主函数，处理命令行参数并执行相应功能
    """
    parser = argparse.ArgumentParser(description="AWQ-H模型量化工具")

    # 基本参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./quantized_model", help="量化后模型保存路径")
    parser.add_argument("--input_text", type=str, default=None, help="用于激活模型的输入文本，优先于数据集")
    parser.add_argument("--dataset_path", type=str, default=None, help="校准数据集路径")
    parser.add_argument("--batch_size", type=int, default=5, help="校准样本数量")

    # 量化参数
    parser.add_argument("--n_bit", type=int, default=8, help="量化位数")
    parser.add_argument("--divide_num", type=int, default=1, help="将模型分成几组进行量化")
    parser.add_argument("--zero_point", action="store_true", help="是否使用零点偏移")
    parser.add_argument("--n_grid", type=int, default=20, help="scale网格搜索数量")

    # 功能选择
    parser.add_argument("--mode", choices=["calibrate", "quantize", "evaluate"], default="quantize",
                        help="模式选择: calibrate(仅计算scale), quantize(量化并保存), evaluate(评估量化效果)")

    # 高级选项
    parser.add_argument("--scales_path", type=str, default=None, help="预先计算好的scales路径，用于直接量化")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16", help="模型加载精度")
    parser.add_argument("--device", type=str, default=None, help="指定设备，例如'cuda:0'，默认自动选择")

    args = parser.parse_args()

    # 创建输出目录
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"已创建输出目录: {args.output_dir}")
    except Exception as e:
        logger.error(f"创建输出目录失败: {e}")
        raise RuntimeError(f"无法创建输出目录 {args.output_dir}: {e}")

    # 加载模型
    logger.info("正在加载模型: {}".format(args.model_path))
    model, tokenizer = load_model(args.model_path, args.dtype)

    if args.device:
        model = model.to(args.device)

    # 根据模式执行不同操作
    if args.mode == "calibrate" or (args.mode == "quantize" and args.scales_path is None):
        logger.info("开始校准计算scales")
        best_scales = awq_h(
            model,
            input_text=args.input_text if args.input_text else "这是一个测试输入，用于计算模型的量化缩放因子。",
            divide_num=args.divide_num
        )

        # 保存scales
        scales_path = os.path.join(args.output_dir, "best_scales.pt")
        torch.save(best_scales, scales_path)
        logger.info("Scales已保存到: {}".format(scales_path))

        if args.mode == "calibrate":
            return

    if args.mode == "quantize" or args.mode == "evaluate":
        # 如果提供了预先计算的scales路径，加载它
        if args.scales_path:
            logger.info("从 {} 加载scales".format(args.scales_path))
            best_scales = torch.load(args.scales_path)

        # 应用量化
        logger.info("应用AWQ-H量化到模型")
        quantized_model = apply_awq_h_quantization(
            model,
            best_scales,
            divide_num=args.divide_num,
            n_bit=args.n_bit,
            zero_point=args.zero_point
        )

        if args.mode == "evaluate":
            # 执行评估
            logger.info("评估量化效果")
            test_input = args.input_text if args.input_text else "这是一个测试输入，用于评估量化效果。"
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

            with torch.no_grad():
                original_output = model(**inputs).logits
                quantized_output = quantized_model(**inputs).logits

            # 简单计算MSE和余弦相似度
            mse = torch.mean((original_output - quantized_output) ** 2)
            cosine_sim = torch.nn.functional.cosine_similarity(
                original_output.flatten(), quantized_output.flatten(), dim=0
            )
            metrics = {"mse": mse.item(), "cosine_similarity": cosine_sim.item()}

            logger.info("量化评估结果: {}".format(metrics))

            # 保存评估结果
            with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        if args.mode == "quantize":
            # 保存量化后的模型
            logger.info("保存量化后的模型到: {}".format(args.output_dir))
            quantized_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # 保存量化配置
            config = {
                "quantization": "awq-h",
                "n_bit": args.n_bit,
                "divide_num": args.divide_num,
                "zero_point": args.zero_point,
                "original_model": args.model_path
            }

            with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            logger.info("量化完成!")

if __name__ == "__main__":
    main()