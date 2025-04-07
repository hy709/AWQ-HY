# AWQ-H 量化工具

这是一个基于 AWQ (Activation-aware Weight Quantization) 改进的量化工具,主要用于大语言模型的量化压缩。

## 功能特点

- 支持模型校准(calibration)、量化(quantization)和评估(evaluation)三种模式
- 基于激活值感知的权重量化策略
- 支持自定义量化比特数(n_bit)和分块数量(divide_num)
- 提供详细的量化评估指标(MSE、余弦相似度等)
- 支持自定义校准数据集

## 使用方法

### 安装依赖

```bash
pip install torch>=2.0.0 transformers>=4.30.0 tqdm matplotlib
```

### 校准模式

在校准模式下，工具会计算模型权重的最佳缩放因子并保存，但不执行量化:

```bash
python main.py --model_path /path/to/model --mode calibrate --divide_num 4 --output_dir ./scales
```

### 量化模式

量化模式下会对模型进行量化并保存量化后的模型:

```bash
python main.py --model_path /path/to/model --mode quantize --n_bit 8 --divide_num 4
```

也可以使用预先计算好的缩放因子进行量化:

```bash
python main.py --model_path /path/to/model --mode quantize --scales_path ./scales/best_scales.pt
```

### 评估模式

评估模式用于量化后与原始模型的性能对比:

```bash
python main.py --model_path /path/to/model --mode evaluate --scales_path ./scales/best_scales.pt
```

## 参数说明

### 基本参数

- `--model_path`: 待量化模型的路径（**必需**）
- `--output_dir`: 输出目录，默认为"./quantized_model"
- `--input_text`: 用于校准的输入文本，优先级高于数据集
- `--dataset_path`: 校准数据集路径
- `--batch_size`: 校准样本数量，默认为5

### 量化参数

- `--n_bit`: 量化位数，默认为8位
- `--divide_num`: 模型分组数量，默认为1
- `--zero_point`: 添加此参数启用零点偏移量化
- `--n_grid`: 缩放因子搜索网格数量，默认为20

### 模式选择

- `--mode`: 运行模式
  - `calibrate`: 仅计算并保存缩放因子
  - `quantize`: 执行量化并保存模型（默认）
  - `evaluate`: 评估量化效果

### 高级选项

- `--scales_path`: 预计算缩放因子的路径
- `--dtype`: 模型加载精度，可选"float16"（默认）或"bfloat16"
- `--device`: 指定设备，如"cuda:0"，默认自动选择

## 项目结构

- `main.py`: 主程序入口
- `data_utils.py`: 校准数据处理
- `model_utils.py`: 模型相关工具函数
- `quant_utils.py`: 量化函数库
- `quantization.py`: AWQ-H核心算法实现
- `judge.py`: 量化质量评估工具
- `logger_config.py`: 日志配置

## 算法原理

AWQ-H算法核心思想是在保持模型性能的前提下进行权重量化，其主要步骤包括：

1. **分层分组**: 将模型层次结构分成多个组，便于分别处理
2. **激活值感知**: 利用模型的激活值来指导权重量化过程
3. **缩放因子搜索**: 对每组层搜索最佳缩放因子
4. **自评估机制**: 使用模型自身评估量化效果，选择最佳参数

与标准AWQ相比，AWQ-H采用了层次化分组处理，可以针对不同层组应用不同的量化策略，提高整体量化效果。

## 使用示例

### 量化Qwen 0.5B模型为8位精度:

```bash
python main.py --model_path Qwen/Qwen-0.5B --mode quantize --n_bit 8 --divide_num 6
```

### 带有自定义校准文本的量化:

```bash
python main.py --model_path /path/to/model --mode quantize --input_text "这是一段用于校准的文本，包含了各种语言结构和词汇以便进行更好的量化。"
```

## 引用

如果您使用了此工具，请考虑引用原始AWQ论文:

```
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Gan, Xingyu and Jia, Xiuyu and Han, Song},
  journal={arXiv preprint arXiv:2306.00978},
  year={2023}
}
```