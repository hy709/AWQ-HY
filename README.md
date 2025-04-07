# AWQ-H 量化工具

这是一个基于 AWQ (Activation-aware Weight Quantization) 改进的量化工具,主要用于大语言模型的量化压缩。

## 功能特点

- 基于激活值感知的权重量化策略
- 支持自定义量化比特数(n_bit)和分块数量(divide_num)
- 支持自定义校准数据集

## 使用方法

### 安装依赖

```bash
pip install torch>=2.0.0 transformers>=4.30.0 
```

## 待定
demo: 基于 awq-h 量化的 qwen2.5-0.5b的实例
awq-h: 预计适配更多的模型以及支持自定义的数据集

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
