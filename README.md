# DistilBERT Fine-tuning on IMDb (MLM)

使用 Hugging Face Transformers 对 DistilBERT 进行掩码语言模型（MLM）微调，基于 IMDb 电影评论数据集。

## 功能
- 标准 MLM 预训练
- 全词掩码 WWM
- 混合精度训练
- Accelerate 分布式训练
- 困惑度评估
- 模型推理

## 数据集
IMDb 电影评论数据集（25000 训练 + 25000 测试 + 50000 无监督）

## 环境安装
```bash
pip install -r requirements.txt
