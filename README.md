# Embedding Demo

三种经典的深度学习模型（MLP、RNN、Transformer）的 embedding 实现演示。

## 项目简介

本项目展示了三种不同的神经网络架构如何学习词向量（word embeddings）：

- **MLP (Multilayer Perceptron)**: 最简单的前馈神经网络，通过平均池化提取特征
- **RNN (Recurrent Neural Network)**: 使用循环神经网络处理序列数据，捕捉时序信息
- **Transformer**: 基于自注意力机制的现代架构，支持并行计算

所有模型都使用相同的玩具数据集进行训练，便于对比不同架构的特性。

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+

### 安装依赖

```bash
pip install torch
```

### 运行示例

```bash
# 运行 MLP 模型
python src/emb_mlp.py

# 运行 RNN 模型
python src/emb_rnn.py

# 运行 Transformer 模型
python src/emb_transformer.py
```

## 数据集

使用一个简单的玩具数据集，包含 7 个短句：

```
- cat love eat
- dog love eat
- cat love run
- dog love run
- cat love fish
- dog love meat
- dog love ball
```

**任务**: 给定前两个词，预测第三个词

**词汇表**: 8 个单词 (cat, dog, love, eat, fish, meat, ball, run)

## 模型架构

### 1. MLP (`src/emb_mlp.py`)

```
输入 (2 个词) → Embedding → 平均池化 → FC1 (ReLU) → FC2 → 输出
```

- 词嵌入维度：8
- 隐藏层维度：16
- 特点：结构简单，不考虑词序信息

### 2. RNN (`src/emb_rnn.py`)

```
输入 (序列) → Embedding → RNN → 最后隐状态 → FC → 输出
```

- 词嵌入维度：8
- 隐藏层维度：16
- 特点：捕捉序列的时序信息

### 3. Transformer (`src/emb_transformer.py`)

```
输入 (序列) → Embedding + 位置编码 → Transformer Encoder → 平均池化 → FC → 输出
```

- 词嵌入维度：8
- 注意力头数：2
- Transformer 层数：2
- 前馈网络维度：16
- 特点：自注意力机制，支持并行计算

## 输出示例

训练完成后，每个模型会输出训练后的词向量：

```
Epoch 100, loss = 1.2345
Epoch 200, loss = 0.8765
Epoch 300, loss = 0.5432
Epoch 400, loss = 0.3210
Epoch 500, loss = 0.1234

词向量 (Embedding 层):
cat: [0.12, -0.34, 0.56, ...]
dog: [0.23, -0.45, 0.67, ...]
...
```

## 项目结构

```
embedding_demo/
├── src/
│   ├── emb_mlp.py          # MLP 模型实现
│   ├── emb_rnn.py          # RNN 模型实现
│   └── emb_transformer.py  # Transformer 模型实现
├── .gitignore
└── README.md
```

## 学习建议

1. **入门**: 从 `emb_mlp.py` 开始，理解最基本的 embedding 训练流程
2. **进阶**: 运行 `emb_rnn.py`，观察序列建模如何影响词向量学习
3. **深入**: 尝试 `emb_transformer.py`，了解现代 Transformer 架构的 embedding 方法

## License

MIT License

