# 三种 Embedding 学习模型对比：MLP vs RNN vs Transformer

> 通过一个极小的玩具数据集（8 个词，7 个句子），直观对比三种神经网络架构如何学习词向量。
> **重点**：每个模型的核心数学计算都附有具体数值例子，让你"看到"内部发生了什么。

## 背景：我们要解决什么任务？

**词汇表：**
- cat(0), dog(1), love(2), eat(3), fish(4), meat(5), ball(6), run(7)

**训练数据（7 个短句，每个句子三个词）：**

```
cat love eat
dog love eat
cat love run
dog love run
cat love fish
dog love meat
dog love ball
```

构造 **7 个样本**，每个样本是 `(前两个词，第三个词)`，例如：

```
[cat, love] -> eat
[dog, love] -> eat
...
```

**任务**：给定前两个词，预测第三个词。模型通过这个任务学会每个词的向量表示（embedding）。

## 模型一：MLP（多层感知机）—— 最简单的"平均池化"版本

### 结构

1. 查表得到每个词的向量（`embed_dim=8`）。
2. 对两个词的向量**求平均**（平均池化）。
3. 将平均后的向量送入两层全连接网络，输出词表大小的概率分布。

### 核心数学：平均池化

假设两个词 `cat` 和 `love` 的向量（简化 `embed_dim=4`）：

```
cat = [0.5, 1.2, -0.3, 0.8]
love = [0.1, 0.6, 1.0, -0.4]
```

平均池化就是逐维度求平均值：

```
pooled = [(0.5+0.1)/2, (1.2+0.6)/2, (-0.3+1.0)/2, (0.8-0.4)/2]
       = [0.3, 0.9, 0.35, 0.2]
```

这个 `pooled` 向量去掉了词序信息（`[cat,love]` 和 `[love,cat]` 结果一样）。

## 模型二：RNN（循环神经网络）—— 保留顺序

### 结构

1. 查表得到每个词的向量。
2. 按顺序送入 RNN 单元，在每一步更新隐状态 `h`。
3. 取最后一个隐状态，通过全连接层输出预测。

### 核心数学：RNN 隐状态更新

RNN 单元使用 `tanh` 激活函数，公式：

```
h_t = tanh( W_h · h_{t-1} + W_x · x_t + b )
```

- `h_t` 是时刻 `t` 的隐状态（这里 `hidden_dim=4`）
- `x_t` 是当前词的向量（`embed_dim=4`）
- `W_h`（4×4）、`W_x`（4×4）、`b` 是训练的参数

**数值例子**（假设已经训练了一部分，参数已知）：

```
初始 h_0 = [0,0,0,0]

处理 cat (x0 = [0.5,1.2,-0.3,0.8])：
h_1 = tanh( W_h·[0,0,0,0] + W_x·x0 + b ) = tanh( some vector )
假设计算后 h_1 = [0.2, 0.8, -0.1, 0.4]

处理 love (x1 = [0.1,0.6,1.0,-0.4])：
h_2 = tanh( W_h·h_1 + W_x·x1 + b ) = 假设得到 [0.5, 0.3, 0.7, -0.2]
```

最后 `h_2` 就是融合了两个词**且保留顺序**的表示。

## 模型三：Transformer（自注意力）—— 并行且全局交互

### 结构

1. 查表得到每个词的向量，加上**位置编码**（因为自注意力本身不关心顺序）。
2. 经过多层 Transformer Encoder，每层内部有多头自注意力和前馈网络。
3. 对输出序列的平均（或取最后一个位置）送入全连接层。

### 核心数学：缩放点积自注意力（单头，简化）

输入两个词（`cat` 和 `love`），加上位置编码后的向量（假设训练好的位置编码）：

```
x0 = [0.6, 1.4, 0.0, 1.2]  # cat + pos0
x1 = [0.6, 1.2, 1.7, 0.4]  # love + pos1
```

为简化，假设 `W_Q = W_K = W_V = I`（恒等矩阵），所以：

```
q0 = x0, k0 = x0, v0 = x0
q1 = x1, k1 = x1, v1 = x1
```

**计算注意力分数**（点积，然后缩放 `√d_k = √4 = 2`）：

```
score00 = q0·k0 = 0.6×0.6 + 1.4×1.4 + 0×0 + 1.2×1.2 = 3.76 → 3.76/2 = 1.88
score01 = q0·k1 = 0.6×0.6 + 1.4×1.2 + 0×1.7 + 1.2×0.4 = 2.52 → 2.52/2 = 1.26
score10 = q1·k0 = 1.26
score11 = q1·k1 = 0.36+1.44+2.89+0.16 = 4.85 → 4.85/2 = 2.425
```

**Softmax** 得到注意力权重（对每行）：

- 行 0：`exp(1.88)=6.55, exp(1.26)=3.52` → 权重 `[0.65, 0.35]`
- 行 1：`exp(1.26)=3.52, exp(2.425)=11.30` → 权重 `[0.24, 0.76]`

**加权求和 Value** 得到每个位置的输出：

```
z0 = 0.65×v0 + 0.35×v1
   = 0.65×[0.6,1.4,0,1.2] + 0.35×[0.6,1.2,1.7,0.4]
   = [0.60, 1.33, 0.595, 0.92]

z1 = 0.24×v0 + 0.76×v1
   = [0.60, 1.248, 1.292, 0.592]
```

可以看到，自注意力让每个词的新表示都**融入了对方的信息**，而且权重由内容动态决定（不像平均池化固定为 0.5）。

## 三者的核心差异总结

| 模型 | 信息融合方式 | 是否保留顺序 | 能否并行 | 数学本质 |
|------|-------------|--------------|----------|----------|
| **MLP** | 固定权重的平均 | ❌ | ✅ | 加权平均 (权重 = 1/长度) |
| **RNN** | 递归更新隐状态 | ✅ | ❌ | `h_t = tanh(W_h h_{t-1} + W_x x_t + b)` |
| **Transformer** | 动态权重的加权和 | ✅ (通过位置编码) | ✅ | `Attention(Q,K,V) = softmax(QK^T/√d)V` |

## 代码层面：共性与差异

### 共性：训练流程完全一致

三个模型的训练代码**一模一样**，都遵循 PyTorch 的标准训练范式：

```python
# 前向传播：输入数据 → 模型 → 预测结果
out = model(inputs)

# 计算损失：预测值与真实标签的差距
loss = criterion(out, targets)

# 梯度清零：清除上一步的梯度，避免累积
optimizer.zero_grad()

# 反向传播：计算损失对每个参数的梯度
loss.backward()

# 参数更新：根据梯度调整参数，让损失下降
optimizer.step()
```

**这意味着**：你只需要修改模型定义（`nn.Module` 子类），训练逻辑完全不需要改动！

### 差异：forward() 方法的实现

#### 1. MLP - 平均池化（不关心位置）

```python
def forward(self, x):
    emb = self.emb(x)           # (batch, seq_len=2, embed_dim)
    pooled = emb.mean(dim=1)    # 关键：对序列维度求平均 → (batch, embed_dim)
    x1 = self.fc1(pooled)
    x = self.relu(x1)
    x2 = self.fc2(x)
    return x2
```

**特点**：
- `emb.mean(dim=1)` 把两个词的向量**简单平均**
- `[cat, love]` 和 `[love, cat]` 得到完全相同的 `pooled`
- 计算完全并行，一次性处理所有词

#### 2. RNN - 顺序处理（关心位置，但效率低）

```python
def forward(self, x):
    emb = self.embedding(x)              # (batch, 2, embed_dim)
    output, h_n = self.rnn(emb)          # 关键：RNN 处理
    last_hidden = output[:, -1, :]       # 取最后一个时间步的隐状态
    logits = self.out(last_hidden)
    return logits
```

**为什么 RNN 比 Transformer 慢？**

```python
output, h_n = self.rnn(emb)
```

这行代码内部是**串行**的：
- `h_1 = tanh(W_h · h_0 + W_x · x_0)` ← 必须先算出 `h_1`
- `h_2 = tanh(W_h · h_1 + W_x · x_1)` ← 依赖 `h_1`，无法并行

**序列越长，RNN 的串行瓶颈越明显**。2 个词时差异不大，但如果是 100 个词的序列，RNN 必须一步步计算 100 次，而 Transformer 可以一次性处理。

#### 3. Transformer - 并行处理（关心位置，效率高）

```python
def forward(self, x):
    batch_size, seq_len = x.shape
    emb = self.embedding(x)              # (batch, 2, embed_dim)

    # 位置编码：让模型知道每个词的位置
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos_emb = self.pos_embed(positions)
    x = emb + pos_emb                    # 词向量 + 位置向量

    x = self.transformer(x)              # 关键：Transformer 并行处理所有位置
    x = x.mean(dim=1)                    # 平均池化
    logits = self.out(x)
    return logits
```

**为什么 Transformer 快？**

自注意力机制一次性计算所有位置之间的关系：

```python
# 注意力矩阵 (2×2) 一次性算出所有词对的相似度
# [cat 和 cat, cat 和 love]
# [love 和 cat, love 和 love]
Attention(Q, K, V) = softmax(QK^T/√d) V
```

**位置编码的作用**：
- 自注意力本身**无法区分** `[cat, love]` 和 `[love, cat]`
- 通过 `emb + pos_embed`，给每个位置加上独特的"位置标签"
- 模型就能学会"第一个词"和"第二个词"的不同

### 效率对比

| 操作 | MLP | RNN | Transformer |
|------|-----|-----|-------------|
| **前向传播** | ✅ 完全并行 | ❌ 串行（时间步依赖） | ✅ 完全并行 |
| **序列长度=2** | 最快 | 较快 | 快（但参数多） |
| **序列长度=100** | 快（但丢失顺序） | 很慢（100 步串行） | 快（仍然并行） |
| **参数量** | 最少 | 中等 | 最多 |

### 为什么选择这些模型？

- **MLP**：快速基线，适合对顺序不敏感的任务
- **RNN**：理论上有顺序，但长序列训练慢，梯度容易消失/爆炸
- **Transformer**：现代主流，并行效率高，长距离依赖建模能力强


## 为什么在这个小任务上 Transformer loss 最低？

用你的实际运行结果（500 轮后）：

- MLP loss ≈ 1.49
- RNN loss ≈ 1.46
- Transformer loss ≈ 1.38

**原因：**

1. Transformer 的参数量最大，拟合能力最强。
2. 自注意力可以直接建模两个输入词之间的交互，而 RNN 需要顺序传递信息（虽然这里只有两步，差异不大）。
3. 平均池化完全丢失顺序，而顺序信息在这个任务中有用（因为 `[cat,love]` 与 `[love,cat]` 不同）。

**但注意**：数据只有 7 条，模型越大越容易过拟合。如果增加数据量，Transformer 的优势会更明显。

## 如何运行？

克隆仓库后，直接运行三个脚本：

```bash
python emb_mlp.py
python emb_rnn.py
python emb_transformer.py
```

每个脚本会自动训练并打印最终的词向量。你可以观察：

1. 训练 loss 的下降曲线。
2. 不同模型学到的 `cat` 和 `dog` 向量是否接近（理论上应该相似，因为它们的上下文都是 `love` + 某个动作/食物）。

## 扩展到更真实的场景

- **增大词汇表和语料**：可以下载 torchtext 里的 AG_NEWS 或 WikiText-2。
- **改变任务**：从"预测下一个词"变成"随机 mask 一个词"（MLM），更接近 BERT。
- **可视化**：用 t-SNE 将 8 维向量降到 2 维，画出词向量的分布。

## 参考资料

- [Word2Vec 论文 (CBOW 和 Skip-gram)](https://arxiv.org/abs/1301.3781)
- [LSTM 论文 (RNN 的改进)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [PyTorch 官方文档：nn.Embedding, nn.RNN, nn.Transformer](https://pytorch.org/docs/stable/nn.html)
