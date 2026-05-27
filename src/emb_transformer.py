import torch
import torch.nn as nn
import torch.optim as optim

# 词汇表
vocab = {'cat': 0, 'dog': 1, 'love': 2, 'eat': 3, 'fish': 4, 'meat': 5, 'ball': 6, 'run': 7}
vocab_size = len(vocab)

# 原始短句
sentences = [
    'cat love eat', 'dog love eat',
    'cat love run', 'dog love run',
    'cat love fish', 'dog love meat', 'dog love ball'
]

# 构造训练数据：输入前两个词，目标第三个词
train_data = []
for s in sentences:
    words = s.split()
    indices = [vocab[w] for w in words]
    train_data.append((indices[:2], indices[2]))

inputs = torch.tensor([d[0] for d in train_data], dtype=torch.long)  # (7, 2)
targets = torch.tensor([d[1] for d in train_data], dtype=torch.long)  # (7,)


# ------------------ Transformer 模型 ------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 位置编码（因为输入长度为2，我们直接学习位置向量）
        self.pos_embed = nn.Embedding(2, embed_dim)

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层：将 Transformer 的输出平均后映射到词汇表大小
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len=2)
        batch_size, seq_len = x.shape

        # 词嵌入 + 位置嵌入
        emb = self.embedding(x)  # (batch, 2, embed_dim)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)  # (batch, 2, embed_dim)
        x = emb + pos_emb  # (batch, 2, embed_dim)

        # Transformer 编码
        x = self.transformer(x)  # (batch, 2, embed_dim)

        # 对两个位置的输出求平均（也可以用只取最后一个位置）
        x = x.mean(dim=1)  # (batch, embed_dim)

        # 预测下一个词
        logits = self.out(x)  # (batch, vocab_size)
        return logits


def train():
    embed_dim = 8  # 必须能被 nhead 整除（nhead=2 时 OK）
    nhead = 2
    num_layers = 2
    hidden_dim = 16  # Transformer 前馈网络维度

    model = SimpleTransformer(vocab_size, embed_dim, nhead, num_layers, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(500):
        out = model(inputs)
        loss = criterion(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:3d}, loss = {loss.item():.4f}")

    # 打印训练后的词向量
    print("\n训练后的词向量 (Embedding 层):")
    for word, idx in vocab.items():
        vec = model.embedding(torch.tensor(idx)).detach().numpy()
        print(f"{word}: {vec}")


if __name__ == "__main__":
    train()