import torch
from torch import optim

# 词汇表
vocab = {
    'cat': 0,
    'dog': 1,
    'love': 2,
    'eat': 3,
    'fish': 4,
    'meat': 5,
    'ball': 6,
    'run': 7
}
vocab_size = len(vocab)

# 原始短句
sentences = [
    'cat love eat',
    'dog love eat',
    'cat love run',
    'dog love run',
    'cat love fish',
    'dog love meat',
    'dog love ball',
]

train_data = []
for s in sentences:
    words = s.split()               # ['cat', 'love', 'eat']
    indices = [vocab[w] for w in words]  # [0,2,3]
    train_data.append((indices[:2], indices[2]))
print(train_data)

inputs = torch.tensor([d[0] for d in train_data], dtype=torch.long)
targets = torch.tensor([d[1] for d in train_data], dtype=torch.long)


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, embed_dim)  # one-hot维度 -> emb dim维度, 高稚到低稚
        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)  # 第一层: emb -> hidden
        self.fc2 = torch.nn.Linear(hidden_dim, vocab_size)  # 第二层: hidden -> one-hot output
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        emb = self.emb(x)
        pooled = emb.mean(dim=1)
        x1 = self.fc1(pooled)
        xx = self.relu(x1)
        x2 = self.fc2(xx)
        return x2

def train():
    embed_dim = 8
    hidden_dim = 16

    model = TinyModel(vocab_size, embed_dim, hidden_dim)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(500):
        out = model(inputs)     # 前向传播：输入数据 → 模型预测
        loss = criterion(out, targets)  # 计算损失：预测值与真实标签的差距
        optimizer.zero_grad() # 梯度清零：清除上一步的梯度，避免累积
        loss.backward()     # 反向传播：计算当前损失对每个参数的梯度
        optimizer.step()     # 参数更新：根据梯度调整参数，让损失下降
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1:3d}, loss = {loss.item():.4f}")

    for word, idx in vocab.items():
        vec = model.emb(torch.tensor(idx)).detach().numpy()
        print(f"{word}: {vec}")

if __name__ == "__main__":
    train()
