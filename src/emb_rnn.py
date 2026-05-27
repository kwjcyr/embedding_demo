import torch
import torch.nn as nn
import torch.optim as optim

# 词汇表
vocab = {'cat':0, 'dog':1, 'love':2, 'eat':3, 'fish':4, 'meat':5, 'ball':6, 'run':7}
vocab_size = len(vocab)

# 短句
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

inputs = torch.tensor([d[0] for d in train_data], dtype=torch.long)   # (7,2)
targets = torch.tensor([d[1] for d in train_data], dtype=torch.long)  # (7,)

# ------------------ RNN 模型 ------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,      # 输入形状 (batch, seq_len, embed_dim)
            num_layers=1,
            nonlinearity='tanh'    # 默认 tanh，也可用 'relu'
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len=2)
        emb = self.embedding(x)               # (batch, 2, embed_dim)
        # RNN 输出: output (batch, 2, hidden_dim), 最后一个隐状态 h_n (1, batch, hidden_dim)
        output, h_n = self.rnn(emb)
        # 取最后一个时间步的输出（即处理完第二个词后的隐状态）
        last_hidden = output[:, -1, :]        # (batch, hidden_dim)
        logits = self.out(last_hidden)        # (batch, vocab_size)
        return logits

def train():
    embed_dim = 8
    hidden_dim = 16

    model = SimpleRNN(vocab_size, embed_dim, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(500):
        out = model(inputs)
        loss = criterion(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1:3d}, loss = {loss.item():.4f}")

    # 打印训练后的词向量
    print("\n词向量 (Embedding层):")
    for word, idx in vocab.items():
        vec = model.embedding(torch.tensor(idx)).detach().numpy()
        print(f"{word}: {vec}")

if __name__ == "__main__":
    train()