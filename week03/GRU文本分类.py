import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据加载和预处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 将标签转换为数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符词汇表
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# --- 改为GRU模型 ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 使用GRU代替LSTM
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # GRU前向传播
        # gru_out: (batch_size, seq_len, hidden_dim) - 所有时间步的输出
        # hidden: (num_layers, batch_size, hidden_dim) - 最后一个时间步的隐藏状态
        gru_out, hidden = self.gru(embedded)

        # 取最后一层的最后一个时间步的隐藏状态
        # 如果是多层GRU，我们取最后一层的输出
        if self.num_layers > 1:
            last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        else:
            last_hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)

        out = self.fc(last_hidden)  # (batch_size, output_dim)
        return out


# --- 训练和预测 ---
# 创建数据集和数据加载器
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

# 定义模型超参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 1  # GRU层数
dropout = 0.2  # 如果是多层，可以添加dropout防止过拟合

# 初始化GRU模型
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(dataloader):.4f}")


# 预测函数
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # 添加batch维度

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建反向标签映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试预测
new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 打印模型结构信息
print(f"\n模型信息:")
print(f"- 词汇表大小: {vocab_size}")
print(f"- 嵌入维度: {embedding_dim}")
print(f"- GRU隐藏层维度: {hidden_dim}")
print(f"- 输出类别数: {output_dim}")
print(f"- GRU层数: {num_layers}")
