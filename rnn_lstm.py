from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
torch.manual_seed(1)

class TextDataset(Dataset):
    def __init__(self, tokenized_text, seq_length):
        self.tokenized_text = tokenized_text
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.tokenized_text) - self.seq_length
        
    def __getitem__(self, idx):
        features = torch.tensor(self.tokenized_text[idx:idx+self.seq_length])
        labels = torch.tensor(self.tokenized_text[idx+1:idx+self.seq_length+1])
        return features, labels

class CharacterLSTM(nn.Module):
    def __init__(self):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=48)
        self.lstm = nn.LSTM(input_size=48, hidden_size=96, batch_first=True)
        self.linear = nn.Linear(96, vocab_size)

    def forward(self, x, states):
        x = self.embedding(x)
        out, states = self.lstm(x, states)
        out = self.linear(out)
        out = out.reshape(-1, out.size(2))
        return out, states

    def init_state(self, batch_size):
        hidden = torch.zeros(1, batch_size, 96)
        cell = torch.zeros(1, batch_size, 96)
        return hidden, cell

# Data Gathering

with open('datasets/frankenstein.txt', 'r', encoding='utf-8') as f:
    frankenstein = f.read()

first_letter_text = frankenstein[1380:8230]
print(first_letter_text)

# Data Cleaning & Pre-Processing

tokenized_text = list(first_letter_text)
print(len(tokenized_text))

unique_char_tokens = sorted(list(set(tokenized_text)))

c2ix = {ch:i for i,ch in enumerate(unique_char_tokens)}
print(c2ix)

vocab_size = len(c2ix)
print(vocab_size)

ix2c = {ix:ch for ch,ix in c2ix.items()}

tokenized_id_text = [c2ix[ch] for ch in tokenized_text]

# Preparing Dataset

seq_length = 48
dataset = TextDataset(tokenized_id_text, seq_length)

batch_size = 36
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training

lstm_model = CharacterLSTM()

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(lstm_model.parameters(), lr=0.015)

num_epochs = 5
for epoch in range(num_epochs):
    for features, labels in dataloader:
        optimizer.zero_grad()
        states = lstm_model.init_state(features.size(0))
        outputs, states = lstm_model(features, states)
        CEloss = loss(outputs, labels.view(-1))
        CEloss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], CELoss: {CEloss.item():.4f}')

# Infer Model

starting_prompt = "You will rejoice to hear"

tokenized_id_prompt = torch.tensor([[c2ix[ch] for ch in starting_prompt]])

lstm_model.eval()

num_generated_chars = 500
with torch.no_grad():
    states = lstm_model.init_state(1)
    for _ in range(num_generated_chars):
        output, states = lstm_model(tokenized_id_prompt, states)
        predicted_id = torch.argmax(output[-1, :], dim=-1).item()
        predicted_char = ix2c[predicted_id]
        starting_prompt += predicted_char
        tokenized_id_prompt = torch.tensor([[predicted_id]])

print(starting_prompt)