import torch
import torch.nn as nn


class SDOEmbedding(nn.Module):
    def __init__(self, channels=6, embedding_dim=512, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cnn1 = nn.Conv2d(channels, 64, 3)
        self.cnn2 = nn.Conv2d(64, 128, 3)
        self.cnn3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(41472, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 3)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 3)
        x = self.cnn3(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 3)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x
    

class SDOSequence(nn.Module):
    def __init__(self, channels=6, embedding_dim=512, sequence_length=10, dropout=0.2):
        super().__init__()
        self.sdo_embedding = SDOEmbedding(channels=channels, embedding_dim=embedding_dim, dropout=dropout)
        self.fc1 = nn.Linear(sequence_length*embedding_dim, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        channels = x.shape[2]
        size = x.shape[3]
        x = x.view(batch_size*seq_len, channels, size, size)
        x = self.sdo_embedding(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
