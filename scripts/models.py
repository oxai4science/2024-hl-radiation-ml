import torch
import torch.nn as nn


class SDOEmbedding(nn.Module):
    def __init__(self, channels=6, embedding_dim=512, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cnn1 = nn.Conv2d(channels, 64, 3)
        self.cnn2 = nn.Conv2d(64, 128, 3)
        self.cnn3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(41472, embedding_dim*2)
        self.fc2 = nn.Linear(embedding_dim*2, embedding_dim)
        # self.dropout = nn.Dropout(dropout)

        
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
        # x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x


class SDOSequence(nn.Module):
    def __init__(self, channels=6, embedding_dim=1024, sequence_length=10, dropout=0.2):
        super().__init__()
        self.sdo_embedding = SDOEmbedding(channels=channels, embedding_dim=embedding_dim, dropout=dropout)
        self.fc1 = nn.Linear(sequence_length*embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        channels = x.shape[2]
        size = x.shape[3]
        x = x.view(batch_size*seq_len, channels, size, size)
        x = self.sdo_embedding(x)
        x = x.view(batch_size, -1)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


class RadRecurrent(nn.Module):
    def __init__(self, data_dim=2, lstm_dim=1024, lstm_depth=2, dropout=0.2, context_window=10, predict_window=10):
        super().__init__()
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        self.context_window = context_window # Not used within model, only for reference
        self.predict_window = predict_window # Not used within model, only for reference

        self.lstm = nn.LSTM(input_size=data_dim, hidden_size=lstm_dim, num_layers=lstm_depth, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(lstm_dim, data_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden = None

    def init(self, batch_size):
        h = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        c = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        device = next(self.parameters()).device
        h = h.to(device)
        c = c.to(device)
        self.hidden = (h, c)

    def init_with_context(self, context):
        # context has shape (batch_size, lstm_dim)
        h = context.unsqueeze(0).repeat(self.lstm_depth, 1, 1)
        c = torch.zeros(self.lstm_depth, context.shape[0], self.lstm_dim)
        device = next(self.parameters()).device
        h = h.to(device)
        c = c.to(device)
        self.hidden = (h, c)

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc1(x)
        return x
    

class RadContext(nn.Module):
    def __init__(self, input_sdo_channels=6, input_sdo_dim=1024, input_other_dim=1, output_dim=1024, lstm_dim=1024, lstm_depth=2):
        super().__init__()
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        self.input_sdo_dim = input_sdo_dim
        self.input_other_dim = input_other_dim

        self.sdo_embedding = SDOEmbedding(channels=input_sdo_channels, embedding_dim=input_sdo_dim)
        self.lstm = nn.LSTM(input_size=input_sdo_dim+input_other_dim, hidden_size=lstm_dim, num_layers=lstm_depth, batch_first=True)
        self.fc1 = nn.Linear(lstm_dim, output_dim)
        self.hidden = None

    def init(self, batch_size):
        h = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        c = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        device = next(self.parameters()).device
        h = h.to(device)
        c = c.to(device)
        self.hidden = (h, c)

    def forward(self, input_sdo, input_other):
        # input_sdo has shape (batch_size, seq_len, channels, size, size)
        # input_other has shape (batch_size, seq_len, self.input_other_dim)
        batch_size = input_sdo.shape[0]
        seq_len = input_sdo.shape[1]
        channels = input_sdo.shape[2]
        size = input_sdo.shape[3]
        input_sdo = input_sdo.view(batch_size*seq_len, channels, size, size)
        input_sdo = self.sdo_embedding(input_sdo)
        input_sdo = input_sdo.view(batch_size, seq_len, -1)
        x = torch.cat([input_sdo, input_other], dim=-1)
        x, self.hidden = self.lstm(x, self.hidden)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        return x



# class RadTransformer