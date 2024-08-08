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
    def __init__(self, data_dim=2, lstm_dim=1024, lstm_depth=2, dropout=0.2, context_window=10, prediction_window=10):
        super().__init__()
        self.data_dim = data_dim
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        self.dropout = dropout
        self.context_window = context_window # Not used within model, only for reference
        self.prediction_window = prediction_window # Not used within model, only for reference

        self.lstm = nn.LSTM(input_size=data_dim, hidden_size=lstm_dim, num_layers=lstm_depth, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(lstm_dim, data_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden = None

    def init(self, batch_size):
        h = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        c = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        device = next(self.parameters()).device
        h = h.to(device)
        c = c.to(device)
        self.hidden = (h, c)

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.fc1(x)
        return x

    def predict(self, context, prediction_window):
        batch_size = context.shape[0]
        # context_length = context.shape[1]
        # print('Running model with context shape: {}'.format(context.shape))
        self.init(batch_size)
        context_input = context
        prediction = [context_input[:, -1, :].unsqueeze(1)] # prepend the prediction values with the last context input
        context_output = self(context_input)
        x = context_output[:, -1, :].unsqueeze(1)
        for _ in range(prediction_window):
            prediction.append(x)
            x = self.forward(x)
        prediction = torch.cat(prediction, dim=1)
        return prediction


class RadRecurrentWithSDO(nn.Module):
    def __init__(self, data_dim=2, lstm_dim=1024, lstm_depth=2, dropout=0.2, sdo_channels=6, sdo_dim=1024, context_window=10, prediction_window=10):
        super().__init__()
        self.data_dim = data_dim
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        self.dropout = dropout
        self.sdo_channels = sdo_channels
        self.sdo_dim = sdo_dim
        self.context_window = context_window # Not used within model, only for reference
        self.prediction_window = prediction_window # Not used within model, only for reference

        self.sdo_embedding = SDOEmbedding(channels=sdo_channels, embedding_dim=sdo_dim)
        self.lstm_context = nn.LSTM(input_size=sdo_dim+data_dim, hidden_size=lstm_dim, num_layers=lstm_depth, batch_first=True)
        self.lstm_predict = nn.LSTM(input_size=data_dim, hidden_size=lstm_dim, num_layers=lstm_depth, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(lstm_dim, data_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden_context = None
        self.hidden_predict = None

    def init(self, batch_size):
        h = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        c = torch.zeros(self.lstm_depth, batch_size, self.lstm_dim)
        device = next(self.parameters()).device
        h = h.to(device)
        c = c.to(device)
        self.hidden_context = (h, c)

    def forward_context(self, sdo, data):
        # sdo has shape (batch_size, seq_len, channels, size, size)
        # data has shape (batch_size, seq_len, self.data_dim)
        batch_size = sdo.shape[0]
        seq_len = sdo.shape[1]
        channels = sdo.shape[2]
        size = sdo.shape[3]
        sdo = sdo.reshape(batch_size*seq_len, channels, size, size)
        sdo = self.sdo_embedding(sdo)
        sdo = sdo.view(batch_size, seq_len, -1)
        x = torch.cat([sdo, data], dim=-1)
        _, self.hidden_context = self.lstm_context(x, self.hidden_context)
        self.hidden_predict = self.hidden_context

    def forward(self, x):
        x, self.hidden_predict = self.lstm_predict(x, self.hidden_predict)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.fc1(x)
        return x

    def predict(self, context_sdo, context_data, prediction_window, num_samples=1):
        context_batch_size = context_sdo.shape[0]
        if context_batch_size != 1:
            raise ValueError('Batch size of context must be 1')
        # context_length = context.shape[1]
        # print('Running model with context shape: {}'.format(context.shape))
        self.init(context_batch_size)
        self.forward_context(context_sdo, context_data)

        h, c = self.hidden_context
        self.hidden_predict = (h.repeat(1, num_samples, 1), c.repeat(1, num_samples, 1))

        x = context_data[:, -1, :].unsqueeze(1) # prepend the prediction values with the last context input
        x = x.repeat(num_samples, 1, 1)
        prediction = [x]
        for _ in range(prediction_window):
            x = self.forward(x)
            prediction.append(x)
        prediction = torch.cat(prediction, dim=1)
        return prediction


# class RadTransformer