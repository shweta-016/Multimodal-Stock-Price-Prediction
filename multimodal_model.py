import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 64)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SentimentNN(nn.Module):
    def __init__(self):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class MultimodalModel(nn.Module):
    def __init__(self, price_input_size):
        super(MultimodalModel, self).__init__()

        self.price_model = PriceLSTM(price_input_size, 128, 2)
        self.sentiment_model = SentimentNN()

        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, price_data, sentiment_data):
        price_features = self.price_model(price_data)
        sentiment_features = self.sentiment_model(sentiment_data)

        combined = torch.cat((price_features, sentiment_features), dim=1)

        out = torch.relu(self.fc1(combined))
        out = self.fc2(out)
        return out