import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

class TextPreprocessor:
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def preprocess(self, texts):
        return [self.clean_text(t) for t in texts]


class SentimentNN(nn.Module):
    def __init__(self, input_size):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SentimentAnalyzer:
    def __init__(self):
        print("Initializing Sentiment Neural Network...")
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=500)  # reduced features
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self, texts):
        texts = self.preprocessor.preprocess(texts)
        X = self.vectorizer.fit_transform(texts).toarray()
        X = self.scaler.fit_transform(X)
        return X

    def train_sentiment_model(self, texts):
        X = self.prepare_data(texts)
        X = torch.tensor(X, dtype=torch.float32)

        input_size = X.shape[1]
        self.model = SentimentNN(input_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        y = torch.randn(len(X), 1)

        for epoch in range(5):
            outputs = self.model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Sentiment Model Epoch {epoch+1}, Loss: {loss.item()}")

    def sentiment_score(self, texts):
        X = self.prepare_data(texts)
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            scores = self.model(X).numpy()

        return scores.flatten()

    def aggregate_daily_sentiment(self, news_df):
        news_df = news_df.dropna()

        print("Training sentiment model...")
        self.train_sentiment_model(news_df['text'].tolist())

        news_df['sentiment'] = self.sentiment_score(news_df['text'].tolist())
        news_df['date'] = pd.to_datetime(news_df['date'])

        daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment'].mean()
        return daily_sentiment.reset_index()