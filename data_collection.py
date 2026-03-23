import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time

class StockDataCollector:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_stock_data(self):
        print("Downloading stock data...")
        stock = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        stock.reset_index(inplace=True)
        self.data = stock
        return stock

    def add_time_features(self):
        print("Adding time features...")
        self.data['Day'] = self.data['Date'].dt.day
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Year'] = self.data['Date'].dt.year
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
        return self.data

    def normalize_data(self):
        print("Normalizing data...")
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
        return self.data

    def create_sequences(self, seq_length=30):
        print("Creating sequences...")
        sequences = []
        targets = []

        data_values = self.data[['Open','High','Low','Close','Volume']].values

        for i in range(len(data_values) - seq_length):
            sequences.append(data_values[i:i+seq_length])
            targets.append(data_values[i+seq_length][3])  # Close price

        return np.array(sequences), np.array(targets)

class NewsDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_news(self, query="stock market"):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}"
        response = requests.get(url)
        data = response.json()

        articles = []
        for article in data['articles']:
            articles.append({
                'title': article['title'],
                'description': article['description'],
                'content': article['content'],
                'publishedAt': article['publishedAt']
            })

        return pd.DataFrame(articles)

    def preprocess_news(self, df):
        df['text'] = df['title'] + " " + df['description']
        df.dropna(inplace=True)
        return df[['publishedAt','text']]

if __name__ == "__main__":
    stock = StockDataCollector("AAPL", "2018-01-01", "2024-01-01")
    df = stock.download_stock_data()
    df = stock.add_time_features()
    df = stock.normalize_data()
    X, y = stock.create_sequences()

    print(X.shape, y.shape)