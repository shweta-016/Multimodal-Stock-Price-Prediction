import pandas as pd
import ta

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df

    def add_rsi(self):
        self.df['RSI'] = ta.momentum.RSIIndicator(self.df['Close']).rsi()
        return self.df

    def add_macd(self):
        macd = ta.trend.MACD(self.df['Close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        return self.df

    def add_moving_averages(self):
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20).mean()
        return self.df

    def add_bollinger_bands(self):
        bb = ta.volatility.BollingerBands(self.df['Close'])
        self.df['BB_High'] = bb.bollinger_hband()
        self.df['BB_Low'] = bb.bollinger_lband()
        return self.df

    def add_all_indicators(self):
        self.add_rsi()
        self.add_macd()
        self.add_moving_averages()
        self.add_bollinger_bands()
        self.df.dropna(inplace=True)
        return self.df