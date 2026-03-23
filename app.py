import streamlit as st
import torch
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from multimodal_model import MultimodalModel
from sentiment_analysis import SentimentAnalyzer

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("📈 Multimodal Stock Price Prediction Dashboard")

ticker = st.text_input("Enter Stock Ticker (Example: AAPL, TSLA, RELIANCE.NS)", "AAPL")

if st.button("Predict"):

    # Load model
    model = MultimodalModel(price_input_size=5)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Download stock data
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

    if len(data) < 30:
        st.error("Not enough data")
    else:
        # Prepare stock input
        seq = data[['Open','High','Low','Close','Volume']].values[-30:]
        seq_mean = seq.mean()
        seq_std = seq.std() if seq.std() != 0 else 1
        seq = (seq - seq_mean) / seq_std
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # Sentiment
        news_df = pd.read_csv("Combined_News_DJIA.csv", encoding='latin-1')
        news_df['text'] = news_df[['Top1','Top2','Top3','Top4','Top5']].astype(str).agg(' '.join, axis=1)
        news_df = news_df[['Date','text']]
        news_df.columns = ['date','text']

        sentiment_analyzer = SentimentAnalyzer()
        daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(news_df)
        sentiment_val = float(daily_sentiment['sentiment'].iloc[-1])
        sentiment_value = torch.tensor([[sentiment_val]], dtype=torch.float32)

        # Prediction
        with torch.no_grad():
            pred = model(seq, sentiment_value)

        pred_value = float(pred.item())
        last_real_price = float(data['Close'].iloc[-1])
        pred_price = last_real_price + pred_value

        # Dashboard Layout
        col1, col2, col3 = st.columns(3)

        col1.metric("Current Price", f"{last_real_price:.2f}")
        col2.metric("Predicted Price", f"{pred_price:.2f}")
        col3.metric("Sentiment Score", f"{sentiment_val:.2f}")

        # Buy/Sell
        if pred_price > last_real_price:
            st.success("📊 BUY SIGNAL")
        else:
            st.error("📉 SELL SIGNAL")

        # Graph
        st.subheader("Stock Price Prediction Graph")
        plt.figure(figsize=(10,5))
        plt.plot(data['Close'].values[-30:], label="Actual Price")
        plt.axhline(y=pred_price, color='r', label="Predicted Price")
        plt.legend()
        plt.title("Actual vs Predicted Price")
        st.pyplot(plt)

        # Show data
        st.subheader("Recent Stock Data")
        st.dataframe(data.tail())