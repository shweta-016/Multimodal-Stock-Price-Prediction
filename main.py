import torch
import pandas as pd
from data_collection import StockDataCollector
from sentiment_analysis import SentimentAnalyzer
from multimodal_model import MultimodalModel
from train_model import Trainer
from evaluation import Evaluator

def main():
    # ================= STOCK DATA =================
    collector = StockDataCollector("AAPL", "2018-01-01", "2024-01-01")
    df = collector.download_stock_data()
    df = collector.add_time_features()
    df = collector.normalize_data()
    X, y = collector.create_sequences()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1,1)

    # ================= NEWS DATA =================
    print("Loading Kaggle news dataset...")
    news_df = pd.read_csv("Combined_News_DJIA.csv", encoding='latin-1')
    # Combine multiple news headlines into one text
    news_df['text'] = news_df[['Top1','Top2','Top3','Top4','Top5','Top6','Top7','Top8','Top9','Top10']].astype(str).agg(' '.join, axis=1)
    # Keep only date and text
    news_df = news_df[['Date','text']]
    news_df.columns = ['date','text']
    # Sentiment Analysis
    sentiment_analyzer = SentimentAnalyzer()
    daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(news_df)
    sentiment_values = torch.tensor(daily_sentiment['sentiment'].values, dtype=torch.float32)
    sentiment_values = sentiment_values.view(-1,1)
    # Make same length
    min_len = min(len(X), len(sentiment_values))
    X = X[:min_len]
    y = y[:min_len]
    sentiment_values = sentiment_values[:min_len]

    # ================= MODEL =================
    model = MultimodalModel(price_input_size=5)
    trainer = Trainer(model)
    trainer.train(X, sentiment_values, y)

    # Save model for real-time prediction
    trainer.save_model("model.pth")
    

    # ================= EVALUATION =================
    evaluator = Evaluator(model)
    preds = evaluator.evaluate(X, sentiment_values, y)

    print("Training Complete")
  

# ===== ADD THIS LINE (SAVE MODEL) =====




if __name__ == "__main__":
    main()