import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import torch
from multimodal_model import MultimodalModel

ticker = "AAPL"

model = MultimodalModel(price_input_size=5)
model.load_state_dict(torch.load("model.pth"))
model.eval()

prices = []
predictions = []

def get_live_data():
    data = yf.download(ticker, period="1d", interval="1m")
    return data

def predict_price(data):
    if len(data) < 30:
        return None
    
    seq = data[['Open','High','Low','Close','Volume']].values[-30:]
    
    # Normalize like training
    seq = (seq - seq.mean()) / seq.std()
    
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    sentiment = torch.tensor([[0.0]])

    with torch.no_grad():
        pred = model(seq, sentiment)

    # De-normalize prediction
    pred_price = pred.item() * data['Close'].std() + data['Close'].mean()

    return pred_price

def update(frame):
    
    data = get_live_data()
    if len(data) == 0:
        return

    current_price = float(data['Close'].iloc[-1])
    pred_price = predict_price(data)

    prices.append(current_price)

    if pred_price is None:
        predictions.append(current_price)
    else:
        predictions.append(float(pred_price))

    plt.cla()
    plt.plot(prices, label="Live Price")
    plt.plot(predictions, label="Predicted Price")
    plt.legend()
    plt.title("Real-Time Stock Prediction")
ani = animation.FuncAnimation(plt.gcf(), update, interval=60000, cache_frame_data=False)
plt.show()