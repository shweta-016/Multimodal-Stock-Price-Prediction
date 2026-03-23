import torch
import torch.nn as nn
import torch.optim as optim
from multimodal_model import MultimodalModel

class Trainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train(self, price_data, sentiment_data, targets, epochs=20):
        for epoch in range(epochs):
            self.model.train()

            outputs = self.model(price_data, sentiment_data)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)