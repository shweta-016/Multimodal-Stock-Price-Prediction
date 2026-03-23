import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, price_data, sentiment_data, targets):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(price_data, sentiment_data)

        predictions = predictions.numpy()
        targets = targets.numpy()

        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)

        print("MSE:", mse)
        print("MAE:", mae)
        print("RMSE:", rmse)

        return predictions