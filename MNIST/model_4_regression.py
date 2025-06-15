import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


# ---------------------
# Surrogate function
# ---------------------
def surrogate_spike_function(us, threshold=1.0, slope=2.75):
    return torch.sigmoid(slope * (us - threshold))


# ---------------------
# LSTMLIFNeuronPaper (simplified from model_4.py)
# ---------------------
class LSTMLIFNeuronPaper(nn.Module):
    def __init__(self, input_size, hidden_size, experiment_params, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.V_th = experiment_params.get('V_th', 0.5)
        self.Gamma = experiment_params.get('Gamma', 0.3)
        self.C1 = nn.Parameter(torch.tensor(experiment_params.get('C1', 0.5)))
        self.C2 = nn.Parameter(torch.tensor(experiment_params.get('C2', 0.5)))

        self.i2h = nn.Linear(input_size, hidden_size)
        self.reset_state()

    def reset_state(self):
        self.UD = None
        self.US = None
        self.spike = None

    def forward(self, x_t):
        x_t = self.i2h(x_t)
        if self.UD is None:
            self.UD = torch.zeros_like(x_t)
            self.US = torch.zeros_like(x_t)
            self.spike = torch.zeros_like(x_t)

        self.UD = self.UD + (-torch.sigmoid(self.C1)) * self.US + x_t
        self.US = self.US + torch.sigmoid(self.C2) * self.UD
        self.spike = surrogate_spike_function(self.US, threshold=self.V_th)
        self.US = self.US - self.Gamma * self.spike
        return self.spike


# ---------------------
# Wrapper for sequence regression
# ---------------------
# class Model4SineRegression(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, output_size=1):
#         super().__init__()
#         experiment_params = {
#             'V_th': 0.5,
#             'Gamma': 0.3,
#             'C1': 0.5,
#             'C2': 0.5
#         }
#         self.spike_core = LSTMLIFNeuronPaper(input_size, hidden_size, experiment_params)
#         self.decoder = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x_seq):
#         self.spike_core.reset_state()
#         spikes = []
#         for t in range(x_seq.size(1)):
#             s_t = self.spike_core(x_seq[:, t, :])
#             spikes.append(s_t)
#         spikes = torch.stack(spikes, dim=1)
#         out = spikes.mean(dim=1)
#         return self.decoder(out)

class Model4SineRegression(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        experiment_params = {
            'V_th': 0.5,
            'Gamma': 0.3,
            'C1': 0.5,
            'C2': 0.5
        }
        self.layer1 = LSTMLIFNeuronPaper(input_size, hidden_size, experiment_params)
        self.layer2 = LSTMLIFNeuronPaper(hidden_size, hidden_size, experiment_params)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        self.layer1.reset_state()
        self.layer2.reset_state()

        spikes = []
        for t in range(x_seq.size(1)):
            s1 = self.layer1(x_seq[:, t, :])
            s2 = self.layer2(s1)
            spikes.append(s2)

        spikes = torch.stack(spikes, dim=1)  # [B, T, H]
        out = spikes.mean(dim=1)
        return self.decoder(out)


# ---------------------
# Data preparation
# ---------------------
def generate_sine_data(num_points=1000, num_cycles=5, sequence_length=50):
    x = np.linspace(0, num_cycles * 2 * np.pi, num_points)
    y = np.sin(x)
    inputs, targets = [], []
    for i in range(len(y) - sequence_length):
        inputs.append(y[i:i + sequence_length])
        targets.append(y[i + sequence_length])
    X = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    return train_test_split(X, Y, test_size=0.2, shuffle=False)


# ---------------------
# Training and evaluation
# ---------------------
def train_and_evaluate():
    X_train, X_test, Y_train, Y_test = generate_sine_data()
    model = Model4SineRegression()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(501):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}")

    with torch.no_grad():
        Y_pred = model(X_test)
        mse = criterion(Y_pred, Y_test).item()
        mae = torch.mean(torch.abs(Y_pred - Y_test)).item()
        print(f"\nTest MSE: {mse:.8f}")
        print(f"Test MAE: {mae:.6f}")

        preds = Y_pred.squeeze().numpy()
        targets = Y_test.squeeze().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(preds[:100], label="Predicted")
        plt.plot(targets[:100], label="Target", linestyle='dashed')
        plt.legend()
        plt.title("Model_4 (LSTMLIFNeuronPaper) Sine Regression")
        plt.show()
# def train_and_evaluate():
#     X_train, X_test, Y_train, Y_test = generate_sine_data()
#     model = Model4SineRegression()
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#     for epoch in range(501):
#         optimizer.zero_grad()
#         output = model(X_train)
#         loss = criterion(output, Y_train)
#         loss.backward()
#         optimizer.step()
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}")
#
#     with torch.no_grad():
#         Y_train_pred = model(X_train)
#         Y_test_pred = model(X_test)
#
#         train_mse = criterion(Y_train_pred, Y_train).item()
#         test_mse = criterion(Y_test_pred, Y_test).item()
#         train_mae = torch.mean(torch.abs(Y_train_pred - Y_train)).item()
#         test_mae = torch.mean(torch.abs(Y_test_pred - Y_test)).item()
#
#         print(f"\nTrain MSE: {train_mse:.8f}, MAE: {train_mae:.6f}")
#         print(f"Test  MSE: {test_mse:.8f}, MAE: {test_mae:.6f}")
#
#         y_train_true = Y_train.squeeze().cpu().numpy()
#         y_train_pred = Y_train_pred.squeeze().cpu().numpy()
#         y_test_true = Y_test.squeeze().cpu().numpy()
#         y_test_pred = Y_test_pred.squeeze().cpu().numpy()
#
#         plt.figure(figsize=(12, 4))
#         plt.plot(y_train_true, label="Y_train (true)", color='blue', linestyle='dashed')
#         plt.plot(y_train_pred, label="Y_train_pred", color='green')
#         plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
#                  y_test_true, label="Y_test (true)", color='orange', linestyle='dashed')
#         plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_pred)),
#                  y_test_pred, label="Y_test_pred", color='red')
#         plt.legend()
#         plt.title("Model_4 (LSTMLIFNeuronPaper) Sine Regression: Train + Test")
#         plt.xlabel("Sample Index")
#         plt.ylabel("Value")
#         plt.tight_layout()
#         plt.show()


if __name__ == '__main__':
    train_and_evaluate()
