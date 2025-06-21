import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Enable Torch Compile and Mixed Precision (AMP)
use_amp = True
use_compile = True


# ---------------------
# Surrogate spike function
# ---------------------
def surrogate_spike_function(us, threshold=1.0, slope=2.75):
    return torch.sigmoid(slope * (us - threshold))


# ---------------------
# LSTM-LIF Neuron V3 Double Dendrite
# ---------------------
class LSTMLIFNeuronV3_DoubleDendrite(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)

        self.c1 = nn.Parameter(torch.tensor(0.5))
        self.c2 = nn.Parameter(torch.tensor(0.5))
        self.c_d2 = nn.Parameter(torch.tensor(0.5))
        self.lambda_ud1 = nn.Parameter(torch.tensor(0.5))
        self.beta_s2 = nn.Parameter(torch.tensor(0.5))

        self.alpha_d1 = 0.9
        self.alpha_d2 = 0.9
        self.alpha_soma = 0.9

        self.gamma_d2 = 0.3
        self.gamma_soma = 0.3

        self.V_th = 0.5

        self.reset_state()

    def reset_state(self):
        self.UD1 = None
        self.UD2 = None
        self.US = None
        self.spike = None

    def forward(self, x_t):
        x_t = self.i2h(x_t)
        if self.UD1 is None:
            self.UD1 = torch.zeros_like(x_t)
            self.UD2 = torch.zeros_like(x_t)
            self.US = torch.zeros_like(x_t)
            self.spike = torch.zeros_like(x_t)

        UD1_new = self.alpha_d1 * self.UD1 + (-torch.sigmoid(self.c1)) * self.US + x_t
        UD2_new = self.alpha_d2 * self.UD2 + torch.sigmoid(
            self.c_d2) * self.US + self.lambda_ud1 * UD1_new - self.gamma_d2 * self.spike
        US_new = self.alpha_soma * self.US + torch.sigmoid(
            self.c2) * UD1_new + self.beta_s2 * UD2_new - self.gamma_soma * self.spike

        spike = surrogate_spike_function(US_new, threshold=self.V_th)

        self.UD1 = UD1_new
        self.UD2 = UD2_new
        self.US = US_new
        self.spike = spike

        return spike


# ---------------------
# Spiking LSTM Regressor with Double Dendrite
# ---------------------
class Model4SineRegression(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.layer1 = LSTMLIFNeuronV3_DoubleDendrite(input_size, hidden_size)
        self.layer2 = LSTMLIFNeuronV3_DoubleDendrite(hidden_size, hidden_size)
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
# Load Sine Data
# ---------------------
def generate_sine_data(num_points=1000, num_cycles=5, sequence_length=50):
    import numpy as np
    from sklearn.model_selection import train_test_split

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
# Train + Evaluate
# ---------------------
def train_and_evaluate():
    X_train, X_test, Y_train, Y_test = generate_sine_data()
    model = Model4SineRegression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(501):
        model.train()
        optimizer.zero_grad()
        output = model(X_train.to(device))
        loss = criterion(output, Y_train.to(device))
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.8f}")

    with torch.no_grad():
        model.eval()
        Y_pred = model(X_test.to(device))
        mse = criterion(Y_pred, Y_test.to(device)).item()
        mae = torch.mean(torch.abs(Y_pred - Y_test.to(device))).item()
        print(f"\nTest MSE: {mse:.8f}")
        print(f"Test MAE: {mae:.6f}")

        preds = Y_pred.squeeze().cpu().numpy()
        targets = Y_test.squeeze().cpu().numpy()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(preds[:100], label="Predicted")
        plt.plot(targets[:100], label="Target", linestyle='dashed')
        plt.legend()
        plt.title("Model_4 (Double Dendrite) Sine Regression")
        plt.show()


if __name__ == '__main__':
    train_and_evaluate()
