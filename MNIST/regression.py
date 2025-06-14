
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from spiking_neuron.TCLIF import TCLIFNode  # Adjust path if needed
from spikingjelly.activation_based import surrogate

# Generate synthetic sine wave dataset
def generate_sine_data(seq_len=100, total_samples=500):
    X = []
    Y = []
    for _ in range(total_samples):
        start = np.random.rand() * 2 * np.pi
        x = np.linspace(start, start + 2 * np.pi, seq_len)
        sine_wave = np.sin(x)
        X.append(sine_wave[:-1])
        Y.append(sine_wave[-1])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# Define Spiking Regression Model
class SineRegressionTCLIF(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(SineRegressionTCLIF, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(1, hidden_size)
        self.spike = TCLIFNode(
            v_threshold=1.0,
            gamma=0.5,
            surrogate_function=surrogate.Sigmoid()
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        batch_size, seq_len = x_seq.shape
        x_seq = x_seq.unsqueeze(-1)  # (B, T, 1)
        mem = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_len):
            x_t = self.encoder(x_seq[:, t])  # (B, H)
            s_t = self.spike(x_t)
            mem = mem + s_t.detach()  # <-- detach here to avoid second backward
        out = self.decoder(mem / seq_len)
        return out

# Training loop
X, Y = generate_sine_data()
model = SineRegressionTCLIF()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Visualize prediction vs target
with torch.no_grad():
    preds = model(X).squeeze().numpy()
    targets = Y.squeeze().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(preds[:100], label='Predicted')
    plt.plot(targets[:100], label='Target', linestyle='dashed')
    plt.legend()
    plt.title("Sine Wave Regression with TCLIFNode")
    plt.show()
