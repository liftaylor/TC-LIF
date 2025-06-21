import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import surrogate
import os
import random
from spiking_neuron.TCLIF import TCLIFNode
from spiking_neuron import base, neuron


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define trainable beta1 and beta2 via decay_factor
class TrainableTCLIFNode(TCLIFNode):
    def __init__(self, v_threshold, surrogate_function, gamma=0.5):
        beta = torch.nn.Parameter(torch.zeros(1, 2))  # β₁ and β₂
        super().__init__(v_threshold=v_threshold,
                         surrogate_function=surrogate_function,
                         hard_reset=False,
                         detach_reset=False,
                         decay_factor=beta,
                         gamma=gamma)
        self.beta = beta  # expose for tracking

    def extra_repr(self):
        return super().extra_repr() + f', beta=({self.beta[0][0].item():.4f}, {self.beta[0][1].item():.4f})'


class DoubleDendriteTCLIFNode(TCLIFNode):
    def __init__(self, v_threshold=1.0, surrogate_function=surrogate.Sigmoid(), gamma=0.5):
        beta = torch.nn.Parameter(torch.zeros(1, 3))  # β₁, β₂, β₃
        super().__init__(v_threshold=v_threshold,
                         surrogate_function=surrogate_function,
                         hard_reset=False,
                         detach_reset=False,
                         decay_factor=beta,
                         gamma=gamma)
        self.beta = beta
        self.register_memory('UD', None)
        self.register_memory('UD2', None)
        self.register_memory('US', None)

    def reset(self):
        super().reset()
        self.UD = None
        self.UD2 = None
        self.US = None

    def neuronal_forward(self, x):
        if self.UD is None:
           self.UD = torch.zeros_like(x)  # UD1

        if self.UD2 is None:
            self.UD2 = torch.zeros_like(x)
        if self.US is None:
            self.US = torch.zeros_like(x)

        beta1 = -torch.sigmoid(self.beta[0][0])  # UD1 ← β₁ * US + x
        beta2 = torch.sigmoid(self.beta[0][1])  # US ← β₂ * UD1 + β₃ * UD2
        beta3 = torch.sigmoid(self.beta[0][2])  # UD2 ← β₃ * UD1

        spike_fn = self.surrogate_function
        self.UD = self.UD + beta1 * self.US + x - self.gamma * spike_fn(self.US - self.v_threshold)
        self.UD2 = self.UD2 + beta3 * self.UD - self.gamma * spike_fn(self.US - self.v_threshold)
        self.US = self.US + beta2 * self.UD + beta3 * self.UD2 - self.v_threshold * spike_fn(self.US - self.v_threshold)

        spike_out = spike_fn(self.US - self.v_threshold)
        self.US = torch.where(spike_out > 0, torch.zeros_like(self.US), self.US)

        return spike_out

        def forward(self, x):
            return self.neuronal_forward(x)

    def extra_repr(self):
        return super().extra_repr() + f', beta=({self.beta[0][0].item():.4f}, {self.beta[0][1].item():.4f}, {self.beta[0][2].item():.4f})'


class LinearRecurrentContainer(nn.Module):
    def __init__(self, spiking_neuron, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spike = spiking_neuron

    def forward(self, x):
        x = self.linear(x)
        return self.spike(x)

    def reset(self):
        if hasattr(self.spike, 'reset'):
            self.spike.reset()


class fbMnistSineRegression(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.encoder1 = nn.Linear(in_dim, 64)
        self.spike1 = LinearRecurrentContainer(DoubleDendriteTCLIFNode(), 64, 64)
        self.encoder2 = nn.Linear(64, 256)
        self.spike2 = LinearRecurrentContainer(DoubleDendriteTCLIFNode(), 256, 256)
        self.decoder = nn.Linear(256, 1)

    def forward(self, x):
        if hasattr(self.spike1, 'reset'):
            self.spike1.reset()
        if hasattr(self.spike2, 'reset'):
            self.spike2.reset()

        output_current = []
        for t in range(0, x.size(1) - 8 + 1):
            x_t = x[:, t:t + 8, :].reshape(-1, 8)
            x_enc1 = self.encoder1(x_t)
            s1 = self.spike1(x_enc1)
            x_enc2 = self.encoder2(s1)
            s2 = self.spike2(x_enc2)
            output_current.append(s2)

        res = torch.stack(output_current, dim=0).mean(0)
        return self.decoder(res)


# Generate and prepare data
def generate_sine_wave(num_points=1000, num_cycles=5):
    x = np.linspace(0, num_cycles * 2 * np.pi, num_points)
    y = np.sin(x)
    return x, y


def generate_sequences(y, sequence_length=50):
    inputs, targets = [], []
    for i in range(len(y) - sequence_length):
        inputs.append(y[i:i + sequence_length])
        targets.append(y[i + sequence_length])
    inputs = np.array(inputs).reshape(-1, sequence_length, 1)
    targets = np.array(targets)
    return inputs, targets


_, y = generate_sine_wave()
X, Y = generate_sequences(y, sequence_length=50)
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True, drop_last=True)

# Training
model = fbMnistSineRegression().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

losses = []
for epoch in range(501):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, beta: {model.spike2.spike.beta.data}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().cpu().numpy()
    true_vals = Y_test.squeeze().cpu().numpy()

plt.figure(figsize=(8, 4))
plt.plot(preds[:100], label='Predicted')
plt.plot(true_vals[:100], label='Target', linestyle='dashed')
plt.title("Prediction vs Target on Test Set")
plt.legend()
plt.grid(True)
plt.savefig("prediction_vs_target.png")
plt.show()

plt.figure()
plt.plot(losses)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss_curve.png")
plt.show()
