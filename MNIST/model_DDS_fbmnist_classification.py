import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import surrogate
import os
import random
from spiking_neuron.TCLIF import TCLIFNode
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create log directory with today's date
run_date = datetime.now().strftime('%Y-%m-%d')
log_dir = Path(f'experiments-{run_date}')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'train_log.txt'


def log_to_file(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {message}"
    with open(log_file, 'a') as f:
        f.write(formatted + '\n')
    print(formatted)


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


class SmnistDDSClassifier(nn.Module):
    def __init__(self, in_dim=1):
        super().__init__()
        self.encoder1 = nn.Linear(in_dim, 64)
        self.spike1 = LinearRecurrentContainer(DoubleDendriteTCLIFNode(), 64, 64)
        self.encoder2 = nn.Linear(64, 256)
        self.spike2 = LinearRecurrentContainer(DoubleDendriteTCLIFNode(), 256, 256)
        self.decoder = nn.Linear(256, 10)

    def forward(self, x):
        if hasattr(self.spike1, 'reset'):
            self.spike1.reset()
        if hasattr(self.spike2, 'reset'):
            self.spike2.reset()

        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            x_enc1 = self.encoder1(x_t)
            s1 = self.spike1(x_enc1)
            x_enc2 = self.encoder2(s1)
            s2 = self.spike2(x_enc2)
            outputs.append(s2)

        res = torch.stack(outputs, dim=0).mean(0)
        return self.decoder(res)


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(784, 1))  # 784 timesteps of 1-dim input
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=data_transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Training
checkpoint_path = "smnist_dds_checkpoint.pt"
start_epoch = 0
model = SmnistDDSClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss()
# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint.get('losses', [])
    start_epoch = checkpoint['epoch'] + 1
    log_to_file(f"Resuming from epoch {start_epoch}")
else:
    losses = []

target_total_epochs = 200

for epoch in range(start_epoch, target_total_epochs):
    scheduler.step()
    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in loop:
        xb = xb.to(device).float()
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)
        avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    log_to_file(
        f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Accuracy: {(correct / total) * 100:.2f}%, Beta2: {[round(x.item(), 4) for x in model.spike2.spike.beta[0]]}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, checkpoint_path)

    # Save best model if accuracy improves
    best_model_path = "smnist_dds_best.pt"
    if epoch == start_epoch or (correct / total) > globals().get('best_acc', 0):
        torch.save(model.state_dict(), best_model_path)
        best_acc = correct / total
        log_to_file(f"Best model saved at epoch {epoch} with accuracy {best_acc * 100:.2f}%")

# Evaluation
model.eval()
correct = 0
samples = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device).float()
        yb = yb.to(device)
        out = model(xb)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        samples += yb.size(0)

final_acc = f"Test Accuracy: {100. * correct / samples:.2f}%"
log_to_file(final_acc)

plt.figure()
plt.plot(losses)
plt.title("Training Loss (CrossEntropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(log_dir / "training_loss_curve.png")
plt.show()
