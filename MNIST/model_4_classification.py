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
# LSTM-LIF Neuron
# ---------------------
class LSTMLIFNeuronPaper(nn.Module):
    def __init__(self, input_size, hidden_size, experiment_params):
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
# Spiking LSTM Classifier for S-MNIST
# ---------------------
class Model4SMNISTClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=10):
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

        spikes = torch.stack(spikes, dim=1)
        out = spikes.mean(dim=1)
        return self.decoder(out)


# ---------------------
# Load S-MNIST Data
# ---------------------
def load_smnist(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1, 1))  # [784, 1]
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


# ---------------------
# Train + Evaluate
# ---------------------
def train_smnist_classifier(epochs=20, checkpoint_path="model_checkpoint.pt"):
    start_time = time.time()
    print("Training started at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    train_loader, test_loader = load_smnist()
    model = Model4SMNISTClassifier().to(device)
    # if use_compile:
    # torch.compile requires PyTorch 2.0+. Commented out for compatibility.
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-5
    )
    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            loop.set_postfix(loss=total_loss / (total / xb.size(0)), acc=correct / total)

        print(
            f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {(correct / total) * 100:.2f}%")

        scheduler.step(correct / total)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Test Accuracy: {(correct / total) * 100:.2f}%")

    end_time = time.time()
    print("Training ended at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("Total duration: {:.2f} seconds".format(end_time - start_time))


if __name__ == '__main__':
    train_smnist_classifier(epochs=450)
