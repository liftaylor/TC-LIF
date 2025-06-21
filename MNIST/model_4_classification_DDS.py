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
def surrogate_spike_function(us, threshold=1.0, slope=1.0):
    return torch.sigmoid(slope * (us - threshold))
#
# def surrogate_spike_function(us, threshold=1.0, slope=2.75):
#     return torch.sigmoid(slope * (us - threshold)) + (us > threshold).float() - torch.sigmoid(slope * (us - threshold)).detach()



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

        # self.gamma_d2 = 0.3
        # self.gamma_soma = 0.3
        #
        # self.V_th = 0.5

        self.V_th = 0.3  # instead of 0.5
        self.gamma_soma = 0.1
        self.gamma_d2 = 0.1

    def reset_state(self, batch_size, device):
        self.UD1 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.UD2 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.US = torch.zeros(batch_size, self.hidden_size, device=device)
        self.spike = torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x_t):
        x_t = self.i2h(x_t)
        if self.UD1 is None:
            self.UD1 = torch.zeros_like(x_t)
            self.UD2 = torch.zeros_like(x_t)
            self.US = torch.zeros_like(x_t)
            self.spike = torch.zeros_like(x_t)

        UD1_new = self.alpha_d1 * self.UD1 + (-torch.sigmoid(self.c1)) * self.US + x_t
        UD2_new = self.alpha_d2 * self.UD2 + torch.sigmoid(self.c_d2) * self.US + self.lambda_ud1 * UD1_new - self.gamma_d2 * self.spike
        US_new  = self.alpha_soma * self.US + torch.sigmoid(self.c2) * UD1_new + self.beta_s2 * UD2_new - self.gamma_soma * self.spike

        # Clamp to prevent sigmoid overflow
        US_new = torch.clamp(torch.nan_to_num(US_new, nan=0.0, posinf=1e6, neginf=-1e6), -100, 100)

        # Sanitize
        UD1_new = torch.nan_to_num(UD1_new, nan=0.0, posinf=1e6, neginf=-1e6)
        UD2_new = torch.nan_to_num(UD2_new, nan=0.0, posinf=1e6, neginf=-1e6)
        US_new  = torch.nan_to_num(US_new,  nan=0.0, posinf=1e6, neginf=-1e6)

        spike = surrogate_spike_function(US_new, threshold=self.V_th)

        self.UD1 = UD1_new
        self.UD2 = UD2_new
        self.US = US_new
        self.spike = spike

        return spike


# ---------------------
# Spiking LSTM Regressor with Double Dendrite
# ---------------------
class Model4SMNISTClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=10):
        super().__init__()
        self.layer1 = LSTMLIFNeuronV3_DoubleDendrite(input_size, hidden_size)
        self.layer2 = LSTMLIFNeuronV3_DoubleDendrite(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        self.layer1.reset_state(B, x_seq.device)
        self.layer2.reset_state(B, x_seq.device)

        spikes = []
        for t in range(x_seq.size(1)):
            s1 = self.layer1(x_seq[:, t, :])
            s2 = self.layer2(s1)
            spikes.append(s2)

        spikes = torch.stack(spikes, dim=1)  # [B, T, H]
        out = spikes.mean(dim=1)
        if self.layer1.spike.mean().item() == 0.0:
            print("Avg spike (layer1):", self.layer1.spike.mean().item())
        return self.decoder(out)


# ---------------------
# Load S-MNIST Data
# ---------------------
# ---------------------
# Load S-MNIST Data
# ---------------------
def load_smnist(batch_size=512):
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
def train_and_evaluate_classifier(epochs=20, checkpoint_path="model_checkpoint_DDS.pt"):
    start_time = time.time()
    print("Training started at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    train_loader, test_loader = load_smnist()
    model = Model4SMNISTClassifier().to(device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Test Accuracy: {(correct / total) * 100:.2f}%")
    end_time = time.time()
    print("Training ended at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("Total duration: {:.2f} seconds".format(end_time - start_time))


if __name__ == '__main__':
    train_and_evaluate_classifier(epochs=50)

