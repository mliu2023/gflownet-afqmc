import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

from utils.visualize import visualize_parity_plot

class EnergyModel(nn.Module):
    def __init__(self, n_fields, hidden_size):
        super().__init__()
        self.n_fields = n_fields
        self.network = nn.Sequential(
            nn.Linear(n_fields, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.network(x)
    
class EnergyModelCNN(nn.Module):
    def __init__(self, n_fields, height, width, hidden_size):
        super().__init__()
        self.n_fields = n_fields
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        # self.conv3 = nn.Conv2d(64, 64, 3)
        # self.conv4 = nn.Conv2d(64, 64, 3)
        self.linear1 = nn.Linear(n_fields * 16, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.reshape(-1, 1, self.height, self.width)
        x = nn.functional.pad(x, (1, 1, 1, 1), "circular")
        x = self.conv1(x)
        x = nn.functional.relu(x)

        x = nn.functional.pad(x, (1, 1, 1, 1), "circular")
        x = self.conv2(x)
        x = nn.functional.relu(x)

        # x = nn.functional.pad(x, (1, 1, 1, 1), "circular")
        # x = self.conv3(x)
        # x = nn.functional.relu(x)

        # x = nn.functional.pad(x, (1, 1, 1, 1), "circular")
        # x = self.conv4(x)
        # x = nn.functional.relu(x)
        
        x = x.reshape(-1, self.n_fields * 16)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x

def train_step(model, optimizer, train_dataloader):
    total_loss = 0
    for fields, energies in train_dataloader:
        pred_energies = model(fields)
        loss = nn.functional.mse_loss(energies, pred_energies)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_dataloader)

def val_step(model, val_dataloader):
    total_loss = 0
    for fields, energies in val_dataloader:
        pred_energies = model(fields)
        loss = nn.functional.mse_loss(energies, pred_energies)
        total_loss += loss.item()
    return total_loss / len(val_dataloader)

if __name__ == "__main__":
    height = 7
    width = 7
    n_fields = 49
    fields = torch.from_numpy(np.load("data/af_7x7_24up_24down_fields.npy"))
    fields = 2 * fields - 1
    fields = fields.to(torch.float32)
    energies = torch.from_numpy(np.load("data/af_7x7_24up_24down_energies.npy")).reshape(-1, 1)
    energies = energies.to(torch.float32)
    print(len(fields))

    # height = 5
    # width = 5
    # n_fields = 25
    # fields = torch.from_numpy(np.load("data/af_5x5_12up_12down_fields.npy"))
    # fields = 2 * fields - 1
    # fields = fields.to(torch.float32)
    # energies = torch.from_numpy(np.load("data/af_5x5_12up_12down_energies.npy")).reshape(-1, 1)
    # energies = energies.to(torch.float32)

    indices = torch.randperm(len(fields))
    train_indices = indices[:int(0.8 * len(fields))]
    val_indices = indices[int(0.8 * len(fields)):]

    train_fields = fields[train_indices]
    train_energies = energies[train_indices]
    val_fields = fields[val_indices]
    val_energies = energies[val_indices]

    train_dataloader = DataLoader(TensorDataset(train_fields, train_energies), batch_size=256, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(val_fields, val_energies), batch_size=256, shuffle=True)

    energy_model = EnergyModel(n_fields, 512)
    # energy_model = EnergyModelCNN(n_fields, height, width, 512)
    optimizer = torch.optim.AdamW(energy_model.parameters(), lr=1e-4)

    tqdm_bar = tqdm(range(25))
    for _ in tqdm_bar:
        train_loss = train_step(energy_model, optimizer, train_dataloader)
        val_loss = val_step(energy_model, val_dataloader)
        tqdm_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

    energies_pred = []
    for field in val_fields:
        energies_pred.append(energy_model(field))
    energies_pred = torch.stack(energies_pred).flatten().detach().numpy()
    val_energies = val_energies.squeeze().numpy()
    visualize_parity_plot(val_energies, energies_pred, os.path.join("train_energy_model", f"parity_plot.png"))