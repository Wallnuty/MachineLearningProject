import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import joblib
import copy

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("train_data.csv")

X = df.drop(columns=["target"]).values
y = df["target"].values.astype(np.float32)
feature_names = df.drop(columns=["target"]).columns.tolist()

# -------------------------
# 60 / 20 / 20 SPLIT
# -------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# -------------------------
# Scaling (NO leakage)
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -------------------------
# Dataset
# -------------------------
class PokemonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(PokemonDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(PokemonDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(PokemonDataset(X_test, y_test), batch_size=64)

# -------------------------
# Neural Network: 
# Deep feedforward neural network with four hidden layers
# (256, 128, 64, and 32 neurons) used to learn battle patterns.
# ReLU activations introduce nonlinearity and improve learning.
# Batch normalization stabilizes training and improves convergence.
# Dropout regularization reduces overfitting by randomly disabling neurons.
# The final output is converted to a probability using a sigmoid function
# to predict the likelihood of one Pokémon defeating another.
# -------------------------
class BattleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BattleNN(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

# -------------------------
# Early stopping setup
# -------------------------
best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())

patience = 10
patience_counter = 0

train_losses = []
val_losses = []

# -------------------------
# TRAINING LOOP
# -------------------------
epochs = 100

for epoch in range(epochs):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).view(-1, 1)

        if torch.rand(1).item() > 0.5:
            X_batch = -X_batch
            y_batch = 1 - y_batch

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            val_correct += (
                preds == y_batch
            ).sum().item()

            val_total += y_batch.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# -------------------------
# Load best model
# -------------------------
model.load_state_dict(best_model_state)

# -------------------------
# TEST EVALUATION
# -------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).view(-1, 1)

        outputs = model(X_batch)
        preds = (torch.sigmoid(outputs) > 0.5).float()

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print("\nFinal Test Accuracy:", correct / total)

# -------------------------
# LOSS CURVE (LT vs LV)
# -------------------------
plt.plot(train_losses, label="Train Loss (LT)")
plt.plot(val_losses, label="Validation Loss (LV)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overtraining Detection")
plt.legend()
plt.show()


joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(scaler, "scaler.pkl")
torch.save(model.state_dict(), "battle_model.pth")

print("Model and scaler saved!")
