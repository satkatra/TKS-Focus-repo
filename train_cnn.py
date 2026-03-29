import pandas as pd
import numpy as np
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

BASE = "datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

# ── Dataset ──────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        try:
            record = wfdb.rdrecord(f"{BASE}/{row['filename_lr']}")
            sig = record.p_signal.T.astype(np.float32)  # (12, 1000)
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        except:
            sig = np.zeros((12, 1000), dtype=np.float32)
        label = int(row["hcm_label"])
        return torch.tensor(sig), torch.tensor(label)

# ── Model ─────────────────────────────────────────────────────────────────
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.net(x))

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(f"{BASE}/ptbxl_database.csv", index_col="ecg_id")
labels = pd.read_csv("datasets/hcm_labels.csv", index_col="ecg_id")
df["hcm_label"] = labels["hcm_label"]

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["hcm_label"], random_state=42)

# Class weights to handle imbalance
pos = (train_df["hcm_label"] == 1).sum()
neg = (train_df["hcm_label"] == 0).sum()
pos_weight = torch.tensor([neg / pos])

train_loader = DataLoader(ECGDataset(train_df), batch_size=32, shuffle=True)
test_loader  = DataLoader(ECGDataset(test_df),  batch_size=32, shuffle=False)

# ── Train ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = ECG_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

for epoch in range(10):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(X).squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────────
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        probs = torch.sigmoid(model(X.to(device))).cpu().squeeze().numpy()
        all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
        all_labels.extend(y.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs > 0.5).astype(int)

print("\n--- CNN Results ---")
print(classification_report(all_labels, preds, target_names=["Control", "HCM"]))
print(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")

torch.save(model.state_dict(), "hcm_cnn.pt")
print("Model saved to hcm_cnn.pt")