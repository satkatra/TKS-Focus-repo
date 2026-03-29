import pandas as pd
import numpy as np
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

BASE = "datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

# ── Symbolic Rules (clinical ECG knowledge) ───────────────────────────────
def symbolic_score(sig):
    """Returns a score 0-3 based on clinical HCM ECG rules."""
    score = 0
    # Rule 1: High R-wave amplitude in lead V5 (index 10) suggests LVH
    if np.max(sig[:, 10]) > 1.5:
        score += 1
    # Rule 2: Deep S-wave in V1 (index 6)
    if np.min(sig[:, 6]) < -1.5:
        score += 1
    # Rule 3: Large voltage range across leads (general hypertrophy sign)
    voltage_range = np.max(sig) - np.min(sig)
    if voltage_range > 3.5:
        score += 1
    return score / 3.0  # normalize to 0-1

# ── Dataset ───────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        try:
            record = wfdb.rdrecord(f"{BASE}/{row['filename_lr']}")
            sig = record.p_signal.astype(np.float32)       # (1000, 12)
            sig_t = sig.T                                   # (12, 1000)
            sig_t = (sig_t - sig_t.mean()) / (sig_t.std() + 1e-8)
            sym = symbolic_score(sig)
        except:
            sig_t = np.zeros((12, 1000), dtype=np.float32)
            sym = 0.0
        label = int(row["hcm_label"])
        return torch.tensor(sig_t), torch.tensor([sym], dtype=torch.float32), torch.tensor(label)

# ── Model: CNN + Symbolic fusion ──────────────────────────────────────────
class NeuroSymbolicECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        # +1 for the symbolic score input
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, sym):
        cnn_out = self.cnn(x)
        combined = torch.cat([cnn_out, sym], dim=1)
        return self.fc(combined)

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(f"{BASE}/ptbxl_database.csv", index_col="ecg_id")
labels = pd.read_csv("datasets/hcm_labels.csv", index_col="ecg_id")
df["hcm_label"] = labels["hcm_label"]

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["hcm_label"], random_state=42)

pos = (train_df["hcm_label"] == 1).sum()
neg = (train_df["hcm_label"] == 0).sum()
pos_weight = torch.tensor([neg / pos])

train_loader = DataLoader(ECGDataset(train_df), batch_size=32, shuffle=True)
test_loader  = DataLoader(ECGDataset(test_df),  batch_size=32, shuffle=False)

# ── Train ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = NeuroSymbolicECG().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

for epoch in range(10):
    model.train()
    total_loss = 0
    for X, sym, y in train_loader:
        X, sym, y = X.to(device), sym.to(device), y.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(X, sym).squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────────
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for X, sym, y in test_loader:
        probs = torch.sigmoid(model(X.to(device), sym.to(device))).cpu().squeeze().numpy()
        all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
        all_labels.extend(y.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs > 0.5).astype(int)

print("\n--- Neuro-Symbolic Results ---")
print(classification_report(all_labels, preds, target_names=["Control", "HCM"]))
print(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")

torch.save(model.state_dict(), "hcm_neurosymbolic.pt")
print("Model saved to hcm_neurosymbolic.pt")