import pandas as pd
import numpy as np
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score

BASE = "datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
MAX_PER_CLASS = 4000

def symbolic_score(sig):
    score = 0
    s_v1 = abs(np.min(sig[:, 6]))
    r_v5 = np.max(sig[:, 10])
    if (s_v1 + r_v5) > 3.5:
        score += 1
    r_avl = np.max(sig[:, 11])
    if (s_v1 + r_avl) > 2.8:
        score += 1
    t_wave_v5 = sig[600:800, 10]
    if np.min(t_wave_v5) < -0.3:
        score += 1
    voltage_sum = sum(np.max(sig[:, i]) - np.min(sig[:, i]) for i in range(12))
    if voltage_sum > 20.0:
        score += 1
    if r_avl > 1.1:
        score += 1
    return score / 5.0

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

class NeuroSymbolicECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock(32),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock(64),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(129, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )
    def forward(self, x, s):
        return self.fc(torch.cat([self.encoder(x), s], dim=1))

df = pd.read_csv(f"{BASE}/ptbxl_database.csv", index_col="ecg_id")
labels = pd.read_csv("datasets/hcm_labels.csv", index_col="ecg_id")
df["hcm_label"] = labels["hcm_label"]

unique_patients = df["patient_id"].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)
n = len(unique_patients)
train_patients = set(unique_patients[:int(0.70*n)])
val_patients   = set(unique_patients[int(0.70*n):int(0.85*n)])
test_patients  = set(unique_patients[int(0.85*n):])

train_df = df[df["patient_id"].isin(train_patients)]
val_df   = df[df["patient_id"].isin(val_patients)]
test_df  = df[df["patient_id"].isin(test_patients)]

hcm_train  = train_df[train_df["hcm_label"]==1].sample(n=min(MAX_PER_CLASS, int((train_df["hcm_label"]==1).sum())), random_state=42)
ctrl_train = train_df[train_df["hcm_label"]==0].sample(n=min(MAX_PER_CLASS, int((train_df["hcm_label"]==0).sum())), random_state=42)
train_df   = pd.concat([hcm_train, ctrl_train]).sample(frac=1, random_state=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

def load_signals(subset, name):
    signals, syms, ys = [], [], []
    for i, (idx, row) in enumerate(subset.iterrows()):
        try:
            record = wfdb.rdrecord(f"{BASE}/{row['filename_lr']}")
            sig = record.p_signal.astype(np.float32)
            sig_t = sig.T
            sig_t = (sig_t - sig_t.mean()) / (sig_t.std() + 1e-8)
            signals.append(sig_t)
            syms.append(symbolic_score(sig))
            ys.append(int(row["hcm_label"]))
        except:
            continue
        if i % 500 == 0:
            print(f"  {name}: {i}/{len(subset)}")
    return np.array(signals), np.array(syms, dtype=np.float32), np.array(ys)

print("Loading training signals...")
X_tr, S_tr, y_tr = load_signals(train_df, "train")
print("Loading validation signals...")
X_val, S_val, y_val = load_signals(val_df, "val")
print("Loading test signals...")
X_te, S_te, y_te = load_signals(test_df, "test")

class ECGDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X)
        self.S = torch.tensor(S).unsqueeze(1)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.S[i], self.y[i]

train_loader = DataLoader(ECGDataset(X_tr, S_tr, y_tr), batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(ECGDataset(X_val, S_val, y_val), batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(ECGDataset(X_te, S_te, y_te), batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cpu")
model = NeuroSymbolicECG().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
pos_weight = torch.tensor([(y_tr==0).sum() / (y_tr==1).sum()]).to(device)
criterion  = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)

best_val_auc = 0
patience     = 4
no_improve   = 0

for epoch in range(20):
    model.train()
    total_loss = 0
    for X_b, S_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b, S_b).squeeze(), y_b.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for X_b, S_b, y_b in val_loader:
            p = torch.sigmoid(model(X_b, S_b)).squeeze().numpy()
            val_probs.extend(p if hasattr(p, '__len__') else [float(p)])
            val_labels.extend(y_b.numpy())
    val_auc = roc_auc_score(val_labels, val_probs)
    print(f"Epoch {epoch+1}/20 - Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "hcm_neurosymbolic_v4.pt")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load("hcm_neurosymbolic_v4.pt"))
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for X_b, S_b, y_b in test_loader:
        p = torch.sigmoid(model(X_b, S_b)).squeeze().numpy()
        all_probs.extend(p if hasattr(p, '__len__') else [float(p)])
        all_labels.extend(y_b.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

print("\n--- Threshold Analysis ---")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (all_probs > threshold).astype(int)
    p = precision_score(all_labels, preds, zero_division=0)
    r = recall_score(all_labels, preds, zero_division=0)
    f = f1_score(all_labels, preds, zero_division=0)
    print(f"  Threshold {threshold} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f}")

preds = (all_probs > 0.5).astype(int)
print("\n--- V4 Results (residual CNN + focal + cosine LR) ---")
print(classification_report(all_labels, preds, target_names=["Control", "HCM"]))
print(f"Test AUC:     {roc_auc_score(all_labels, all_probs):.4f}")
print(f"Best Val AUC: {best_val_auc:.4f}")
print("Model saved to hcm_neurosymbolic_v4.pt")
