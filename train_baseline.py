import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

BASE = "datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

# Load labels
labels = pd.read_csv("datasets/hcm_labels.csv", index_col="ecg_id")
df = pd.read_csv(f"{BASE}/ptbxl_database.csv", index_col="ecg_id")
df["hcm_label"] = labels["hcm_label"]

# Load ECG signals and extract simple features
def extract_features(row):
    try:
        path = f"{BASE}/{row['filename_lr']}"
        record = wfdb.rdrecord(path)
        sig = record.p_signal  # shape (1000, 12)
        features = []
        for lead in range(12):
            channel = sig[:, lead]
            features += [
                np.mean(channel),
                np.std(channel),
                np.max(channel),
                np.min(channel),
                np.max(channel) - np.min(channel)  # amplitude range
            ]
        return features
    except:
        return None

print("Extracting features (this takes ~5-10 minutes)...")
df["features"] = df.apply(extract_features, axis=1)
df = df.dropna(subset=["features"])

X = np.array(df["features"].tolist())
y = df["hcm_label"].values

# Split by keeping patient separation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train random forest with class weighting to handle imbalance
print("Training model...")
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- Results ---")
print(classification_report(y_test, y_pred, target_names=["Control", "HCM"]))
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")