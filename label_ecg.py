import pandas as pd
import ast

BASE = "datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

df = pd.read_csv(f"{BASE}/ptbxl_database.csv", index_col="ecg_id")
df.scp_codes = df.scp_codes.apply(ast.literal_eval)

HCM_CODES = {"LVH", "RVH", "LVOLT", "SEHYP"}

def has_hcm_code(scp_dict):
    return int(any(code in HCM_CODES for code in scp_dict.keys()))

df["hcm_label"] = df.scp_codes.apply(has_hcm_code)

print(df["hcm_label"].value_counts())
df[["hcm_label"]].to_csv("datasets/hcm_labels.csv")
print("Labels saved.")