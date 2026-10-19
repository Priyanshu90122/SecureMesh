"""
Baseline training example for SecureMesh IDS synthetic data.
This demonstrates the pipeline only; metrics on synthetic data are not real-world validation.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "securemesh_ids_synthetic_dataset.csv"
df = pd.read_csv(DATA_PATH)

feature_cols = [
    "message_rate_per_min",
    "failed_handshakes_per_min",
    "mac_signature_failures_per_min",
    "rssi_variance",
    "hop_count_mean",
    "forwarded_to_originated_ratio",
    "duplicate_packet_rate",
    "route_error_rate",
    "ttl_expiry_rate",
    "unique_sender_ratio",
    "packet_drop_rate",
]

X = df[feature_cols]
y = df["is_malicious"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    min_samples_leaf=2,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Classification report:")
print(classification_report(y_test, pred, target_names=["normal", "malicious"]))
print("Confusion matrix [rows=true, cols=predicted]:")
print(confusion_matrix(y_test, pred))
print("\\nFeature importance:")
for name, score in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f"{name:35s} {score:.4f}")
