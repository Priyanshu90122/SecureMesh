"""
SecureMesh ML-IDS — Interactive Predictor
============================================
Run this AFTER train_recommendation_model.py (it needs the saved model in
./outputs/securemesh_rf_model.joblib).

Prompts you for one device's feature values, then prints:
  - predicted scenario (normal / flooding / replay / blackhole / brute_force_pairing)
  - probability breakdown across all classes
  - anomaly_score = 1 - P(normal)
  - recommended_action (none / log / warn / confirm) based on the same
    thresholds used in training

Usage:
    python predict.py
Then either answer the prompts one field at a time, or pass --demo to see
a few example devices run through the model without typing anything.
"""

import os
import sys
import joblib
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL_PATH = os.path.join(OUT_DIR, "securemesh_rf_model.joblib")

LOG_THRESH = 0.30
WARN_THRESH = 0.60
CONFIRM_THRESH = 0.85

FEATURES = [
    ("messages_per_min", "Messages per minute (e.g. 9 = normal, 140+ = flooding)"),
    ("failed_handshakes", "Failed handshake/pairing attempts in this window (int)"),
    ("mac_failures", "MAC/signature verification failures in this window (int)"),
    ("rssi_mean_dbm", "Mean BLE RSSI in dBm (e.g. -55)"),
    ("rssi_variance", "RSSI variance (signal stability, e.g. 4)"),
    ("hop_anomaly_score", "Hop-count anomaly score, 0 to 1 (e.g. 0.1 normal, 0.85+ blackhole)"),
    ("forward_ratio", "Forwarded-vs-originated ratio, 0 to 1"),
]


def recommend_action(score):
    if score < LOG_THRESH:
        return "none"
    elif score < WARN_THRESH:
        return "log"
    elif score < CONFIRM_THRESH:
        return "warn"
    else:
        return "confirm"


def predict_one(clf, row: dict):
    X = pd.DataFrame([row])[[f for f, _ in FEATURES]]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    normal_idx = list(classes).index("normal")
    anomaly_score = 1 - proba[normal_idx]
    predicted = clf.predict(X)[0]

    print("\n" + "=" * 50)
    print(f"Predicted type     : {predicted}")
    print(f"Anomaly score      : {anomaly_score:.4f}")
    print(f"Recommended action : {recommend_action(anomaly_score)}")
    print("-" * 50)
    print("Probability by class:")
    for c, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        print(f"  {c:22s} {p:.4f}")
    print("=" * 50)


def prompt_row():
    row = {}
    print("\nEnter this device's feature values for the current 60s window:")
    for key, hint in FEATURES:
        while True:
            raw = input(f"  {key} [{hint}]: ").strip()
            try:
                row[key] = float(raw)
                break
            except ValueError:
                print("    Please enter a number.")
    return row


def demo(clf):
    examples = {
        "looks normal": dict(messages_per_min=9, failed_handshakes=0, mac_failures=0,
                              rssi_mean_dbm=-58, rssi_variance=4, hop_anomaly_score=0.1, forward_ratio=0.5),
        "looks like flooding": dict(messages_per_min=145, failed_handshakes=0, mac_failures=1,
                                     rssi_mean_dbm=-56, rssi_variance=5, hop_anomaly_score=0.15, forward_ratio=0.2),
        "looks like blackhole": dict(messages_per_min=11, failed_handshakes=0, mac_failures=0,
                                      rssi_mean_dbm=-55, rssi_variance=4, hop_anomaly_score=0.9, forward_ratio=0.92),
        "looks like brute-force pairing": dict(messages_per_min=6, failed_handshakes=10, mac_failures=1,
                                                rssi_mean_dbm=-60, rssi_variance=4, hop_anomaly_score=0.15, forward_ratio=0.1),
    }
    for label, row in examples.items():
        print(f"\n--- Demo case: {label} ---")
        predict_one(clf, row)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Run 'python train_recommendation_model.py' first to train and save it.")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)

    if "--demo" in sys.argv:
        demo(clf)
    else:
        row = prompt_row()
        predict_one(clf, row)

        again = input("\nPredict another? (y/n): ").strip().lower()
        while again == "y":
            row = prompt_row()
            predict_one(clf, row)
            again = input("\nPredict another? (y/n): ").strip().lower()
