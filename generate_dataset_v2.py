"""
SecureMesh ML-IDS — Blueprint-Aligned Synthetic Dataset (v2, corrected)
=========================================================================
Fixes vs the previous "Evidence_Aligned" file:
  1. Only the 4 attack types your blueprint actually defines
     (flooding, replay, blackhole/route-poisoning, brute_force_pairing) —
     dropped decreased_rank / dodag_version (those are RPL/6LoWPAN attacks,
     not part of your BLE+LoRa AODV-style design).
  2. Realistic class balance: normal traffic is the MAJORITY class (60%),
     attacks are the minority (40% combined) — not inverted like before.
  3. Multiclass `label` = the scenario itself (normal / flooding / replay /
     blackhole / brute_force_pairing) so a Random Forest can learn attack
     TYPE, not just attack/not-attack. A binary `is_attack` column is also
     included in case you want binary framing for a specific evaluation.
  4. `source_basis` is now honest: this is fully synthetic, generated from
     parameterized distributions to match your blueprint's feature schema —
     no external/published dataset is cited, because none was actually used.

Feature schema (Step 3 of your explanation doc):
    messages_per_min, failed_handshakes, mac_failures,
    rssi_mean_dbm, rssi_variance, hop_anomaly_score, forward_ratio
Rolling window = 60s (Step 3), per-device (Section 2.5).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_NORMAL = 6000
N_PER_ATTACK = 1000  # x4 attack types = 4000 -> total 10,000, 60:40 normal:attack


def clip0(x):
    return np.clip(x, 0, None)


def gen_normal(n):
    return pd.DataFrame({
        "messages_per_min": clip0(RNG.normal(9, 3.5, n)),
        "failed_handshakes": clip0(RNG.poisson(0.3, n)),
        "mac_failures": clip0(RNG.poisson(0.2, n)),
        "rssi_mean_dbm": RNG.normal(-58, 8, n),
        "rssi_variance": clip0(RNG.normal(4, 2, n)),
        "hop_anomaly_score": clip0(RNG.normal(0.10, 0.08, n)),
        "forward_ratio": np.clip(RNG.normal(0.5, 0.15, n), 0, 1),
        "scenario": "normal",
    })


def gen_flooding(n):
    return pd.DataFrame({
        "messages_per_min": clip0(RNG.normal(140, 30, n)),
        "failed_handshakes": clip0(RNG.poisson(0.5, n)),
        "mac_failures": clip0(RNG.poisson(1.0, n)),
        "rssi_mean_dbm": RNG.normal(-56, 8, n),
        "rssi_variance": clip0(RNG.normal(5, 2, n)),
        "hop_anomaly_score": clip0(RNG.normal(0.15, 0.1, n)),
        "forward_ratio": np.clip(RNG.normal(0.2, 0.1, n), 0, 1),
        "scenario": "flooding",
    })


def gen_replay(n):
    return pd.DataFrame({
        "messages_per_min": clip0(RNG.normal(18, 6, n)),
        "failed_handshakes": clip0(RNG.poisson(0.4, n)),
        "mac_failures": clip0(RNG.poisson(6, n)),
        "rssi_mean_dbm": RNG.normal(-59, 8, n),
        "rssi_variance": clip0(RNG.normal(4, 2, n)),
        "hop_anomaly_score": clip0(RNG.normal(0.2, 0.12, n)),
        "forward_ratio": np.clip(RNG.normal(0.6, 0.15, n), 0, 1),
        "scenario": "replay",
    })


def gen_blackhole(n):
    return pd.DataFrame({
        "messages_per_min": clip0(RNG.normal(11, 4, n)),
        "failed_handshakes": clip0(RNG.poisson(0.3, n)),
        "mac_failures": clip0(RNG.poisson(0.3, n)),
        "rssi_mean_dbm": RNG.normal(-55, 8, n),
        "rssi_variance": clip0(RNG.normal(4, 2, n)),
        "hop_anomaly_score": clip0(RNG.normal(0.85, 0.1, n)),
        "forward_ratio": np.clip(RNG.normal(0.9, 0.08, n), 0, 1),
        "scenario": "blackhole",
    })


def gen_bruteforce(n):
    return pd.DataFrame({
        "messages_per_min": clip0(RNG.normal(6, 3, n)),
        "failed_handshakes": clip0(RNG.poisson(9, n)),
        "mac_failures": clip0(RNG.poisson(1.0, n)),
        "rssi_mean_dbm": RNG.normal(-60, 8, n),
        "rssi_variance": clip0(RNG.normal(4, 2, n)),
        "hop_anomaly_score": clip0(RNG.normal(0.15, 0.1, n)),
        "forward_ratio": np.clip(RNG.normal(0.1, 0.08, n), 0, 1),
        "scenario": "brute_force_pairing",
    })


parts = [
    gen_normal(N_NORMAL),
    gen_flooding(N_PER_ATTACK),
    gen_replay(N_PER_ATTACK),
    gen_blackhole(N_PER_ATTACK),
    gen_bruteforce(N_PER_ATTACK),
]
df = pd.concat(parts, ignore_index=True)

# rounding
df["messages_per_min"] = df["messages_per_min"].round(2)
df["failed_handshakes"] = df["failed_handshakes"].round(0).astype(int)
df["mac_failures"] = df["mac_failures"].round(0).astype(int)
df["rssi_mean_dbm"] = df["rssi_mean_dbm"].round(2)
df["rssi_variance"] = df["rssi_variance"].round(2)
df["hop_anomaly_score"] = df["hop_anomaly_score"].round(3)
df["forward_ratio"] = df["forward_ratio"].round(3)

# metadata
df["device_id"] = ["device_" + str(i % 100).zfill(3) for i in range(len(df))]
df["window_id"] = range(len(df))
df["window_seconds"] = 60
df["label"] = df["scenario"]                       # multiclass target
df["is_attack"] = (df["scenario"] != "normal").astype(int)  # binary target, optional
df["source_basis"] = "Synthetic - parameterized simulation per SecureMesh blueprint Sec 2.5 / Step 4 (no external dataset used)"

# shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "sample_id", ["SM_" + str(i + 1).zfill(6) for i in range(len(df))])

cols = ["sample_id", "device_id", "window_id", "window_seconds",
        "messages_per_min", "failed_handshakes", "mac_failures",
        "rssi_mean_dbm", "rssi_variance", "hop_anomaly_score", "forward_ratio",
        "scenario", "label", "is_attack", "source_basis"]
df = df[cols]

out_path = "/mnt/user-data/outputs/SecureMesh_IDS_Dataset_v2.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df["label"].value_counts())
print("normal : attack ratio =", (df['is_attack']==0).sum(), ":", (df['is_attack']==1).sum())
