
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="SecureMesh IDS Lab",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SecureMesh — ML Intrusion Detection Lab")
st.caption("Experimental intrusion detection using network telemetry.")

DATA_PATH = Path(__file__).parent / "securemesh_ids_synthetic_dataset.csv"

FEATURES = [
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

FRACTION_FEATURES = [
    "duplicate_packet_rate",
    "route_error_rate",
    "ttl_expiry_rate",
    "unique_sender_ratio",
    "packet_drop_rate",
]

NORMAL_ATTACK_NAMES = {
    "normal", "benign", "none", "no_attack", "no attack"
}


def normalize_binary_labels(series):
    """Convert common binary labels into 0=normal, 1=malicious."""
    mapping = {
        "0": 0,
        "0.0": 0,
        "normal": 0,
        "benign": 0,
        "safe": 0,
        "1": 1,
        "1.0": 1,
        "malicious": 1,
        "attack": 1,
        "suspicious": 1,
    }

    result = series.astype(str).str.strip().str.lower().map(mapping)

    if result.isna().any():
        raise ValueError(
            "is_malicious must contain 0/1 or normal/malicious labels."
        )

    return result.astype(int)


@st.cache_resource
def train_models(dataset_path, dataset_mtime):
    data = pd.read_csv(dataset_path)

    required = FEATURES + ["is_malicious", "attack_type"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["is_malicious"] = normalize_binary_labels(
        data["is_malicious"]
    )
    data["attack_type"] = data["attack_type"].astype(str).str.strip()

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    if data.empty:
        raise ValueError("No valid rows found in the dataset.")

    if data["is_malicious"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both normal (0) and malicious (1) rows."
        )

    X = data[FEATURES]
    y_binary = data["is_malicious"]
    y_attack = data["attack_type"]

    # IMPORTANT: split all labels together to preserve row alignment.
    (
        X_train,
        X_test,
        yb_train,
        yb_test,
        ya_train,
        ya_test,
    ) = train_test_split(
        X,
        y_binary,
        y_attack,
        test_size=0.20,
        random_state=42,
        stratify=y_binary,
    )

    binary_model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    binary_model.fit(X_train, yb_train)

    # Train attack-type model on malicious rows only.
    malicious_mask = yb_train == 1
    attack_X = X_train.loc[malicious_mask]
    attack_y = ya_train.loc[malicious_mask]

    # Do not train the attack classifier on labels named "Normal".
    valid_attack_mask = ~attack_y.str.lower().isin(
        NORMAL_ATTACK_NAMES
    )
    attack_X = attack_X.loc[valid_attack_mask]
    attack_y = attack_y.loc[valid_attack_mask]

    attack_model = None

    if len(attack_X) > 0:
        attack_model = RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=43,
            n_jobs=-1,
        )
        attack_model.fit(attack_X, attack_y)

    # Evaluate attack types only on held-out malicious samples.
    test_malicious_mask = yb_test == 1
    attack_test_X = X_test.loc[test_malicious_mask]
    attack_test_y = ya_test.loc[test_malicious_mask]

    valid_test_mask = ~attack_test_y.str.lower().isin(
        NORMAL_ATTACK_NAMES
    )
    attack_test_X = attack_test_X.loc[valid_test_mask]
    attack_test_y = attack_test_y.loc[valid_test_mask]

    return (
        binary_model,
        attack_model,
        X_test,
        yb_test,
        attack_test_X,
        attack_test_y,
    )


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH.name}. "
        "Keep the CSV in the same folder as this Python file."
    )
    st.stop()

try:
    (
        binary_model,
        attack_model,
        X_test,
        yb_test,
        attack_test_X,
        attack_test_y,
    ) = train_models(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


st.warning(
    "Prototype only: training data is synthetic, not real device traffic. "
    "Predictions are experimental and must not automatically block devices."
)

with st.sidebar:
    st.header("Input mode")
    mode = st.radio(
        "Choose how to provide telemetry",
        ["Manual input", "Paste one CSV row"],
    )
    st.caption("Rate values are per minute.")
    st.caption("Fields ending in _rate are fractions from 0 to 1.")
    st.caption("Use a consistent rolling window, such as 60 seconds.")


defaults = {
    "message_rate_per_min": 8.0,
    "failed_handshakes_per_min": 0.0,
    "mac_signature_failures_per_min": 0.0,
    "rssi_variance": 5.0,
    "hop_count_mean": 2.0,
    "forwarded_to_originated_ratio": 1.0,
    "duplicate_packet_rate": 0.03,
    "route_error_rate": 0.02,
    "ttl_expiry_rate": 0.03,
    "unique_sender_ratio": 0.75,
    "packet_drop_rate": 0.04,
}

row = {}
submitted = False

if mode == "Manual input":
    st.subheader("1. Enter network observations")

    col1, col2, col3 = st.columns(3)

    with col1:
        row["message_rate_per_min"] = st.number_input(
            "Messages per minute",
            min_value=0.0,
            max_value=10000.0,
            value=defaults["message_rate_per_min"],
        )
        row["failed_handshakes_per_min"] = st.number_input(
            "Failed handshakes per minute",
            min_value=0.0,
            max_value=10000.0,
            value=defaults["failed_handshakes_per_min"],
        )
        row["mac_signature_failures_per_min"] = st.number_input(
            "MAC/signature failures per minute",
            min_value=0.0,
            max_value=10000.0,
            value=defaults["mac_signature_failures_per_min"],
        )
        row["rssi_variance"] = st.number_input(
            "RSSI variance",
            min_value=0.0,
            max_value=10000.0,
            value=defaults["rssi_variance"],
        )

    with col2:
        row["hop_count_mean"] = st.number_input(
            "Mean hop count",
            min_value=0.0,
            max_value=255.0,
            value=defaults["hop_count_mean"],
        )
        row["forwarded_to_originated_ratio"] = st.number_input(
            "Forwarded / originated ratio",
            min_value=0.0,
            max_value=10000.0,
            value=defaults["forwarded_to_originated_ratio"],
        )
        row["duplicate_packet_rate"] = st.slider(
            "Duplicate packet rate", 0.0, 1.0,
            defaults["duplicate_packet_rate"], 0.01
        )
        row["route_error_rate"] = st.slider(
            "Route error rate", 0.0, 1.0,
            defaults["route_error_rate"], 0.01
        )

    with col3:
        row["ttl_expiry_rate"] = st.slider(
            "TTL expiry rate", 0.0, 1.0,
            defaults["ttl_expiry_rate"], 0.01
        )
        row["unique_sender_ratio"] = st.slider(
            "Unique sender ratio", 0.0, 1.0,
            defaults["unique_sender_ratio"], 0.01
        )
        row["packet_drop_rate"] = st.slider(
            "Packet drop rate", 0.0, 1.0,
            defaults["packet_drop_rate"], 0.01
        )

    submitted = st.button("Analyze device", type="primary")

else:
    st.subheader("1. Paste a CSV row")

    st.write("Enter 11 numeric values in this exact order:")
    st.code(",".join(FEATURES))

    pasted = st.text_input(
        "CSV values",
        value="8,0,0,5,2,1,0.03,0.02,0.03,0.75,0.04",
    )

    submitted = st.button("Analyze pasted row", type="primary")

    if submitted:
        try:
            values = [float(v.strip()) for v in pasted.split(",")]

            if len(values) != len(FEATURES):
                st.error(
                    f"Expected {len(FEATURES)} values, got {len(values)}."
                )
                submitted = False
            elif not np.isfinite(values).all():
                st.error("All values must be finite numbers.")
                submitted = False
            else:
                row = dict(zip(FEATURES, values))

        except ValueError:
            st.error("Every CSV value must be numeric.")
            submitted = False


if submitted:
    valid = True

    for key in FEATURES:
        value = row.get(key)

        if value is None or not np.isfinite(value) or value < 0:
            st.error(f"{key} must be a finite, non-negative number.")
            valid = False
        elif key in FRACTION_FEATURES and value > 1:
            st.error(f"{key} must be between 0 and 1.")
            valid = False

    if valid:
        input_df = pd.DataFrame(
            [[row[column] for column in FEATURES]],
            columns=FEATURES,
        )

        classes = list(binary_model.classes_)

        if 1 not in classes:
            st.error("The trained model does not contain class 1 (malicious).")
            st.stop()

        probabilities = binary_model.predict_proba(input_df)[0]
        malicious_probability = float(
            probabilities[classes.index(1)]
        )
        predicted_binary = int(binary_model.predict(input_df)[0])

        st.subheader("2. Result")
        left, right = st.columns(2)

        with left:
            if predicted_binary == 1:
                st.error("⚠️ Suspicious / potentially malicious pattern")
            else:
                st.success("✅ Pattern classified as likely normal")

            st.metric(
                "Model's malicious probability",
                f"{malicious_probability:.1%}",
            )
            st.progress(
                min(max(malicious_probability, 0.0), 1.0)
            )

            st.caption(
                "The verdict uses the binary model's predicted class. "
                "The score is not a verified real-world probability."
            )

        with right:
            if predicted_binary == 0:
                st.info("Most likely class: **Normal**")
            elif attack_model is None:
                st.info("Likely attack class: **Unknown**")
            else:
                predicted_attack = str(
                    attack_model.predict(input_df)[0]
                )
                st.info(
                    "Likely attack class: **"
                    + predicted_attack.replace("_", " ").title()
                    + "**"
                )

            st.write(
                "A prediction is not proof that a device is safe or malicious."
            )

        st.subheader("3. Most influential model features")

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": binary_model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        st.dataframe(
            importance.head(6),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("4. Entered values")
        st.dataframe(
            input_df,
            hide_index=True,
            use_container_width=True,
        )


with st.expander("Model evaluation on a held-out synthetic test split"):
    st.warning(
        "These metrics describe performance on this synthetic dataset only. "
        "They do not establish real-world detection accuracy."
    )

    binary_predictions = binary_model.predict(X_test)

    st.write("Binary classification report")
    st.code(
        classification_report(
            yb_test,
            binary_predictions,
            labels=[0, 1],
            target_names=["normal", "malicious"],
            zero_division=0,
        )
    )

    st.write("Binary confusion matrix: rows=true, columns=predicted")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(
                yb_test,
                binary_predictions,
                labels=[0, 1],
            ),
            index=["true normal", "true malicious"],
            columns=["pred normal", "pred malicious"],
        )
    )

    st.write("Attack-type report for held-out malicious rows")
    if attack_model is not None and len(attack_test_X) > 0:
        st.code(
            classification_report(
                attack_test_y,
                attack_model.predict(attack_test_X),
                zero_division=0,
            )
        )
    else:
        st.info(
            "Not enough held-out malicious examples to calculate "
            "attack-type metrics."
        )
