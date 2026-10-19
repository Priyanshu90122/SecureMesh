# SecureMesh Interactive IDS

## Run
1. Install Python 3.10+.
2. Put `securemesh_ids_app.py` and `securemesh_ids_synthetic_dataset.csv` in the same folder.
3. Run:
   `pip install -r requirements_securemesh_ids.txt`
   `streamlit run securemesh_ids_app.py`

The browser UI allows manual entry of 11 telemetry features or pasting a single comma-separated row. It reports a binary normal/suspicious prediction, estimated malicious probability, and a likely attack class.

## Feature alignment with your project
The repository contains mesh packet fields such as TTL, hop count, message IDs, packet timestamps, and MeshNode counters for bad packets, duplicates, relayed packets, decrypt failures, stale/replay packets, flood events, forged announces, RSSI and delivery stats. The blueprint also recommends per-device rolling-window features such as message rate, failed-handshake rate, MAC/signature failure rate, RSSI variance, hop count, and forwarding ratio. The app's 11 input fields follow that blueprint and include additional useful IDS rates.

## Important
The supplied dataset is synthetic. Values should be computed from logs/simulation over consistent windows. The app is a research prototype; don't automatically block real devices based on its prediction. Verify alerts with rule-based checks and controlled tests.
