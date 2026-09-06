# SecureMesh ML-IDS — Package

## Files
- `SecureMesh_IDS_Dataset_v2_clean.csv` — final dataset (10,000 rows, 60:40 normal:attack, 4 attack types matching the blueprint)
- `generate_dataset_v2.py` — regenerates the dataset from scratch (tune means/stds per class here)
- `train_recommendation_model.py` — trains the Random Forest, evaluates it (precision/recall/FPR), and produces graduated-response recommendations (`none/log/warn/confirm/auto_block`)
- `securemesh_rf_model.joblib` — the trained model (load with `joblib.load(...)`)
- `SecureMesh_IDS_recommendations.csv` — every row scored with `anomaly_score` + `recommended_action`
- `model_evaluation_report.txt` — classification report, confusion matrix, FPR per class, feature importances

## Setup
```bash
pip install pandas numpy scikit-learn joblib
```

## Run
```bash
python generate_dataset_v2.py          # regenerate dataset (optional, already included)
python train_recommendation_model.py   # trains model, saves everything into ./outputs/
```
This creates an `outputs/` folder next to the scripts with:
`securemesh_rf_model.joblib`, `model_evaluation_report.txt`, `SecureMesh_IDS_recommendations.csv`.

## Interactive prediction (enter values, get a live prediction)
```bash
python predict.py          # prompts you for feature values, prints prediction + recommendation
python predict.py --demo   # runs 4 pre-built example devices with no typing needed
```
Needs `outputs/securemesh_rf_model.joblib` to exist first — run `train_recommendation_model.py` once before this.

## Loading the trained model yourself (e.g. in a notebook)
```python
import joblib
clf = joblib.load("outputs/securemesh_rf_model.joblib")
proba = clf.predict_proba(new_feature_rows)
```

## Note
Dataset is fully synthetic (parameterized simulation), not sourced from any external published dataset — matches your blueprint's Step 4 approach. Recalibrate the distributions in `generate_dataset_v2.py` once you have real Phase-1 traffic to compare against.
