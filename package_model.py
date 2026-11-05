import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments, build_features

# Load Configuration
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
k = int(cfg["selector_k"])
n_est = int(cfg["n_estimators"])
max_d = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
min_samples = int(cfg["min_samples_leaf"])
threshold = float(cfg["threshold"])

# Rebuild Exact Split
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42, stratify=y_train_val)

# Fit Final Pipeline
pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=k)),
    ("model", RandomForestClassifier(n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_samples, random_state=42))
])
pipe.fit(X_train, y_train)

# Export Model Artifacts
joblib.dump(pipe, "models/delay_model.pkl")

pd.DataFrame([{"threshold": threshold}]).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

# Compute & Export Training Data Imputation Medians
raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame([{"column": k, "median": v} for k, v in medians.items()]).to_csv("models/impute_values.csv", index=False)

# Reload and Compare Predictions
reloaded_pipe = joblib.load("models/delay_model.pkl")

orig_probs = pipe.predict_proba(X_test)[:, 1]
reload_probs = reloaded_pipe.predict_proba(X_test)[:, 1]
max_abs_diff = float(np.max(np.abs(orig_probs - reload_probs)))

reload_check = [
    ("test_rows_scored", len(X_test)),
    ("reload_max_abs_probability_diff", round(max_abs_diff, 9)),
    ("threshold", threshold)
]
pd.DataFrame(reload_check, columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)
print("Model packaged successfully!")
