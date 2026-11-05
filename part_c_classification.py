import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Stratified Splits
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42, stratify=y_train_val)

# Helper function to compute metrics
def get_metrics_row(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"model": model_name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}

model_rows = []

# 1. Blind Baseline
model_rows.append(get_metrics_row("blind_all_on_time", y_val, np.zeros_like(y_val)))

# 2. Logistic Regression
lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
lr_pipe.fit(X_train, y_train)
model_rows.append(get_metrics_row("logistic_regression", y_val, lr_pipe.predict(X_val)))

# 3. Decision Tree Tuning
best_dt_f1, best_dt_depth = -1, None
for d in [2, 3, 4, 5, 6, 8, 10]:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    f1 = f1_score(y_val, dt.predict(X_val), zero_division=0)
    if f1 > best_dt_f1:
        best_dt_f1 = f1
        best_dt_depth = d

dt_best = DecisionTreeClassifier(criterion="gini", max_depth=best_dt_depth, random_state=42)
dt_best.fit(X_train, y_train)
model_rows.append(get_metrics_row(f"decision_tree_depth_{best_dt_depth}", y_val, dt_best.predict(X_val)))

# 4. Default Random Forest
rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
rf_def.fit(X_train, y_train)
model_rows.append(get_metrics_row("random_forest_default", y_val, rf_def.predict(X_val)))

pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)

# Feature Selection
rf_def.fit(X_train, y_train)
rf_importances = rf_def.feature_importances_

skb_all = SelectKBest(f_classif, k="all")
skb_all.fit(X_train, y_train)
f_scores = skb_all.scores_

feat_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances,
    "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
})
feat_df["f_rank"] = feat_df["f_score"].rank(ascending=False, method="min").astype(int)
feat_df = feat_df.sort_values("f_rank")
feat_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    val_f1 = f1_score(y_val, pipe.predict(X_val), zero_division=0)
    selectk_rows.append({"k": k, "val_f1": round(val_f1, 4)})
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

# Hyperparameter Tuning with k=7
selector_7 = SelectKBest(f_classif, k=7)
X_tr_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Grid Search
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
gs.fit(X_tr_7, y_train)
gs_val_f1 = f1_score(y_val, gs.predict(X_val_7), zero_division=0)

# Randomized Search
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
rs.fit(X_tr_7, y_train)
rs_val_f1 = f1_score(y_val, rs.predict(X_val_7), zero_division=0)

# Default Forest (Same 7 features)
rf_untuned = RandomForestClassifier(n_estimators=100, random_state=42)
rf_untuned.fit(X_tr_7, y_train)
untuned_val_f1 = f1_score(y_val, rf_untuned.predict(X_val_7), zero_division=0)

tuning_rows = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
     "best_params": str(gs.best_params_), "best_cv_accuracy": round(gs.best_score_, 4), "val_f1": round(gs_val_f1, 4)},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
     "best_params": str(rs.best_params_), "best_cv_accuracy": round(rs.best_score_, 4), "val_f1": round(rs_val_f1, 4)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "defaults", "best_cv_accuracy": "", "val_f1": round(untuned_val_f1, 4)}
]
pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)

# Select Winning Model Configuration
tuning_df = pd.DataFrame(tuning_rows)
best_row = tuning_df.sort_values("val_f1", ascending=False).iloc[0]
chosen_from = best_row["search"]

if chosen_from == "grid_search":
    best_params = gs.best_params_
elif chosen_from == "randomized_search":
    best_params = rs.best_params_
else:
    best_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

final_clf_model = RandomForestClassifier(random_state=42, **best_params)
final_pipeline = Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", final_clf_model)])
final_pipeline.fit(X_train, y_train)

# Threshold Tuning
val_probs = final_pipeline.predict_proba(X_val)[:, 1]
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresh_rows = []
chosen_threshold = 0.1

for t in threshold_list:
    y_pred_t = (val_probs >= t).astype(int)
    row = get_metrics_row(f"thresh_{t}", y_val, y_pred_t)
    thresh_rows.append({"threshold": t, "TN": row["TN"], "FP": row["FP"], "FN": row["FN"], "TP": row["TP"],
                        "accuracy": row["accuracy"], "precision": row["precision"], "recall": row["recall"], "f1": row["f1"]})
    if row["recall"] >= 0.80:
        chosen_threshold = t

pd.DataFrame(thresh_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

# Final Evaluation on Test Set
test_probs = final_pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= chosen_threshold).astype(int)
test_metrics = get_metrics_row("final_test", y_test, test_preds)
pd.DataFrame([{
    "threshold": chosen_threshold, "TN": test_metrics["TN"], "FP": test_metrics["FP"],
    "FN": test_metrics["FN"], "TP": test_metrics["TP"], "accuracy": test_metrics["accuracy"],
    "precision": test_metrics["precision"], "recall": test_metrics["recall"], "f1": test_metrics["f1"]
}]).to_csv("outputs/part_c_final_test.csv", index=False)

# Final Configuration Export
config_data = [{
    "selector_k": 7,
    "n_estimators": best_params.get("n_estimators", 100),
    "max_depth": "unlimited" if best_params.get("max_depth") is None else best_params.get("max_depth"),
    "min_samples_leaf": best_params.get("min_samples_leaf", 1),
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}]
pd.DataFrame(config_data).to_csv("outputs/part_c_final_config.csv", index=False)
print("Part C complete!")
