import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

# Train, validation, test splits (70% / 15% / 15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

# Check data leakage in scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_max = np.max(np.abs(np.mean(X_train_scaled, axis=0)))
test_max = np.max(np.abs(np.mean(X_test_scaled, axis=0)))

checks_b = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", round(train_max, 9)),
    ("test_scaled_max_abs_mean", round(test_max, 4)),
]
pd.DataFrame(checks_b, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

# Baseline model
mean_val_pred = np.full_like(y_val, y_train.mean())
base_mae = mean_absolute_error(y_val, mean_val_pred)
base_mse = mean_squared_error(y_val, mean_val_pred)
base_rmse = np.sqrt(base_mse)
base_r2 = r2_score(y_val, mean_val_pred)

# Decision Tree Depth tuning
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_records = []
best_depth, best_dt_val_rmse = None, float("inf")

for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    tr_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
    v_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
    depth_records.append({"max_depth": "unlimited" if d is None else str(d), "train_rmse": round(tr_rmse, 4), "val_rmse": round(v_rmse, 4)})
    if v_rmse < best_dt_val_rmse:
        best_dt_val_rmse = v_rmse
        best_depth = d

pd.DataFrame(depth_records).to_csv("outputs/part_b_tree_depth.csv", index=False)

# Model comparison
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

metrics_list = [{
    "model": "mean_baseline", "split": "val",
    "MAE": round(base_mae, 4), "MSE": round(base_mse, 4),
    "RMSE": round(base_rmse, 4), "R2": round(base_r2, 4)
}]

best_model_name, lowest_val_rmse = None, float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, preds)
    metrics_list.append({
        "model": name, "split": "val",
        "MAE": round(mae, 4), "MSE": round(mse, 4),
        "RMSE": round(rmse, 4), "R2": round(r2, 4)
    })
    if rmse < lowest_val_rmse:
        lowest_val_rmse = rmse
        best_model_name = name

# Learning Rate Experiment
lr_records = []
for lr in [0.01, 0.1, 1.0]:
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gbr.fit(X_train, y_train)
    preds = gbr.predict(X_val)
    lr_records.append({
        "learning_rate": lr,
        "val_rmse": round(np.sqrt(mean_squared_error(y_val, preds)), 4),
        "val_r2": round(r2_score(y_val, preds), 4)
    })
pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# Cross-Validation on top 2 models
val_metrics_df = pd.DataFrame(metrics_list)
top_2_names = val_metrics_df[val_metrics_df["model"] != "mean_baseline"].sort_values("RMSE").head(2)["model"].tolist()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_records = []
for name in top_2_names:
    mdl = models[name]
    scores = cross_val_score(mdl, X_train_val, y_train_val, cv=cv, scoring="r2")
    cv_records.append({
        "model": name,
        "mean_r2": round(np.mean(scores), 4),
        "std_r2": round(np.std(scores), 4)
    })
pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)

# Final test scoring on the best model
final_model = models[best_model_name]
test_preds = final_model.predict(X_test)
metrics_list.append({
    "model": best_model_name, "split": "test",
    "MAE": round(mean_absolute_error(y_test, test_preds), 4),
    "MSE": round(mean_squared_error(y_test, test_preds), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, test_preds)), 4),
    "R2": round(r2_score(y_test, test_preds), 4)
})

pd.DataFrame(metrics_list).to_csv("outputs/part_b_metrics.csv", index=False)
print("Part B complete!")
