import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features, FEATURES

# Read raw inputs
raw_df = pd.read_csv("data/raw_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# Record "Before" metrics
raw_rows = len(raw_df)
dedup_df = raw_df.drop_duplicates()
duplicate_rows_removed = raw_rows - len(dedup_df)
rows_after_dedup = len(dedup_df)

missing_distance_km_before = dedup_df["distance_km"].isna().sum()
sentinel_weight_rows = (dedup_df["parcel_weight_kg"] > 100).sum()
missing_parcel_weight_before = (dedup_df["parcel_weight_kg"].isna() & (dedup_df["parcel_weight_kg"] <= 100)).sum()
missing_declared_value_before = dedup_df["declared_value"].isna().sum()
missing_payment_mode_before = dedup_df["payment_mode"].isna().sum()
service_type_spellings_before = dedup_df["service_type"].nunique()

# Clean shipments
clean_df, medians = clean_shipments(raw_df)
service_type_spellings_after = clean_df["service_type"].nunique()

# Build features
modelling_df = build_features(clean_df, hubs_df)
modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)

# Write modelling table
modelling_df.to_csv("data/modelling_table.csv", index=False)

# Compute checks metrics
rows_after_hub_merge = len(modelling_df)
nulls_in_features_and_target = modelling_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
delayed_count = modelling_df["is_delayed"].sum()
delay_rate = round(modelling_df["is_delayed"].mean(), 4)

q1 = modelling_df["delivery_hours"].quantile(0.25)
q3 = modelling_df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_b = q1 - 1.5 * iqr
upper_b = q3 + 1.5 * iqr
outliers = ((modelling_df["delivery_hours"] < lower_b) | (modelling_df["delivery_hours"] > upper_b)).sum()

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", round(medians["distance_km"], 4)),
    ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
    ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", round(q1, 4)),
    ("hours_q3", round(q3, 4)),
    ("hours_lower_bound", round(lower_b, 4)),
    ("hours_upper_bound", round(upper_b, 4)),
    ("hours_outlier_count", outliers),
    ("declared_value_mean_inr", round(modelling_df["declared_value_inr"].mean(), 4)),
    ("declared_value_median_inr", round(modelling_df["declared_value_inr"].median(), 4)),
    ("log_declared_value_mean", round(modelling_df["log_declared_value"].mean(), 4)),
    ("log_declared_value_median", round(modelling_df["log_declared_value"].median(), 4)),
]
pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

# Delay evidence by group table
def get_group_stats(df, col, dim_name):
    res = df.groupby(col).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", lambda x: round(x.mean(), 4)),
        mean_hours=("delivery_hours", lambda x: round(x.mean(), 4))
    ).reset_index()
    res.rename(columns={col: "group"}, inplace=True)
    res.insert(0, "dimension", dim_name)
    return res

modelling_df["distance_band"] = pd.cut(
    modelling_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    include_lowest=True
)

st_df = get_group_stats(modelling_df, "service_type", "service_type")
ht_df = get_group_stats(modelling_df, "hub_tier", "hub_tier")
db_df = get_group_stats(modelling_df, "distance_band", "distance_band")

delay_by_group = pd.concat([st_df, ht_df, db_df], ignore_index=True)
delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# Numeric correlations
num_cols = modelling_df.select_dtypes(include=[np.number]).columns
corr_matrix = modelling_df[num_cols].corr().round(4)
corr_matrix.reset_index().to_csv("outputs/part_a_correlations.csv", index=False)
print("Part A complete!")
