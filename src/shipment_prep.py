import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    df = raw.copy()
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Clean service_type
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # 3. Handle payment_mode missing values
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # 4. Clean declared_value into declared_value_inr
    df["declared_value_inr"] = (
        df["declared_value"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["declared_value_inr"] = pd.to_numeric(df["declared_value_inr"], errors="coerce")
    df = df.drop(columns=["declared_value"])
    
    # 5. Convert dispatch_time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    
    # 6. Treat parcel_weight_kg > 100 as missing sentinel
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # 7. Impute missing values with medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median())
        }
    
    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    return df, medians

def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df
