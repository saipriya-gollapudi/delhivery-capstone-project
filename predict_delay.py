import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

def main():
    # Load Input Files and Models
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    segments = pd.read_csv("outputs/hub_segments.csv")
    
    model = joblib.load("models/delay_model.pkl")
    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df["threshold"].iloc[0])
    
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    # Clean & Feature Engineer Data
    clean_df, _ = clean_shipments(new_raw, medians=medians)
    featured_df = build_features(clean_df, hubs)
    
    X_new = featured_df[feature_order]
    
    # Generate Probabilities and Action Recommendations
    probs = model.predict_proba(X_new)[:, 1]
    featured_df["delay_probability"] = probs.round(4)
    featured_df["flagged"] = (featured_df["delay_probability"] >= threshold).astype(int)
    
    # Map Segment Data
    featured_df = featured_df.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")
    
    def assign_action(row):
        if row["flagged"] == 1:
            return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"
        return "No action"

    featured_df["action"] = featured_df.apply(assign_action, axis=1)
    
    # Save Action Report
    cols_to_save = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    featured_df[cols_to_save].to_csv("outputs/new_shipment_actions.csv", index=False)
    print("New shipments successfully scored!")

if __name__ == "__main__":
    main()
