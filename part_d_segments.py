import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("data/modelling_table.csv")

# Hub Profile Generation
hub_prof = df.groupby("hub_id").agg(
    shipments=("is_delayed", "count"),
    mean_delivery_hours=("delivery_hours", lambda x: round(x.mean(), 4)),
    delay_rate=("is_delayed", lambda x: round(x.mean(), 4)),
    mean_distance_km=("distance_km", lambda x: round(x.mean(), 4)),
    mean_n_stops=("n_stops", lambda x: round(x.mean(), 4))
).reset_index().sort_values("hub_id")

hub_prof.to_csv("outputs/hub_profile.csv", index=False)

# K-Means Clustering on Scaled Features
features_to_scale = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaler = StandardScaler()
X_hub_scaled = scaler.fit_transform(hub_prof[features_to_scale])

# Separation metric computation
dist_matrix = squareform(pdist(X_hub_scaled, metric="euclidean"))
sep_rows = []

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_hub_scaled)
    
    # Calculate Cluster Sizes
    counts = pd.Series(labels).value_counts().sort_values(ascending=False).astype(str).tolist()
    c_sizes = "-".join(counts)
    
    across_dists, within_dists = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within_dists.append(dist_matrix[i, j])
            else:
                across_dists.append(dist_matrix[i, j])
    
    ratio = np.mean(across_dists) / np.mean(within_dists)
    sep_rows.append({"K": k, "across_over_within": round(ratio, 4), "cluster_sizes": c_sizes})

pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

# Segment assignment for K=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
hub_prof["cluster"] = km3.fit_predict(X_hub_scaled)

# Order segments by mean delay rate
cluster_order = hub_prof.groupby("cluster")["delay_rate"].mean().sort_values().index
segment_map = {cluster_order[0]: "Standard", cluster_order[1]: "Watch", cluster_order[2]: "Escalate"}
hub_prof["segment"] = hub_prof["cluster"].map(segment_map)

# Merge Hub Metadata
hubs_info = pd.read_csv("data/hubs.csv")
hub_seg = hub_prof.merge(hubs_info[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
hub_seg[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)

# Segment Profiles
seg_prof = hub_prof.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", lambda x: round(x.mean(), 4)),
    delay_rate=("delay_rate", lambda x: round(x.mean(), 4)),
    mean_distance_km=("mean_distance_km", lambda x: round(x.mean(), 4)),
    mean_n_stops=("mean_n_stops", lambda x: round(x.mean(), 4))
).reset_index()
seg_prof.to_csv("outputs/segment_profile.csv", index=False)

# Check Tier Mismatches
tier_expected = {1: "Standard", 2: "Watch", 3: "Escalate"}
hub_seg["expected_segment"] = hub_seg["hub_tier"].map(tier_expected)
mismatches = hub_seg[hub_seg["segment"] != hub_seg["expected_segment"]]
mismatches[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("Part D complete!")
