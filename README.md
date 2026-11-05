# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run
1. Clone the repository and navigate to root:
   git clone https://github.com/saipriya-gollapudi/delhivery-capstone-project.git
   cd delhivery-capstone-project
2. Install exact pinned dependencies:
   pip install -r requirements.txt
3. Execute the full end-to-end pipeline in sequence:
   python generate_data.py
   python part_a_clean_eda.py
   python part_b_regression.py
   python part_c_classification.py
   python part_d_segments.py
   python package_model.py
   python predict_delay.py

## Data findings
- Highest Delay Service Type: Express (48.97%)
- Delay Progression by Hub Tier: Tier 1 (4.00%), Tier 2 (21.50%), Tier 3 (44.00%)
- Distance Band Progression: 0-200 km (2.19%), 201-400 km (13.68%), 401-600 km (43.64%), 601+ km (85.23%)
- Top Correlated Features: promised_hours (+0.8370), distance_km (+0.4010), n_stops (+0.3120)

## Model results (regression)
- Baseline R2: -0.0210
- Winner: Gradient Boosting Regressor (Val RMSE: 6.28 hours, R2: 0.9310)

## Model results (classification)
- Blind Baseline Accuracy: 76.67%
- Selected Features: Top 7 features by F-score
- Chosen Alert Threshold: 0.5 (Validation Recall >= 80%)
- Test Performance: F1 = 0.7210, Precision = 0.6890, Recall = 0.7560

## Operations recommendation
- Escalate Hub Segment Delay Rate: 46.00%
- Mismatched Hubs: HUB07 (Tier 1 -> Watch), HUB13 (Tier 2 -> Escalate), HUB15 (Tier 2 -> Escalate), HUB22 (Tier 3 -> Watch)

## Production plan
- Monthly retraining schedule with fixed random_state seeds for reproducibility.

## Limitations
- Evaluated on synthetic data with limited delayed test samples.
