# train.py
import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load Data
# ------------------------------
df = pd.read_csv("data/transactions.csv")
print("Data loaded.")

features = ["price", "discount_percent", "refund_flag", "profit_margin"]
X = df[features]

# ------------------------------
# 2. Create Pseudo-Labels (Anomaly-based opportunity detection)
# ------------------------------
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
df['opportunity_label'] = iso.fit_predict(X)
df['opportunity_label'] = df['opportunity_label'].apply(lambda x: 1 if x==-1 else 0)
y = df['opportunity_label']

# ------------------------------
# 3. Train XGBoost Classifier
# ------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X, y)
print("XGBoost model trained.")

# ------------------------------
# 4. Save Model
# ------------------------------
joblib.dump(model, "models/xgb_model.pkl")
print("Model saved to models/xgb_model.pkl")

# ------------------------------
# 5. SHAP Explainer
# ------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
joblib.dump(explainer, "models/shap_explainer.pkl")
print("SHAP explainer saved.")

# Optional: Save SHAP dicts for inspection
df['shap_dict'] = [{f: val for f, val in zip(features, shap_val)} for shap_val in shap_values]
df.to_csv("data/shap_values.csv", index=False)
print("SHAP values saved.")
