# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load Data & Model
# ------------------------------
df = pd.read_csv("data/transactions.csv")
features = ["price", "discount_percent", "refund_flag", "profit_margin"]

model = joblib.load("models/xgb_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

# ------------------------------
# 2. Predict Opportunity Scores
# ------------------------------
df['opportunity_score'] = model.predict_proba(df[features])[:,1]*100

# ------------------------------
# 3. SHAP Explanations
# ------------------------------
shap_values = explainer.shap_values(df[features])
df['shap_dict'] = [{f: val for f, val in zip(features, shap_val)} for shap_val in shap_values]

# ------------------------------
# 4. Dynamic Recommendations Based on Top SHAP Features
# ------------------------------
def dynamic_recommendation(row, top_n=2):
    shap_vals = row['shap_dict']
    top_features = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    rec = []
    for f, val in top_features:
        if f == 'price':
            rec.append(f"Review pricing (impact {val:.2f})")
        elif f == 'discount_percent':
            rec.append(f"Adjust discount strategy (impact {val:.2f})")
        elif f == 'refund_flag':
            rec.append(f"Investigate refunds (impact {val:.2f})")
        elif f == 'profit_margin':
            rec.append(f"Optimize cost/profit margin (impact {val:.2f})")
    return ", ".join(rec)

df['recommendation'] = df.apply(dynamic_recommendation, axis=1)

# ------------------------------
# 5. Streamlit Dashboard
# ------------------------------
st.set_page_config(page_title="AI Profit Opportunity Detector", layout="wide")
st.title("💰 AI Profit Opportunity Detector")
st.write("Industry-level dashboard with dynamic, SHAP-based recommendations.")

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Opportunity Score", 0, 100, 20)
region_filter = st.sidebar.multiselect("Region", options=df['region'].unique(), default=df['region'].unique())

filtered_df = df[(df['opportunity_score'] >= min_score) & (df['region'].isin(region_filter))]

# ------------------------------
# Top Opportunities Table
# ------------------------------
st.subheader(f"Top Opportunities ({len(filtered_df)})")
def color_score(val):
    if val > 90:
        color = 'background-color: #FF4B4B; color:white'
    elif val > 70:
        color = 'background-color: #FFA500; color:white'
    else:
        color = 'background-color: #90EE90; color:black'
    return color

st.dataframe(
    filtered_df[['transaction_id','product_id','region','opportunity_score','recommendation']].style.applymap(color_score, subset=['opportunity_score'])
)

# ------------------------------
# Top 10 Priority Opportunities
# ------------------------------
st.subheader("Top 10 Priority Opportunities")
top10 = filtered_df.sort_values('opportunity_score', ascending=False).head(10)
st.dataframe(top10[['transaction_id','product_id','region','opportunity_score','recommendation']])

# ------------------------------
# SHAP Feature Importance Summary (Top 10)
# ------------------------------
st.subheader("Feature Contribution Summary (Top 10 Transactions)")
top10_shap = top10[['shap_dict']]
feature_data = pd.DataFrame(top10_shap['shap_dict'].tolist())
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=feature_data, ci=None, estimator=lambda x: sum(abs(x)))
plt.title("Total SHAP Contribution per Feature (Top 10)")
st.pyplot(fig)

# ------------------------------
# 6. SHAP Waterfall for Individual Top Transaction
# ------------------------------
st.subheader("Individual Transaction Feature Impact")

transaction_id = st.selectbox(
    "Select Transaction ID (Top 10 Only)",
    top10['transaction_id'].tolist()
)

selected_row = top10[top10['transaction_id'] == transaction_id].iloc[0]
selected_index = selected_row.name  # row index in dataframe

# Generate waterfall plot for this transaction
st.write(f"**Transaction ID:** {transaction_id}")
st.write("Feature contributions to opportunity score:")

shap_vals_individual = shap_values[selected_index]
shap.initjs()
fig2, ax2 = plt.subplots(figsize=(6,4))
shap.bar_plot(shap_vals_individual, feature_names=features, max_display=len(features))
st.pyplot(fig2)

# ------------------------------
# Opportunity Score Distribution
# ------------------------------
st.subheader("Opportunity Score Distribution")
st.bar_chart(filtered_df['opportunity_score'])

# ------------------------------
# CSV Export
# ------------------------------
st.download_button(
    label="Export Filtered Opportunities",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_opportunities.csv",
    mime="text/csv"
)
