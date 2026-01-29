import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import base64
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ================= LOGO =================
logo = Image.open("logo.jpeg")  
buffered = io.BytesIO()
logo.save(buffered, format="PNG")
logo_base64 = base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    f"""
    <style>
    .logo-img {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        border: 3px solid #4CAF50;
        margin-bottom: 10px;
    }}
    .title {{
        text-align: center;
        color: #2E86C1;
    }}
    .subtitle {{
        text-align: center;
        color: #566573;
        font-size: 18px;
    }}
    .footer {{
        text-align: center;
        color: #999999;
        font-size: 14px;
        margin-top: 20px;
    }}
    .result-card {{
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }}
    .summary-card {{
        background-color:#f2f2f2;
        border-radius:10px;
        padding:10px;
        margin-bottom:10px;
    }}
    </style>
    <img src="data:image/png;base64,{logo_base64}" class="logo-img">
    """,
    unsafe_allow_html=True
)

# ================= TITLE =================
st.markdown('<div class="title"><h1>📊 Customer Churn Prediction System</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a customer will <b>Churn: YES or NO</b></div>', unsafe_allow_html=True)
st.markdown("---")

# ================= LOAD FILES =================
model = pickle.load(open("gb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ================= SIDEBAR INPUT =================
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# ================= CUSTOMER SUMMARY CARD =================
st.markdown("### Customer Profile Summary")
st.markdown(f"""
<div class="summary-card">
**Gender:** {gender}  <br>
**Senior Citizen:** {SeniorCitizen}  <br>
**Partner:** {Partner}  <br>
**Dependents:** {Dependents}  <br>
**Tenure:** {tenure} months  <br>
**Contract:** {Contract}  <br>
**Monthly Charges:** ${MonthlyCharges}  <br>
**Total Charges:** ${TotalCharges}  <br>
</div>
""", unsafe_allow_html=True)

# ================= THRESHOLD SLIDER =================
threshold = st.slider("Set Churn Threshold", 0.0, 1.0, 0.5, 0.01)

# ================= INPUT DATA =================
input_data = {
    "gender": gender,
    "SeniorCitizen": int(SeniorCitizen == "Yes"),
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

df = pd.DataFrame([input_data])

# ================= DATA PREPROCESS =================
df = pd.get_dummies(df, drop_first=True)
for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# Scale
df_scaled = scaler.transform(df)

# ================= PREDICTION =================
if st.button("🔮 Predict Churn"):
    prob = model.predict_proba(df_scaled)[0][1]

    # Animated probability bar
    st.markdown("### Churn Probability")
    progress = st.progress(0)
    for i in range(int(prob*100)+1):
        progress.progress(i)

    # Risk Category
    if prob >= threshold:
        risk = "High Risk"
    elif prob >= threshold*0.7:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    # Churn YES / NO
    churn_pred = "YES" if prob >= threshold else "NO"

    # Color-coded card
    if churn_pred == "YES":
        color_bg = "#F1948A"
        color_text = "#7B241C"
    else:
        color_bg = "#ABEBC6"
        color_text = "#145A32"

    st.markdown(
        f'<div class="result-card" style="background-color:{color_bg}; color:{color_text};">'
        f'Prediction: {churn_pred} | {risk} | Probability: {prob:.2f}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.write("DEBUG Probability:", prob)

# ================= FOOTER =================
st.markdown('<div class="footer">Developed by  Mahnoor Khan</div>', unsafe_allow_html=True)
