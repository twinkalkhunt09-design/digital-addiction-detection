import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression

st.set_page_config("Digital Addiction Intelligence System", layout="wide")

# ================= 🎨 MAIN AREA CSS ONLY =================
st.markdown("""
<style>

/* 🌟 Background (LIGHT PREMIUM - Peach Cream) */
body {
    background: linear-gradient(135deg, #fff6f0, #fdfdfd);
}

/* Container */
.block-container {
    padding: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: #5a3e36;
    font-weight: 700;
}

/* 🌟 MAIN CONTENT CARDS ONLY */
[data-testid="stMetric"],
.stDataFrame,
.stPlotlyChart,
.stTextInput,
.stSlider,
.stSelectbox {
    background: #fffaf7;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    border: 1px solid #f1e3dc;
    transition: 0.3s;
}

/* Hover effect */
.stPlotlyChart:hover,
.stDataFrame:hover,
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.12);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff9a8b, #ff6a88);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #ff9a8b, #ff6a88);
}

/* Labels */
label {
    color: #4a3f3a !important;
}

/* Table text */
.stDataFrame div {
    color: black !important;
}

/* Input text */
input, textarea {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
df = pd.read_csv("Digital_Addiction_Dataset - User_Behavior_Dataset.csv", skiprows=2)
df.columns = df.columns.str.strip()

# ================= CLEANING =================
df["Risk Category"] = df["Risk Category"].astype(str).str.strip().str.capitalize()

df["Risk Label"] = df["Risk Category"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

df = df.dropna(subset=["Risk Label"])

# ================= MODEL =================
features = ["Age", "Total Screen Time hrs", "Addiction Risk Score"]
features = [f for f in features if f in df.columns]

X = df[features].dropna()
y = df.loc[X.index, "Risk Label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ================= SIDEBAR (UNCHANGED) =================
with st.sidebar:
    selected = option_menu("Main Menu",
                           ["Dataset", "Overview", "Behavior Analysis", "Prediction", "AI Assistant"],
                           icons=["table", "bar-chart", "graph-up", "cpu", "robot"],
                           menu_icon="activity")

# ================= DATASET =================
if selected == "Dataset":
    st.title("📊 Data Explorer")

    selected_column = st.multiselect("Select Columns", df.columns, default=df.columns)
    filtered_df = df[selected_column]

    search = st.text_input("Search")

    if search:
        filtered_df = filtered_df[filtered_df.astype(str).apply(
            lambda row: row.str.contains(search, case=False).any(), axis=1
        )]

    st.dataframe(filtered_df)

# ================= OVERVIEW =================
if selected == "Overview":
    st.title("📈 Digital Addiction Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Users", len(df))
    col2.metric("Avg Screen Time", f"{df['Total Screen Time hrs'].mean():.1f} hrs")
    col3.metric("Avg Addiction Score", f"{df['Addiction Risk Score'].mean():.1f}")
    col4.metric("High Risk Users", len(df[df["Risk Category"] == "High"]))

    st.plotly_chart(px.pie(df, names="Risk Category"))

# ================= BEHAVIOR =================
if selected == "Behavior Analysis":
    st.title("🧠 Behavior Insights")

    st.plotly_chart(px.histogram(df, x="Total Screen Time hrs"))
    st.plotly_chart(px.scatter(df,
                               x="Total Screen Time hrs",
                               y="Addiction Risk Score",
                               color="Risk Category"))

# ================= PREDICTION =================
if selected == "Prediction":
    st.title("🔮 Smart Addiction Prediction")

    col1, col2, col3 = st.columns(3)

    age = col1.slider("Age", 10, 60, 25)
    screen = col2.slider("Screen Time (hrs)", 1.0, 15.0, 5.0)
    score = col3.slider("Addiction Score", 0.0, 100.0, 50.0)

    if st.button("🚀 Predict Risk"):

        input_df = pd.DataFrame([[age, screen, score]], columns=features)

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        label_map = {0: "Low", 1: "Medium", 2: "High"}
        result = label_map[pred]

        confidence = round(max(prob)*100, 2)
        risk_score = int((screen*5 + score*0.5) / 2)

        st.subheader("📊 Prediction Result")

        if result == "High":
            st.error(f"⚠ HIGH RISK ({confidence}%)")
        elif result == "Medium":
            st.warning(f"⚠ MEDIUM RISK ({confidence}%)")
        else:
            st.success(f"✅ LOW RISK ({confidence}%)")

        st.progress(min(risk_score/100, 1.0))
        st.write(f"Risk Meter: {risk_score}%")

        st.subheader("💡 Recommendations")

        if risk_score > 70:
            st.error("Reduce screen time immediately, avoid late-night usage.")
        elif risk_score > 40:
            st.warning("Take breaks, control social media usage.")
        else:
            st.success("Healthy usage pattern 👍")

# ================= AI =================
if selected == "AI Assistant":
    st.title("🤖 AI Assistant")

    q = st.text_input("Ask anything")

    if q:
        q = q.lower()

        if "screen" in q:
            st.success(f"Avg Screen Time: {df['Total Screen Time hrs'].mean():.2f}")
        elif "risk" in q:
            st.success(f"High Risk Users: {len(df[df['Risk Category']=='High'])}")
        else:
            st.info("Try: screen time, risk")