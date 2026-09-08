
# ============================================================
# DIGITAL ADDICTION INTELLIGENCE SYSTEM
# Machine Learning + Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Digital Addiction Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #fff7f2, #ffffff);
    }

    /* Main container */
    .block-container {
        padding: 2rem 3rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #5a3e36;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #fffaf7;
        border: 1px solid #f0ddd3;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 700;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 15px;
    }

    /* Input fields */
    input, textarea {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

DATASET_FILE = "Digital_Addiction_Dataset - User_Behavior_Dataset.csv"

RISK_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

RISK_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}


# ============================================================
# DATA LOADING FUNCTION
# ============================================================

@st.cache_data
def load_dataset(file_path):
    """
    Load and clean the digital addiction dataset.
    """

    try:
        data = pd.read_csv(file_path, skiprows=2)

        # Remove unnecessary spaces from column names
        data.columns = data.columns.str.strip()

        return data

    except FileNotFoundError:
        st.error(
            f"Dataset not found: {file_path}"
        )
        st.stop()

    except Exception as error:
        st.error(
            f"Error while loading dataset: {error}"
        )
        st.stop()


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(data):
    """
    Clean risk category and create numerical risk labels.
    """

    data = data.copy()

    if "Risk Category" not in data.columns:
        st.error("Required column 'Risk Category' is missing.")
        st.stop()

    # Clean Risk Category
    data["Risk Category"] = (
        data["Risk Category"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    # Convert category into numerical labels
    data["Risk Label"] = data["Risk Category"].map(RISK_MAPPING)

    # Remove rows without valid risk labels
    data = data.dropna(subset=["Risk Label"])

    return data


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(data):
    """
    Train Logistic Regression model.
    """

    required_features = [
        "Age",
        "Total Screen Time hrs",
        "Addiction Risk Score"
    ]

    # Check available features
    available_features = [
        feature
        for feature in required_features
        if feature in data.columns
    ]

    if len(available_features) < 3:
        return None, None, None, None

    model_data = data[
        available_features + ["Risk Label"]
    ].dropna()

    if model_data.empty:
        return None, None, None, None

    X = model_data[available_features]
    y = model_data["Risk Label"]

    # Check whether enough classes are available
    if y.nunique() < 2:
        return None, None, None, None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(X_train, y_train)

    # Accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, available_features, accuracy, X


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk_meter(screen_time, addiction_score):
    """
    Calculate a simple risk meter for visualization.
    """

    risk_score = (
        screen_time * 5
        + addiction_score * 0.5
    ) / 2

    return int(
        max(0, min(risk_score, 100))
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def show_recommendation(risk_score):
    """
    Display personalized recommendation.
    """

    st.subheader("💡 Personalized Recommendation")

    if risk_score >= 70:

        st.error(
            "🔴 High digital usage detected. "
            "Reduce unnecessary screen time, avoid late-night "
            "device usage and take regular breaks."
        )

    elif risk_score >= 40:

        st.warning(
            "🟡 Moderate digital usage detected. "
            "Try setting screen-time limits and take regular "
            "breaks during device usage."
        )

    else:

        st.success(
            "🟢 Your current usage pattern appears relatively healthy. "
            "Continue maintaining a balanced digital routine."
        )


# ============================================================
# LOAD DATA
# ============================================================

df = load_dataset(DATASET_FILE)

df = preprocess_data(df)


# ============================================================
# TRAIN MODEL
# ============================================================

model, features, model_accuracy, model_data = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Digital Intelligence")

    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Dataset",
            "📈 Overview",
            "🧠 Behavior Analysis",
            "🔮 Prediction",
            "🤖 AI Assistant"
        ]
    )

    st.markdown("---")

    st.info(
        "Digital Addiction Intelligence System\n\n"
        "Machine Learning based behavioral analysis "
        "and risk prediction dashboard."
    )


# ============================================================
# PAGE 1 - DATASET
# ============================================================

if selected_page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    st.write(
        "Explore, filter and search the digital addiction dataset."
    )

    # Dataset information
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Total Columns",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("---")

    # Column selection
    selected_columns = st.multiselect(
        "Select Columns",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    if selected_columns:

        filtered_df = df[selected_columns].copy()

        # Search
        search_text = st.text_input(
            "🔎 Search Dataset"
        )

        if search_text:

            mask = filtered_df.astype(str).apply(
                lambda row: row.str.contains(
                    search_text,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )

            filtered_df = filtered_df[mask]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )

    else:

        st.warning(
            "Please select at least one column."
        )


# ============================================================
# PAGE 2 - OVERVIEW
# ============================================================

elif selected_page == "📈 Overview":

    st.title("📈 Digital Addiction Overview")

    st.write(
        "High-level statistics and distribution of digital addiction risk."
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Users",
        len(df)
    )

    if "Total Screen Time hrs" in df.columns:

        avg_screen_time = df[
            "Total Screen Time hrs"
        ].mean()

        col2.metric(
            "📱 Avg Screen Time",
            f"{avg_screen_time:.2f} hrs"
        )

    if "Addiction Risk Score" in df.columns:

        avg_score = df[
            "Addiction Risk Score"
        ].mean()

        col3.metric(
            "⚠ Avg Addiction Score",
            f"{avg_score:.2f}"
        )

    high_risk_users = len(
        df[df["Risk Category"] == "High"]
    )

    col4.metric(
        "🔴 High Risk Users",
        high_risk_users
    )

    st.markdown("---")

    # Risk distribution
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        risk_counts = (
            df["Risk Category"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Category",
            "Users"
        ]

        fig_pie = px.pie(
            risk_counts,
            names="Risk Category",
            values="Users",
            hole=0.45
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:

        st.subheader("Risk Category Count")

        fig_bar = px.bar(
            risk_counts,
            x="Risk Category",
            y="Users",
            text="Users"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # Model performance
    if model_accuracy is not None:

        st.subheader("🤖 Machine Learning Model")

        st.metric(
            "Logistic Regression Accuracy",
            f"{model_accuracy * 100:.2f}%"
        )


# ============================================================
# PAGE 3 - BEHAVIOR ANALYSIS
# ============================================================

elif selected_page == "🧠 Behavior Analysis":

    st.title("🧠 Behavioral Analysis")

    st.write(
        "Analyze screen-time behavior and addiction risk patterns."
    )

    if "Total Screen Time hrs" in df.columns:

        st.subheader("📱 Screen Time Distribution")

        histogram = px.histogram(
            df,
            x="Total Screen Time hrs",
            nbins=20,
            title="Distribution of Screen Time"
        )

        st.plotly_chart(
            histogram,
            use_container_width=True
        )

    if (
        "Total Screen Time hrs" in df.columns
        and "Addiction Risk Score" in df.columns
    ):

        st.subheader(
            "📊 Screen Time vs Addiction Risk"
        )

        scatter = px.scatter(
            df,
            x="Total Screen Time hrs",
            y="Addiction Risk Score",
            color="Risk Category",
            hover_data=df.columns.tolist(),
            title="Relationship Between Screen Time and Addiction Score"
        )

        st.plotly_chart(
            scatter,
            use_container_width=True
        )

    if (
        "Age" in df.columns
        and "Addiction Risk Score" in df.columns
    ):

        st.subheader(
            "👤 Age vs Addiction Risk"
        )

        age_chart = px.scatter(
            df,
            x="Age",
            y="Addiction Risk Score",
            color="Risk Category",
            title="Age and Addiction Risk Relationship"
        )

        st.plotly_chart(
            age_chart,
            use_container_width=True
        )


# ============================================================
# PAGE 4 - PREDICTION
# ============================================================

elif selected_page == "🔮 Prediction":

    st.title("🔮 Smart Addiction Risk Prediction")

    st.write(
        "Enter user information to predict the digital addiction risk."
    )

    if model is None or features is None:

        st.error(
            "Machine Learning model could not be trained. "
            "Please check the required dataset columns."
        )

    else:

        col1, col2, col3 = st.columns(3)

        age = col1.slider(
            "👤 Age",
            min_value=10,
            max_value=60,
            value=25
        )

        screen_time = col2.slider(
            "📱 Screen Time (Hours)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )

        addiction_score = col3.slider(
            "⚠ Addiction Risk Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        st.markdown("---")

        if st.button(
            "🚀 Predict Addiction Risk",
            use_container_width=True
        ):

            input_data = pd.DataFrame(
                [[
                    age,
                    screen_time,
                    addiction_score
                ]],
                columns=features
            )

            prediction = model.predict(
                input_data
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            result = RISK_LABELS.get(
                prediction,
                "Unknown"
            )

            confidence = (
                max(probabilities) * 100
            )

            risk_meter = calculate_risk_meter(
                screen_time,
                addiction_score
            )

            st.subheader(
                "📊 Prediction Result"
            )

            col1, col2 = st.columns(2)

            with col1:

                if result == "High":

                    st.error(
                        f"🔴 HIGH RISK\n\n"
                        f"Confidence: {confidence:.2f}%"
                    )

                elif result == "Medium":

                    st.warning(
                        f"🟡 MEDIUM RISK\n\n"
                        f"Confidence: {confidence:.2f}%"
                    )

                else:

                    st.success(
                        f"🟢 LOW RISK\n\n"
                        f"Confidence: {confidence:.2f}%"
                    )

            with col2:

                st.metric(
                    "Risk Meter",
                    f"{risk_meter}%"
                )

                st.progress(
                    risk_meter / 100
                )

            show_recommendation(
                risk_meter
            )


# ============================================================
# PAGE 5 - AI ASSISTANT
# ============================================================

elif selected_page == "🤖 AI Assistant":

    st.title("🤖 Digital Addiction AI Assistant")

    st.write(
        "Ask questions about the dataset and digital addiction statistics."
    )

    question = st.text_input(
        "💬 Ask your question",
        placeholder="Example: What is the average screen time?"
    )

    if question:

        question = question.lower().strip()

        if "screen" in question:

            if "Total Screen Time hrs" in df.columns:

                average = df[
                    "Total Screen Time hrs"
                ].mean()

                st.success(
                    f"📱 Average screen time is "
                    f"**{average:.2f} hours**."
                )

        elif "high risk" in question:

            count = len(
                df[df["Risk Category"] == "High"]
            )

            st.success(
                f"🔴 There are **{count} high-risk users**."
            )

        elif "medium risk" in question:

            count = len(
                df[df["Risk Category"] == "Medium"]
            )

            st.success(
                f"🟡 There are **{count} medium-risk users**."
            )

        elif "low risk" in question:

            count = len(
                df[df["Risk Category"] == "Low"]
            )

            st.success(
                f"🟢 There are **{count} low-risk users**."
            )

        elif "user" in question or "users" in question:

            st.success(
                f"👥 Total users in the dataset: **{len(df)}**"
            )

        elif "score" in question:

            if "Addiction Risk Score" in df.columns:

                average_score = df[
                    "Addiction Risk Score"
                ].mean()

                st.success(
                    f"⚠ Average addiction risk score is "
                    f"**{average_score:.2f}**."
                )

        elif "accuracy" in question:

            if model_accuracy is not None:

                st.success(
                    f"🤖 Logistic Regression model accuracy is "
                    f"**{model_accuracy * 100:.2f}%**."
                )

        else:

            st.info(
                "I can answer questions about:\n\n"
                "• Screen time\n"
                "• High risk users\n"
                "• Medium risk users\n"
                "• Low risk users\n"
                "• Total users\n"
                "• Addiction score\n"
                "• Model accuracy"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Digital Addiction Intelligence System | "
    "Python • Pandas • Plotly • Scikit-learn • Streamlit"
)

