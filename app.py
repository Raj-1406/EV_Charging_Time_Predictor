import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from datetime import date, time


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart EV Charging Time Predictor",
    page_icon="🔋",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ev_charging_time_predictor.pkl"
PREVIEW_PATH = BASE_DIR / "dataset_preview.csv"


# --------------------------------------------------
# 2. LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()


# --------------------------------------------------
# 3. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 3.5rem;
    padding-bottom: 2.5rem;
}

.hero {
    padding: 1.5rem 1.8rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 18px;
    margin-bottom: 1.5rem;
    background: linear-gradient(
        120deg,
        rgba(34, 197, 94, 0.12),
        rgba(59, 130, 246, 0.06)
    );
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 750;
    line-height: 1.25;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    opacity: 0.8;
    margin: 0;
}

.result-card {
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(34, 197, 94, 0.45);
    background: rgba(34, 197, 94, 0.09);
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.result-label {
    font-size: 1rem;
    opacity: 0.8;
}

.result-value {
    font-size: 2.5rem;
    font-weight: 750;
    margin: 0.4rem 0;
    line-height: 1.25;
}

.result-detail {
    font-size: 1rem;
    opacity: 0.8;
}
</style>

<div class="hero">
    <div class="hero-title">
        🔋 Smart EV Charging Time Predictor
    </div>
    <p class="hero-subtitle">
        Estimate EV charging-session duration using
        a trained machine learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. THREE MAIN TABS
# --------------------------------------------------

tab_predict, tab_dataset, tab_about = st.tabs([
    "⚡ Predict Duration",
    "📊 Dataset & Model",
    "ℹ️ About Project"
])


# ==================================================
# TAB 1: PREDICT DURATION
# ==================================================

with tab_predict:

    st.subheader("Plan Your Charging Session")

    st.write(
        "Enter the details below to estimate "
        "your expected EV charging duration."
    )

    # User inputs
    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            company = st.selectbox(
                "Charger Company Category",
                options=[0, 1],
                format_func=lambda x: f"Category {x}"
            )

            charger_type = st.selectbox(
                "Charger Type",
                options=[0, 1],
                format_func=lambda x: (
                    "Slow (0)" if x == 0 else "Fast (1)"
                )
            )

            locations = [
                "hotel",
                "company",
                "apartment",
                "resort",
                "public area",
                "public institution",
                "restaurant",
                "accommodation",
                "golf",
                "sightseeing",
                "market",
                "camping",
                "public parking lot",
                "bus garage"
            ]

            location = st.selectbox(
                "Charging Location",
                options=locations
            )

        with col2:

            start_date = st.date_input(
                "Planned Charging Date",
                value=date.today()
            )

            start_time = st.time_input(
                "Planned Start Time",
                value=time(18, 0),
                step=900
            )

            st.caption(
                "Day, month and weekend status are "
                "calculated automatically from the date."
            )

        submitted = st.form_submit_button(
            "⚡ Predict Charging Duration",
            type="primary",
            use_container_width=True
        )

    # Automatically derive time features
    day_of_week = start_date.strftime("%A")
    month = start_date.month
    is_weekend = start_date.weekday() >= 5
    start_hour = start_time.hour

    st.caption(
        f"Selected: {day_of_week} · "
        f"{start_date.strftime('%B')} · "
        f"{'Weekend' if is_weekend else 'Weekday'} · "
        f"Start hour: {start_hour:02d}:00"
    )

    # Generate prediction
    if submitted:

        input_data = pd.DataFrame([{
            "ChargerCompany": company,
            "Location": location,
            "ChargerType": charger_type,
            "StartHour": start_hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            # Convert predicted minutes to a non-negative
            # integer for display.
            predicted_minutes = max(0, int(round(prediction)))

            hours, minutes = divmod(predicted_minutes, 60)

            # Correct singular/plural wording
            if hours == 0:
                duration_text = (
                    f"{minutes} "
                    f"{'minute' if minutes == 1 else 'minutes'}"
                )
            else:
                hour_text = "hour" if hours == 1 else "hours"
                minute_text = "minute" if minutes == 1 else "minutes"

                duration_text = (
                    f"{hours} {hour_text} "
                    f"{minutes} {minute_text}"
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Estimated Charging Duration
                    </div>
                    <div class="result-value">
                        {duration_text}
                    </div>
                    <div class="result-detail">
                        Expected session duration based
                        on your selected details.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Total Duration",
                    f"{predicted_minutes:,} minutes"
                )

            with metric2:
                st.metric(
                    "Selected Model",
                    "Gradient Boosting"
                )

            st.info(
                "This is an estimate, not a guaranteed charging "
                "time. The model's test-set MAE is about "
                "84 minutes, so actual duration may differ "
                "considerably from the prediction."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==================================================
# TAB 2: DATASET & MODEL
# ==================================================

with tab_dataset:

    st.header("Dataset & Model Information")

    st.write(
        "Explore the dataset used for training and compare "
        "the performance of the tested regression algorithms."
    )

    # Dataset overview
    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Original Records", "72,856")
    metric2.metric("Cleaned Sessions", "72,108")
    metric3.metric("Input Features", "7")
    metric4.metric("Target Variable", "Duration")

    st.divider()

    # Dataset preview
    st.subheader("Training Dataset Preview")

    if PREVIEW_PATH.exists():

        try:
            preview_df = pd.read_csv(PREVIEW_PATH)

            st.dataframe(
                preview_df.head(10),
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:
            st.warning(f"Could not read the dataset preview: {error}")

    else:
        st.info(
            "The dataset preview file is optional. To show the "
            "actual cleaned training records here, export the first "
            "10 rows of model_df from your notebook as "
            "'dataset_preview.csv' and place it beside app.py."
        )

    st.markdown(
        "[Open the original dataset on Figshare]"
        "(https://figshare.com/articles/dataset/"
        "A_dataset_for_multi-faceted_analysis_of_electric_vehicle_"
        "charging_transactions/22495141)"
    )

    # Features table
    st.subheader("Features Used by the Model")

    feature_info = pd.DataFrame({
        "Feature": [
            "ChargerCompany",
            "Location",
            "ChargerType",
            "StartHour",
            "DayOfWeek",
            "Month",
            "IsWeekend",
            "Duration"
        ],
        "Description": [
            "Charger company category",
            "Charging location",
            "Slow or fast charger category",
            "Hour when the session starts",
            "Day derived from the selected date",
            "Month derived from the selected date",
            "Whether the selected date is a weekend",
            "Target: session duration in minutes"
        ],
        "Role": [
            "Input",
            "Input",
            "Input",
            "Input",
            "Input",
            "Input",
            "Input",
            "Target"
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Actual model comparison results
    st.subheader("Model Performance Comparison")

    model_results = pd.DataFrame({
        "Model": [
            "Gradient Boosting",
            "Linear Regression",
            "Random Forest"
        ],
        "MAE (minutes)": [
            83.804321,
            89.524633,
            89.425553
        ],
        "RMSE (minutes)": [
            125.939938,
            132.217115,
            135.735652
        ],
        "R²": [
            0.266565,
            0.191631,
            0.148034
        ]
    })

    st.dataframe(
        model_results.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Lower MAE and RMSE are better. Higher R² is better. "
        "These are the recorded results from the test-set evaluation."
    )

    st.subheader("Model Error Comparison")

    chart_data = model_results.set_index("Model")[
        ["MAE (minutes)", "RMSE (minutes)"]
    ]

    st.bar_chart(chart_data)

    st.success(
        "Gradient Boosting was selected because it achieved "
        "the lowest MAE and RMSE and the highest R² among "
        "the three tested models."
    )

    st.caption(
        "Test performance: MAE ≈ 83.80 minutes, "
        "RMSE ≈ 125.94 minutes, R² ≈ 0.2666."
    )


# ==================================================
# TAB 3: ABOUT PROJECT
# ==================================================

with tab_about:

    st.header("About the Project")

    st.write(
        "Smart EV Charging Time Predictor is a classical "
        "machine learning regression project that estimates "
        "the expected duration of an electric vehicle "
        "charging session."
    )

    st.divider()

    # Main project details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Objective")
        st.write(
            "Estimate charging-session duration using "
            "information available when the session starts."
        )

    with col2:
        st.markdown("### 🤖 ML Task")
        st.write(
            "Supervised learning using regression algorithms "
            "to predict a numerical duration in minutes."
        )

    with col3:
        st.markdown("### 🏆 Selected Model")
        st.write(
            "Gradient Boosting Regressor, selected after "
            "comparing three regression algorithms."
        )

    st.divider()

    # Project workflow
    st.subheader("Project Workflow")

    st.code(
        "EV Charging Dataset\n"
        "        ↓\n"
        "Data Loading & Understanding\n"
        "        ↓\n"
        "Data Cleaning & EDA\n"
        "        ↓\n"
        "Feature Engineering\n"
        "        ↓\n"
        "Train/Test Split & Preprocessing\n"
        "        ↓\n"
        "Train and Compare Regression Models\n"
        "        ↓\n"
        "Evaluate and Select Best Model\n"
        "        ↓\n"
        "Save Pipeline using Joblib\n"
        "        ↓\n"
        "Streamlit Application\n"
        "        ↓\n"
        "GitHub & Streamlit Community Cloud",
        language="text"
    )

    # Technologies
    st.subheader("Technologies Used")

    st.markdown("""
    - **Python** — core programming language
    - **Pandas and NumPy** — data handling and numerical operations
    - **Matplotlib** — exploratory data visualization
    - **Scikit-learn** — preprocessing, regression and evaluation
    - **Joblib** — saving and loading the trained pipeline
    - **Streamlit** — interactive web application
    - **GitHub** — project version control and hosting repository
    - **Streamlit Community Cloud** — application deployment
    """)

    st.divider()

    # Algorithms
    st.subheader("Algorithms Compared")

    st.markdown("""
    1. Linear Regression
    2. Random Forest Regressor
    3. Gradient Boosting Regressor
    """)

    # Limitations
    st.subheader("Limitations")

    st.markdown("""
    - The model uses only seven selected input features.
    - Battery state, requested energy and other potentially
      useful charging details are not available as model inputs.
    - The current test-set R² is approximately 0.267, indicating
      limited predictive performance.
    - Actual charging duration may differ considerably from
      the predicted duration.
    """)

    # Future scope
    st.subheader("Future Scope")

    st.write(
        "Future versions could evaluate additional verified "
        "features, such as battery state of charge, requested "
        "energy and charger capacity, if suitable data becomes "
        "available at prediction time."
    )

    st.divider()

    st.caption(
        "Smart EV Charging Time Predictor | "
        "Classical Machine Learning Project"
    )