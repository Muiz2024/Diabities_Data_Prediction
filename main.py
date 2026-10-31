import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.datasets import load_diabetes

MODEL_PATH = "best_diabetes_model.joblib"

FEATURE_INFO = {
    "age": "Age (years, mean-centered & scaled)",
    "sex": "Sex (mean-centered & scaled)",
    "bmi": "Body mass index (mean-centered & scaled)",
    "bp": "Average blood pressure (mean-centered & scaled)",
    "s1": "Total serum cholesterol (mean-centered & scaled)",
    "s2": "LDL - low-density lipoproteins (mean-centered & scaled)",
    "s3": "HDL - high-density lipoproteins (mean-centered & scaled)",
    "s4": "Total cholesterol / HDL (mean-centered & scaled)",
    "s5": "Log of serum triglycerides (mean-centered & scaled)",
    "s6": "Blood sugar level (mean-centered & scaled)",
}

TARGET_MIN, TARGET_MAX, TARGET_MEAN, TARGET_STD = 25.0, 346.0, 152.13, 77.09

MODEL_META = {
    "name": "Ridge Regression (tuned)",
    "params": "alpha=1, solver=svd",
    "pipeline": "StandardScaler -> Ridge",
    "cv_r2": 0.4808,
    "test_r2": 0.4541,
    "test_rmse": 53.78,
    "test_mae": 42.81,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    data = load_diabetes(as_frame=True)
    df = data.frame
    features = [c for c in df.columns if c != "target"]
    summary = {}
    for col in features:
        summary[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "mean": float(df[col].mean()),
        }
    return df, features, summary


def risk_category(value):
    if value < TARGET_MEAN - 0.5 * TARGET_STD:
        return "Low", "#2ecc71"
    elif value > TARGET_MEAN + 0.5 * TARGET_STD:
        return "High", "#e74c3c"
    return "Medium", "#f39c12"


def main():
    st.set_page_config(
        page_title="Diabetes Progression Predictor",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🩺 Diabetes Progression Predictor")
    st.markdown(
        "Predicts a **quantitative measure of disease progression one year "
        "after baseline** using the tuned Ridge regression model trained in "
        "`diabetes_regression_project.ipynb`."
    )

    model = load_model()
    df, features, summary = load_dataset()

    with st.sidebar:
        st.header("Model Information")
        st.markdown(f"**Algorithm:** {MODEL_META['name']}")
        st.markdown(f"**Best params:** `{MODEL_META['params']}`")
        st.markdown(f"**Pipeline:** `{MODEL_META['pipeline']}`")
        st.markdown("---")
        st.markdown("**Performance (test set)**")
        col_a, col_b = st.columns(2)
        col_a.metric("R²", f"{MODEL_META['test_r2']:.4f}")
        col_b.metric("CV R²", f"{MODEL_META['cv_r2']:.4f}")
        col_a.metric("RMSE", f"{MODEL_META['test_rmse']:.2f}")
        col_b.metric("MAE", f"{MODEL_META['test_mae']:.2f}")
        st.markdown("---")
        st.caption(
            "Features are mean-centered & scaled (sklearn diabetes dataset). "
            "Sliders are bounded to the training data range."
        )

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab_single:
        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            st.subheader("Patient Features")
            input_values = {}
            cols = st.columns(2)
            for idx, feat in enumerate(features):
                with cols[idx % 2]:
                    s = summary[feat]
                    step = (s["max"] - s["min"]) / 100.0
                    val = st.slider(
                        f"{feat}",
                        min_value=round(s["min"], 6),
                        max_value=round(s["max"], 6),
                        value=round(s["median"], 6),
                        step=round(step, 6),
                        format="%.6f",
                        help=FEATURE_INFO.get(feat, feat),
                    )
                    input_values[feat] = val

            btn_col1, btn_col2 = st.columns(2)
            predict_btn = btn_col1.button("🔮 Predict", type="primary")
            sample_btn = btn_col2.button("🎲 Random Sample")

            if sample_btn:
                st.session_state["sample_row"] = df.sample(1).iloc[0]

        with col_right:
            st.subheader("Prediction Result")

            sample = st.session_state.get("sample_row")
            if sample is not None:
                for feat in features:
                    input_values[feat] = float(sample[feat])
                actual = float(sample["target"])
                st.info(f"Loaded random patient (actual progression: **{actual:.0f}**)")

            if predict_btn or sample is not None:
                X = pd.DataFrame([input_values], columns=features)
                pred = float(model.predict(X)[0])
                pred_clipped = float(np.clip(pred, TARGET_MIN, TARGET_MAX))
                category, color = risk_category(pred_clipped)

                st.markdown(
                    f"<div style='text-align:center;padding:1.2em;border-radius:12px;"
                    f"background:#1e1e2e;border:1px solid #333;'>"
                    f"<span style='font-size:1.1em;color:#aaa;'>Predicted Progression</span><br>"
                    f"<span style='font-size:3em;font-weight:bold;color:{color};'>"
                    f"{pred_clipped:.1f}</span><br>"
                    f"<span style='font-size:1.3em;color:{color};'>{category} Risk</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**Reference scale**")
                ref_cols = st.columns(3)
                ref_cols[0].metric("Low", f"< {TARGET_MEAN - 0.5 * TARGET_STD:.0f}")
                ref_cols[1].metric(
                    "Medium",
                    f"{TARGET_MEAN - 0.5 * TARGET_STD:.0f}–{TARGET_MEAN + 0.5 * TARGET_STD:.0f}",
                )
                ref_cols[2].metric("High", f"> {TARGET_MEAN + 0.5 * TARGET_STD:.0f}")

                st.markdown("**Input summary**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Feature": feat,
                                "Value": input_values[feat],
                                "Description": FEATURE_INFO.get(feat, feat),
                            }
                            for feat in features
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Adjust the feature sliders and click **Predict**.")

    with tab_batch:
        st.subheader("Batch Prediction via CSV")
        st.markdown(
            "Upload a CSV with columns: `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6` "
            "(values in the same scaled space as the training data)."
        )
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                missing = [f for f in features if f not in batch_df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                else:
                    X_batch = batch_df[features]
                    preds = model.predict(X_batch)
                    result_df = batch_df.copy()
                    result_df["predicted_progression"] = np.clip(
                        preds, TARGET_MIN, TARGET_MAX
                    )
                    result_df["risk_category"] = [
                        risk_category(p)[0] for p in result_df["predicted_progression"]
                    ]
                    st.success(f"Generated {len(result_df)} predictions.")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download predictions",
                        result_df.to_csv(index=False).encode("utf-8"),
                        file_name="diabetes_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.caption("No file uploaded yet.")
            if st.button("Download sample input template"):
                template = pd.DataFrame(
                    {feat: [summary[feat]["median"]] for feat in features}
                )
                st.download_button(
                    "⬇️ sample_input.csv",
                    template.to_csv(index=False).encode("utf-8"),
                    file_name="sample_input.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()