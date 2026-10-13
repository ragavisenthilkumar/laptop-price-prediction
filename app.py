import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("/content/laptop_price_model.pkl")

model = load_model()

st.title("💻 Laptop Price Prediction")
st.write("Enter the laptop specifications to estimate its price.")

# Find the input column names
preprocessor = (
    model.named_steps.get("preprocessor")
    if hasattr(model, "named_steps")
    else None
)

if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
elif preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
    features = list(preprocessor.feature_names_in_)
else:
    st.error("Cannot identify model input columns. Check the training code.")
    st.stop()

# Identify numerical and categorical columns from the fitted pipeline
numeric_cols = []
categorical_cols = []

if preprocessor is not None and hasattr(preprocessor, "transformers_"):
    for name, transformer, cols in preprocessor.transformers_:
        if isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, (list, tuple)):
            try:
                cols = list(cols)
            except TypeError:
                continue

        if name in ["num", "numeric"]:
            numeric_cols.extend(cols)
        elif name in ["cat", "categorical"]:
            categorical_cols.extend(cols)

if not numeric_cols and not categorical_cols:
    st.error(
        "This code needs the fitted preprocessing pipeline "
        "to identify numeric and categorical inputs."
    )
    st.stop()

with st.form("laptop_form"):
    values = {}
    left, right = st.columns(2)

    for i, feature in enumerate(features):
        container = left if i % 2 == 0 else right

        with container:
            if feature in numeric_cols:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0
                )
            else:
                values[feature] = st.text_input(
                    feature.replace("_", " ").title()
                )

    submitted = st.form_submit_button("Predict Laptop Price")

if submitted:
    if any(
        isinstance(values[c], str) and not values[c].strip()
        for c in features
    ):
        st.warning("Please fill in all fields.")
    else:
        try:
            input_df = pd.DataFrame(
                [{c: values[c] for c in features}],
                columns=features
            )

            for c in numeric_cols:
                input_df[c] = pd.to_numeric(input_df[c])

            prediction = float(model.predict(input_df)[0])

            st.success(f"Estimated Laptop Price: {prediction:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
