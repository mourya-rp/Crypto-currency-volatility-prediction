import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="xx",
    layout="wide"
)

# --- Load Resources (Cached for performance) ---
@st.cache_resource
def load_model():
    try:
        # Try loading the optimized model first
        return joblib.load('optimized_volatility_model-3.pkl')
    except FileNotFoundError:
        # Fallback to standard model if optimized not found
        try:
            return joblib.load('volatility_model.pkl')
        except FileNotFoundError:
            st.error("Error: Model file (optimized_volatility_model-3.pkl or volatility_model.pkl) not found.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading volatility_model.pkl: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Error loading optimized_volatility_model-3.pkl: {e}")
        st.stop()
    return None # Ensure None is returned if st.stop() somehow doesn't halt execution

@st.cache_data
def load_data():
    try:
        df_loaded = pd.read_csv('processed_cryptocurrency_data.csv')
        df_loaded['date'] = pd.to_datetime(df_loaded['date'])
        return df_loaded
    except FileNotFoundError:
        st.error("Error: Data file 'processed_cryptocurrency_data.csv' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or processing 'processed_cryptocurrency_data.csv': {e}")
        st.stop()
    return None # Ensure None is returned

# Initialize model and df to None
model = None
df = None

# Attempt to load resources
model = load_model()
df = load_data()

# Only run the main application logic if both model and df are successfully loaded
if model is not None and df is not None:
    # --- Sidebar: User Inputs ---
    st.sidebar.header("User Input")
    crypto_list = df['crypto_name'].unique()
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", crypto_list)

    # Filter data for selected crypto
    crypto_data = df[df['crypto_name'] == selected_crypto].sort_values(by='date')

    # --- Main Dashboard ---
    st.title(f"xx Cryptocurrency Volatility Prediction: {selected_crypto}")
    st.markdown("This tool uses a Random Forest model to predict the **rolling volatility** based on market indicators.")

    # 1. Show Recent Data
    st.subheader(f"Recent Market Data for {selected_crypto}")
    st.dataframe(crypto_data.tail(5).sort_values(by='date', ascending=False))

    # 2. Prediction Section
    st.subheader("xx Live Volatility Prediction")
    st.write("Based on the latest available market data:")

    # Get the latest data point (features only)
    # We exclude non-feature columns used in training
    latest_row = crypto_data.iloc[[-1]]
    ignore_cols = ['date', 'crypto_name', 'rolling_volatility', 'daily_return']
    features = [col for col in df.columns if col not in ignore_cols]

    # Prepare input for model
    X_input = latest_row[features]

    if st.button("Predict Volatility"):
        prediction = model.predict(X_input)[0]

        # Display Result
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Volatility (Scaled)", value=f"{prediction:.4f}")
        with col2:
            # Simple interpretation based on scaled value (0-1)
            if prediction > 0.05:
                st.error("Risk Level: HIGH")
            elif prediction > 0.02:
                st.warning("Risk Level: MODERATE")
            else:
                st.success("Risk Level: LOW")


 









