import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="📈",
    layout="wide"
)

# --- Load Resources ---
@st.cache_resource
def load_model():
    # Attempt to load whatever model name is on your GitHub
    for name in ['optimized_volatility_model.pkl', 'volatility_model.pkl', 'optimized_volatility_model-3.pkl']:
        try:
            return joblib.load(name)
        except:
            continue
    st.error("Model file not found on GitHub!")
    return None

@st.cache_data
def load_data():
    try:
        df_loaded = pd.read_csv('processed_cryptocurrency_data.csv')
        df_loaded['date'] = pd.to_datetime(df_loaded['date'])
        return df_loaded
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

model = load_model()
df = load_data()

if model is not None and df is not None:
    # --- Sidebar ---
    st.sidebar.header("User Input")
    crypto_list = df['crypto_name'].unique()
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", crypto_list)

    crypto_data = df[df['crypto_name'] == selected_crypto].sort_values(by='date')

    # --- Main Dashboard ---
    st.title(f"Cryptocurrency Volatility Prediction: {selected_crypto}")
    
    # 1. Recent Data
    st.subheader("Latest Market Snapshot")
    st.dataframe(crypto_data.tail(5).sort_values(by='date', ascending=False))

    # 2. Prediction
    st.subheader("Live Risk Assessment")
    latest_row = crypto_data.iloc[[-1]]
    ignore_cols = ['date', 'crypto_name', 'rolling_volatility', 'daily_return']
    features = [col for col in df.columns if col not in ignore_cols]
    X_input = latest_row[features]

    if st.button("Calculate Risk Level"):
        prediction = model.predict(X_input)[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Volatility", f"{prediction:.4f}")
        with col2:
            if prediction > 0.05:
                st.error("Risk Level: HIGH")
            elif prediction > 0.02:
                st.warning("Risk Level: MODERATE")
            else:
                st.success("Risk Level: LOW")

    # 3. Visualization (Optimized for Web)
    st.subheader("Historical Trends")
    tab1, tab2 = st.tabs(["Volatility Chart", "Price vs Risk"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        # We use standard matplotlib plot for better stability on Cloud
        ax1.plot(crypto_data['date'], crypto_data['rolling_volatility'], color='orange', linewidth=2)
        ax1.set_title("Volatility Over Time")
        st.pyplot(fig1)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(crypto_data['date'], crypto_data['close'], label='Price', color='blue')
        ax2.set_ylabel('Price (Scaled)')
        
        ax3 = ax2.twinx()
        ax3.plot(crypto_data['date'], crypto_data['rolling_volatility'], label='Volatility', color='red', alpha=0.4)
        ax3.set_ylabel('Volatility')
        
        fig2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        st.pyplot(fig2)


 









