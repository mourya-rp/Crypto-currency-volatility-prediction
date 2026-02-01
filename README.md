# Crypto-currency-volatility-prediction

Objective
An interactive web application designed to predict and visualize market risk (volatility) for various cryptocurrencies using machine learning.

Live App
Access the live dashboard here: https://crypto-currency-volatility-prediction-mourya.streamlit.app/

Technical Overview
1. Machine Learning

Developed a Random Forest Regressor to predict rolling volatility. The model analyzes historical price action and market indicators to categorize risk levels.

2. Data Engineering

Feature Extraction: Engineered rolling windows and daily returns from raw market data.

Data Cleaning: Implemented a pipeline to handle time-series data and prepare features for model inference.

3. Application & Deployment

Frontend: Built a responsive dashboard using the Streamlit framework.

Optimization: Utilized resource caching to ensure high-speed model loading and data retrieval.

Cloud Hosting: Deployed via Streamlit Cloud with automatic updates synced to this repository.

Project Structure
crptocurrency_volatility_project.py: The main application and UI logic.

optimized_volatility_model.pkl: The trained Random Forest model.

processed_cryptocurrency_data.csv: The historical dataset used for analysis.

requirements.txt: List of Python dependencies for the environment.

Skills & Tools
Languages: Python

Libraries: Pandas, Scikit-Learn, Joblib, Matplotlib, Seaborn

Platforms: GitHub, Streamlit Cloud
