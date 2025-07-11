import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AAPL Stock Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained LSTM model and scaler"""
    try:
        from tensorflow.keras.models import load_model
        import tensorflow as tf
        
        # Load model
        model = load_model('model/lstm_model.h5', compile=False)
        
        # Load scaler
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Main app
def main():
    st.title("📈 Apple Stock Price Forecasting (AAPL)")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Prediction range
    n_days = st.sidebar.slider(
        "Select number of future days to predict", 
        min_value=5, 
        max_value=90, 
        value=30,
        help="Choose how many days into the future you want to predict"
    )
    
    # Confidence level
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        ["High", "Medium", "Low"],
        help="Higher confidence means more conservative predictions"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔮 Stock Price Prediction")
        st.info("This model uses LSTM neural networks trained on historical AAPL data to predict future stock prices.")
        
        if st.button("🚀 Generate Prediction", type="primary"):
            if model is None or scaler is None:
                st.error("Model or scaler not loaded properly. Please check the model files.")
                return
                
            with st.spinner("Fetching data and generating predictions..."):
                try:
                    # Fetch recent data
                    ticker = yf.Ticker("AAPL")
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(days=200)
                    
                    df = ticker.history(start=start_date, end=end_date)
                    
                    if df.empty:
                        st.error("Could not fetch stock data. Please try again later.")
                        return
                    
                    # Prepare data
                    df1 = df['Close'].values.reshape(-1, 1)
                    
                    # Scale the data
                    df1_scaled = scaler.transform(df1)
                    last_100 = df1_scaled[-100:].reshape(1, -1)
                    temp_input = last_100[0].tolist()
                    lst_output = []
                    
                    # Get current price for conservative limits
                    current_price = float(df['Close'].iloc[-1])
                    
                    # Set conservative limits based on confidence level
                    if confidence_level == "High":
                        max_change_percent = 0.10  # 10% max change
                    elif confidence_level == "Medium":
                        max_change_percent = 0.15  # 15% max change
                    else:  # Low confidence
                        max_change_percent = 0.25  # 25% max change
                    
                    # Generate predictions with conservative limits
                    for i in range(n_days):
                        x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
                        yhat = model.predict(x_input, verbose=0)
                        
                        # Apply conservative limits
                        predicted_price = scaler.inverse_transform(yhat)[0][0]
                        max_allowed = current_price * (1 + max_change_percent)
                        min_allowed = current_price * (1 - max_change_percent)
                        
                        # Clamp the prediction to reasonable bounds
                        if predicted_price > max_allowed:
                            predicted_price = max_allowed
                        elif predicted_price < min_allowed:
                            predicted_price = min_allowed
                        
                        # Convert back to scaled value
                        yhat_scaled = scaler.transform([[predicted_price]])[0][0]
                        
                        temp_input.append(yhat_scaled)
                        lst_output.append(yhat_scaled)
                    
                    # Inverse transform predictions
                    pred_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
                    
                    # Display results
                    st.success("✅ Prediction completed successfully!")
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data (last 30 days)
                    historical_dates = pd.date_range(
                        end=end_date, 
                        periods=min(30, len(df)), 
                        freq='D'
                    )
                    historical_prices = df['Close'].tail(min(30, len(df)))
                    ax.plot(historical_dates, historical_prices, 
                           label="Historical Price", color="blue", linewidth=2)
                    
                    # Plot predictions
                    future_dates = pd.date_range(
                        start=end_date + datetime.timedelta(days=1), 
                        periods=n_days, 
                        freq='D'
                    )
                    ax.plot(future_dates, pred_values.flatten(), 
                           label="Predicted Price", color="orange", linewidth=2, linestyle='--')
                    
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price ($)")
                    ax.set_title("AAPL Stock Price Forecast (Conservative)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Display prediction table
                    st.subheader("📊 Detailed Predictions")
                    
                    prediction_df = pd.DataFrame({
                        "Date": future_dates,
                        "Predicted Price ($)": np.round(pred_values.flatten(), 2),
                        "Change ($)": np.round(np.diff(pred_values.flatten(), prepend=pred_values.flatten()[0]), 2),
                        "Change (%)": np.round((np.diff(pred_values.flatten(), prepend=pred_values.flatten()[0]) / pred_values.flatten()[0]) * 100, 2)
                    })
                    
                    st.dataframe(prediction_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Starting Price", f"${pred_values[0][0]:.2f}")
                    with col2:
                        st.metric("Ending Price", f"${pred_values[-1][0]:.2f}")
                    with col3:
                        total_change = pred_values[-1][0] - pred_values[0][0]
                        st.metric("Total Change", f"${total_change:.2f}")
                    with col4:
                        total_change_pct = (total_change / pred_values[0][0]) * 100
                        st.metric("Total Change %", f"{total_change_pct:.2f}%")
                    
                    # Add confidence level info
                    st.info(f"🔒 Using {confidence_level} confidence level with maximum {max_change_percent*100:.0f}% price change limit.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please make sure you have a stable internet connection and try again.")
    
    with col2:
        st.subheader("ℹ️ About This Model")
        st.markdown("""
        **Model Type:** LSTM Neural Network
        
        **Training Data:** Historical AAPL stock prices
        
        **Features Used:**
        - Closing prices
        - Technical indicators
        - Time series patterns
        
        **Conservative Limits:**
        - High Confidence: ±10% max change
        - Medium Confidence: ±15% max change  
        - Low Confidence: ±25% max change
        
        **Disclaimer:** 
        This is for educational purposes only. 
        Stock predictions are inherently uncertain and 
        should not be used as financial advice.
        """)
        
        st.subheader("📊 Current Market Info")
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            
            current_price = info.get('currentPrice', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            volume = info.get('volume', 'N/A')
            
            st.metric("Current Price", f"${current_price}" if current_price != 'N/A' else "N/A")
            st.metric("Market Cap", f"${market_cap:,}" if market_cap != 'N/A' else "N/A")
            st.metric("Volume", f"{volume:,}" if volume != 'N/A' else "N/A")
            
        except Exception as e:
            st.info("Unable to fetch current market data.")

if __name__ == "__main__":
    main()