# 📈 Apple Stock Price Prediction App

A Streamlit-based web application that uses LSTM (Long Short-Term Memory) neural networks to predict Apple Inc. (AAPL) stock prices.

## 🚀 Features

- **Real-time Data**: Fetches current AAPL stock data using Yahoo Finance
- **LSTM Predictions**: Uses trained LSTM model for accurate price forecasting
- **Interactive UI**: Beautiful Streamlit interface with customizable parameters
- **Visual Analytics**: Charts showing historical data and future predictions
- **Detailed Reports**: Comprehensive prediction tables with change metrics

## 📋 Requirements

- Python 3.8+
- pip (Python package installer)

## 🛠️ Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd stock_price_app
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

1. **Activate your virtual environment** (if not already activated)
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in the terminal

## 📊 How to Use

1. **Adjust Settings**: Use the sidebar to customize prediction parameters
   - Number of days to predict (5-90 days)
   - Confidence level (High/Medium/Low)

2. **Generate Predictions**: Click the "🚀 Generate Prediction" button

3. **View Results**: 
   - Interactive chart showing historical and predicted prices
   - Detailed prediction table with price changes
   - Summary statistics

## 🧠 Model Information

- **Model Type**: LSTM (Long Short-Term Memory) Neural Network
- **Training Data**: Historical AAPL stock prices
- **Features**: Closing prices, technical indicators, time series patterns
- **Architecture**: Sequential model with LSTM layers

## 📁 Project Structure

```
stock_price_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── model/             # Trained model files
│   ├── lstm_model.h5  # LSTM model weights
│   └── scaler.pkl     # Data scaler
└── venv/              # Virtual environment (created during setup)
```

## ⚠️ Important Notes

- **Educational Purpose**: This app is for educational and demonstration purposes only
- **No Financial Advice**: Predictions should not be used as financial advice
- **Model Limitations**: Stock predictions are inherently uncertain due to market volatility
- **Data Source**: Uses Yahoo Finance API for real-time data

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Loading Errors**: Ensure model files exist in the `model/` directory

3. **Data Fetching Issues**: Check internet connection for real-time data

4. **Port Already in Use**: If port 8501 is busy, Streamlit will automatically use the next available port

### Dependencies Issues:

If you encounter version conflicts, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📈 Model Performance

The LSTM model has been trained on historical AAPL data and includes:
- Feature scaling for better training
- Time series preprocessing
- Optimized hyperparameters

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving the UI/UX
- Enhancing the model

## 📄 License

This project is for educational purposes. Please use responsibly.

---

**Disclaimer**: This application is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as financial advice. Always consult with qualified financial advisors before making investment decisions.
