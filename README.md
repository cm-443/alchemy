# alchemy

###Instructions
1. Open Oanda account and get api keys (Spread is built in, no fees, $10 is enough to run tests)
2. Add api keys to places
3. Run the get_data.py script to get training data
4. Train the model with the collected data
5. Run the trade script


### OANDA Forex Data Collector

#### How It Works

The script:
1. Connects to OANDA's API
2. Fetches EUR/USD price data in 5-minute chunks
3. Saves data to individual CSV files with timestamp naming
4. Creates a list of all generated files for later processing
5. Skips Saturdays when forex markets are closed

#### Data Collection
- Granularity: 5 seconds
- Fields: timestamp, open, high, low, close
- Auto-creates required directories
- Files stored in 'csv' directory
- File list saved to '5s_list.py'
- Manages data in 5-minute chunks to avoid timeout

### Forest Machine Learning Model 

#### How It Works

The script:
1. Loads historical forex data from CSV files
2. Calculates technical indicators (EMAs, RSI, MACD, Bollinger Bands, etc.)
3. Labels price movements as Up/Down/Neutral using dynamic thresholds based on volatility
4. Uses 5-fold time series cross-validation to train and evaluate the model
5. Implements class balancing using TomekLinks and SMOTETomek
6. Saves the best performing model for later use in trading

#### Performance Metrics
- Average Training Score: 0.8990 ± 0.0135
- Average Validation Score: 0.7420 ± 0.0102 
- Best model achieves ~74% accuracy with balanced precision/recall

#### Technical Details
- Uses Bayesian optimization for hyperparameter tuning
- Generates confusion matrices and feature importance plots
- Handles class imbalance using hybrid sampling approach
- Saves trained model as `randomforest_best_model_with_scaler.joblib`


### Trading Script

#### How It Works

1. Streams EUR/USD price data from OANDA's API
2. Maintains a buffer of price data and calculates technical indicators (EMAs, RSI, MACD, Bollinger Bands)
3. Uses a pre-trained Random Forest model to predict price movements
4. Makes trading decisions when:
  - No open trades exist
  - Model predicts with high confidence (>75%)
  - Three of the same signals (buy or sell) appear in the queue

#### Trade Parameters
- Size: 10 units (fractions of a penny)
- Take Profit: 15 pips
- Trailing Stop Loss: 5 pips

#### Technical Details
- Runs two async tasks: price streaming and data processing
- Checks for trading signals every 5 seconds (to match the 5 sec data it was trained on)
- Requires a pre-trained Random Forest model (`randomforest_best_model_with_scaler.joblib`)




