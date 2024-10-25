# alchemy

## Trading Script

### How It Works

1. Streams EUR/USD price data from OANDA's API
2. Maintains a buffer of price data and calculates technical indicators (EMAs, RSI, MACD, Bollinger Bands)
3. Uses a pre-trained Random Forest model to predict price movements
4. Makes trading decisions when:
  - No open trades exist
  - Model predicts with high confidence (>75%)
  - Three of the same signals (buy or sell) appear in the queue

### Trade Parameters
- Size: 10 units
- Take Profit: 15 pips
- Trailing Stop Loss: 5 pips

## Technical Details
- Runs two async tasks: price streaming and data processing
- Checks for trading signals every 5 seconds (to match the 5 sec data it was trained on)
- Requires a pre-trained Random Forest model (`randomforest_best_model_with_scaler.joblib`)
