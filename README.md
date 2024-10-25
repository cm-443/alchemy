# alchemy

Trading Script: Streams EUR/USD price data from OANDA's API Maintains a buffer of price data and calculates technical indicators (EMAs, RSI, MACD, Bollinger Bands, etc.)
Uses a pre-trained Random Forest model to predict price movements based on these indicators
Makes trading decisions when:

There are no open trades
The model predicts with high confidence (>75%)
For "Up" predictions: Places a buy order
For "Down" predictions: Places a sell order

Each trade is placed with:

10 units size
15 pip take profit
5 pip trailing stop loss

The script runs two main async tasks: one for streaming prices and another for processing the data/making trading decisions, checking for signals every 5 seconds.
