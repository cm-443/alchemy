import asyncio
from collections import deque
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import pandas as pd
import joblib
import requests
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, CCIIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


# Initialize OANDA client
accountID = 'dont store your keys like this'
access_token = 'use aws'
client = oandapyV20.API(access_token=access_token, environment="live")

#Some shit like this assumimg you have aws cli configed 
# import boto3
# import json

# def get_secret(secret_name, region_name="us-east-1"):
#     client = boto3.client('secretsmanager', region_name=region_name)
#     response = client.get_secret_value(SecretId=AccessKey)
#     return response['SecretString']

# secret = get_secret('AccessKey')
# secret_dict = json.loads(secret)


# Load the Random Forest model (includes scaler)
model = joblib.load('randomforest_best_model_with_scaler.joblib')
print("Model loaded successfully")


# Add at the top level of your script with other initializations
signal_queue = deque(maxlen=3)
probability_queue = deque(maxlen=3)

def initialize_data_buffer():
    """Initialize the data buffer with required columns"""
    return pd.DataFrame(columns=[
        'timestamp', 'open', 'high', 'low', 'close'
    ])

# Make sure to initialize the data buffer at the start of your script
data_buffer = initialize_data_buffer()

def has_open_trade(instrument):
    params = {"instrument": instrument}
    r = trades.TradesList(accountID, params=params)
    open_trades = client.request(r)["trades"]
    return len(open_trades) > 0


def add_technical_indicators(df):
    """Add technical indicators as per the training model"""
    df['EMA_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['RSI_5'] = ta.momentum.RSIIndicator(df['close'], window=5).rsi()

    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'],
                                             window=14, smooth_window=3)
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()

    # Crossover features
    df['EMA_Crossover'] = (df['EMA_9'] > df['EMA_21']).astype(int)
    df['EMA_Crossover_Change'] = df['EMA_Crossover'].diff()

    df['MACD_Crossover'] = (df['MACD'] > df['MACD_signal']).astype(int)
    df['MACD_Crossover_Change'] = df['MACD_Crossover'].diff()

    df['BB_width'] = df['BB_high'] - df['BB_low']
    df['BB_Tight'] = (df['BB_width'] < df['BB_width'].rolling(window=20).mean()).astype(int)

    df['Breakout_Buy_Signal'] = ((df['BB_Tight'] == 1) &
                                 (df['MACD_Crossover_Change'] == 1)).astype(int)
    df['Breakout_Sell_Signal'] = ((df['BB_Tight'] == 1) &
                                  (df['MACD_Crossover_Change'] == -1)).astype(int)

    df['RSI_Trend'] = (df['RSI_5'].diff() > 0).astype(int)
    df['Price_Above_BB_Mid'] = (df['close'] > df['BB_mid']).astype(int)
    df['Stoch_Crossover'] = (df['Stoch_k'] > df['Stoch_d']).astype(int)

    df.dropna(inplace=True)
    return df


async def process_data():
    global data_buffer
    params = {"instruments": "EUR_USD"}

    # Define feature columns in the same order as during training
    feature_columns = [
        'EMA_9', 'EMA_21', 'RSI_5', 'MACD', 'MACD_signal', 'MACD_diff',
        'BB_high', 'BB_low', 'BB_mid', 'Stoch_k', 'Stoch_d',
        'BB_width', 'EMA_Crossover', 'MACD_Crossover',
        'EMA_Crossover_Change', 'MACD_Crossover_Change',
        'BB_Tight', 'Breakout_Buy_Signal', 'Breakout_Sell_Signal',
        'RSI_Trend', 'Price_Above_BB_Mid', 'Stoch_Crossover'
    ]

    while True:
        try:
            if len(data_buffer) >= 30:  # Minimum data needed for indicators
                # Prepare data for prediction
                data = data_buffer.copy()
                data = add_technical_indicators(data)

                if len(data) > 0:
                    # Prepare features with correct column names
                    X = prepare_features(data)

                    # Ensure we're using only the last row and preserving feature names
                    X_last = pd.DataFrame(X.iloc[-1:].values, columns=feature_columns)

                    # Get prediction
                    prediction = model.predict(X_last)[0]
                    probabilities = model.predict_proba(X_last)[0]

                    print(f"Prediction: {prediction} (Down: {probabilities[0]:.3f}, "
                          f"Neutral: {probabilities[1]:.3f}, Up: {probabilities[2]:.3f})")

                    # Trading logic
                    if not has_open_trade(params["instruments"]):
                        # Strong signals only
                        if prediction == 2 and probabilities[2] > 0.75:  # Up with high confidence
                            await execute_trade(True, False)
                        elif prediction == 0 and probabilities[0] > 0.75:  # Down with high confidence
                            await execute_trade(False, True)
                        else:
                            print("No strong signals detected")
                    else:
                        print("Trade already open, skipping signal")

        except Exception as e:
            print(f"Error in process_data: {str(e)}")

        await asyncio.sleep(5)  # Adjust the sleep time as needed


def prepare_features(df):
    """Prepare features in the same order as training"""
    feature_columns = [
        'EMA_9', 'EMA_21', 'RSI_5', 'MACD', 'MACD_signal', 'MACD_diff',
        'BB_high', 'BB_low', 'BB_mid', 'Stoch_k', 'Stoch_d',
        'BB_width', 'EMA_Crossover', 'MACD_Crossover',
        'EMA_Crossover_Change', 'MACD_Crossover_Change',
        'BB_Tight', 'Breakout_Buy_Signal', 'Breakout_Sell_Signal',
        'RSI_Trend', 'Price_Above_BB_Mid', 'Stoch_Crossover'
    ]
    return df[feature_columns]




async def stream_pricing():
    """Stream price data and maintain buffer for technical indicators"""
    global data_buffer
    params = {"instruments": "EUR_USD"}
    buffer_size = 100  # Enough data points for longest technical indicator calculation

    while True:
        try:
            r = pricing.PricingStream(accountID=accountID, params=params)
            response_stream = client.request(r)

            # Use regular for loop with async sleep instead of async for
            for ticks in response_stream:
                if ticks['type'] == 'PRICE':
                    timestamp = pd.to_datetime(ticks['time'])
                    close_price = float(ticks['bids'][0]['price'])

                    # Initialize prices
                    if not data_buffer.empty:
                        # Use actual high/low for current period
                        last_close = data_buffer.iloc[-1]['close']
                        high_price = max(close_price, last_close)
                        low_price = min(close_price, last_close)
                        open_price = last_close
                    else:
                        high_price = close_price
                        low_price = close_price
                        open_price = close_price

                    # Create new row
                    new_row = {
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price
                    }

                    # Add new data to buffer
                    data_buffer = pd.concat([data_buffer, pd.DataFrame([new_row])], ignore_index=True)

                    # Keep only needed amount of historical data
                    if len(data_buffer) > buffer_size:
                        data_buffer = data_buffer.iloc[-buffer_size:]

                    # Ensure data is sorted by timestamp
                    data_buffer = data_buffer.sort_values('timestamp').reset_index(drop=True)

                    print(f"Latest close price for EUR_USD: {close_price}")
                    print(f"Buffer size: {len(data_buffer)} rows")

                    # Add small delay to prevent overwhelming the API
                    await asyncio.sleep(5)

        except requests.exceptions.ChunkedEncodingError:
            print("Connection broken (ChunkedEncodingError), attempting to reconnect...")
            await asyncio.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"Connection error ({type(e).__name__}): {e}, attempting to reconnect...")
            await asyncio.sleep(5)
        except ConnectionError:
            print("Connection error, attempting to reconnect...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("Script stopped by the user.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(f"Error details: {type(e)}")
            await asyncio.sleep(5)
        finally:
            await asyncio.sleep(5)  # Ensure we always have a delay between reconnection attempts


async def get_latest_close_price(instrument="EUR_USD"):
    params = {"instruments": instrument}
    try:
        r = pricing.PricingInfo(accountID=accountID, params=params)
        response = client.request(r)
        if 'bids' in response['prices'][0]:
            return float(response['prices'][0]['bids'][0]['price'])
        else:
            return None
    except Exception as e:
        print(f"Error fetching latest close price: {e}")
        return None

async def execute_trade(buy_signal, sell_signal):
    try:
        # close_price = await get_latest_close_price()
        close_price = await get_latest_close_price()
        pip_value = 0.0001  # 1 pip for EUR/USD
        trailing_stop_loss_distance = 5 * pip_value  # 5 pips
        tp_distance = 15 * pip_value  # 10 pips

        if buy_signal:
            units = "10"
            tp_price = round(close_price + tp_distance, 3)
        elif sell_signal:
            units = "-10"
            tp_price = round(close_price - tp_distance, 3)
        else:
            return


        order_request = {
            "order": {
                "instrument": "EUR_USD",
                "units": units,
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "takeProfitOnFill": {
                    "price": str(tp_price)
                },
                "trailingStopLossOnFill": {
                    "distance": trailing_stop_loss_distance
                }
            }
        }

        # Execute the order
        response = client.request(orders.OrderCreate(accountID, data=order_request))
        print("API Response", response)

    except oandapyV20.exceptions.V20Error as e:
        print(f"Trading error (V20Error): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


async def main():
    try:
        task1 = asyncio.create_task(stream_pricing())
        task2 = asyncio.create_task(process_data())
        await asyncio.gather(task1, task2)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        # Cleanup code if needed
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
