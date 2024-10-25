import csv
import os
from datetime import datetime, timedelta
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# OANDA API account details 
access_token = 'A smart person would use AWS secrets manager'
accountID = 'Oanda Account ID'

# Or use an env file stored in S3 for a task def on fargate with the correct iam policy in AWS for the ultimate debugging funnnnnnnn
# accountID = os.environ.get('accountID')
# access_token = os.environ.get('access_token')

# Initialize the API client
client = API(access_token=access_token, environment="live")

# Directory to save CSV files
directory_path = 'csv'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# List to keep track of the CSV files
csv_files = []

# Specify the number of days you want to fetch data for
NUM_DAYS = 30  # Set this to the desired number of days (e.g., 7 for one week)

# Function to fetch and save data for a given date
def fetch_and_save_data_for_date(date):
    currency_pair = 'EUR_USD'
    start_date = date
    end_date = date + timedelta(days=1)

    subinterval_duration = 300  # 5 minutes
    granularity = 'S5'  # 5 seconds granularity

    current_date = start_date
    while current_date < end_date:
        subinterval_start = current_date
        subinterval_end = subinterval_start + timedelta(seconds=subinterval_duration)

        params = {
            'granularity': granularity,
            'from': subinterval_start.isoformat('T') + 'Z',
            'to': subinterval_end.isoformat('T') + 'Z'
        }

        endpoint = instruments.InstrumentsCandles(instrument=currency_pair, params=params)

        try:
            response = client.request(endpoint)
        except Exception as e:
            print(f"Error fetching data for {subinterval_start}: {e}")
            return

        if response and response['candles']:
            prices = []
            for candle in response['candles']:
                prices.append({
                    'timestamp': datetime.fromisoformat(candle['time'][:-4]),
                    'open': candle['mid']['o'],
                    'high': candle['mid']['h'],
                    'low': candle['mid']['l'],
                    'close': candle['mid']['c']
                })

            csv_file_name = os.path.join(directory_path, f'{subinterval_start.strftime("%Y%m%d_%H%M%S")}.csv')
            csv_files.append(csv_file_name)  # Add the file name to the list
            with open(csv_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close'])
                for price in prices:
                    writer.writerow([price['timestamp'], price['open'], price['high'], price['low'], price['close']])

            print(f"Data for {subinterval_start} written to {csv_file_name}")

        current_date = subinterval_end

# Main loop to fetch data
start_date = datetime.now() - timedelta(days=NUM_DAYS)
end_date = datetime.now()

current_date = start_date
while current_date <= end_date:
    if current_date.weekday() != 5:  # Skip Saturdays
        print(f"Fetching data for {current_date.date()}")
        fetch_and_save_data_for_date(current_date)
    current_date += timedelta(days=1)

# After fetching is complete, append the list of CSV files to 'one_year_5s_list.py'
now = datetime.now()
list_str = f"\nlist_{now.strftime('%Y%m%d_%H%M%S')} = " + str([os.path.basename(file) for file in csv_files])

with open('5s_list.py', 'a') as f:
    f.write(list_str)
