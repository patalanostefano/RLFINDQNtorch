import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import yfinance as yf
import ta


class DataLoader:
    def __init__(self, dataset_name, split_point, begin_date=None, end_date=None, load_from_file=False, crypto=False, include_indicators=True):
        warnings.filterwarnings('ignore')
        self.DATA_NAME = dataset_name
        self.DATA_PATH = os.path.join(
            Path(__file__).parents[2], 'data', dataset_name)
        self.DATA_FILE = dataset_name + '.csv'
        self.split_point = split_point
        self.begin_date = begin_date
        self.end_date = end_date
        self.crypto = crypto
        self.include_indicators = include_indicators

        os.makedirs(self.DATA_PATH, exist_ok=True)

        if not load_from_file:
            self.data = self.load_data()
            if self.include_indicators:
                self.add_technical_indicators()
                if self.crypto:
                    self.add_adntrat()

            #self.remove_extra_columns()
            self.normalize_data()
            self.data.to_csv(os.path.join(
                self.DATA_PATH, 'data_processed.csv'), index=True)
        else:
            self.data = pd.read_csv(os.path.join(
                self.DATA_PATH, 'data_processed.csv'))
            self.data.set_index('Date', inplace=True)

        self.filter_data()
        self.split_data()
        self.print_data_info()

    def load_data(self):
        print(f"Loading data for {self.DATA_NAME}")
        data = yf.download(
            self.DATA_NAME, start=self.begin_date, end=self.end_date)
        data.dropna(inplace=True)
        data.columns = data.columns.droplevel(1)

        data.rename(columns={
            #'Adj Close': 'adj_close',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Open': 'open',
            'Volume': 'volume'
        }, inplace=True)
        
        zero_close = data[data['close'] == 0]

        if not zero_close.empty:
            print(f"Warning: Found {len(zero_close)} rows with zero close price.")
            print(zero_close)
        
        print(f"Data loaded. Shape: {data.shape}")
        print("\nSample of the data:")
        print(data.head())
        return data

    #def remove_extra_columns(self):
        #self.data = self.data.drop(columns=['adj_close'])

    def add_technical_indicators(self):
        print("Adding technical indicators")
        self.data['MACD'] = ta.trend.macd(self.data['close'])
        indicator_bb = ta.volatility.BollingerBands(
            close=self.data["close"], window=20, window_dev=2)
        self.data['BB_Upper'] = indicator_bb.bollinger_hband()
        self.data['BB_Lower'] = indicator_bb.bollinger_lband()
        self.data['RSI'] = ta.momentum.rsi(self.data['close'])
        self.data['EMA'] = ta.trend.ema_indicator(
            self.data['close'], window=20)
        print("Technical indicators added")

    def add_adntrat(self):
        self.data['NTRAT'] = self.data['close'].rolling(window=30).mean()

    def filter_data(self):
        print("Filtering data")
        if self.begin_date:
            self.data = self.data[self.data.index >= self.begin_date]
        if self.end_date:
            self.data = self.data[self.data.index <= self.end_date]
        print(f"Data filtered. New shape: {self.data.shape}")

    def split_data(self):
        print("Splitting data")
        if isinstance(self.split_point, str):
            self.data_train = self.data[self.data.index < self.split_point]
            self.data_test = self.data[self.data.index >= self.split_point]
        elif isinstance(self.split_point, int):
            self.data_train = self.data[:self.split_point]
            self.data_test = self.data[self.split_point:]
        else:
            raise ValueError('Split point should be either str (date) or int!')

        self.data_train_with_date = self.data_train.reset_index()
        self.data_test_with_date = self.data_test.reset_index()

        self.data_train.reset_index(drop=True, inplace=True)
        self.data_test.reset_index(drop=True, inplace=True)
        print(
            f"Data split. Train shape: {self.data_train.shape}, Test shape: {self.data_test.shape}")

    def normalize_data(self):
        print("Normalizing data")
        min_max_scaler = MinMaxScaler()
        columns_to_normalize = ['open', 'high', 'low', 'close', 'Volume']
        if self.include_indicators:
            columns_to_normalize.extend(
                ['MACD', 'Signal', 'OBV', 'BB_Upper', 'BB_Lower', 'RSI', 'EMA'])
            if self.crypto:
                columns_to_normalize.append('NTRAT')

        for column in columns_to_normalize:
            if column in self.data.columns:
                if self.data[column].min() == self.data[column].max():
                    print(
                        f"Warning: All values in {column} are the same. Skipping normalization for this column.")
                else:
                    self.data[column] = min_max_scaler.fit_transform(
                        self.data[[column]].fillna(method='ffill'))

        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        self.data = self.data.dropna()
        print("Data normalized")

    def print_data_info(self):
        print("\nData Information:")
        print(self.data.info())
        print("\nFirst rows:")
        print(self.data.head())
        print("\nLast rows:")
        print(self.data.tail())
