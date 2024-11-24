from src.env.trading_env import TradingEnv
import numpy as np
import pandas as pd
from talipp.indicators import Aroon
from talipp.ohlcv import OHLCV


def Prepare(data, action_name, gamma, n_step, batch_size, window_size, transaction_cost, windowed):
    if windowed:

        ohlcv_data = [OHLCV(o, h, l, c, v) for o, h, l, c, v in zip(
            data['open'], data['high'], data['low'], data['close'], data['Volume'])]

        aroon = Aroon(window_size, ohlcv_data)

        # Extract Aroon Up and Down values, filling None values with 0
        aroon_up = [val.up if val is not None else 0 for val in aroon]
        aroon_down = [val.down if val is not None else 0 for val in aroon]

        # Define trend direction: 1 for Uptrend, -1 for Downtrend, 0 for Sideways
        trend = np.where(np.array(aroon_up) > np.array(aroon_down), 1,
                         np.where(np.array(aroon_up) < np.array(aroon_down), -1, 0))

        # Initialize the trend duration column with zeros
        data['trend_duration'] = 0.0

        # Calculate trend duration
        current_duration = 0.0
        current_trend = trend[0]
        max_duration = window_size  # Set the maximum duration to match the window size

        for i in range(1, len(data)):
            if trend[i] == current_trend:
                current_duration = min(current_duration + 1.0, max_duration)
            else:
                current_duration = 1.0
                current_trend = trend[i]

            data.at[i, 'trend_duration'] = current_duration

        print('trended data:', data)

        env = TradingEnv(
            data=data,
            action_name=action_name,
            gamma=gamma,
            n_step=n_step,
            batch_size=batch_size,
            start_index_reward=0,
            transaction_cost=transaction_cost
        )

    else:
        env = TradingEnv(
            data=data,
            action_name=action_name,
            gamma=gamma,
            n_step=n_step,
            batch_size=batch_size,
            start_index_reward=0,
            transaction_cost=transaction_cost
        )

    return env
