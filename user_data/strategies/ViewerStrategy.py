from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
import talib.abstract as ta


class ViewerStrategy(IStrategy):
    """
    Trend + pullback momentum strategy inspired by a "viewer-submitted" idea.

    Core idea:
    - Trade in the direction of the higher-timeframe trend (via EMAs / ADX filter).
    - Enter on pullbacks confirmed by momentum re-acceleration (RSI/MACD).
    - Use dynamic ROI and protective stoploss; optional trailing stop.
    """

    # --- Timeframes & order behavior ---
    timeframe = '15m'
    can_short: bool = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_buying_expired_candle_after = 3
    startup_candle_count: int = 200

    # --- ROI / Stoploss / Trailing ---
    minimal_roi = {
        "0": 0.05,   # take profits from the start when available
        "60": 0.03,
        "180": 0.02,
        "480": 0.01,
    }

    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # --- Hyperoptable params ---
    ema_fast_period = IntParameter(8, 21, default=12, space="buy")
    ema_slow_period = IntParameter(34, 89, default=50, space="buy")
    rsi_pullback = IntParameter(20, 40, default=32, space="buy")
    adx_min = IntParameter(15, 35, default=20, space="buy")

    exit_rsi_high = IntParameter(60, 85, default=72, space="sell")

    # --- Protections (soft) ---
    protections = [
        {"method": "StoplossGuard", "lookback_period_candles": 48, "trade_limit": 3, "stop_duration_candles": 24, "only_per_pair": False},
        {"method": "LowProfitPairs", "lookback_period_candles": 72, "trade_limit": 2, "required_profit": 0.005},
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
    ]

    # --- Informative pairs (optional) ---
    informative_timeframe = '1h'

    def informative_pairs(self) -> List[tuple]:
        # Use the same pair on 1h for higher-timeframe context
        return []

    # --- Indicators ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        # EMAs for trend direction
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_fast_period.value))
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_slow_period.value))

        # ADX to ensure trend strength
        adx = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['adx'] = adx

        # RSI for pullback + momentum recapture
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        # MACD for momentum confirmation
        macd = ta.MACD(dataframe['close'])
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # ATR for potential volatility filters / stop sizing insight
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        # Pullback measure: distance to EMA fast
        dataframe['distance_ema_fast'] = (dataframe['close'] - dataframe['ema_fast']) / dataframe['ema_fast']

        # Rolling min/max for simple pivot context
        dataframe['lower_wick'] = dataframe['open'].where(dataframe['open'] < dataframe['close'], dataframe['close']) - dataframe['low']
        dataframe['upper_wick'] = dataframe['high'] - dataframe['open'].where(dataframe['open'] > dataframe['close'], dataframe['close'])

        return dataframe

    # --- Entries ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe.loc[:, 'enter_long'] = 0

        # Conditions:
        # 1) Trend: ema_fast > ema_slow, ADX >= threshold
        # 2) Pullback: RSI below configurable threshold (oversold in uptrend)
        # 3) Momentum re-acceleration: MACD histogram rising or macd > signal
        # 4) Avoid extended candles: wick/atr sanity

        adx_ok = dataframe['adx'] >= self.adx_min.value
        trend_up = dataframe['ema_fast'] > dataframe['ema_slow']
        pullback = dataframe['rsi'] < self.rsi_pullback.value
        momentum_up = (dataframe['macd'] > dataframe['macdsignal']) | (dataframe['macdhist'] > dataframe['macdhist'].shift(1))

        # Simple wick sanity: prefer decent lower wick (buyers step in) or small upper wick
        wick_ok = (dataframe['lower_wick'] > 0) & (dataframe['upper_wick'] < dataframe['atr'] * 1.5)

        dataframe.loc[
            trend_up & adx_ok & pullback & momentum_up & wick_ok,
            'enter_long'
        ] = 1

        return dataframe

    # --- Exits ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe.loc[:, 'exit_long'] = 0

        # Exit on RSI stretch or momentum loss
        rsi_stretch = dataframe['rsi'] > self.exit_rsi_high.value
        momentum_loss = (dataframe['macd'] < dataframe['macdsignal']) & (dataframe['macdhist'] < 0)

        dataframe.loc[
            rsi_stretch | momentum_loss,
            'exit_long'
        ] = 1

        return dataframe

    # Optional: custom stoploss could be added here for ATR-based dynamic stops
    # def custom_stoploss(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
    #     return self.stoploss

