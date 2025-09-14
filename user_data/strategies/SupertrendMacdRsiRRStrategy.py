"""
SupertrendMacdRsiRRStrategy

Implements the YouTube-described strategy:
- Indicators: SuperTrend, MACD (line vs signal), RSI 50 filter
- Entries on 1h timeframe (default) with candle-size filter
- Stoploss at recent swing low/high (configurable lookback)
- Take-profit via risk-reward multiple of stop distance

Optimization targets exposed via hyperopt parameters:
- SuperTrend length/multiplier
- RSI mid threshold (default 50)
- Maximum candle body/ATR size filter
- Swing point lookback
- Risk-reward ratio for TP

Backtesting example:
    freqtrade backtesting --config user_data/config.json \
        --strategy SupertrendMacdRsiRRStrategy --timerange 20240101-20241231

Hyperopt example:
    freqtrade hyperopt --config user_data/config.json \
        --strategy SupertrendMacdRsiRRStrategy -e 1000 \
        --spaces buy sell stoploss protection --timerange 20240101-20240901
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.strategy.interface import SellType


class SupertrendMacdRsiRRStrategy(IStrategy):
    # Use 1h candles per the video
    timeframe: str = "1h"

    # Allow both long and short to reflect video. Set can_short True if exchange supports shorts.
    can_short: bool = True

    # Startup candles to cover ST/MACD calculations and swing lookbacks
    startup_candle_count: int = 300

    # Hyperopt parameters
    st_length = IntParameter(5, 30, default=10, optimize=True, load=True)
    st_multiplier = DecimalParameter(1.0, 5.0, default=3.0, decimals=2, optimize=True, load=True)

    rsi_period = IntParameter(7, 21, default=14, optimize=True, load=True)
    rsi_mid = IntParameter(45, 60, default=50, optimize=True, load=True)

    # MACD standard ranges
    macd_fast = IntParameter(8, 20, default=12, optimize=True, load=True)
    macd_slow = IntParameter(18, 30, default=26, optimize=True, load=True)
    macd_signal = IntParameter(6, 14, default=9, optimize=True, load=True)

    # Candle-size filter to avoid oversized entries: ratio of body to ATR
    max_body_atr = DecimalParameter(0.5, 3.0, default=1.5, decimals=2, optimize=True, load=True)
    atr_period = IntParameter(10, 30, default=14, optimize=True, load=True)

    # Swing lookback (bars left/right) for stop placement
    swing_lookback = IntParameter(2, 20, default=5, optimize=True, load=True)

    # Risk-reward multiple for target
    rr_multiple = DecimalParameter(1.0, 3.0, default=1.5, decimals=2, optimize=True, load=True)

    # Global fallback stoploss (hard stop) - strategy uses custom_stoploss anyway
    stoploss: float = -0.25

    # Enable custom stoploss/exit
    use_custom_stoploss: bool = True

    plot_config = {
        "main_plot": {
            "st_upper": {"color": "red"},
            "st_lower": {"color": "green"},
            "st_trend": {"color": "blue"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "MACD": {
                "macd": {"color": "blue"},
                "macd_signal": {"color": "orange"},
                "macd_hist": {"color": "grey"},
            },
        },
    }

    @staticmethod
    def _ema(series: Series, length: int) -> Series:
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def _true_range(df: DataFrame) -> Series:
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _atr(self, df: DataFrame, length: int) -> Series:
        tr = self._true_range(df)
        return tr.ewm(alpha=1.0 / float(length), adjust=False).mean()

    def _rsi(self, df: DataFrame, length: int) -> Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / float(length), adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(length), adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _macd(self, series: Series, fast: int, slow: int, signal: int) -> DataFrame:
        ema_fast = self._ema(series, fast)
        ema_slow = self._ema(series, slow)
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        out = DataFrame(index=series.index)
        out["macd"] = macd
        out["macd_signal"] = macd_signal
        out["macd_hist"] = macd_hist
        return out

    def _supertrend(self, df: DataFrame, length: int, multiplier: float) -> DataFrame:
        atr = self._atr(df, length)
        hl2 = (df["high"] + df["low"]) / 2.0
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        trend = Series(index=df.index, dtype="float64")
        direction = Series(index=df.index, dtype="int8")

        trend.iloc[:] = np.nan
        direction.iloc[:] = 0

        for i in range(1, len(df)):
            prev_trend = trend.iat[i - 1] if i > 0 else np.nan
            prev_dir = int(direction.iat[i - 1]) if i > 0 else 0

            curr_upper = upperband.iat[i]
            curr_lower = lowerband.iat[i]
            prev_upper = upperband.iat[i - 1]
            prev_lower = lowerband.iat[i - 1]

            if np.isnan(prev_trend):
                # initialize with lower band
                direction.iat[i] = 1
                trend.iat[i] = curr_lower
                continue

            if prev_trend == prev_upper:
                curr_upper = min(curr_upper, prev_upper)
            if prev_trend == prev_lower:
                curr_lower = max(curr_lower, prev_lower)

            if df["close"].iat[i] > prev_trend:
                direction.iat[i] = 1
                trend.iat[i] = curr_lower
            else:
                direction.iat[i] = -1
                trend.iat[i] = curr_upper

        out = DataFrame(index=df.index)
        out["st_trend"] = trend
        out["st_dir"] = direction
        out["st_upper"] = upperband
        out["st_lower"] = lowerband
        return out

    def populate_indicators(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        st = self._supertrend(
            df,
            int(self.st_length.value),
            float(self.st_multiplier.value),
        )
        df = pd.concat([df, st], axis=1)

        macd = self._macd(
            df["close"],
            int(self.macd_fast.value),
            int(self.macd_slow.value),
            int(self.macd_signal.value),
        )
        df = pd.concat([df, macd], axis=1)

        df["rsi"] = self._rsi(df, int(self.rsi_period.value))
        df["atr"] = self._atr(df, int(self.atr_period.value))

        # Candle body / ATR filter
        body = (df["close"] - df["open"]).abs()
        df["body_atr_ratio"] = body / df["atr"].replace(0, np.nan)

        # Convenience flags
        df["close_above_st"] = df["close"] > df["st_trend"]
        df["close_below_st"] = df["close"] < df["st_trend"]
        df["macd_above_signal"] = df["macd"] > df["macd_signal"]
        df["macd_below_signal"] = df["macd"] < df["macd_signal"]

        return df

    def _find_swing_low(self, df: DataFrame, idx: int, lookback: int) -> Optional[float]:
        start = max(0, idx - lookback)
        window_low = df["low"].iloc[start : idx + 1]
        if window_low.empty:
            return None
        return float(window_low.min())

    def _find_swing_high(self, df: DataFrame, idx: int, lookback: int) -> Optional[float]:
        start = max(0, idx - lookback)
        window_high = df["high"].iloc[start : idx + 1]
        if window_high.empty:
            return None
        return float(window_high.max())

    def populate_entry_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None

        rsi_mid = int(self.rsi_mid.value)
        max_body = float(self.max_body_atr.value)

        long_cond = (
            (df["close_above_st"]) &
            (df["rsi"] > rsi_mid) &
            (df["macd_above_signal"]) &
            (df["body_atr_ratio"] <= max_body)
        )
        short_cond = (
            (df["close_below_st"]) &
            (df["rsi"] < rsi_mid) &
            (df["macd_below_signal"]) &
            (df["body_atr_ratio"] <= max_body)
        )

        df.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "long_signal")
        df.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "short_signal")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        # Exits handled via custom_exit and hard stoploss; keep signals off by default
        df["exit_long"] = 0
        df["exit_short"] = 0
        df["exit_tag"] = None
        return df

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> Optional[float]:
        # Dynamic: based on swing high/low distance from entry
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            # Locate the row index of current candle
            last = df.iloc[-1]
        except Exception:
            return None

        lookback = int(self.swing_lookback.value)
        rr = float(self.rr_multiple.value)

        # Compute distance from entry to swing-based stop
        entry_rate = float(trade.open_rate)
        idx = df.index.get_loc(df.index[-1])

        if trade.is_short:
            swing_high = self._find_swing_high(df, idx, lookback)
            if swing_high is None:
                return None
            stop = swing_high
            # Convert to stoploss percent (negative for loss)
            stop_dist = (stop - entry_rate) / entry_rate
            if stop_dist <= 0:
                return None
            return -float(stop_dist)
        else:
            swing_low = self._find_swing_low(df, idx, lookback)
            if swing_low is None:
                return None
            stop = swing_low
            stop_dist = (entry_rate - stop) / entry_rate
            if stop_dist <= 0:
                return None
            return -float(stop_dist)

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        # Exit when price reaches RR target relative to swing-based stop
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception:
            return None

        lookback = int(self.swing_lookback.value)
        rr = float(self.rr_multiple.value)

        # Use last complete candle
        idx = df.index.get_loc(df.index[-1])
        entry = float(trade.open_rate)

        if trade.is_short:
            swing_high = self._find_swing_high(df, idx, lookback)
            if swing_high is None:
                return None
            risk = (swing_high - entry) / entry
            if risk <= 0:
                return None
            tp_rate = entry * (1.0 - rr * risk)
            if current_rate <= tp_rate:
                return {"exit_tag": "rr_tp_short", "sell_type": SellType.SELL_SIGNAL}
        else:
            swing_low = self._find_swing_low(df, idx, lookback)
            if swing_low is None:
                return None
            risk = (entry - swing_low) / entry
            if risk <= 0:
                return None
            tp_rate = entry * (1.0 + rr * risk)
            if current_rate >= tp_rate:
                return {"exit_tag": "rr_tp_long", "sell_type": SellType.SELL_SIGNAL}

        return None

