"""
RegimeAdaptiveTriadStrategy

One strategy class combining 3 sub-strategies (BULL/SIDEWAYS/BEAR) selected by a
regime detector computed on informative 1h candles. Uses informative timeframes
15m/1h/4h and a single base timeframe (default 5m) at runtime.

Regime detection (1h):
- EMA200 slope (% over N candles), ADX, ATR% (ATR/Close*100)
- BULL  if (close > ema200) and (ema200_slope > ema_slope_pos) and (adx > adx_bull)
- BEAR  if (close < ema200) and (ema200_slope < -ema_slope_neg) and (adx > adx_bear)
- If ATR% <= atr_pct_sideways_max => SIDEWAYS, else default to SIDEWAYS when not bull/bear

Sub-strategies
- BULL (5m/15m): Trend-following. EMA cross + pullback with RSI filter, ADX/BB width gates.
- SIDEWAYS (15m): Mean reversion. Lower BB touch with RSI floor; tight TP, time-based exit.
- BEAR (15m/1h): Counter-trend. Oversold RSI + rebound from 1h EMA200 - ATR band.

Risk
- ROI segments, global stoploss, regime-aware custom stoploss for BEAR, optional trailing stop.
- Protections exposed via parameters (CooldownPeriod, MaxDrawdown, LowProfitPairs) if supported.

Usage
Backtest example:
    freqtrade backtesting --config user_data/config.json \
        --strategy RegimeAdaptiveTriadStrategy --timerange 20240101-20240901

Hyperopt example:
    freqtrade hyperopt --config user_data/config.json \
        --strategy RegimeAdaptiveTriadStrategy -e 1000 \
        --spaces roi stoploss buy sell protection --timerange 20240101-20240901
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
    BooleanParameter,
)
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.helpers import merge_informative_pair
from pandas import DataFrame, Series
# Indicators implemented with pandas/numpy only (no external 'ta').


class RegimeAdaptiveTriadStrategy(IStrategy):
    timeframe: str = "5m"

    # Informative TFs
    _inf_tf_15m: str = "15m"
    _inf_tf_1h: str = "1h"
    _inf_tf_4h: str = "4h"

    # Run on long only by default. Adapt as needed.
    can_short: bool = False

    # Expose ROI via parameters
    roi_t1_mins = IntParameter(10, 240, default=60, optimize=True, load=True)
    roi_t1 = DecimalParameter(0.002, 0.08, default=0.02, decimals=3, optimize=True, load=True)
    roi_t2_mins = IntParameter(60, 720, default=180, optimize=True, load=True)
    roi_t2 = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, optimize=True, load=True)
    roi_t3_mins = IntParameter(120, 1440 * 3, default=720, optimize=True, load=True)
    roi_t3 = DecimalParameter(0.000, 0.04, default=0.005, decimals=3, optimize=True, load=True)

    # Global risk
    stoploss_param = DecimalParameter(-0.30, -0.02, default=-0.10, decimals=3, optimize=True, load=True)

    # Trailing stop controls
    use_trailing_param = BooleanParameter(default=True, optimize=True, load=True)
    trailing_stop_positive_param = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, optimize=True, load=True)
    trailing_stop_positive_offset_param = DecimalParameter(0.002, 0.10, default=0.03, decimals=3, optimize=True, load=True)
    trailing_only_offset_is_reached_param = BooleanParameter(default=True, optimize=True, load=True)

    # Regime detection thresholds (1h)
    ema_slope_pos = DecimalParameter(0.00, 0.50, default=0.05, decimals=3, optimize=True, load=True)
    ema_slope_neg = DecimalParameter(0.00, 0.50, default=0.05, decimals=3, optimize=True, load=True)
    adx_bull = IntParameter(10, 50, default=20, optimize=True, load=True)
    adx_bear = IntParameter(10, 50, default=20, optimize=True, load=True)
    atr_pct_sideways_max = DecimalParameter(0.10, 5.00, default=1.00, decimals=2, optimize=True, load=True)
    ema_slope_lookback = IntParameter(3, 48, default=12, optimize=True, load=True)

    # BULL params
    buy_ema_fast = IntParameter(5, 21, default=9, optimize=True, load=True)
    buy_ema_slow = IntParameter(20, 200, default=50, optimize=True, load=True)
    buy_rsi_max_pullback = IntParameter(40, 70, default=55, optimize=True, load=True)
    buy_bb_width_min = DecimalParameter(0.002, 0.10, default=0.01, decimals=3, optimize=True, load=True)
    buy_adx_min = IntParameter(10, 40, default=18, optimize=True, load=True)

    # SIDEWAYS params
    sideways_logic = CategoricalParameter(["bb_touch", "bb_cross"], default="bb_touch", optimize=True, load=True)
    buy_rsi_min = IntParameter(20, 45, default=30, optimize=True, load=True)
    buy_bb_dev = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, optimize=True, load=True)
    sell_tp_pct_side = DecimalParameter(0.002, 0.05, default=0.01, decimals=3, optimize=True, load=True)
    sell_time_minutes_side = IntParameter(30, 1440, default=180, optimize=True, load=True)

    # BEAR params
    buy_rsi_min_bear = IntParameter(10, 35, default=20, optimize=True, load=True)
    buy_atr_mult_bear = DecimalParameter(1.0, 4.0, default=2.0, decimals=2, optimize=True, load=True)
    sell_tp_pct_bear = DecimalParameter(0.003, 0.08, default=0.02, decimals=3, optimize=True, load=True)
    stoploss_bear = DecimalParameter(-0.40, -0.05, default=-0.15, decimals=3, optimize=True, load=True)

    # Protections toggles
    use_cooldown = BooleanParameter(default=True, optimize=True, load=True)
    cooldown_candles = IntParameter(5, 720, default=60, optimize=True, load=True)

    use_max_drawdown = BooleanParameter(default=True, optimize=True, load=True)
    max_drawdown_lookback = IntParameter(20, 200, default=100, optimize=True, load=True)
    max_drawdown_protection_pct = DecimalParameter(0.02, 0.50, default=0.20, decimals=2, optimize=True, load=True)
    max_drawdown_trade_limit = IntParameter(1, 50, default=20, optimize=True, load=True)

    use_low_profit_pairs = BooleanParameter(default=False, optimize=True, load=True)
    low_profit_pairs_lookback = IntParameter(20, 200, default=40, optimize=True, load=True)
    low_profit_pairs_min_avg_profit = DecimalParameter(-0.05, 0.05, default=0.0, decimals=3, optimize=True, load=True)
    low_profit_pairs_stop_duration = IntParameter(10, 1440, default=120, optimize=True, load=True)

    use_custom_stoploss: bool = True

    startup_candle_count: int = 300

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "blue"},
            "ema_slow": {"color": "orange"},
            "ema200_1h": {"color": "purple"},
            "bb_high_15m": {"color": "grey"},
            "bb_low_15m": {"color": "grey"},
        },
        "subplots": {
            "RSI": {"rsi_15m": {"color": "green"}},
            "ADX": {"adx_1h": {"color": "red"}},
            "ATR%": {"atr_pct_1h": {"color": "black"}},
        },
    }

    @property
    def minimal_roi(self) -> Dict[str, float]:
        roi: Dict[str, float] = {
            str(self.roi_t3_mins.value): float(self.roi_t3.value),
            str(self.roi_t2_mins.value): float(self.roi_t2.value),
            str(self.roi_t1_mins.value): float(self.roi_t1.value),
            "0": 0.0,
        }
        return roi

    @property
    def trailing_stop(self) -> bool:
        return bool(self.use_trailing_param.value)

    @property
    def trailing_stop_positive(self) -> float:
        return float(self.trailing_stop_positive_param.value)

    @property
    def trailing_stop_positive_offset(self) -> float:
        return float(self.trailing_stop_positive_offset_param.value)

    @property
    def trailing_only_offset_is_reached(self) -> bool:
        return bool(self.trailing_only_offset_is_reached_param.value)

    def informative_pairs(self) -> List[tuple[str, str]]:
        pairs = self.dp.current_whitelist() if self.dp else []
        informative: List[tuple[str, str]] = []
        for p in pairs:
            informative.append((p, self._inf_tf_15m))
            informative.append((p, self._inf_tf_1h))
            informative.append((p, self._inf_tf_4h))
        return informative

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

    @staticmethod
    def _compute_adx(df: DataFrame, window: int = 14) -> Series:
        high = df["high"]
        low = df["low"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = RegimeAdaptiveTriadStrategy._true_range(df)
        atr = tr.ewm(alpha=1.0 / float(window), adjust=False).mean()
        plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / float(window), adjust=False).mean() / atr)
        minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / float(window), adjust=False).mean() / atr)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / float(window), adjust=False).mean().fillna(0.0)
        return adx

    @staticmethod
    def _compute_atr(df: DataFrame, window: int = 14) -> Series:
        tr = RegimeAdaptiveTriadStrategy._true_range(df)
        return tr.ewm(alpha=1.0 / float(window), adjust=False).mean()

    @staticmethod
    def _compute_bbands(df: DataFrame, window: int = 20, dev: float = 2.0) -> DataFrame:
        out = DataFrame(index=df.index)
        mid = df["close"].rolling(window=window, min_periods=window).mean()
        std = df["close"].rolling(window=window, min_periods=window).std(ddof=0)
        high = mid + dev * std
        low = mid - dev * std
        out["bb_high"] = high
        out["bb_low"] = low
        out["bb_mid"] = mid
        out["bb_width"] = (high - low) / mid.replace(0, np.nan)
        return out

    @staticmethod
    def _compute_rsi(df: DataFrame, window: int = 14) -> Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / float(window), adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(window), adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _compute_regime_on_1h(self, df_1h: DataFrame) -> DataFrame:
        df = df_1h.copy()
        df["ema200_1h"] = self._ema(df["close"], 200)
        lookback = int(self.ema_slope_lookback.value)
        df["ema200_slope"] = (
            (df["ema200_1h"] - df["ema200_1h"].shift(lookback)) / df["ema200_1h"].shift(lookback)
        )
        df["adx_1h"] = self._compute_adx(df, 14)
        df["atr_1h"] = self._compute_atr(df, 14)
        df["atr_pct_1h"] = (df["atr_1h"] / df["close"]).abs() * 100.0

        regime = Series(index=df.index, dtype="int8")
        regime.values[:] = 0

        # SIDEWAYS if volatility very low
        regime = regime.where(df["atr_pct_1h"] > float(self.atr_pct_sideways_max.value), 0)

        bull_cond = (
            (df["close"] > df["ema200_1h"]) &
            (df["ema200_slope"] > float(self.ema_slope_pos.value)) &
            (df["adx_1h"] > int(self.adx_bull.value))
        )
        bear_cond = (
            (df["close"] < df["ema200_1h"]) &
            (df["ema200_slope"] < -float(self.ema_slope_neg.value)) &
            (df["adx_1h"] > int(self.adx_bear.value))
        )

        regime = regime.where(~bull_cond, 1)
        regime = regime.where(~bear_cond, -1)

        df["regime"] = regime.fillna(0).astype("int8")
        df["regime_name"] = df["regime"].map({1: "BULL", 0: "SIDEWAYS", -1: "BEAR"}).fillna("SIDEWAYS")
        return df

    def populate_indicators(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        pair = metadata["pair"]

        # Base timeframe indicators (5m)
        df["ema_fast"] = self._ema(df["close"], int(self.buy_ema_fast.value))
        df["ema_slow"] = self._ema(df["close"], int(self.buy_ema_slow.value))

        # 15m informative
        informative_15m = self.dp.get_pair_dataframe(pair, self._inf_tf_15m)
        informative_15m = informative_15m.copy()
        informative_15m["rsi_15m"] = self._compute_rsi(informative_15m, 14)
        bb_15m = self._compute_bbands(informative_15m, window=20, dev=float(self.buy_bb_dev.value))
        informative_15m = pd.concat([informative_15m, bb_15m], axis=1)
        informative_15m["bb_width_15m"] = informative_15m["bb_width"]
        informative_15m.rename(
            columns={
                "bb_high": "bb_high_15m",
                "bb_low": "bb_low_15m",
                "bb_mid": "bb_mid_15m",
            },
            inplace=True,
        )
        informative_15m["adx_15m"] = self._compute_adx(informative_15m, 14)

        df, inf_15m = merge_informative_pair(
            df, informative_15m, self._inf_tf_15m, self.timeframe, ffill=True
        )

        # 1h informative for regime
        informative_1h = self.dp.get_pair_dataframe(pair, self._inf_tf_1h)
        informative_1h = self._compute_regime_on_1h(informative_1h)

        df, inf_1h = merge_informative_pair(
            df, informative_1h, self._inf_tf_1h, self.timeframe, ffill=True
        )

        # 4h informative (reserved, minimal calculations)
        informative_4h = self.dp.get_pair_dataframe(pair, self._inf_tf_4h)
        informative_4h = informative_4h.copy()
        informative_4h["ema200_4h"] = self._ema(informative_4h["close"], 200)

        df, _ = merge_informative_pair(
            df, informative_4h[["date", "ema200_4h"]], self._inf_tf_4h, self.timeframe, ffill=True
        )

        # Convenience regime flags on base timeframe
        df["regime_1h"] = df[f"regime_{self._inf_tf_1h}"]
        df["regime_name_1h"] = df[f"regime_name_{self._inf_tf_1h}"]

        # 1h ATR band for BEAR entry guard
        df["atr_1h"] = df[f"atr_1h_{self._inf_tf_1h}"]
        df["ema200_1h"] = df[f"ema200_1h_{self._inf_tf_1h}"]
        df["atr_band_lower_1h"] = df["ema200_1h"] - float(self.buy_atr_mult_bear.value) * df["atr_1h"]

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = None

        bull = df["regime_1h"] == 1
        side = df["regime_1h"] == 0
        bear = df["regime_1h"] == -1

        # BULL entries: EMA cross + pullback with RSI gate and ADX/BB width checks
        bull_cond = (
            bull &
            (df["ema_fast"] > df["ema_slow"]) &
            (df["close"] <= df["ema_fast"]) &
            (df["rsi_15m"] < int(self.buy_rsi_max_pullback.value)) &
            (df["bb_width_15m"] > float(self.buy_bb_width_min.value)) &
            (df["adx_15m"] > int(self.buy_adx_min.value))
        )
        df.loc[bull_cond, ["enter_long", "enter_tag"]] = (1, "BULL")

        # SIDEWAYS entries: Lower BB touch/cross + RSI floor
        if self.sideways_logic.value == "bb_touch":
            side_cond = (
                side &
                (df["close"] <= df["bb_low_15m"]) &
                (df["rsi_15m"] >= int(self.buy_rsi_min.value))
            )
        else:  # bb_cross
            side_cond = (
                side &
                (df["close"].shift(1) < df["bb_low_15m"].shift(1)) &
                (df["close"] >= df["bb_low_15m"]) &
                (df["rsi_15m"] >= int(self.buy_rsi_min.value))
            )
        df.loc[side_cond, ["enter_long", "enter_tag"]] = (1, "SIDEWAYS")

        # BEAR entries: oversold RSI + rebound from ATR band below EMA200_1h
        bear_cond = (
            bear &
            (df["rsi_15m"] <= int(self.buy_rsi_min_bear.value)) &
            (df["close"] <= df["atr_band_lower_1h"]) &
            (df["close"] > df["close"].shift(1))
        )
        df.loc[bear_cond, ["enter_long", "enter_tag"]] = (1, "BEAR")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = None

        # Optional signal-based exits can be added here; primary exits via ROI/trailing/custom_exit
        return df

    def protections(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        prot: List[Dict[str, Any]] = []
        if bool(self.use_cooldown.value):
            prot.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": int(self.cooldown_candles.value),
            })
        if bool(self.use_max_drawdown.value):
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": int(self.max_drawdown_lookback.value),
                "trade_limit": int(self.max_drawdown_trade_limit.value),
                "stop_duration_candles": int(self.cooldown_candles.value),
                "max_allowed_drawdown": float(self.max_drawdown_protection_pct.value),
            })
        if bool(self.use_low_profit_pairs.value):
            prot.append({
                "method": "LowProfitPairs",
                "lookback_period_candles": int(self.low_profit_pairs_lookback.value),
                "stop_duration_candles": int(self.low_profit_pairs_stop_duration.value),
                "trade_limit": 1,
                "required_profit": float(self.low_profit_pairs_min_avg_profit.value),
            })
        return prot

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
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_row = df.iloc[-1]
            regime_val = int(last_row.get("regime_1h", 0))
        except Exception:
            regime_val = 0

        if regime_val == -1:
            return float(self.stoploss_bear.value)
        return float(self.stoploss_param.value)

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
        enter_tag = (trade.enter_tag or "").upper()
        trade_duration_m = int((current_time - trade.open_date_utc).total_seconds() // 60)

        if enter_tag == "SIDEWAYS":
            if current_profit >= float(self.sell_tp_pct_side.value):
                return {"exit_tag": "side_tp", "sell_type": SellType.SELL_SIGNAL}
            if trade_duration_m >= int(self.sell_time_minutes_side.value) and current_profit > 0:
                return {"exit_tag": "side_time", "sell_type": SellType.TIMEOUT}

        if enter_tag == "BEAR":
            if current_profit >= float(self.sell_tp_pct_bear.value):
                return {"exit_tag": "bear_tp", "sell_type": SellType.SELL_SIGNAL}

        return None

