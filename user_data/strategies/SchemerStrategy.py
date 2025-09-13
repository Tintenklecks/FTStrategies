from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IntParameter


class SchemerStrategy(IStrategy):
    """SchemerStrategy (Smoothed Heikin-Ashi + EMA50 touch/cross, 1.5R TP)

    Rules distilled from the provided transcript:
    - Timeframe: 5 minutes
    - Trend filter: Double-smoothed Heikin-Ashi (EMA, length 50 for both smoothings)
      - Longs only when SHA is green (bullish)
      - Shorts only when SHA is red (bearish)
    - EMA signal: EMA(50) acts as dynamic S/R and trigger
      - Before entry, EMA must be clearly on the S/R side of the SHA candle (no touch)
        - For long: EMA below the SHA candle low (support)
        - For short: EMA above the SHA candle high (resistance)
      - Trigger: price touches/crosses EMA50, then enter when a confirming candle closes
        - Long: bullish close above EMA after touch
        - Short: bearish close below EMA after touch
    - Stoploss (configurable):
      - "wide": at SHA low/high of the signal candle (previous candle to entry)
      - "midbody": at 50% of the SHA body of the signal candle
    - Take profit: fixed 1.5R (risk-reward 1:1.5) using custom_exit
    - Sessions: Only allow new entries during London and New York sessions
      - Simple UTC approximation: London 07:00-16:00, New York 12:00-20:00
      - On Fridays, do not open new trades at/after 20:00 UTC
      - Positions may remain open overnight; no forced session close
    - Optional volume/POC proxy filter: daily anchored VWAP (as proxy for POC)
      - Long only if price > daily VWAP; Short only if price < daily VWAP

    Notes:
    - This strategy uses custom_stoploss to anchor the initial stop at the signal candle
      and keep it fixed (no trailing), and custom_exit for the 1.5R target.
    - The POC mentioned in the video is approximated here with daily anchored VWAP,
      which avoids repaint when computed incrementally per day.
    """

    # --- Strategy metadata ---
    can_short: bool = True
    timeframe: str = "5m"

    # Avoid ROI exits; use custom_exit for 1.5R target
    minimal_roi: Dict[str, float] = {"0": 10.0}

    # We set a very loose base stop; custom_stoploss will set the true stop each tick
    stoploss: float = -0.99

    # Hyperparameters / user-toggles
    ema_period: IntParameter = IntParameter(10, 200, default=50, space="buy", optimize=False)
    sha_smooth_length: IntParameter = IntParameter(10, 200, default=50, space="buy", optimize=False)
    use_vwap_filter: BooleanParameter = BooleanParameter(default=True, space="buy", optimize=False)
    stop_mode: CategoricalParameter = CategoricalParameter(["wide", "midbody"], default="midbody", space="buy", optimize=False)

    process_only_new_candles = True
    startup_candle_count: int = 300

    plot_config = {
        "main_plot": {
            "ema50": {"color": "yellow"},
            "daily_vwap": {"color": "#8888ff"},
        },
        "subplots": {
            "sha_color": {"sha_green": {"color": "#00aa00"}, "sha_red": {"color": "#aa0000"}},
        },
    }

    # --------------------------- Helpers ---------------------------------
    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False, min_periods=length).mean()

    @staticmethod
    def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """Compute classic Heikin-Ashi OHLC from raw OHLC.

        Returns DataFrame with columns: ha_open, ha_close, ha_high, ha_low.
        """
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        ha_open = ha_close.copy()
        # Initialize ha_open with first valid value
        if not ha_open.empty:
            ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        # Recursive definition for ha_open
        for i in range(1, len(df)):
            ha_open.iat[i] = (ha_open.iat[i - 1] + ha_close.iat[i - 1]) / 2.0
        ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
        return pd.DataFrame({
            "ha_open": ha_open,
            "ha_close": ha_close,
            "ha_high": ha_high,
            "ha_low": ha_low,
        }, index=df.index)

    def _smoothed_heikin_ashi(self, df: pd.DataFrame, length: int) -> pd.DataFrame:
        """Double-smoothed Heikin-Ashi using EMA for both smoothing steps.

        We smooth ha_open and ha_close twice; ha_high/ha_low are derived from smoothed
        open/close and raw highs/lows, following common SHA implementations.
        """
        ha = self._heikin_ashi(df)
        # First smoothing
        s_open_1 = self._ema(ha["ha_open"], length)
        s_close_1 = self._ema(ha["ha_close"], length)
        # Second smoothing
        s_open_2 = self._ema(s_open_1, length)
        s_close_2 = self._ema(s_close_1, length)
        # High/Low from smoothed open/close and raw highs/lows
        s_high = pd.concat([df["high"], s_open_2, s_close_2], axis=1).max(axis=1)
        s_low = pd.concat([df["low"], s_open_2, s_close_2], axis=1).min(axis=1)
        out = pd.DataFrame({
            "sha_open": s_open_2,
            "sha_close": s_close_2,
            "sha_high": s_high,
            "sha_low": s_low,
        }, index=df.index)
        return out

    @staticmethod
    def _daily_vwap(df: pd.DataFrame) -> pd.Series:
        """Compute daily anchored VWAP (proxy for daily POC dominance).

        Resets each UTC day at 00:00.
        """
        # Typical price
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].fillna(0.0)
        day = df.index.tz_convert(timezone.utc).normalize()
        cum_pv = (tp * vol).groupby(day, sort=False).cumsum()
        cum_v = vol.groupby(day, sort=False).cumsum().replace(0.0, np.nan)
        vwap = cum_pv / cum_v
        return vwap

    @staticmethod
    def _is_london_ny_session_utc(dt: pd.Timestamp) -> bool:
        """Allow entries only during London (07-16 UTC) and New York (12-20 UTC)."""
        hour = dt.hour
        in_london = 7 <= hour < 16
        in_ny = 12 <= hour < 20
        return in_london or in_ny

    @staticmethod
    def _friday_after_cutoff_utc(dt: pd.Timestamp) -> bool:
        """Block new entries on Fridays at/after 20:00 UTC."""
        # Monday=0 ... Sunday=6
        return dt.weekday() == 4 and dt.hour >= 20

    # ------------------------- Populate indicators ------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:  # type: ignore[override]
        # EMA 50 on close
        ema_len = int(self.ema_period.value)
        dataframe["ema50"] = self._ema(dataframe["close"], ema_len)

        # Double-smoothed Heikin-Ashi
        sha_len = int(self.sha_smooth_length.value)
        sha = self._smoothed_heikin_ashi(dataframe, sha_len)
        dataframe = dataframe.join(sha)

        # SHA color helpers
        dataframe["sha_green"] = (dataframe["sha_close"] > dataframe["sha_open"]).astype(int)
        dataframe["sha_red"] = (dataframe["sha_close"] < dataframe["sha_open"]).astype(int)

        # VWAP proxy (daily anchored)
        dataframe["daily_vwap"] = self._daily_vwap(dataframe)

        # Touch/cross detection relative to EMA
        dataframe["touch_ema"] = (
            (dataframe["low"] <= dataframe["ema50"]) & (dataframe["high"] >= dataframe["ema50"])  # touch or cross
        ).astype(int)

        # Confirming closes
        dataframe["bull_close_above_ema"] = (
            (dataframe["close"] > dataframe["ema50"]) & (dataframe["close"] > dataframe["open"])  # bullish body closing above EMA
        ).astype(int)
        dataframe["bear_close_below_ema"] = (
            (dataframe["close"] < dataframe["ema50"]) & (dataframe["close"] < dataframe["open"])  # bearish body closing below EMA
        ).astype(int)

        # EMA location vs SHA (strict: no touching)
        dataframe["ema_below_sha"] = (dataframe["ema50"] < dataframe["sha_low"]).astype(int)
        dataframe["ema_above_sha"] = (dataframe["ema50"] > dataframe["sha_high"]).astype(int)

        # Session filters (UTC index expected by Freqtrade)
        idx_utc = dataframe.index.tz_convert(timezone.utc)
        dataframe["is_session"] = [self._is_london_ny_session_utc(ts) for ts in idx_utc]
        dataframe["friday_after_cutoff"] = [self._friday_after_cutoff_utc(ts) for ts in idx_utc]

        return dataframe

    # ------------------------- Entry conditions ---------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:  # type: ignore[override]
        dataframe.loc[:, "enter_long"] = False
        dataframe.loc[:, "enter_short"] = False
        dataframe.loc[:, "enter_tag"] = None

        # VWAP filter flag
        use_vwap = bool(self.use_vwap_filter.value)

        # Long conditions
        long_cond = (
            (dataframe["sha_green"] == 1) &  # trend filter
            (dataframe["ema_below_sha"] == 1) &  # EMA clearly acting as support
            (
                # touch then confirm: either last candle touched EMA, and we now close bullish above
                (dataframe["touch_ema"].shift(1) == 1) & (dataframe["bull_close_above_ema"] == 1)
            )
        )
        if use_vwap:
            long_cond &= (dataframe["close"] > dataframe["daily_vwap"])

        # Session gating
        long_cond &= (dataframe["is_session"] == True) & (dataframe["friday_after_cutoff"] == False)

        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (True, "sha+ema50_long")

        # Short conditions
        short_cond = (
            (dataframe["sha_red"] == 1) &  # trend filter
            (dataframe["ema_above_sha"] == 1) &  # EMA clearly acting as resistance
            (
                (dataframe["touch_ema"].shift(1) == 1) & (dataframe["bear_close_below_ema"] == 1)
            )
        )
        if use_vwap:
            short_cond &= (dataframe["close"] < dataframe["daily_vwap"])

        short_cond &= (dataframe["is_session"] == True) & (dataframe["friday_after_cutoff"] == False)

        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (True, "sha+ema50_short")

        return dataframe

    # ------------------------- Exit conditions ----------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:  # type: ignore[override]
        # We use custom_exit (1.5R), so no static exit conditions
        dataframe.loc[:, "exit_long"] = False
        dataframe.loc[:, "exit_short"] = False
        return dataframe

    # ---------------------- Custom Stoploss/Exit ---------------------------
    use_custom_stoploss: bool = True

    def _get_signal_candle(self, df: pd.DataFrame, trade_open_time: datetime) -> Optional[pd.Series]:
        """Return the signal candle (previous candle to the entry candle)."""
        # Entry occurs at the open of the candle following the signal.
        # So we locate the candle with index strictly less than open_time and take the last.
        df_sub = df.loc[df.index < pd.Timestamp(trade_open_time, tz=timezone.utc)]
        if df_sub.empty:
            return None
        # The last candle before open_time is the signal candle.
        return df_sub.iloc[-1]

    def _compute_signal_stop_price(self, side: str, sig: pd.Series) -> Optional[float]:
        mode = str(self.stop_mode.value)
        if mode == "wide":
            if side == "long":
                return float(sig.get("sha_low", np.nan))
            else:
                return float(sig.get("sha_high", np.nan))
        elif mode == "midbody":
            sha_open = float(sig.get("sha_open", np.nan))
            sha_close = float(sig.get("sha_close", np.nan))
            if np.isnan(sha_open) or np.isnan(sha_close):
                return None
            mid = (sha_open + sha_close) / 2.0
            return float(mid)
        return None

    def custom_stoploss(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:  # type: ignore[override]
        """Anchor initial stop at the signal candle and keep it fixed.

        Returns relative stoploss (negative for long, positive for short) vs current_rate.
        """
        # Retrieve dataframe for the pair
        df: Optional[pd.DataFrame] = kwargs.get("dataframe", None)
        if df is None:
            return 1  # fall back to base stoploss

        sig = self._get_signal_candle(df, trade.open_date_utc)
        if sig is None:
            return 1

        side = "long" if trade.is_long else "short"
        sl_price = self._compute_signal_stop_price(side, sig)
        if sl_price is None or np.isnan(sl_price):
            return 1

        # Compute relative stoploss to maintain absolute sl_price
        if trade.is_long:
            rel = (sl_price / current_rate) - 1.0
            # Ensure not above current (which would instantly stop out)
            rel = min(rel, -0.001)
            return rel
        else:
            # For shorts, loss occurs when price rises. Relative positive value.
            rel = 1.0 - (sl_price / current_rate)
            rel = min(rel, -0.001)  # freqtrade expects negative? guard by mirroring
            # Note: Freqtrade expects negative values for stoploss. For shorts, we convert to negative distance.
            return -abs(rel)

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[str]:  # type: ignore[override]
        """Exit at 1.5R based on anchored stop from signal candle."""
        df: Optional[pd.DataFrame] = kwargs.get("dataframe", None)
        if df is None:
            return None

        sig = self._get_signal_candle(df, trade.open_date_utc)
        if sig is None:
            return None

        side = "long" if trade.is_long else "short"
        sl_price = self._compute_signal_stop_price(side, sig)
        if sl_price is None or np.isnan(sl_price):
            return None

        open_rate = float(trade.open_rate)
        if trade.is_long:
            risk = open_rate - sl_price
            if risk <= 0:
                return None
            target = open_rate + 1.5 * risk
            # If current candle traded at/above target, exit
            row = df.loc[df.index == pd.Timestamp(current_time, tz=timezone.utc)]
            if not row.empty:
                high_now = float(row["high"].iloc[0])
                if high_now >= target:
                    return "tp_1.5R"
        else:
            risk = sl_price - open_rate
            if risk <= 0:
                return None
            target = open_rate - 1.5 * risk
            row = df.loc[df.index == pd.Timestamp(current_time, tz=timezone.utc)]
            if not row.empty:
                low_now = float(row["low"].iloc[0])
                if low_now <= target:
                    return "tp_1.5R"
        return None

