"""Short-horizon technical signal and backtest utilities.

The module intentionally separates short-horizon signals from the long-term
quality/valuation model.  It only uses information available on or before the
signal date.  A signal is an observation state (breakout/pullback/trend), not a
promise of profit or a substitute for checking announcements and trading risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


HISTORY_ALIASES = {
    "date": ("日期", "date", "Date", "时间"),
    "open": ("开盘", "open", "Open"),
    "high": ("最高", "high", "High"),
    "low": ("最低", "low", "Low"),
    "close": ("收盘", "close", "Close"),
    "volume": ("成交量", "volume", "Volume", "vol", "Vol"),
    "amount": ("成交额", "amount", "Amount"),
    "turnover": ("换手率", "turnover", "Turnover"),
}


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize AKShare Eastmoney/Tencent daily bars to a stable schema."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "turnover"])
    out = pd.DataFrame()
    for target, aliases in HISTORY_ALIASES.items():
        source = _first_column(frame, aliases)
        if source is not None:
            out[target] = frame[source]
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(out.columns):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "turnover"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[(out["close"] > 0) & (out["high"] > 0) & (out["low"] > 0)]
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add daily indicators used by the after-close overnight plan.

    The strategy is intentionally end-of-day.  It uses the completed signal-day
    bar to prepare the next trading day's plan.  It does not pretend to be an
    intraday or 14:50 signal.
    """
    df = normalize_history_frame(frame).copy()
    if df.empty:
        return df
    open_, close, high, low = df["open"], df["close"], df["high"], df["low"]
    for days in (3, 5, 10, 20, 60, 120, 250):
        df[f"ma{days}"] = close.rolling(days, min_periods=days).mean()
        df[f"ret{days}"] = close.pct_change(days) * 100
    df["ma10AboveMA20"] = (df["ma10"] > df["ma20"]).astype(float)
    df["ma20Slope"] = (df["ma20"] / df["ma20"].shift(5) - 1) * 100
    df["ma60Slope"] = (df["ma60"] / df["ma60"].shift(10) - 1) * 100

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macdDif"] = ema12 - ema26
    df["macdDea"] = df["macdDif"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macdHist"] = (df["macdDif"] - df["macdDea"]) * 2

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    df.loc[(loss == 0) & (gain > 0), "rsi14"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi14"] = 50

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    df["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atrPct"] = df["atr14"] / close * 100
    df["volatility20"] = close.pct_change().rolling(20, min_periods=20).std() * math.sqrt(20) * 100

    df["high20Prev"] = high.shift(1).rolling(20, min_periods=20).max()
    df["high60Prev"] = high.shift(1).rolling(60, min_periods=60).max()
    df["low20"] = low.rolling(20, min_periods=20).min()
    df["rollingHigh60"] = close.rolling(60, min_periods=60).max()
    df["maxDrawdown60"] = (close / df["rollingHigh60"] - 1) * 100
    df["breakout20"] = (close > df["high20Prev"]).astype(float)
    df["breakout60"] = (close > df["high60Prev"]).astype(float)
    df["distanceMA10"] = (close / df["ma10"] - 1) * 100
    df["distanceMA20"] = (close / df["ma20"] - 1) * 100
    df["distanceHigh20"] = (close / df["high20Prev"] - 1) * 100
    df["distanceLow20"] = (close / df["low20"] - 1) * 100
    df["pullback20"] = (
        (df["ma10"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
        & (df["ma20Slope"] > 0)
        & (df["distanceMA20"].between(-2.8, 2.2))
        & (close > df["ma60"])
    ).astype(float)

    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std()
    upper, lower = middle + 2 * deviation, middle - 2 * deviation
    df["bollingerPctB"] = (close - lower) / (upper - lower).replace(0, np.nan) * 100

    df["amountAvg5"] = df["amount"].rolling(5, min_periods=3).mean()
    df["amountAvg20"] = df["amount"].rolling(20, min_periods=10).mean()
    df["volumeAvg5"] = df["volume"].rolling(5, min_periods=3).mean()
    df["volumeAvg20"] = df["volume"].rolling(20, min_periods=10).mean()
    df["volumeRatio5_20"] = df["volumeAvg5"] / df["volumeAvg20"].replace(0, np.nan)
    df["amountRatio5_20"] = df["amountAvg5"] / df["amountAvg20"].replace(0, np.nan)
    df["volumeTodayRatio20"] = df["volume"] / df["volumeAvg20"].replace(0, np.nan)
    df["amountTodayRatio20"] = df["amount"] / df["amountAvg20"].replace(0, np.nan)
    df["turnoverAvg5"] = df["turnover"].rolling(5, min_periods=3).mean()

    # End-of-day strength.  These are especially useful for a next-day plan.
    day_range = (high - low).replace(0, np.nan)
    df["dayPct"] = close.pct_change() * 100
    df["gapOpenPct"] = (open_ / previous_close - 1) * 100
    df["bodyPct"] = (close / open_ - 1) * 100
    df["closeLocation"] = (close - low) / day_range * 100
    df["upperShadowPct"] = (high - np.maximum(open_, close)) / previous_close.replace(0, np.nan) * 100
    df["lowerShadowPct"] = (np.minimum(open_, close) - low) / previous_close.replace(0, np.nan) * 100
    df["closeNearHigh"] = (df["closeLocation"] >= 75).astype(float)
    df["greenDay"] = (close > open_).astype(float)
    df["limitUp"] = (df["dayPct"] >= 9.5).astype(float)
    df["limitDown"] = (df["dayPct"] <= -9.5).astype(float)
    return df


def _amount_score(amount_yuan: float | None) -> float | None:
    if amount_yuan is None:
        return None
    anchors = [(2e7, 15), (5e7, 35), (1e8, 52), (3e8, 72), (8e8, 88), (2e9, 100)]
    if amount_yuan <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if amount_yuan <= x1:
            return y0 + (amount_yuan - x0) / (x1 - x0) * (y1 - y0)
    return 100.0


def _linear(value, bad, good):
    number = _finite(value)
    if number is None:
        return None
    if good == bad:
        return 50.0
    return _clip((number - bad) / (good - bad) * 100)


def _inverse(value, good, bad):
    number = _finite(value)
    if number is None:
        return None
    return _clip((bad - number) / (bad - good) * 100)


def _weighted(items: Iterable[tuple[float | None, float]], neutral: float = 50.0) -> tuple[float, float]:
    items = list(items)
    total = sum(max(0.0, weight) for _, weight in items) or 1.0
    score, covered = 0.0, 0.0
    for value, weight in items:
        weight = max(0.0, weight)
        if value is None or not math.isfinite(float(value)):
            score += neutral * weight
        else:
            score += _clip(float(value)) * weight
            covered += weight
    return score / total, covered / total


def _rsi_score(value) -> float | None:
    value = _finite(value)
    if value is None:
        return None
    if 50 <= value <= 68:
        return 90 - abs(value - 59) * 1.5
    if 42 <= value < 50:
        return 55 + (value - 42) * 4
    if 68 < value <= 76:
        return 82 - (value - 68) * 4
    if value > 76:
        return max(5, 50 - (value - 76) * 5)
    return max(10, 50 - (42 - value) * 3)


def _metrics_from_indicator_frame(df: pd.DataFrame, index: int = -1) -> dict:
    if df is None or df.empty:
        return {"shortDataLevel": "未取得日K", "historyBars": 0, "shortDataCoverage": 0}
    if index < 0:
        index = len(df) + index
    index = max(0, min(int(index), len(df) - 1))
    row = df.iloc[index]
    history_bars = index + 1
    close = _finite(row.get("close"))
    atr = _finite(row.get("atr14"))
    low20 = _finite(row.get("low20"))
    ma10 = _finite(row.get("ma10"))
    ma20 = _finite(row.get("ma20"))
    ma60 = _finite(row.get("ma60"))

    stop_candidates = []
    if close and atr:
        stop_candidates.append(close - 1.6 * atr)
    if low20:
        stop_candidates.append(low20 * 0.995)
    if ma20:
        stop_candidates.append(ma20 - 0.8 * (atr or 0))
    stop_candidates = [x for x in stop_candidates if x is not None and close and 0 < x < close]
    technical_stop = max(stop_candidates) if stop_candidates else None
    risk_pct = (close - technical_stop) / close * 100 if close and technical_stop else None
    target_15r = close + 1.5 * (close - technical_stop) if close and technical_stop else None

    fields = {
        "historyBars": int(history_bars),
        "historyStart": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "historyEnd": df["date"].iloc[index].strftime("%Y-%m-%d"),
        "signalDate": df["date"].iloc[index].strftime("%Y-%m-%d"),
        "shortDataLevel": "精确日K" if history_bars >= 250 else ("部分日K" if history_bars >= 120 else "日K不足"),
        "signalClose": close,
        "ma5": _finite(row.get("ma5")), "ma10": ma10, "ma20": ma20,
        "ma60": ma60, "ma120": _finite(row.get("ma120")), "ma250": _finite(row.get("ma250")),
        "aboveMA10": None if close is None or ma10 is None else int(close > ma10),
        "aboveMA20": None if close is None or ma20 is None else int(close > ma20),
        "aboveMA60": None if close is None or ma60 is None else int(close > ma60),
        "aboveMA250": None if close is None or _finite(row.get("ma250")) is None else int(close > float(row.get("ma250"))),
        "ma10AboveMA20": None if ma10 is None or ma20 is None else int(ma10 > ma20),
        "ma20AboveMA60": None if ma20 is None or ma60 is None else int(ma20 > ma60),
        "ma20Slope": _finite(row.get("ma20Slope")), "ma60Slope": _finite(row.get("ma60Slope")),
        "ret3": _finite(row.get("ret3")), "ret5": _finite(row.get("ret5")), "ret10": _finite(row.get("ret10")),
        "ret20": _finite(row.get("ret20")), "ret60": _finite(row.get("ret60")), "ret120": _finite(row.get("ret120")),
        "dayPct": _finite(row.get("dayPct")), "gapOpenPct": _finite(row.get("gapOpenPct")),
        "bodyPct": _finite(row.get("bodyPct")), "closeLocation": _finite(row.get("closeLocation")),
        "upperShadowPct": _finite(row.get("upperShadowPct")), "lowerShadowPct": _finite(row.get("lowerShadowPct")),
        "closeNearHigh": int(row.get("closeNearHigh", 0) == 1), "greenDay": int(row.get("greenDay", 0) == 1),
        "rsi14": _finite(row.get("rsi14")), "macdDif": _finite(row.get("macdDif")),
        "macdDea": _finite(row.get("macdDea")), "macdHist": _finite(row.get("macdHist")),
        "atr14": atr, "atrPct": _finite(row.get("atrPct")), "volatility20": _finite(row.get("volatility20")),
        "high20": _finite(row.get("high20Prev")), "high60": _finite(row.get("high60Prev")), "low20": low20,
        "distanceMA10": _finite(row.get("distanceMA10")), "distanceMA20": _finite(row.get("distanceMA20")),
        "distanceHigh20": _finite(row.get("distanceHigh20")), "distanceLow20": _finite(row.get("distanceLow20")),
        "bollingerPctB": _finite(row.get("bollingerPctB")),
        "breakout20": int(row.get("breakout20", 0) == 1), "breakout60": int(row.get("breakout60", 0) == 1),
        "pullback20": int(row.get("pullback20", 0) == 1),
        "amountAvg5": _finite(row.get("amountAvg5")), "amountAvg20": _finite(row.get("amountAvg20")),
        "volumeRatio5_20": _finite(row.get("volumeRatio5_20")), "amountRatio5_20": _finite(row.get("amountRatio5_20")),
        "volumeTodayRatio20": _finite(row.get("volumeTodayRatio20")), "amountTodayRatio20": _finite(row.get("amountTodayRatio20")),
        "turnoverAvg5": _finite(row.get("turnoverAvg5")), "maxDrawdown60": _finite(row.get("maxDrawdown60")),
        "limitUp": int(row.get("limitUp", 0) == 1), "limitDown": int(row.get("limitDown", 0) == 1),
        "technicalStop": technical_stop, "technicalRiskPct": risk_pct, "target15R": target_15r,
        "nextOpenMinGapPct": -4.0, "nextOpenMaxGapPct": 2.5,
        "entryReferenceLow": close * 0.98 if close else None,
        "entryReferenceHigh": close * 1.025 if close else None,
        "shortMetricMethod": "收盘后隔日计划：前复权日K、真实均线、收盘强度、量价、RSI/MACD/ATR",
    }
    core_names = [
        "ma10", "ma20", "ma60", "ma20Slope", "ret3", "ret5", "rsi14", "macdHist", "atrPct",
        "amountAvg20", "volumeTodayRatio20", "amountTodayRatio20", "closeLocation", "upperShadowPct", "dayPct",
    ]
    fields["shortDataCoverage"] = round(sum(fields.get(k) is not None for k in core_names) / len(core_names) * 100)
    return fields


def latest_short_metrics(frame: pd.DataFrame) -> dict:
    return _metrics_from_indicator_frame(add_indicators(frame), -1)


def _ideal_band(value, low, ideal_low, ideal_high, high) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    if ideal_low <= number <= ideal_high:
        return 92.0
    if number < ideal_low:
        return _clip(20 + (number - low) / max(ideal_low - low, 1e-9) * 72)
    return _clip(92 - (number - ideal_high) / max(high - ideal_high, 1e-9) * 82)


def score_short_metrics(metrics: dict, market_score: float | None = None, industry_score: float | None = None) -> dict:
    """Score a realistic after-close plan for the next trading day.

    A signal is generated only after the signal-day bar is complete.  The plan
    assumes a possible entry at the next session's open and, because A-shares
    are T+1, the earliest modeled exit is the following trading day.
    """
    setup_score, setup_cov = _weighted([
        (84 if metrics.get("aboveMA20") == 1 else 22 if metrics.get("aboveMA20") == 0 else None, 16),
        (82 if metrics.get("aboveMA60") == 1 else 24 if metrics.get("aboveMA60") == 0 else None, 14),
        (88 if metrics.get("ma10AboveMA20") == 1 and metrics.get("ma20AboveMA60") == 1 else 38, 16),
        (_linear(metrics.get("ma20Slope"), -2.5, 3.5), 10),
        (_linear(metrics.get("closeLocation"), 25, 90), 18),
        (96 if metrics.get("breakout20") else 84 if metrics.get("pullback20") else _linear(metrics.get("distanceHigh20"), -12, 0.5), 18),
        (_inverse(abs(float(metrics.get("distanceMA20"))) if _finite(metrics.get("distanceMA20")) is not None else None, 0.5, 12), 8),
    ])
    volume_score, volume_cov = _weighted([
        (_amount_score(_finite(metrics.get("amountAvg20"))), 35),
        (_ideal_band(metrics.get("volumeTodayRatio20"), 0.45, 0.9, 2.2, 4.5), 22),
        (_ideal_band(metrics.get("amountTodayRatio20"), 0.45, 0.9, 2.3, 4.5), 18),
        (_ideal_band(metrics.get("turnoverAvg5"), 0.15, 0.8, 6.0, 16), 15),
        (_linear(metrics.get("amountRatio5_20"), 0.65, 1.6), 10),
    ])
    momentum_score, momentum_cov = _weighted([
        (_rsi_score(metrics.get("rsi14")), 24),
        (_ideal_band(metrics.get("ret3"), -7, 0.5, 8.0, 15), 20),
        (_ideal_band(metrics.get("ret5"), -10, 1.0, 12.0, 22), 18),
        (_ideal_band(metrics.get("dayPct"), -5, 0.5, 6.5, 10), 18),
        (86 if _finite(metrics.get("macdHist")) is not None and float(metrics.get("macdHist")) > 0 else 38 if _finite(metrics.get("macdHist")) is not None else None, 12),
        (_linear(metrics.get("closeLocation"), 30, 90), 8),
    ])
    environment_score, env_cov = _weighted([(market_score, 50), (industry_score, 50)])

    risk = 100.0
    blocks: list[str] = []
    atr_pct = _finite(metrics.get("atrPct"))
    rsi = _finite(metrics.get("rsi14"))
    distance = _finite(metrics.get("distanceMA20"))
    ret3 = _finite(metrics.get("ret3"))
    ret5 = _finite(metrics.get("ret5"))
    day_pct = _finite(metrics.get("dayPct"))
    close_location = _finite(metrics.get("closeLocation"))
    upper_shadow = _finite(metrics.get("upperShadowPct"))
    amount20 = _finite(metrics.get("amountAvg20"))
    if metrics.get("limitUp"):
        risk -= 40; blocks.append("信号日接近涨停，次日容易高开或无法获得合理成交")
    if rsi is not None and rsi > 76:
        risk -= 22; blocks.append("RSI偏热")
    if distance is not None and distance > 10:
        risk -= 20; blocks.append("偏离20日均线较大")
    if ret3 is not None and ret3 > 12:
        risk -= 20; blocks.append("3日涨幅过快")
    if ret5 is not None and ret5 > 18:
        risk -= 18; blocks.append("5日涨幅过快")
    if atr_pct is not None and atr_pct > 6:
        risk -= 18; blocks.append("ATR波动偏高")
    if close_location is not None and close_location < 38:
        risk -= 18; blocks.append("收盘位置偏弱")
    if upper_shadow is not None and upper_shadow > 4.5:
        risk -= 16; blocks.append("上影线偏长，追涨承接需核查")
    if metrics.get("aboveMA20") == 0 and metrics.get("aboveMA60") == 0:
        risk -= 30; blocks.append("同时位于20日与60日均线下方")
    if amount20 is not None and amount20 < 1e8:
        risk -= 30; blocks.append("20日平均成交额不足1亿元")
    if day_pct is not None and day_pct <= -4:
        risk -= 22; blocks.append("信号日明显走弱")
    risk = _clip(risk)

    overall = _clip(environment_score * 0.20 + setup_score * 0.30 + volume_score * 0.25 + momentum_score * 0.15 + risk * 0.10)
    coverage = round((env_cov * 0.20 + setup_cov * 0.30 + volume_cov * 0.25 + momentum_cov * 0.15 + 1.0 * 0.10) * 100)
    confidence = 0.55 + 0.45 * coverage / 100
    trusted = 50 + (overall - 50) * confidence if overall > 50 else overall
    if metrics.get("historyBars", 0) < 120:
        trusted = min(trusted, 54)
    trusted = round(_clip(trusted), 1)

    supports: list[str] = []
    if metrics.get("breakout20"):
        supports.append("收盘突破前20日高点")
    if metrics.get("pullback20"):
        supports.append("多头结构中回踩MA20附近")
    if close_location is not None and close_location >= 75:
        supports.append("收盘位于当日区间上部")
    if metrics.get("aboveMA20") == 1 and metrics.get("aboveMA60") == 1:
        supports.append("位于MA20和MA60上方")
    if _finite(metrics.get("volumeTodayRatio20")) is not None and float(metrics.get("volumeTodayRatio20")) >= 1.1:
        supports.append("当日量能高于20日均量")
    if amount20 is not None and amount20 >= 2e8:
        supports.append("成交活跃度适合隔日策略")

    market_value = _finite(market_score) or 50.0
    industry_value = _finite(industry_score) or 50.0
    volume_today = _finite(metrics.get("volumeTodayRatio20"))
    amount_today = _finite(metrics.get("amountTodayRatio20"))
    ma20_slope = _finite(metrics.get("ma20Slope"))
    macd_hist = _finite(metrics.get("macdHist"))

    decision, group = "等待隔日形态", "wait"
    action = "收盘形态尚未达到隔日计划门槛，等待突破、回踩企稳或收盘转强。"
    if metrics.get("historyBars", 0) < 120 or coverage < 70:
        decision, group = "隔日数据不足", "data"
        action = "日K或量价字段不足，不能形成隔日计划。"
    elif amount20 is not None and amount20 < 1e8:
        decision, group = "流动性不足", "avoid"
        action = "20日平均成交额不足1亿元，不作为常规隔日交易标的。"
    elif metrics.get("limitUp") or (rsi is not None and rsi > 76) or (distance is not None and distance > 10) or (ret3 is not None and ret3 > 12):
        decision, group = "过热不追", "avoid"
        action = "信号日已明显过热；次日即使高开也不追，等待换手和重新形成结构。"
    elif metrics.get("aboveMA20") == 0 and metrics.get("aboveMA60") == 0:
        decision, group = "弱势规避", "avoid"
        action = "价格结构偏弱，不以低估或反弹预期替代隔日止损纪律。"
    else:
        breakout_ready = (
            bool(metrics.get("breakout20"))
            and (close_location is not None and close_location >= 70)
            and max(volume_today or 0, amount_today or 0) >= 1.15
            and metrics.get("aboveMA60") == 1
            and (ma20_slope is not None and ma20_slope > 0.2)
            and (day_pct is not None and 0.8 <= day_pct <= 8.0)
            and (rsi is not None and 50 <= rsi <= 74)
            and (upper_shadow is None or upper_shadow <= 4.5)
            and market_value >= 48 and industry_value >= 47
            and trusted >= 68
        )
        pullback_ready = (
            bool(metrics.get("pullback20"))
            and (close_location is not None and close_location >= 55)
            and (volume_today is None or volume_today <= 1.18)
            and (day_pct is None or -2.5 <= day_pct <= 4.5)
            and (ret5 is not None and -7 <= ret5 <= 11)
            and (rsi is not None and 44 <= rsi <= 66)
            and (macd_hist is None or macd_hist > -0.04)
            and market_value >= 45 and industry_value >= 44
            and trusted >= 62
        )
        strong_close_ready = (
            metrics.get("aboveMA20") == 1 and metrics.get("aboveMA60") == 1
            and metrics.get("ma10AboveMA20") == 1
            and (close_location is not None and close_location >= 82)
            and (day_pct is not None and 1.0 <= day_pct <= 6.5)
            and max(volume_today or 0, amount_today or 0) >= 1.05
            and (ret5 is None or ret5 <= 14)
            and market_value >= 50 and industry_value >= 48
            and trusted >= 65
        )
        if breakout_ready:
            decision, group = "隔日突破备选", "signal"
            action = "次日仅在开盘相对信号收盘价-4%至+2.5%、未跌回突破位且市场不转弱时观察；高开超过2.5%不追。"
        elif pullback_ready:
            decision, group = "隔日回踩备选", "signal"
            action = "次日仅在MA20附近企稳、开盘不过度跳空且不跌破保护位时观察；跌破MA60则取消计划。"
        elif strong_close_ready:
            decision, group = "收盘强势观察", "signal"
            action = "收盘强度较好但不是标准突破/回踩；次日若平开或小幅低开后承接稳定可观察，高开超过2.5%不追。"
        elif metrics.get("aboveMA20") == 1 and metrics.get("aboveMA60") == 1 and trusted >= 58:
            decision, group = "等待隔日触发", "wait"
            action = "趋势尚可但缺少明确触发点，等待收盘突破、缩量回踩或量价共振。"

    return {
        "shortScore": trusted,
        "shortRawScore": round(overall, 1),
        "shortCoverage": coverage,
        "shortDecision": decision,
        "shortGroup": group,
        "shortAction": action,
        "shortSupports": supports[:5],
        "shortBlocks": list(dict.fromkeys(blocks))[:6],
        "shortDimensions": {
            "environment": round(environment_score, 1), "price": round(setup_score, 1),
            "volume": round(volume_score, 1), "momentum": round(momentum_score, 1), "risk": round(risk, 1),
        },
        "overnightModel": "收盘后选股→次日开盘确认→最早再下一交易日退出（A股T+1）",
    }


def market_environment_score(frame: pd.DataFrame) -> tuple[float, dict]:
    df = add_indicators(frame)
    if df.empty:
        return 50.0, {"marketTrendLabel": "市场数据不足"}
    row = df.iloc[-1]
    score, coverage = _weighted([
        (80 if row.get("close") > row.get("ma20") else 25 if pd.notna(row.get("ma20")) else None, 25),
        (85 if row.get("close") > row.get("ma60") else 25 if pd.notna(row.get("ma60")) else None, 25),
        (_linear(row.get("ma20Slope"), -3, 3), 20),
        (_linear(row.get("ret20"), -10, 10), 20),
        (_rsi_score(row.get("rsi14")), 10),
    ])
    label = "偏强" if score >= 65 else "中性" if score >= 45 else "偏弱"
    return round(score, 1), {"marketTrendLabel": label, "marketTrendCoverage": round(coverage * 100)}


def market_environment_series(frame: pd.DataFrame) -> dict[pd.Timestamp, float]:
    """Compute market environment scores for every date in one vectorized pass."""
    df = add_indicators(frame)
    result: dict[pd.Timestamp, float] = {}
    if df.empty:
        return result
    for _, row in df.iterrows():
        close = _finite(row.get("close"))
        ma20 = _finite(row.get("ma20"))
        ma60 = _finite(row.get("ma60"))
        score, _ = _weighted([
            (80 if close is not None and ma20 is not None and close > ma20 else 25 if close is not None and ma20 is not None else None, 25),
            (85 if close is not None and ma60 is not None and close > ma60 else 25 if close is not None and ma60 is not None else None, 25),
            (_linear(row.get("ma20Slope"), -3, 3), 20),
            (_linear(row.get("ret20"), -10, 10), 20),
            (_rsi_score(row.get("rsi14")), 10),
        ])
        result[pd.Timestamp(row["date"])] = round(score, 1)
    return result


@dataclass
class BacktestConfig:
    # holding_days=1 means: enter at T+1 open and exit at T+2 close at the latest.
    # This is the shortest realistic after-close plan under A-share T+1 rules.
    holding_days: int = 1
    fee_bps: float = 20.0
    stop_atr: float = 1.6
    target_r: float = 1.5
    minimum_amount: float = 1e8
    max_gap_up: float = 2.5
    max_gap_down: float = -4.0


def backtest_frame(frame: pd.DataFrame, config: BacktestConfig | None = None, market_frame: pd.DataFrame | None = None) -> dict:
    """Walk-forward backtest for the after-close overnight plan.

    Signal: completed T-day close.
    Entry: T+1 open, only when the opening gap is inside the allowed band.
    Exit: because A-shares are T+1, no exit is allowed on the entry day; the
    earliest exit is T+2.  With holding_days=1 the trade exits at T+2 close,
    or at a stop/target reached on T+2.  If stop and target are both touched on
    the same daily bar, the conservative assumption is that the stop occurs first.
    """
    config = config or BacktestConfig()
    df = add_indicators(frame)
    if len(df) < 160:
        return {"trades": 0, "error": "日K少于160根，无法有效回测"}
    market_by_date = market_environment_series(market_frame) if market_frame is not None and not market_frame.empty else {}

    trades = []
    i = 120
    while i < len(df) - config.holding_days - 2:
        row = df.iloc[i]
        metrics = _metrics_from_indicator_frame(df, i)
        date = pd.Timestamp(row["date"])
        market_score = market_by_date.get(date, 50.0)
        result = score_short_metrics(metrics, market_score, 50.0)
        if result["shortDecision"] not in {"隔日突破备选", "隔日回踩备选", "收盘强势观察"}:
            i += 1
            continue
        if market_score < 45:
            i += 1
            continue
        if _finite(row.get("amountAvg20")) is not None and float(row.get("amountAvg20")) < config.minimum_amount:
            i += 1
            continue

        entry_index = i + 1
        entry = float(df.iloc[entry_index]["open"])
        signal_close = float(row["close"])
        gap_pct = (entry / signal_close - 1) * 100 if signal_close else 0
        if gap_pct > config.max_gap_up or gap_pct < config.max_gap_down:
            i += 1
            continue

        atr = _finite(row.get("atr14"))
        ma20 = _finite(row.get("ma20"))
        stop = entry - config.stop_atr * atr if atr else None
        if ma20:
            stop = max(stop or 0, ma20 - 0.8 * (atr or 0))
        if stop is not None and stop >= entry:
            stop = entry * 0.975
        if stop is not None:
            risk_pct = (entry - stop) / entry * 100
            if risk_pct < 1.5:
                stop = entry * 0.985
                risk_pct = 1.5
            if risk_pct > 8:
                i += 1
                continue
        else:
            risk_pct = None
        target = entry + config.target_r * (entry - stop) if stop else None

        # A-share T+1: do not stop out or take profit on the purchase day.
        earliest_exit_index = entry_index + 1
        exit_index = min(entry_index + config.holding_days, len(df) - 1)
        exit_index = max(exit_index, earliest_exit_index)
        exit_price = float(df.iloc[exit_index]["close"])
        exit_reason = f"T+{config.holding_days + 1}收盘"
        for j in range(earliest_exit_index, exit_index + 1):
            open_j = float(df.iloc[j]["open"])
            low_j = float(df.iloc[j]["low"])
            high_j = float(df.iloc[j]["high"])
            # Gap risk is filled at the opening price rather than an unavailable stop price.
            if stop and open_j <= stop:
                exit_index, exit_price, exit_reason = j, open_j, "跳空低开保护"
                break
            if target and open_j >= target:
                exit_index, exit_price, exit_reason = j, open_j, f"跳空达到{config.target_r:.1f}R"
                break
            # Conservative ordering when a daily bar touches both levels.
            if stop and low_j <= stop:
                exit_index, exit_price, exit_reason = j, float(stop), "保护位"
                break
            if target and high_j >= target:
                exit_index, exit_price, exit_reason = j, float(target), f"{config.target_r:.1f}R目标"
                break

        gross_return = exit_price / entry - 1
        net_return = gross_return - config.fee_bps / 10000.0
        entry_day_close = float(df.iloc[entry_index]["close"])
        entry_day_return = (entry_day_close / entry - 1) * 100
        trades.append({
            "signalDate": date.strftime("%Y-%m-%d"),
            "entryDate": pd.Timestamp(df.iloc[entry_index]["date"]).strftime("%Y-%m-%d"),
            "exitDate": pd.Timestamp(df.iloc[exit_index]["date"]).strftime("%Y-%m-%d"),
            "signal": result["shortDecision"], "score": result["shortScore"],
            "entry": round(entry, 4), "exit": round(exit_price, 4), "returnPct": round(net_return * 100, 3),
            "entryDayReturnPct": round(entry_day_return, 3),
            "gapPct": round(gap_pct, 3), "riskPct": round(risk_pct, 3) if risk_pct is not None else None,
            "exitReason": exit_reason,
        })
        i = exit_index + 1

    if not trades:
        return {"trades": 0, "winRate": None, "averageReturnPct": None, "compoundReturnPct": 0, "maxDrawdownPct": 0, "details": []}
    returns = np.array([t["returnPct"] / 100 for t in trades], dtype=float)
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.size and abs(losses.sum()) > 0 else None
    return {
        "trades": len(trades),
        "winRate": round(float((returns > 0).mean() * 100), 2),
        "averageReturnPct": round(float(returns.mean() * 100), 3),
        "medianReturnPct": round(float(np.median(returns) * 100), 3),
        "compoundReturnPct": round(float((equity[-1] - 1) * 100), 2),
        "maxDrawdownPct": round(float(drawdown.min() * 100), 2),
        "profitFactor": round(float(profit_factor), 3) if profit_factor is not None else None,
        "averageEntryDayReturnPct": round(float(np.mean([t["entryDayReturnPct"] for t in trades])), 3),
        "details": trades,
    }
