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
    df = normalize_history_frame(frame).copy()
    if df.empty:
        return df
    close, high, low = df["close"], df["high"], df["low"]
    for days in (5, 10, 20, 60, 120, 250):
        df[f"ma{days}"] = close.rolling(days, min_periods=days).mean()
        df[f"ret{days}"] = close.pct_change(days) * 100
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
    df["distanceMA20"] = (close / df["ma20"] - 1) * 100
    df["distanceHigh20"] = (close / df["high20Prev"] - 1) * 100
    df["distanceLow20"] = (close / df["low20"] - 1) * 100
    df["pullback20"] = (
        (df["ma20"] > df["ma60"])
        & (df["ma20Slope"] > 0)
        & (df["distanceMA20"].between(-2.5, 2.5))
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
    df["dayPct"] = close.pct_change() * 100
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
    ma20 = _finite(row.get("ma20"))
    stop_candidates = []
    if close and atr:
        stop_candidates.append(close - 1.8 * atr)
    if low20:
        stop_candidates.append(low20 * 0.995)
    if ma20:
        stop_candidates.append(ma20 - (atr or 0))
    stop_candidates = [x for x in stop_candidates if x is not None and close and 0 < x < close]
    technical_stop = max(stop_candidates) if stop_candidates else None
    risk_pct = (close - technical_stop) / close * 100 if close and technical_stop else None
    target_2r = close + 2 * (close - technical_stop) if close and technical_stop else None
    fields = {
        "historyBars": int(history_bars),
        "historyStart": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "historyEnd": df["date"].iloc[index].strftime("%Y-%m-%d"),
        "shortDataLevel": "精确日K" if history_bars >= 250 else ("部分日K" if history_bars >= 120 else "日K不足"),
        "ma5": _finite(row.get("ma5")), "ma10": _finite(row.get("ma10")), "ma20": ma20,
        "ma60": _finite(row.get("ma60")), "ma120": _finite(row.get("ma120")), "ma250": _finite(row.get("ma250")),
        "aboveMA20": None if close is None or ma20 is None else int(close > ma20),
        "aboveMA60": None if close is None or _finite(row.get("ma60")) is None else int(close > float(row.get("ma60"))),
        "aboveMA250": None if close is None or _finite(row.get("ma250")) is None else int(close > float(row.get("ma250"))),
        "ma20AboveMA60": None if ma20 is None or _finite(row.get("ma60")) is None else int(ma20 > float(row.get("ma60"))),
        "ma20Slope": _finite(row.get("ma20Slope")), "ma60Slope": _finite(row.get("ma60Slope")),
        "ret5": _finite(row.get("ret5")), "ret10": _finite(row.get("ret10")), "ret20": _finite(row.get("ret20")),
        "ret60": _finite(row.get("ret60")), "ret120": _finite(row.get("ret120")),
        "rsi14": _finite(row.get("rsi14")), "macdDif": _finite(row.get("macdDif")),
        "macdDea": _finite(row.get("macdDea")), "macdHist": _finite(row.get("macdHist")),
        "atr14": atr, "atrPct": _finite(row.get("atrPct")), "volatility20": _finite(row.get("volatility20")),
        "high20": _finite(row.get("high20Prev")), "high60": _finite(row.get("high60Prev")), "low20": low20,
        "distanceMA20": _finite(row.get("distanceMA20")), "distanceHigh20": _finite(row.get("distanceHigh20")),
        "distanceLow20": _finite(row.get("distanceLow20")), "bollingerPctB": _finite(row.get("bollingerPctB")),
        "breakout20": int(row.get("breakout20", 0) == 1), "breakout60": int(row.get("breakout60", 0) == 1),
        "pullback20": int(row.get("pullback20", 0) == 1),
        "amountAvg5": _finite(row.get("amountAvg5")), "amountAvg20": _finite(row.get("amountAvg20")),
        "volumeRatio5_20": _finite(row.get("volumeRatio5_20")), "amountRatio5_20": _finite(row.get("amountRatio5_20")),
        "volumeTodayRatio20": _finite(row.get("volumeTodayRatio20")), "amountTodayRatio20": _finite(row.get("amountTodayRatio20")),
        "turnoverAvg5": _finite(row.get("turnoverAvg5")), "maxDrawdown60": _finite(row.get("maxDrawdown60")),
        "limitUp": int(row.get("limitUp", 0) == 1), "limitDown": int(row.get("limitDown", 0) == 1),
        "technicalStop": technical_stop, "technicalRiskPct": risk_pct, "target2R": target_2r,
        "shortMetricMethod": "前复权日K；真实均线、RSI、MACD、ATR、20/60日突破与量价结构",
    }
    core_names = [
        "ma20", "ma60", "ma20Slope", "ret20", "rsi14", "macdHist", "atrPct",
        "amountAvg20", "volumeRatio5_20", "high20", "low20",
    ]
    fields["shortDataCoverage"] = round(sum(fields.get(k) is not None for k in core_names) / len(core_names) * 100)
    return fields


def latest_short_metrics(frame: pd.DataFrame) -> dict:
    return _metrics_from_indicator_frame(add_indicators(frame), -1)


def score_short_metrics(metrics: dict, market_score: float | None = None, industry_score: float | None = None) -> dict:
    """Return a transparent short-horizon score and state label."""
    price_score, price_cov = _weighted([
        (80 if metrics.get("aboveMA20") == 1 else 25 if metrics.get("aboveMA20") == 0 else None, 20),
        (85 if metrics.get("aboveMA60") == 1 else 25 if metrics.get("aboveMA60") == 0 else None, 20),
        (85 if metrics.get("ma20AboveMA60") == 1 else 30 if metrics.get("ma20AboveMA60") == 0 else None, 15),
        (_linear(metrics.get("ma20Slope"), -3, 4), 15),
        (_linear(metrics.get("ma60Slope"), -5, 5), 10),
        (95 if metrics.get("breakout20") else 82 if metrics.get("pullback20") else _linear(metrics.get("distanceHigh20"), -15, 1), 20),
    ])
    volume_score, volume_cov = _weighted([
        (_amount_score(_finite(metrics.get("amountAvg20"))), 45),
        (_linear(metrics.get("volumeRatio5_20"), 0.65, 1.6), 15),
        (_linear(metrics.get("amountRatio5_20"), 0.65, 1.6), 10),
        (_linear(metrics.get("volumeTodayRatio20"), 0.65, 1.8), 10),
        (_linear(metrics.get("amountTodayRatio20"), 0.65, 1.8), 10),
        (_linear(metrics.get("turnoverAvg5"), 0.2, 5), 15),
    ])
    momentum_score, momentum_cov = _weighted([
        (_rsi_score(metrics.get("rsi14")), 30),
        (85 if _finite(metrics.get("macdHist")) is not None and float(metrics.get("macdHist")) > 0 else 35 if _finite(metrics.get("macdHist")) is not None else None, 25),
        (_linear(metrics.get("ret20"), -12, 15), 20),
        (_linear(metrics.get("relativeStrength20"), -12, 12), 15),
        (_inverse(metrics.get("atrPct"), 1.2, 7), 10),
    ])
    environment_score, env_cov = _weighted([(market_score, 50), (industry_score, 50)])

    risk = 100.0
    blocks: list[str] = []
    atr_pct = _finite(metrics.get("atrPct"))
    rsi = _finite(metrics.get("rsi14"))
    distance = _finite(metrics.get("distanceMA20"))
    ret5 = _finite(metrics.get("ret5"))
    amount20 = _finite(metrics.get("amountAvg20"))
    if metrics.get("limitUp"):
        risk -= 28; blocks.append("当日接近涨停，次日追高风险较高")
    if rsi is not None and rsi > 78:
        risk -= 24; blocks.append("RSI进入过热区")
    if distance is not None and distance > 12:
        risk -= 22; blocks.append("股价偏离20日均线过大")
    if ret5 is not None and ret5 > 18:
        risk -= 18; blocks.append("5日涨幅过快")
    if atr_pct is not None and atr_pct > 6:
        risk -= 20; blocks.append("短期波动率偏高")
    if metrics.get("aboveMA20") == 0 and metrics.get("aboveMA60") == 0:
        risk -= 28; blocks.append("同时位于20日与60日均线下方")
    if amount20 is not None and amount20 < 5e7:
        risk -= 25; blocks.append("20日平均成交额偏低")
    risk = _clip(risk)

    overall = _clip(environment_score * 0.20 + price_score * 0.30 + volume_score * 0.25 + momentum_score * 0.15 + risk * 0.10)
    coverage = round((env_cov * 0.20 + price_cov * 0.30 + volume_cov * 0.25 + momentum_cov * 0.15 + 1 * 0.10) * 100)
    confidence = 0.55 + 0.45 * coverage / 100
    trusted = 50 + (overall - 50) * confidence if overall > 50 else overall
    if metrics.get("historyBars", 0) < 120:
        trusted = min(trusted, 54)
    trusted = round(_clip(trusted), 1)

    supports: list[str] = []
    if metrics.get("breakout20"):
        supports.append("突破前20日高点")
    if metrics.get("pullback20"):
        supports.append("上升趋势中回踩20日线附近")
    if metrics.get("aboveMA20") == 1 and metrics.get("aboveMA60") == 1:
        supports.append("股价位于20日和60日均线上方")
    if _finite(metrics.get("macdHist")) is not None and float(metrics.get("macdHist")) > 0:
        supports.append("MACD动能为正")
    if amount20 is not None and amount20 >= 1e8:
        supports.append("短线流动性基本达标")

    decision = "等待信号"
    group = "wait"
    action = "保持观察，等待价格结构、量能或市场环境进一步确认。"
    if metrics.get("historyBars", 0) < 120 or coverage < 65:
        decision, group, action = "短线数据不足", "data", "缺少足够日K或量价字段，不能据此做短线判断。"
    elif amount20 is not None and amount20 < 5e7:
        decision, group, action = "流动性不足", "avoid", "成交额较低，滑点和冲击成本可能明显，不宜作为常规短线标的。"
    elif (rsi is not None and rsi > 78) or (distance is not None and distance > 12) or (ret5 is not None and ret5 > 18) or metrics.get("limitUp"):
        decision, group, action = "过热勿追", "avoid", "已有短线过热或涨停风险，等待回落和换手后重新判断。"
    elif metrics.get("aboveMA20") == 0 and metrics.get("aboveMA60") == 0:
        decision, group, action = "弱势规避", "avoid", "价格结构仍弱，不以基本面低估替代短线止损纪律。"
    else:
        ret60 = _finite(metrics.get("ret60"))
        ma20_slope = _finite(metrics.get("ma20Slope"))
        ma60_slope = _finite(metrics.get("ma60Slope"))
        macd_hist = _finite(metrics.get("macdHist"))
        volume_5_20 = _finite(metrics.get("volumeRatio5_20"))
        volume_today = _finite(metrics.get("volumeTodayRatio20"))
        breakout_volume = max(volume_5_20 or 0, volume_today or 0)
        breakout_ready = (
            bool(metrics.get("breakout20"))
            and breakout_volume >= 1.12
            and metrics.get("aboveMA60") == 1
            and (ret60 is not None and ret60 > 3)
            and (ma20_slope is not None and ma20_slope > 0.35)
            and (rsi is not None and 50 <= rsi <= 75)
            and trusted >= 66
        )
        pullback_ready = (
            bool(metrics.get("pullback20"))
            and (volume_5_20 is not None and volume_5_20 <= 1.02)
            and (ret60 is not None and ret60 > 5)
            and (ma60_slope is not None and ma60_slope > 0)
            and (rsi is not None and 44 <= rsi <= 66)
            and (macd_hist is None or macd_hist > -0.03)
            and trusted >= 61
        )
        if breakout_ready:
            decision, group, action = "突破确认", "signal", "形成放量突破观察信号；次日若高开过大或跌回平台，则不追入。"
        elif pullback_ready:
            decision, group, action = "回踩关注", "signal", "中期趋势向上且回调量能收缩；跌破保护位或MA60转弱则信号失效。"
        elif metrics.get("aboveMA20") == 1 and metrics.get("aboveMA60") == 1 and metrics.get("ma20AboveMA60") == 1 and trusted >= 58:
            decision, group, action = "趋势跟踪", "wait", "处于多头结构但尚无触发点，等待回踩或放量突破，不把持续上涨本身当作新买点。"

    return {
        "shortScore": trusted,
        "shortRawScore": round(overall, 1),
        "shortCoverage": coverage,
        "shortDecision": decision,
        "shortGroup": group,
        "shortAction": action,
        "shortSupports": supports[:4],
        "shortBlocks": list(dict.fromkeys(blocks))[:5],
        "shortDimensions": {
            "environment": round(environment_score, 1), "price": round(price_score, 1),
            "volume": round(volume_score, 1), "momentum": round(momentum_score, 1), "risk": round(risk, 1),
        },
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
    holding_days: int = 10
    fee_bps: float = 18.0
    stop_atr: float = 1.8
    minimum_amount: float = 5e7


def backtest_frame(frame: pd.DataFrame, config: BacktestConfig | None = None, market_frame: pd.DataFrame | None = None) -> dict:
    """Walk-forward backtest for the exact same price/volume signal states.

    Entry uses the next trading day's open. Signals never use future rows. A
    stock cannot open a new trade while a prior trade is active.
    """
    config = config or BacktestConfig()
    df = add_indicators(frame)
    if len(df) < 160:
        return {"trades": 0, "error": "日K少于160根，无法有效回测"}
    market_by_date = market_environment_series(market_frame) if market_frame is not None and not market_frame.empty else {}

    trades = []
    i = 120
    while i < len(df) - config.holding_days - 1:
        row = df.iloc[i]
        metrics = _metrics_from_indicator_frame(df, i)
        date = pd.Timestamp(row["date"])
        market_score = market_by_date.get(date, 50.0)
        result = score_short_metrics(metrics, market_score, 50.0)
        if result["shortDecision"] not in {"突破确认", "回踩关注"}:
            i += 1
            continue
        if market_score < 42:
            i += 1
            continue
        if _finite(row.get("amountAvg20")) is not None and float(row.get("amountAvg20")) < config.minimum_amount:
            i += 1
            continue
        entry_index = i + 1
        entry = float(df.iloc[entry_index]["open"])
        signal_close = float(row["close"])
        gap_pct = (entry / signal_close - 1) * 100 if signal_close else 0
        if gap_pct > 3.5 or gap_pct < -6:
            i += 1
            continue
        atr = _finite(row.get("atr14"))
        ma20 = _finite(row.get("ma20"))
        stop = entry - config.stop_atr * atr if atr else None
        if ma20:
            stop = max(stop or 0, ma20 - (atr or 0))
        if stop is not None and stop >= entry:
            stop = entry * 0.97
        if stop is not None:
            risk_pct = (entry - stop) / entry * 100
            if risk_pct < 1.5:
                stop = entry * 0.985
                risk_pct = 1.5
            if risk_pct > 9:
                i += 1
                continue
        exit_index = min(entry_index + config.holding_days, len(df) - 1)
        exit_price = float(df.iloc[exit_index]["close"])
        exit_reason = f"持有{config.holding_days}日"
        for j in range(entry_index, exit_index + 1):
            if stop and float(df.iloc[j]["low"]) <= stop:
                exit_index, exit_price, exit_reason = j, float(stop), "保护位"
                break
        gross_return = exit_price / entry - 1
        net_return = gross_return - config.fee_bps / 10000.0
        trades.append({
            "signalDate": date.strftime("%Y-%m-%d"),
            "entryDate": pd.Timestamp(df.iloc[entry_index]["date"]).strftime("%Y-%m-%d"),
            "exitDate": pd.Timestamp(df.iloc[exit_index]["date"]).strftime("%Y-%m-%d"),
            "signal": result["shortDecision"], "score": result["shortScore"],
            "entry": round(entry, 4), "exit": round(exit_price, 4), "returnPct": round(net_return * 100, 3),
            "gapPct": round(gap_pct, 3), "exitReason": exit_reason,
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
        "details": trades,
    }
