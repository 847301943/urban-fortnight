from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from short_strategy import BacktestConfig, backtest_frame, latest_short_metrics, score_short_metrics  # noqa: E402


def make_history(kind: str, seed: int = 7, bars: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=bars)
    if kind == "uptrend":
        drift = np.linspace(0, 0.55, bars)
        cycle = np.sin(np.arange(bars) / 17) * 0.025
        noise = rng.normal(0, 0.006, bars).cumsum() * 0.12
        close = 20 * np.exp(drift + cycle + noise)
    elif kind == "downtrend":
        drift = np.linspace(0, -0.5, bars)
        noise = rng.normal(0, 0.008, bars).cumsum() * 0.10
        close = 25 * np.exp(drift + noise)
    elif kind == "breakout":
        close = 18 + np.sin(np.arange(bars) / 9) * 0.35 + rng.normal(0, 0.08, bars)
        close[-45:] += np.linspace(0, 7, 45)
    else:
        close = 20 + np.sin(np.arange(bars) / 10) * 0.6 + rng.normal(0, 0.12, bars)
    close = np.maximum(close, 2)
    open_ = close * (1 + rng.normal(0, 0.003, bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.015, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.015, bars))
    volume = rng.lognormal(16.0, 0.25, bars)
    if kind == "breakout":
        volume[-45:] *= np.linspace(1.0, 2.0, 45)
    amount = volume * close
    turnover = np.clip(rng.normal(2.2, 0.55, bars), 0.2, 7)
    return pd.DataFrame({"日期": dates, "开盘": open_, "最高": high, "最低": low, "收盘": close, "成交量": volume, "成交额": amount, "换手率": turnover})


def main() -> None:
    result = {}
    for kind in ("uptrend", "downtrend", "breakout", "range"):
        frame = make_history(kind)
        metrics = latest_short_metrics(frame)
        score = score_short_metrics(metrics, 65 if kind != "downtrend" else 35, 60 if kind != "downtrend" else 35)
        bt = backtest_frame(frame, BacktestConfig(holding_days=1, fee_bps=20), market_frame=make_history("uptrend", seed=99))
        result[kind] = {"decision": score["shortDecision"], "score": score["shortScore"], "coverage": score["shortCoverage"], "trades": bt.get("trades", 0), "winRate": bt.get("winRate"), "averageReturnPct": bt.get("averageReturnPct")}
    assert result["downtrend"]["decision"] in {"弱势规避", "等待隔日形态", "隔日数据不足", "等待隔日触发"}
    assert result["uptrend"]["coverage"] >= 80
    assert result["breakout"]["score"] >= result["downtrend"]["score"]
    out = ROOT / "data" / "synthetic_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
