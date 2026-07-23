"""Backtest the short-horizon module and audit generated data coverage.

The report is an engineering validation, not a claim of future profitability.
It uses the current surviving shortlist, adjusted daily bars, next-day-open
entries, fixed costs, and no event/announcement data. Results therefore retain
survivorship, selection, suspension and execution biases and are labelled as
such in the JSON shown by the web page.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from short_strategy import BacktestConfig, backtest_frame, normalize_history_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STOCKS_FILE = DATA / "stocks.json"
CACHE_FILE = DATA / ".short_history_cache.pkl"
BACKTEST_FILE = DATA / "backtest.json"
AUDIT_FILE = DATA / "data_audit.json"
BACKTEST_LIMIT = int(os.getenv("BACKTEST_LIMIT", "500"))
BACKTEST_HOLDING_DAYS = int(os.getenv("BACKTEST_HOLDING_DAYS", "1"))
BACKTEST_FEE_BPS = float(os.getenv("BACKTEST_FEE_BPS", "20"))


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def percentage(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total else 0.0


def summary_from_trades(trades: list[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "winRate": None,
            "averageReturnPct": None,
            "medianReturnPct": None,
            "profitFactor": None,
            "averageWinPct": None,
            "averageLossPct": None,
            "maxTradeLossPct": None,
            "maxTradeGainPct": None,
        }
    returns = np.array([float(x["returnPct"]) for x in trades], dtype=float)
    wins, losses = returns[returns > 0], returns[returns < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.size and abs(losses.sum()) > 0 else None
    return {
        "trades": int(len(trades)),
        "winRate": round(float((returns > 0).mean() * 100), 2),
        "averageReturnPct": round(float(returns.mean()), 3),
        "medianReturnPct": round(float(np.median(returns)), 3),
        "profitFactor": round(float(profit_factor), 3) if profit_factor is not None else None,
        "averageWinPct": round(float(wins.mean()), 3) if wins.size else None,
        "averageLossPct": round(float(losses.mean()), 3) if losses.size else None,
        "maxTradeLossPct": round(float(returns.min()), 3),
        "maxTradeGainPct": round(float(returns.max()), 3),
    }


def group_summary(trades: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for trade in trades:
        groups[str(trade.get(field) or "未分类")].append(trade)
    return {key: summary_from_trades(rows) for key, rows in sorted(groups.items())}


def validation_split(trades: list[dict]) -> dict:
    if len(trades) < 20:
        return {"splitDate": None, "earlier": summary_from_trades(trades), "laterValidation": summary_from_trades([])}
    ordered = sorted(trades, key=lambda x: x["signalDate"])
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * 0.70)))
    split_date = ordered[split_index]["signalDate"]
    earlier = [x for x in ordered if x["signalDate"] < split_date]
    later = [x for x in ordered if x["signalDate"] >= split_date]
    return {"splitDate": split_date, "earlier": summary_from_trades(earlier), "laterValidation": summary_from_trades(later)}


def benchmark_summary(frame: pd.DataFrame) -> dict:
    frame = normalize_history_frame(frame)
    if frame.empty:
        return {"available": False}
    start, end = float(frame.iloc[0]["close"]), float(frame.iloc[-1]["close"])
    return {
        "available": True,
        "startDate": frame.iloc[0]["date"].strftime("%Y-%m-%d"),
        "endDate": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "returnPct": round((end / start - 1) * 100, 2) if start else None,
        "bars": int(len(frame)),
    }


def make_backtest(payload: dict, cache: dict) -> dict:
    stock_meta = cache.get("stockMeta", {})
    histories: dict[str, pd.DataFrame] = cache.get("stocks", {})
    market = cache.get("market", pd.DataFrame())
    candidates = []
    for stock in payload.get("stocks", []):
        code = str(stock.get("code") or "")
        if code not in histories:
            continue
        amount = finite(stock.get("amountAvg20")) or finite(stock.get("amount")) or 0
        short_score = finite(stock.get("shortScore")) or 0
        candidates.append((amount, short_score, code, stock))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    candidates = candidates[:max(1, BACKTEST_LIMIT)]

    config = BacktestConfig(holding_days=BACKTEST_HOLDING_DAYS, fee_bps=BACKTEST_FEE_BPS)
    all_trades: list[dict] = []
    tested = 0
    errors: list[dict] = []
    period_start, period_end = None, None
    for _, _, code, stock in candidates:
        try:
            result = backtest_frame(histories[code], config=config, market_frame=market)
            tested += 1
            for trade in result.get("details", []):
                trade.update({
                    "code": code,
                    "name": stock.get("name") or stock_meta.get(code, {}).get("name") or "",
                    "industry": stock.get("industry") or stock_meta.get(code, {}).get("industry") or "未分类",
                })
                all_trades.append(trade)
            normalized = normalize_history_frame(histories[code])
            if not normalized.empty:
                first = normalized.iloc[0]["date"].strftime("%Y-%m-%d")
                last = normalized.iloc[-1]["date"].strftime("%Y-%m-%d")
                period_start = first if period_start is None or first < period_start else period_start
                period_end = last if period_end is None or last > period_end else period_end
        except Exception as exc:
            errors.append({"code": code, "error": str(exc)[:300]})

    all_trades.sort(key=lambda x: (x["signalDate"], x["code"]))
    overall = summary_from_trades(all_trades)
    report = {
        "schemaVersion": 3,
        "generatedAt": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "完成" if tested else "无可用历史数据",
        "strategy": "收盘后隔日计划回测：隔日突破／隔日回踩／收盘强势",
        "execution": {
            "entry": "信号日收盘后生成计划；下一交易日开盘，跳空范围-4%至+2.5%才允许模拟介入",
            "exit": f"A股T+1约束：买入当日不允许卖出；默认在再下一交易日收盘或保护位/1.5R目标退出（holding_days={BACKTEST_HOLDING_DAYS}）",
            "feeBpsRoundTrip": BACKTEST_FEE_BPS,
            "positionRule": "同一股票上一笔交易退出后才允许新开仓",
            "lookAhead": "指标只使用信号日及以前数据；不使用信号日收盘价虚拟成交",
        },
        "sample": {
            "eligibleHistories": len(histories),
            "testedStocks": tested,
            "limit": BACKTEST_LIMIT,
            "periodStart": period_start,
            "periodEnd": period_end,
            "marketSource": cache.get("marketSource") or "未知",
            "failedStocks": len(errors),
        },
        "overall": overall,
        "bySignal": group_summary(all_trades, "signal"),
        "byIndustry": group_summary(all_trades, "industry"),
        "validation": validation_split(all_trades),
        "benchmark": benchmark_summary(market),
        "limitations": [
            "使用当前仍上市且进入本次高流动性候选池的股票，存在幸存者偏差和当前样本选择偏差。",
            "没有纳入历史退市股、ST状态变化、停牌、涨跌停无法成交、除权异常、融资约束和真实滑点。",
            "没有纳入公告、减持、解禁、业绩披露窗口、龙虎榜和盘中盘口，因此不能据此直接下单。",
            "行业环境在历史回测中采用中性值，当前页面的行业广度分只用于最新截面信号。",
            "这是收盘后生成、次日介入的现实模型，不是买在信号日尾盘并次日卖出的理想化回测。",
            "回测用于检查代码、信号方向和数据覆盖，不代表未来收益，也不是投资建议。",
        ],
        "errors": errors[:20],
        "recentTrades": all_trades[-50:],
    }
    return report


def audit_data(payload: dict, backtest: dict) -> dict:
    stocks = payload.get("stocks", [])
    total = len(stocks)
    fields = {
        "ROE": "roe", "ROIC": "roic", "经营现金流/净利润": "ocfToProfit", "自由现金流收益率": "fcfYield",
        "PE": "pe", "PB": "pb", "股息率": "dividendYield", "3年营收增速": "revenueCagr3", "3年利润增速": "profitCagr3",
        "最新利润增速": "profitGrowthQ", "精确日K": "shortScore", "EPS预期调整": "epsRevision",
        "应收异常": "receivableRisk", "存货异常": "inventoryRisk", "股本稀释": "dilution",
    }
    coverage = {}
    for label, key in fields.items():
        count = sum(finite(x.get(key)) is not None for x in stocks)
        coverage[label] = {"count": count, "pct": percentage(count, total)}
    exact_cash = sum(x.get("cashFlowDataLevel") == "精确" for x in stocks)
    financial = [x for x in stocks if str(x.get("financialType") or "general") != "general"]
    financial_special = sum(x.get("financialMetricLevel") in {"专项", "部分专项"} for x in financial)
    short = [x for x in stocks if finite(x.get("shortScore")) is not None]
    short_pool_eligible = [x for x in stocks if bool(x.get("shortPoolEligible"))]
    short_pool_selected = [x for x in stocks if bool(x.get("shortPoolSelected"))]
    short_failed = [x for x in stocks if x.get("shortDataLevel") == "日K获取失败"]
    exact_short = sum(x.get("shortDataLevel") == "精确日K" for x in stocks)
    static_pe = sum("静态PE" in str(x.get("peMethod") or "") for x in stocks)
    proxy_fcf = sum("代理" in str(x.get("fcfMethod") or "") for x in stocks)
    event_covered = sum((finite(x.get("eventRiskCoverage")) or 0) > 0 for x in stocks)

    issues: list[dict] = []
    def issue(level: str, title: str, detail: str, action: str):
        issues.append({"level": level, "title": title, "detail": detail, "action": action})

    selected_count = len(short_pool_selected)
    selected_coverage = percentage(len(short), selected_count)
    if selected_count == 0:
        issue("高", "隔日池为空", "没有股票进入当日隔日池。", "检查行情快照、成交额单位以及SHORT_MIN_AMOUNT设置。")
    elif selected_coverage < 85:
        issue("高", "隔日池日K刷新率偏低", f"当日选中{selected_count}只，成功形成指标{len(short)}只，刷新率{selected_coverage}%。", "检查东方财富/腾讯日K接口限流；本日失败股票不得沿用旧信号。")
    elif len(short_pool_eligible) > selected_count:
        issue("提示", "隔日池达到容量上限", f"基础合格{len(short_pool_eligible)}只，本次选择{selected_count}只。", "提高SHORT_HISTORY_LIMIT或提高最低成交额；不要用轮换旧信号补足。")
    if event_covered == 0:
        issue("高", "事件风险尚未覆盖", "公告、减持、解禁、停复牌和财报披露窗口均未自动纳入。", "短线信号旁固定显示未覆盖警告；下单前人工检查公告。")
    if coverage["EPS预期调整"]["pct"] < 10:
        issue("中", "盈利预测调整缺失", "该指标不能有效参与排名。", "保持零权重或待核验，不得用默认中性分伪装为已覆盖。")
    if coverage["应收异常"]["pct"] < 30 or coverage["存货异常"]["pct"] < 30:
        issue("中", "应收和存货风险覆盖不足", "财务质量的预警字段大范围缺失。", "后续从资产负债表与利润表计算同比差额；当前缺失不加分。")
    if proxy_fcf > exact_cash:
        issue("中", "多数自由现金流仍是代理口径", f"精确现金流{exact_cash}只，代理口径{proxy_fcf}只。", "继续优先补全高分候选；网页保留口径标签，金融股不使用FCF。")
    if static_pe > total * 0.3:
        issue("中", "较多PE为静态估算", f"静态PE约{static_pe}只。", "显示PE口径与财报日期，避免把静态PE误标为TTM。")
    if financial and percentage(financial_special, len(financial)) < 60:
        issue("中", "金融专项指标覆盖不足", f"金融股专项/部分专项覆盖{financial_special}/{len(financial)}只。", "银行、保险、券商优先全部补齐；综合金融可按市值补。")
    if backtest.get("overall", {}).get("trades", 0) < 100:
        issue("提示", "回测交易样本偏少", f"当前交易数{backtest.get('overall', {}).get('trades', 0)}。", "扩大历史跨度或候选数量后再评价稳定性，不能据少量交易调参数。")

    return {
        "schemaVersion": 3,
        "generatedAt": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "stockCount": total,
        "coverage": coverage,
        "specialCoverage": {
            "exactCashFlow": {"count": exact_cash, "pct": percentage(exact_cash, total)},
            "proxyFreeCashFlow": {"count": proxy_fcf, "pct": percentage(proxy_fcf, total)},
            "financialStocks": len(financial),
            "financialSpecial": {"count": financial_special, "pctOfFinancial": percentage(financial_special, len(financial))},
            "shortPoolEligible": {"count": len(short_pool_eligible), "pctOfMarket": percentage(len(short_pool_eligible), total)},
            "shortPoolSelected": {"count": len(short_pool_selected), "pctOfEligible": percentage(len(short_pool_selected), len(short_pool_eligible))},
            "shortHistory": {"count": len(short), "pctOfSelected": percentage(len(short), len(short_pool_selected))},
            "shortHistoryFailed": {"count": len(short_failed), "pctOfSelected": percentage(len(short_failed), len(short_pool_selected))},
            "shortExact250Bars": {"count": exact_short, "pctOfSelected": percentage(exact_short, len(short_pool_selected))},
            "eventRisk": {"count": event_covered, "pct": percentage(event_covered, total)},
            "staticPE": {"count": static_pe, "pct": percentage(static_pe, total)},
        },
        "issues": issues,
        "interpretation": [
            "覆盖率低不代表公司差，只代表当前自动数据不能支持该项结论。",
            "未进入隔日池通常代表流动性、价格或风险门槛不适合本策略，不应统计成数据缺失。",
            "长期价值结论和隔日信号必须分开阅读；两者方向冲突时不自动合并为买入建议。",
            "隔日计划必须检查次日开盘跳空、成交可行性和事件风险。",
        ],
    }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if not STOCKS_FILE.exists():
        raise SystemExit("stocks.json does not exist")
    payload = json.loads(STOCKS_FILE.read_text(encoding="utf-8"))
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("rb") as handle:
                cache = pickle.load(handle)
        except Exception as exc:
            cache = {"stocks": {}, "market": pd.DataFrame(), "loadError": str(exc)}
    else:
        cache = {"stocks": {}, "market": pd.DataFrame(), "loadError": "short history cache missing"}
    backtest = make_backtest(payload, cache)
    audit = audit_data(payload, backtest)
    BACKTEST_FILE.write_text(json.dumps(backtest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    AUDIT_FILE.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    try:
        CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"backtest written: {BACKTEST_FILE}; trades={backtest.get('overall', {}).get('trades', 0)}")
    print(f"audit written: {AUDIT_FILE}; issues={len(audit.get('issues', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
