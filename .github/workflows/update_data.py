"""Build the token-free A-share screener dataset.

The updater uses public AKShare interfaces in two stages:
1. Bulk market and financial-report tables cover the whole A-share universe.
2. A limited set of high-quality preliminary candidates is enriched with full
   annual financial statements to calculate a more precise ROIC and free cash
   flow. Failure of optional enrichment never destroys the complete bulk file.

ROIC is a calculated metric rather than an accounting line item. The script
always records the calculation method. FCF is OCF minus cash capital spending
when detailed statements are available; otherwise a conservative bulk proxy
(OCF + net investing cash flow) is used and clearly labelled.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "stocks.json"
MIN_STOCKS = int(os.getenv("MIN_STOCKS", "4000"))
QUARTERS_TO_TRY = int(os.getenv("QUARTERS_TO_TRY", "6"))
ANNUAL_YEARS = int(os.getenv("ANNUAL_YEARS", "4"))
DETAIL_LIMIT = int(os.getenv("DETAIL_LIMIT", "200"))
DETAIL_WORKERS = int(os.getenv("DETAIL_WORKERS", "5"))

FINANCIAL_KEYWORDS = ("银行", "保险", "证券", "多元金融", "信托", "期货")


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def call_with_retry(
    name: str,
    func: Callable[[], pd.DataFrame],
    attempts: int = 6,
    base_wait: float = 4.0,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            log(f"{name}: attempt {attempt}/{attempts}")
            frame = func()
            if frame is None or frame.empty:
                raise RuntimeError("empty dataframe")
            log(f"{name}: received {len(frame)} rows")
            return frame
        except Exception as exc:
            last_error = exc
            wait = min(75, base_wait * (2 ** (attempt - 1))) + random.uniform(0.4, 2.5)
            log(f"{name}: failed: {exc}; retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"{name} failed after {attempts} attempts: {last_error}")


def numeric(value):
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def clip(value, low, high):
    if value is None:
        return None
    return max(low, min(high, value))


def clean_code(value) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    # 新浪实时行情返回 sh600000 / sz000001 / bj430017；东财返回纯数字。
    if len(text) >= 8 and text[:2] in {"sh", "sz", "bj"} and text[2:].isdigit():
        text = text[2:]
    return text.zfill(6) if text.isdigit() else text


def is_a_share(code: str) -> bool:
    return code.startswith(("00", "001", "002", "003", "30", "60", "68", "8", "4", "92"))


def market_symbol(code: str) -> str:
    if code.startswith(("6", "68")):
        return "SH" + code
    if code.startswith(("4", "8", "92")):
        return "BJ" + code
    return "SZ" + code


def quarter_ends(now: datetime, count: int) -> list[str]:
    values: list[datetime] = []
    for year in range(now.year, now.year - 5, -1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            dt = datetime(year, month, day)
            if dt <= now:
                values.append(dt)
    values.sort(reverse=True)
    return [x.strftime("%Y%m%d") for x in values[:count]]


def annual_ends(now: datetime, count: int) -> list[str]:
    latest_year = now.year - 1 if now.month < 5 else now.year - 1
    return [f"{latest_year - i}1231" for i in range(count)]


def first_present(record: dict, names: Iterable[str]):
    for name in names:
        if name in record:
            value = record.get(name)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return None


def first_number(record: dict, names: Iterable[str]):
    return numeric(first_present(record, names))


def latest_by_code(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [x.copy() for x in frames if x is not None and not x.empty and "股票代码" in x.columns]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable, ignore_index=True)
    merged["股票代码"] = merged["股票代码"].map(clean_code)
    if "__report_date" in merged.columns:
        merged = merged.sort_values("__report_date", ascending=False)
    return merged.drop_duplicates("股票代码", keep="first")


def dict_by_code(frame: pd.DataFrame) -> dict[str, dict]:
    if frame is None or frame.empty or "股票代码" not in frame.columns:
        return {}
    return {
        clean_code(record.get("股票代码")): record
        for record in frame.to_dict(orient="records")
    }


def fetch_date_tables(
    dates: list[str], function_name: str, attempts: int = 3
) -> dict[str, pd.DataFrame]:
    function = getattr(ak, function_name)
    result: dict[str, pd.DataFrame] = {}
    for report_date in dates:
        try:
            frame = call_with_retry(
                f"{function_name}({report_date})",
                lambda d=report_date: function(date=d),
                attempts=attempts,
            )
            if "股票代码" in frame.columns:
                frame["股票代码"] = frame["股票代码"].map(clean_code)
            frame["__report_date"] = report_date
            result[report_date] = frame
        except Exception as exc:
            log(f"optional table skipped: {function_name} {report_date}: {exc}")
    return result


def merge_latest(table_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return latest_by_code(table_map.values())


def annual_history_by_code(
    table_map: dict[str, pd.DataFrame], value_names: list[str]
) -> dict[str, list[tuple[str, float]]]:
    history: dict[str, list[tuple[str, float]]] = {}
    for date, frame in sorted(table_map.items()):
        if frame is None or frame.empty:
            continue
        for record in frame.to_dict(orient="records"):
            code = clean_code(record.get("股票代码"))
            value = first_number(record, value_names)
            if code and value is not None:
                history.setdefault(code, []).append((date, value))
    for code in history:
        history[code].sort(key=lambda x: x[0])
    return history


def cagr_from_history(items: list[tuple[str, float]], years: int = 3):
    if len(items) < years + 1:
        return None
    selected = items[-(years + 1):]
    first, last = selected[0][1], selected[-1][1]
    if first <= 0 or last <= 0:
        return None
    return ((last / first) ** (1 / years) - 1) * 100


def trend_from_values(values: list[float]):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    denominator = sum((i - x_mean) ** 2 for i in range(len(values)))
    if denominator == 0:
        return 0.0
    slope = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values)) / denominator
    scale = max(sum(abs(v) for v in values) / len(values), 1.0)
    return clip(slope / scale * 100, -150, 150)


def round_list(values: list[float | None], divisor: float = 1e8) -> list[float | None]:
    return [None if v is None else round(v / divisor, 2) for v in values]


def effective_tax_rate(total_profit, net_profit, income_tax=None):
    if income_tax is not None and total_profit not in (None, 0) and total_profit > 0:
        return clip(income_tax / total_profit, 0.0, 0.35)
    if total_profit not in (None, 0) and total_profit > 0 and net_profit is not None:
        return clip((total_profit - net_profit) / total_profit, 0.0, 0.35)
    return 0.25


def is_financial(industry: str) -> bool:
    return any(keyword in str(industry or "") for keyword in FINANCIAL_KEYWORDS)


def bulk_roic(report: dict, balance: dict, income: dict, industry: str):
    if is_financial(industry):
        return None, "金融行业不适用"
    operating_profit = first_number(income, ["营业利润", "OPERATE_PROFIT"])
    total_profit = first_number(income, ["利润总额", "TOTAL_PROFIT"])
    net_profit = first_number(report, ["净利润-净利润", "净利润"])
    total_assets = first_number(balance, ["资产-总资产", "总资产"])
    cash = first_number(balance, ["资产-货币资金", "货币资金"]) or 0.0
    accounts_payable = first_number(balance, ["负债-应付账款", "应付账款"]) or 0.0
    advances = first_number(balance, ["负债-预收账款", "预收账款", "合同负债"]) or 0.0
    if operating_profit is None or total_assets is None:
        return None, "数据不足"
    invested = total_assets - cash - accounts_payable - advances
    if invested <= 0:
        return None, "投入资本异常"
    tax = effective_tax_rate(total_profit, net_profit)
    return clip(operating_profit * (1 - tax) / invested * 100, -100, 100), "估算：税后营业利润/经营投入资本"


def record_by_latest_date(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {}
    work = frame.copy()
    date_col = next((c for c in ["REPORT_DATE", "报告期", "报告日期"] if c in work.columns), None)
    if date_col:
        work["__sort_date"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values("__sort_date", ascending=False)
    return work.iloc[0].to_dict()


def last_annual_records(frame: pd.DataFrame, count: int = 3) -> list[dict]:
    if frame is None or frame.empty:
        return []
    work = frame.copy()
    date_col = next((c for c in ["REPORT_DATE", "报告期", "报告日期"] if c in work.columns), None)
    if date_col:
        work["__sort_date"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values("__sort_date", ascending=False)
    return work.head(count).iloc[::-1].to_dict(orient="records")


def detailed_metric_for_stock(stock: dict) -> tuple[str, dict]:
    code = stock["code"]
    symbol = market_symbol(code)
    result: dict = {}
    if is_financial(stock.get("industry", "")):
        return code, {
            "roic": None,
            "roicMethod": "金融行业不适用",
            "fcfYield": None,
            "fcfMethod": "金融行业不适用",
            "cashFlowDataLevel": "不适用",
        }
    try:
        # Each function returns multiple annual periods. Optional enrichment is
        # deliberately limited to preliminary quality candidates.
        balance_df = call_with_retry(
            f"detail balance {symbol}",
            lambda: ak.stock_balance_sheet_by_yearly_em(symbol=symbol),
            attempts=2,
            base_wait=2.0,
        )
        income_df = call_with_retry(
            f"detail income {symbol}",
            lambda: ak.stock_profit_sheet_by_yearly_em(symbol=symbol),
            attempts=2,
            base_wait=2.0,
        )
        cash_df = call_with_retry(
            f"detail cashflow {symbol}",
            lambda: ak.stock_cash_flow_sheet_by_yearly_em(symbol=symbol),
            attempts=2,
            base_wait=2.0,
        )

        balance = record_by_latest_date(balance_df)
        income = record_by_latest_date(income_df)
        cash = record_by_latest_date(cash_df)

        operating_profit = first_number(income, ["OPERATE_PROFIT", "营业利润"])
        total_profit = first_number(income, ["TOTAL_PROFIT", "利润总额"])
        net_profit = first_number(income, ["PARENT_NETPROFIT", "NETPROFIT", "净利润"])
        income_tax = first_number(income, ["INCOME_TAX", "INCOME_TAX_EXPENSE", "所得税费用"])
        equity = first_number(
            balance,
            ["TOTAL_EQUITY", "TOTAL_PARENT_EQUITY", "TOTAL_EQUITY_ATTR_P", "股东权益合计"],
        )
        cash_balance = first_number(balance, ["MONETARYFUNDS", "CASH_EQUIV", "货币资金"]) or 0.0
        debt_fields = [
            "SHORT_LOAN", "NONCURRENT_LIAB_1YEAR", "LONG_LOAN", "BOND_PAYABLE",
            "LEASE_LIAB", "SHORT_BOND_PAYABLE", "短期借款", "一年内到期的非流动负债",
            "长期借款", "应付债券", "租赁负债",
        ]
        interest_debt = sum(first_number(balance, [field]) or 0.0 for field in debt_fields)
        invested = None
        if equity is not None:
            invested = equity + interest_debt - cash_balance
        if invested is None or invested <= 0:
            total_assets = first_number(balance, ["TOTAL_ASSETS", "资产总计", "总资产"])
            current_free = sum(
                first_number(balance, [field]) or 0.0
                for field in ["ACCOUNTS_PAYABLE", "ADVANCE_RECEIVABLES", "CONTRACT_LIAB", "应付账款", "预收款项", "合同负债"]
            )
            if total_assets is not None:
                invested = total_assets - cash_balance - current_free
        if operating_profit is not None and invested not in (None, 0) and invested > 0:
            tax = effective_tax_rate(total_profit, net_profit, income_tax)
            result["roic"] = clip(operating_profit * (1 - tax) / invested * 100, -100, 100)
            result["roicMethod"] = "精算：税后营业利润/(权益+有息负债-现金)"

        ocf = first_number(cash, ["NETCASH_OPERATE", "NET_CASH_FLOWS_OPER_ACT", "经营活动产生的现金流量净额"])
        capex = first_number(
            cash,
            [
                "CONSTRUCT_LONG_ASSET", "CASH_PAY_ACQUIRE_CONST_FIOLTA",
                "购建固定资产、无形资产和其他长期资产支付的现金",
            ],
        )
        if ocf is not None and capex is not None:
            capex = abs(capex)
            fcf = ocf - capex
            result["freeCashFlow"] = fcf
            market_cap_yuan = numeric(stock.get("marketCap"))
            market_cap_yuan = market_cap_yuan * 1e8 if market_cap_yuan is not None else None
            result["fcfYield"] = fcf / market_cap_yuan * 100 if market_cap_yuan not in (None, 0) else None
            result["fcfMethod"] = "精算：经营现金流-资本开支"

        annual_cash = last_annual_records(cash_df, 3)
        ocf_values = [
            first_number(r, ["NETCASH_OPERATE", "NET_CASH_FLOWS_OPER_ACT", "经营活动产生的现金流量净额"])
            for r in annual_cash
        ]
        capex_values = [
            first_number(r, ["CONSTRUCT_LONG_ASSET", "CASH_PAY_ACQUIRE_CONST_FIOLTA", "购建固定资产、无形资产和其他长期资产支付的现金"])
            for r in annual_cash
        ]
        fcf_values = [
            None if o is None or c is None else o - abs(c)
            for o, c in zip(ocf_values, capex_values)
        ]
        if any(v is not None for v in ocf_values):
            result["ocfYears3"] = round_list(ocf_values)
            result["ocfPositiveYears3"] = sum(v is not None and v > 0 for v in ocf_values)
            result["ocfTrend3"] = trend_from_values([v for v in ocf_values if v is not None])
        if any(v is not None for v in fcf_values):
            result["fcfYears3"] = round_list(fcf_values)
            result["fcfPositiveYears3"] = sum(v is not None and v > 0 for v in fcf_values)
            result["fcfTrend3"] = trend_from_values([v for v in fcf_values if v is not None])
        has_exact_roic = numeric(result.get("roic")) is not None
        has_exact_fcf = numeric(result.get("freeCashFlow")) is not None
        result["cashFlowDataLevel"] = "精确" if has_exact_roic and has_exact_fcf else ("部分精确" if has_exact_roic or has_exact_fcf else stock.get("cashFlowDataLevel", "估算"))
    except Exception as exc:
        result["detailError"] = str(exc)[:180]
        result["cashFlowDataLevel"] = stock.get("cashFlowDataLevel", "估算")
    return code, result


def preliminary_detail_candidates(stocks: list[dict], limit: int) -> list[dict]:
    eligible = []
    for stock in stocks:
        if is_financial(stock.get("industry", "")):
            continue
        roe = numeric(stock.get("roe"))
        debt = numeric(stock.get("debtRatio"))
        ocf = numeric(stock.get("ocfToProfit"))
        pe = numeric(stock.get("pe"))
        if roe is None or roe < 8 or (debt is not None and debt > 80):
            continue
        score = (
            min(max(roe, 0), 35) * 2.2
            + min(max(numeric(stock.get("grossMargin")) or 0, 0), 80) * 0.35
            + (12 if ocf is not None and ocf >= 1 else 5 if ocf is not None and ocf > 0 else 0)
            + (10 if pe is not None and 0 < pe <= 30 else 4 if pe is not None and pe > 0 else 0)
            + min(math.log10(max(numeric(stock.get("marketCap")) or 1, 1)) * 5, 20)
        )
        eligible.append((score, stock))
    eligible.sort(key=lambda item: item[0], reverse=True)
    return [stock for _, stock in eligible[:limit]]


def fetch_spot_snapshot() -> tuple[pd.DataFrame, str]:
    """Fetch the A-share universe without depending on one quote provider.

    GitHub-hosted runners are sometimes rejected by Eastmoney's push2 quote
    service.  Try it only briefly, then switch to AKShare's Sina snapshot,
    which also returns all Shanghai, Shenzhen and Beijing A shares in one call.
    Missing valuation/trend fields are calculated later from financial reports
    where possible, or left blank rather than fabricated.
    """
    try:
        frame = call_with_retry(
            "stock_zh_a_spot_em", ak.stock_zh_a_spot_em, attempts=2, base_wait=3.0
        )
        return frame, "东方财富实时行情"
    except Exception as exc:
        log(f"Eastmoney quote snapshot unavailable, switching to Sina: {exc}")

    frame = call_with_retry(
        "stock_zh_a_spot (Sina fallback)", ak.stock_zh_a_spot, attempts=4, base_wait=5.0
    )
    return frame, "新浪实时行情（东财不可用时自动切换）"


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def build() -> dict:
    now = datetime.now()
    spot, spot_source = fetch_spot_snapshot()
    if "代码" not in spot.columns:
        raise RuntimeError(f"quote snapshot has no code column; columns={list(spot.columns)}")
    spot["代码"] = spot["代码"].map(clean_code)
    spot = spot[spot["代码"].map(is_a_share)].drop_duplicates("代码").copy()
    if len(spot) < MIN_STOCKS:
        raise RuntimeError(f"snapshot contains only {len(spot)} stocks, below safety threshold {MIN_STOCKS}")

    quarter_dates = quarter_ends(now, QUARTERS_TO_TRY)
    annual_dates = annual_ends(now, ANNUAL_YEARS)
    log(f"quarter dates: {quarter_dates}")
    log(f"annual dates: {annual_dates}")

    q_reports = fetch_date_tables(quarter_dates, "stock_yjbb_em")
    q_balances = fetch_date_tables(quarter_dates, "stock_zcfz_em")
    try:
        q_balances_bj = fetch_date_tables(quarter_dates, "stock_zcfz_bj_em")
        for date, frame in q_balances_bj.items():
            q_balances[date] = pd.concat([q_balances.get(date, pd.DataFrame()), frame], ignore_index=True)
    except Exception as exc:
        log(f"Beijing balance tables skipped: {exc}")
    q_cashflows = fetch_date_tables(quarter_dates, "stock_xjll_em")
    q_incomes = fetch_date_tables(quarter_dates, "stock_lrb_em")

    a_reports = fetch_date_tables(annual_dates, "stock_yjbb_em")
    a_balances = fetch_date_tables(annual_dates, "stock_zcfz_em")
    try:
        a_balances_bj = fetch_date_tables(annual_dates, "stock_zcfz_bj_em")
        for date, frame in a_balances_bj.items():
            a_balances[date] = pd.concat([a_balances.get(date, pd.DataFrame()), frame], ignore_index=True)
    except Exception as exc:
        log(f"Beijing annual balance tables skipped: {exc}")
    a_cashflows = fetch_date_tables(annual_dates, "stock_xjll_em")
    a_incomes = fetch_date_tables(annual_dates, "stock_lrb_em")

    report_map = dict_by_code(merge_latest(q_reports))
    balance_map = dict_by_code(merge_latest(q_balances))
    cash_map = dict_by_code(merge_latest(q_cashflows))
    income_map = dict_by_code(merge_latest(q_incomes))

    revenue_history = annual_history_by_code(a_reports, ["营业总收入-营业总收入", "营业总收入"])
    profit_history = annual_history_by_code(a_reports, ["净利润-净利润", "净利润"])
    ocf_history = annual_history_by_code(a_cashflows, ["经营性现金流-现金流量净额", "经营活动产生的现金流量净额"])
    investing_history = annual_history_by_code(a_cashflows, ["投资性现金流-现金流量净额", "投资活动产生的现金流量净额"])

    annual_report_maps = {date: dict_by_code(frame) for date, frame in a_reports.items()}
    annual_cash_maps = {date: dict_by_code(frame) for date, frame in a_cashflows.items()}
    latest_annual_report = dict_by_code(merge_latest(a_reports))

    ret60_values = numeric_series(spot, "60日涨跌幅").dropna()
    med60 = float(ret60_values.median()) if not ret60_values.empty else 0.0
    updated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    trade_date = datetime.now().strftime("%Y-%m-%d")

    stocks: list[dict] = []
    for q in spot.to_dict(orient="records"):
        code = clean_code(q.get("代码"))
        report = report_map.get(code, {})
        annual_report = latest_annual_report.get(code, {})
        balance = balance_map.get(code, {})
        cash = cash_map.get(code, {})
        income = income_map.get(code, {})
        industry = str(report.get("所处行业") or annual_report.get("所处行业") or "未分类")

        revenue = first_number(report, ["营业总收入-营业总收入", "营业总收入", "营业收入-营业收入"])
        profit = first_number(report, ["净利润-净利润", "净利润"])
        ocf = first_number(cash, ["经营性现金流-现金流量净额", "经营活动产生的现金流量净额"])
        investing_cf = first_number(cash, ["投资性现金流-现金流量净额", "投资活动产生的现金流量净额"])
        fcf_proxy = None if ocf is None or investing_cf is None else ocf + investing_cf

        ret60 = numeric(q.get("60日涨跌幅"))
        ytd = numeric(q.get("年初至今涨跌幅"))
        pct = numeric(q.get("涨跌幅"))
        volume_ratio = numeric(q.get("量比"))
        turnover = numeric(q.get("换手率"))
        market_cap_yuan = numeric(q.get("总市值"))
        market_cap = market_cap_yuan / 1e8 if market_cap_yuan is not None else None
        price = numeric(q.get("最新价"))
        # 新浪快照没有 PE/PB。PB 可由最新每股净资产计算；PE 采用最近完整
        # 年度每股收益计算静态 PE，并在数据说明中明确口径。
        bvps = first_number(report, ["每股净资产"])
        annual_eps = first_number(annual_report, ["每股收益"])
        derived_pb = price / bvps if price is not None and bvps not in (None, 0) else None
        derived_pe = price / annual_eps if price is not None and annual_eps not in (None, 0) else None

        volume_confirm = 2.5
        if pct is not None and pct > 0:
            volume_confirm += 0.5
        if volume_ratio is not None and volume_ratio > 1.2:
            volume_confirm += 1.0
        if turnover is not None and turnover > 2:
            volume_confirm += 0.5
        if pct is not None and pct < 0 and volume_ratio is not None and volume_ratio > 1.5:
            volume_confirm -= 1.0
        volume_confirm = clip(volume_confirm, 0.0, 5.0)

        report_date = first_present(report, ["__report_date", "最新公告日期", "公告日期"])
        net_margin = profit / revenue * 100 if revenue not in (None, 0) and profit is not None else None
        ocf_to_profit = ocf / profit if profit not in (None, 0) and ocf is not None else None
        roic, roic_method = bulk_roic(report, balance, income, industry)

        annual_ocf_values = []
        annual_fcf_values = []
        for date in sorted(annual_dates)[-3:]:
            cash_record = annual_cash_maps.get(date, {}).get(code, {})
            yearly_ocf = first_number(cash_record, ["经营性现金流-现金流量净额", "经营活动产生的现金流量净额"])
            yearly_investing = first_number(cash_record, ["投资性现金流-现金流量净额", "投资活动产生的现金流量净额"])
            annual_ocf_values.append(yearly_ocf)
            annual_fcf_values.append(None if yearly_ocf is None or yearly_investing is None else yearly_ocf + yearly_investing)

        financial = is_financial(industry)
        if financial:
            roic = None
            roic_method = "金融行业不适用"
            fcf_proxy = None
            fcf_method = "金融行业不适用"
            cash_level = "不适用"
        else:
            fcf_method = "代理：经营现金流+投资现金流净额"
            cash_level = "估算"

        stocks.append({
            "code": code,
            "name": str(q.get("名称") or report.get("股票简称") or ""),
            "industry": industry,
            "price": price,
            "pctChange": pct,
            "marketCap": market_cap,
            "turnover": turnover,
            "reportDate": str(report_date or ""),
            "dataDate": updated_at,
            "dataSource": f"GitHub Actions / AKShare公开接口 / {spot_source}",
            "autoData": 1,
            "roe": numeric(report.get("净资产收益率")),
            "roic": roic,
            "roicMethod": roic_method,
            "grossMargin": numeric(report.get("销售毛利率")),
            "netMargin": net_margin,
            "debtRatio": numeric(balance.get("资产负债率")),
            "operatingCashFlow": ocf,
            "ocfToProfit": ocf_to_profit,
            "freeCashFlow": fcf_proxy,
            "fcfYield": None if fcf_proxy is None or market_cap_yuan in (None, 0) else fcf_proxy / market_cap_yuan * 100,
            "fcfMethod": fcf_method,
            "cashFlowDataLevel": cash_level,
            "ocfYears3": round_list(annual_ocf_values),
            "fcfYears3": round_list(annual_fcf_values),
            "ocfPositiveYears3": sum(v is not None and v > 0 for v in annual_ocf_values),
            "fcfPositiveYears3": sum(v is not None and v > 0 for v in annual_fcf_values),
            "ocfTrend3": trend_from_values([v for v in annual_ocf_values if v is not None]),
            "fcfTrend3": trend_from_values([v for v in annual_fcf_values if v is not None]),
            "revenueCagr3": cagr_from_history(revenue_history.get(code, []), 3),
            "profitCagr3": cagr_from_history(profit_history.get(code, []), 3),
            "revenueGrowthQ": first_number(report, ["营业总收入-同比增长", "营业收入-同比增长"]),
            "profitGrowthQ": first_number(report, ["净利润-同比增长", "净利润同比"]),
            "epsRevision": None,
            "pe": numeric(q.get("市盈率-动态")) if numeric(q.get("市盈率-动态")) is not None else derived_pe,
            "peMethod": "动态PE" if numeric(q.get("市盈率-动态")) is not None else ("静态PE：股价/最近完整年度EPS" if derived_pe is not None else "暂无"),
            "pb": numeric(q.get("市净率")) if numeric(q.get("市净率")) is not None else derived_pb,
            "pbMethod": "行情PB" if numeric(q.get("市净率")) is not None else ("股价/最新每股净资产" if derived_pb is not None else "暂无"),
            "dividendYield": numeric(first_present(q, ["股息率", "股息率(TTM)"])),
            "aboveMA60": None if ret60 is None else int(ret60 > 0),
            "ma60Slope": None if ret60 is None else clip(ret60 / 6.0, -5.0, 5.0),
            "aboveMA250": None if ytd is None else int(ytd > 0),
            "relativeStrength": None if ret60 is None else ret60 - med60,
            "volumeConfirm": volume_confirm,
            "moat": 3,
            "management": 3,
            "predictability": 3,
            "understandable": 3,
            "capitalAllocation": 3,
            "receivableRisk": None,
            "inventoryRisk": None,
            "dilution": None,
            "ocfPositive": None if ocf is None else int(ocf > 0),
            "raw60Return": ret60,
            "ytdReturn": ytd,
            "pcf": None,
        })

    stocks = [x for x in stocks if x["code"] and x["name"]]
    if len(stocks) < MIN_STOCKS:
        raise RuntimeError(f"final dataset contains only {len(stocks)} stocks")

    exact_count = 0
    if DETAIL_LIMIT > 0:
        candidates = preliminary_detail_candidates(stocks, DETAIL_LIMIT)
        log(f"detailed enrichment candidates: {len(candidates)}")
        updates: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, DETAIL_WORKERS)) as executor:
            futures = {executor.submit(detailed_metric_for_stock, stock): stock["code"] for stock in candidates}
            for index, future in enumerate(as_completed(futures), 1):
                code = futures[future]
                try:
                    result_code, payload = future.result()
                    updates[result_code] = payload
                    if payload.get("cashFlowDataLevel") == "精确":
                        exact_count += 1
                except Exception as exc:
                    log(f"detail enrichment {code} failed: {exc}")
                if index % 20 == 0:
                    log(f"detail enrichment progress: {index}/{len(candidates)}, exact={exact_count}")
        for stock in stocks:
            if stock["code"] in updates:
                stock.update({k: v for k, v in updates[stock["code"]].items() if v is not None or k.endswith("Method") or k == "cashFlowDataLevel"})

    roic_count = sum(numeric(x.get("roic")) is not None for x in stocks)
    fcf_count = sum(numeric(x.get("fcfYield")) is not None for x in stocks)
    return {
        "schemaVersion": 2,
        "updatedAt": updated_at,
        "tradeDate": trade_date,
        "source": f"AKShare公开接口（{spot_source}；GitHub后台整理，无需数据Token）",
        "stockCount": len(stocks),
        "coverage": {
            "roe": sum(numeric(x.get("roe")) is not None for x in stocks),
            "roic": roic_count,
            "fcfYield": fcf_count,
            "exactCashFlow": exact_count,
        },
        "notes": [
            f"行情入口：{spot_source}。东财 push2 拒绝 GitHub 服务器时自动切换新浪；新浪缺失的估值字段从财报计算或留空，不伪造。",
            "ROIC为计算指标：全市场采用经营投入资本估算口径，优先候选在完整报表可用时采用权益+有息负债-现金口径。",
            "自由现金流：优先候选采用经营现金流减资本开支；其余非金融股采用经营现金流加投资现金流净额的保守代理，并在网页明确标注口径。",
            "银行、保险、券商等金融企业不套用工业企业ROIC和自由现金流公式。",
            "三年现金流趋势采用最近三个完整年度；单位数组为亿元。",
            "均线字段仍使用60日涨幅和年初至今涨幅作为趋势代理。",
        ],
        "stocks": stocks,
    }


def main() -> int:
    try:
        payload = build()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temp = OUTPUT.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temp.replace(OUTPUT)
        log(
            f"wrote {OUTPUT} with {payload['stockCount']} stocks; "
            f"exact cash flow={payload.get('coverage', {}).get('exactCashFlow', 0)}"
        )
        return 0
    except Exception as exc:
        log(f"UPDATE FAILED: {exc}")
        if OUTPUT.exists():
            log("existing stocks.json is preserved")
            try:
                cached = json.loads(OUTPUT.read_text(encoding="utf-8"))
                cached_count = len(cached.get("stocks", []))
                if cached_count >= MIN_STOCKS:
                    log(f"valid cached dataset contains {cached_count} stocks; deploy cache instead")
                    return 0
            except Exception as cache_exc:
                log(f"cached dataset validation failed: {cache_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
