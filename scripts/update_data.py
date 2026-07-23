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
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import akshare as ak
import pandas as pd

from short_strategy import (
    latest_short_metrics,
    market_environment_score,
    normalize_history_frame,
    score_short_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "stocks.json"
MIN_STOCKS = int(os.getenv("MIN_STOCKS", "4000"))
QUARTERS_TO_TRY = int(os.getenv("QUARTERS_TO_TRY", "6"))
ANNUAL_YEARS = int(os.getenv("ANNUAL_YEARS", "4"))
DETAIL_LIMIT = int(os.getenv("DETAIL_LIMIT", "200"))
DETAIL_WORKERS = int(os.getenv("DETAIL_WORKERS", "5"))
FINANCIAL_DETAIL_LIMIT = int(os.getenv("FINANCIAL_DETAIL_LIMIT", "260"))
FINANCIAL_DETAIL_WORKERS = int(os.getenv("FINANCIAL_DETAIL_WORKERS", "4"))
DIVIDEND_DETAIL_LIMIT = int(os.getenv("DIVIDEND_DETAIL_LIMIT", "220"))
DIVIDEND_DETAIL_WORKERS = int(os.getenv("DIVIDEND_DETAIL_WORKERS", "4"))
SHORT_HISTORY_LIMIT = int(os.getenv("SHORT_HISTORY_LIMIT", "320"))
SHORT_HISTORY_WORKERS = int(os.getenv("SHORT_HISTORY_WORKERS", "4"))
SHORT_HISTORY_CALENDAR_DAYS = int(os.getenv("SHORT_HISTORY_CALENDAR_DAYS", "1250"))
SHORT_CACHE = ROOT / "data" / ".short_history_cache.pkl"

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


def financial_type(industry: str) -> str:
    text = str(industry or "")
    if "银行" in text:
        return "bank"
    if "保险" in text:
        return "insurance"
    if "证券" in text or "券商" in text:
        return "broker"
    if any(keyword in text for keyword in ("多元金融", "信托", "期货", "金融服务")):
        return "diversified"
    return "general"


def is_financial(industry: str) -> bool:
    return financial_type(industry) != "general"


def eastmoney_secu_code(code: str) -> str:
    if code.startswith(("6", "68")):
        return f"{code}.SH"
    if code.startswith(("4", "8", "92")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def xueqiu_symbol(code: str) -> str:
    if code.startswith(("6", "68")):
        return f"SH{code}"
    if code.startswith(("4", "8", "92")):
        return f"BJ{code}"
    return f"SZ{code}"


def financial_dividend_yield(stock: dict) -> tuple[float | None, str]:
    """Return a financial stock's TTM dividend yield with source labels.

    The bulk Eastmoney snapshot normally supplies dividend yield, but the Sina
    fallback does not.  Financial stocks are therefore enriched separately:
    1) Xueqiu individual snapshot: current TTM dividend yield;
    2) Eastmoney dividend-plan detail: latest implemented plan yield fallback.
    Missing values remain blank rather than being estimated from unrelated data.
    """
    existing = numeric(stock.get("dividendYield"))
    if existing is not None:
        return existing, stock.get("dividendYieldMethod") or "全市场行情股息率"

    code = stock.get("code", "")
    try:
        frame = ak.stock_individual_spot_xq(symbol=xueqiu_symbol(code), timeout=12)
        if frame is not None and not frame.empty and {"item", "value"}.issubset(frame.columns):
            record = dict(zip(frame["item"].astype(str), frame["value"]))
            value = numeric(record.get("股息率(TTM)"))
            if value is not None and value >= 0:
                return value, "雪球个股行情：股息率(TTM)"
    except Exception as exc:
        log(f"financial dividend Xueqiu {code} skipped: {exc}")

    try:
        frame = ak.stock_fhps_detail_em(symbol=code)
        if frame is not None and not frame.empty and "现金分红-股息率" in frame.columns:
            work = frame.copy()
            if "方案进度" in work.columns:
                implemented = work[work["方案进度"].astype(str).str.contains("实施", na=False)]
                if not implemented.empty:
                    work = implemented
            date_col = next((c for c in ("除权除息日", "报告期", "最新公告日期") if c in work.columns), None)
            if date_col:
                work["__date"] = pd.to_datetime(work[date_col], errors="coerce")
                work = work.sort_values("__date", ascending=False)
            values = pd.to_numeric(work["现金分红-股息率"], errors="coerce").dropna()
            if not values.empty:
                value = float(values.iloc[0])
                if math.isfinite(value) and value >= 0:
                    return value, "东方财富：最近已实施分红方案股息率（备用口径）"
    except Exception as exc:
        log(f"financial dividend Eastmoney {code} skipped: {exc}")

    return None, "雪球TTM与分红方案接口均未取得数据"


def latest_indicator_record(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {}
    work = frame.copy()
    for col in ("REPORT_DATE", "REPORTDATE", "报告日期"):
        if col in work.columns:
            work["__date"] = pd.to_datetime(work[col], errors="coerce")
            work = work.sort_values("__date", ascending=False)
            break
    return work.iloc[0].to_dict() if not work.empty else {}


def fuzzy_number(record: dict, exact: Iterable[str] = (), token_groups: Iterable[tuple[str, ...]] = ()):
    value = first_number(record, exact)
    if value is not None:
        return value
    normalized = {str(k).upper().replace("_", ""): k for k in record.keys()}
    for tokens in token_groups:
        norm_tokens = tuple(str(x).upper().replace("_", "") for x in tokens)
        for norm_key, original in normalized.items():
            if all(token in norm_key for token in norm_tokens):
                value = numeric(record.get(original))
                if value is not None:
                    return value
    return None


def financial_metric_for_stock(stock: dict) -> tuple[str, dict]:
    code = stock["code"]
    model = financial_type(stock.get("industry", ""))
    payload = {
        "financialType": model,
        "financialMetricLevel": "基础",
        "financialMetricMethod": "ROE、PB、利润趋势和股息率基础模型",
    }
    if model == "general":
        return code, payload

    dividend_yield, dividend_method = financial_dividend_yield(stock)
    payload["dividendYieldMethod"] = dividend_method
    if dividend_yield is not None:
        payload["dividendYield"] = dividend_yield

    try:
        frame = call_with_retry(
            f"financial_indicator({code})",
            lambda: ak.stock_financial_analysis_indicator_em(
                symbol=eastmoney_secu_code(code), indicator="按报告期"
            ),
            attempts=2,
            base_wait=2.5,
        )
        record = latest_indicator_record(frame)
        payload["roa"] = fuzzy_number(record, ("ZZCJLL", "ROA", "JROA", "TOTAL_ASSET_ROA"), (("ROA",), ("TOTAL", "ASSET", "RETURN")))
        f10_roe = fuzzy_number(record, ("ROEJQ", "ROEKCJQ", "ROE"), (("ROE",),))
        f10_profit_growth = fuzzy_number(record, ("PARENTNETPROFITTZ", "NET_PROFIT_YOY"), (("NET", "PROFIT", "YOY"),))
        f10_revenue_growth = fuzzy_number(record, ("TOTALOPERATEREVETZ", "OPERATE_INCOME_YOY"), (("OPERATE", "INCOME", "YOY"),))
        if f10_roe is not None:
            payload["roe"] = f10_roe
        if f10_profit_growth is not None:
            payload["profitGrowthQ"] = f10_profit_growth
        if f10_revenue_growth is not None:
            payload["revenueGrowthQ"] = f10_revenue_growth
        if model == "bank":
            payload.update({
                "nplRatio": fuzzy_number(record, ("NON_PERFORMING_LOAN_RATIO", "NONPERFORMING_LOAN_RATIO", "NONPERFORM_LOAN_RATIO", "NPL_RATIO", "BLDKBL"), (("NON", "PERFORM", "LOAN", "RATIO"), ("BAD", "LOAN", "RATIO"))),
                "provisionCoverage": fuzzy_number(record, ("PROVISION_COVERAGE", "PROVISION_COVERAGE_RATIO", "BAD_LOAN_COVERAGE", "LOAN_PROVISION_COVERAGE", "BLDKBBL"), (("PROVISION", "COVER"), ("BAD", "LOAN", "COVER"))),
                "netInterestMargin": fuzzy_number(record, ("NET_INTEREST_MARGIN", "NET_INTEREST_SPREAD", "NET_INTEREST_MARGIN_RATIO", "JXCL"), (("NET", "INTEREST", "MARGIN"),)),
                "coreTier1CapitalAdequacy": fuzzy_number(record, ("CORE_TIER1_CAPITAL_ADEQUACY_RATIO", "CORE_TIER1_CAPITAL_ADEQUACY", "CORE_CAPITAL_ADEQUACY_RATIO"), (("CORE", "TIER1", "CAPITAL"),)),
                "capitalAdequacy": fuzzy_number(record, ("CAPITAL_ADEQUACY_RATIO", "CAPITAL_ADEQUACY"), (("CAPITAL", "ADEQUACY"),)),
            })
        elif model == "insurance":
            payload.update({
                "solvencyRatio": fuzzy_number(record, ("COMPREHENSIVE_SOLVENCY_ADEQUACY_RATIO", "SOLVENCY_ADEQUACY_RATIO"), (("SOLVENCY", "ADEQUACY"),)),
                "coreSolvencyRatio": fuzzy_number(record, ("CORE_SOLVENCY_ADEQUACY_RATIO",), (("CORE", "SOLVENCY"),)),
                "nbvGrowth": fuzzy_number(record, ("NBV_GROWTH", "NEW_BUSINESS_VALUE_GROWTH"), (("NEW", "BUSINESS", "VALUE", "GROWTH"),)),
                "embeddedValueGrowth": fuzzy_number(record, ("EMBEDDED_VALUE_GROWTH", "EV_GROWTH"), (("EMBEDDED", "VALUE", "GROWTH"),)),
                "combinedRatio": fuzzy_number(record, ("COMBINED_RATIO",), (("COMBINED", "RATIO"),)),
                "pev": fuzzy_number(record, ("PEV", "P_EV"), (("PEV",),)),
            })
        elif model == "broker":
            payload.update({
                "riskCoverageRatio": fuzzy_number(record, ("RISK_COVERAGE_RATIO",), (("RISK", "COVERAGE"),)),
                "capitalLeverageRatio": fuzzy_number(record, ("CAPITAL_LEVERAGE_RATIO",), (("CAPITAL", "LEVERAGE"),)),
                "liquidityCoverageRatio": fuzzy_number(record, ("LIQUIDITY_COVERAGE_RATIO",), (("LIQUIDITY", "COVERAGE"),)),
                "netStableFundingRatio": fuzzy_number(record, ("NET_STABLE_FUNDING_RATIO", "NET_FUNDING_RATIO"), (("NET", "STABLE", "FUNDING"),)),
                "netCapital": fuzzy_number(record, ("NET_CAPITAL", "PROPRIETARY_CAPITAL"), (("NET", "CAPITAL"),)),
            })
        specialty_keys = {
            "bank": ("nplRatio", "provisionCoverage", "netInterestMargin", "coreTier1CapitalAdequacy"),
            "insurance": ("solvencyRatio", "coreSolvencyRatio", "nbvGrowth", "embeddedValueGrowth"),
            "broker": ("riskCoverageRatio", "capitalLeverageRatio", "liquidityCoverageRatio", "netStableFundingRatio"),
            "diversified": ("roa",),
        }.get(model, ())
        found = sum(numeric(payload.get(k)) is not None for k in specialty_keys)
        payload["financialMetricLevel"] = "专项" if found >= max(2, len(specialty_keys) // 2) else ("部分专项" if found else "基础")
        payload["financialMetricMethod"] = ("东方财富F10主要指标；股息率单独按TTM补全；缺失字段不推算" if found else "专项接口未返回可识别字段；股息率单独按TTM补全")
    except Exception as exc:
        payload["financialMetricMethod"] = f"专项接口失败，保留基础模型：{exc}"
    return code, payload


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



def _history_market_prefix(code: str) -> str:
    if code.startswith(("6", "68")):
        return "sh" + code
    if code.startswith(("4", "8", "92")):
        return "bj" + code
    return "sz" + code


def fetch_stock_history(stock: dict, start_date: str, end_date: str) -> tuple[str, pd.DataFrame, str | None]:
    """Fetch adjusted daily bars with an independent fallback source."""
    code = stock["code"]
    errors: list[str] = []
    attempts = [
        (
            "东方财富前复权日K",
            lambda: ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
            ),
        ),
        (
            "腾讯前复权日K",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=_history_market_prefix(code), start_date=start_date, end_date=end_date, adjust="qfq"
            ),
        ),
    ]
    for label, call in attempts:
        try:
            frame = normalize_history_frame(call())
            if len(frame) >= 80:
                return code, frame, label
            errors.append(f"{label}: only {len(frame)} bars")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    return code, pd.DataFrame(), "; ".join(errors)[-500:]


def fetch_market_history(start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    attempts = [
        ("上证指数东方财富日K", lambda: ak.stock_zh_index_daily_em(symbol="sh000001")),
        ("上证指数新浪日K", lambda: ak.stock_zh_index_daily(symbol="sh000001")),
    ]
    errors: list[str] = []
    for label, call in attempts:
        try:
            frame = normalize_history_frame(call())
            if not frame.empty:
                start = pd.to_datetime(start_date, errors="coerce")
                end = pd.to_datetime(end_date, errors="coerce")
                if pd.notna(start):
                    frame = frame[frame["date"] >= start]
                if pd.notna(end):
                    frame = frame[frame["date"] <= end]
                if len(frame) >= 120:
                    return frame.reset_index(drop=True), label
            errors.append(f"{label}: insufficient rows")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    log("market history unavailable: " + "; ".join(errors))
    return pd.DataFrame(), "市场日K未取得"


def short_history_candidates(stocks: list[dict], limit: int) -> list[dict]:
    """Choose a liquid, diversified shortlist for exact daily-bar enrichment."""
    if limit <= 0:
        return []
    eligible: list[tuple[float, dict]] = []
    for stock in stocks:
        name = str(stock.get("name") or "")
        price = numeric(stock.get("price"))
        amount = numeric(stock.get("amount"))
        market_cap = numeric(stock.get("marketCap"))
        turnover = numeric(stock.get("turnover"))
        if not price or price <= 0 or name.upper().startswith(("ST", "*ST")) or "退" in name:
            continue
        if amount is not None and amount < 2e7:
            continue
        liquidity = math.log10(max(amount or 3e7, 1e6)) * 12
        size = math.log10(max((market_cap or 10) * 1e8, 1e8)) * 4
        quality = min(max(numeric(stock.get("roe")) or 0, -10), 35) * 0.7
        momentum = min(max(numeric(stock.get("raw60Return")) or 0, -30), 50) * 0.25
        activity = min(max(turnover or 0, 0), 12) * 1.1
        eligible.append((liquidity + size + quality + momentum + activity, stock))
    eligible.sort(key=lambda x: x[0], reverse=True)

    # Preserve industry breadth instead of allowing one hot sector to occupy the list.
    chosen: list[dict] = []
    industry_count: dict[str, int] = {}
    soft_cap = max(8, limit // 18)
    for _, stock in eligible:
        industry = str(stock.get("industry") or "未分类")
        if industry_count.get(industry, 0) >= soft_cap:
            continue
        chosen.append(stock)
        industry_count[industry] = industry_count.get(industry, 0) + 1
        if len(chosen) >= limit:
            return chosen
    selected_codes = {x["code"] for x in chosen}
    for _, stock in eligible:
        if stock["code"] in selected_codes:
            continue
        chosen.append(stock)
        if len(chosen) >= limit:
            break
    return chosen


def _industry_environment(metrics_by_code: dict[str, dict], stock_by_code: dict[str, dict]) -> tuple[dict[str, dict], float]:
    ret20_values = [numeric(m.get("ret20")) for m in metrics_by_code.values()]
    ret20_values = [x for x in ret20_values if x is not None]
    overall_ret20 = float(pd.Series(ret20_values).median()) if ret20_values else 0.0
    buckets: dict[str, list[dict]] = {}
    for code, metrics in metrics_by_code.items():
        industry = str(stock_by_code.get(code, {}).get("industry") or "未分类")
        buckets.setdefault(industry, []).append(metrics)
    output: dict[str, dict] = {}
    for industry, rows in buckets.items():
        valid20 = [numeric(x.get("aboveMA20")) for x in rows if numeric(x.get("aboveMA20")) is not None]
        valid60 = [numeric(x.get("aboveMA60")) for x in rows if numeric(x.get("aboveMA60")) is not None]
        returns = [numeric(x.get("ret20")) for x in rows if numeric(x.get("ret20")) is not None]
        breadth20 = sum(x > 0 for x in valid20) / len(valid20) * 100 if valid20 else None
        breadth60 = sum(x > 0 for x in valid60) / len(valid60) * 100 if valid60 else None
        median_ret = float(pd.Series(returns).median()) if returns else None
        components = []
        if breadth20 is not None:
            components.append(clip(20 + breadth20 * 0.8, 0, 100))
        if breadth60 is not None:
            components.append(clip(20 + breadth60 * 0.8, 0, 100))
        if median_ret is not None:
            components.append(clip(50 + (median_ret - overall_ret20) * 3, 0, 100))
        score = sum(components) / len(components) if components else 50.0
        output[industry] = {
            "industryShortScore": round(score, 1),
            "industryBreadth20": round(breadth20, 1) if breadth20 is not None else None,
            "industryBreadth60": round(breadth60, 1) if breadth60 is not None else None,
            "industryMedianRet20": round(median_ret, 2) if median_ret is not None else None,
        }
    return output, overall_ret20


def enrich_short_history(stocks: list[dict], updated_at: str) -> tuple[int, float, str, dict[str, pd.DataFrame], pd.DataFrame]:
    candidates = short_history_candidates(stocks, SHORT_HISTORY_LIMIT)
    if not candidates:
        return 0, 50.0, "未启用短线日K补全", {}, pd.DataFrame()
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=SHORT_HISTORY_CALENDAR_DAYS)).strftime("%Y%m%d")
    market_frame, market_source = fetch_market_history(start, end)
    market_score, market_meta = market_environment_score(market_frame)
    log(f"short history candidates: {len(candidates)}; market={market_source} score={market_score}")
    histories: dict[str, pd.DataFrame] = {}
    metrics_by_code: dict[str, dict] = {}
    source_by_code: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, SHORT_HISTORY_WORKERS)) as executor:
        futures = {executor.submit(fetch_stock_history, stock, start, end): stock["code"] for stock in candidates}
        for index, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                result_code, frame, label = future.result()
                if frame is not None and not frame.empty:
                    histories[result_code] = frame
                    metrics_by_code[result_code] = latest_short_metrics(frame)
                    source_by_code[result_code] = label or "日K"
                else:
                    errors[result_code] = label or "日K未取得"
            except Exception as exc:
                errors[code] = str(exc)
            if index % 25 == 0:
                log(f"short history progress: {index}/{len(candidates)}, valid={len(histories)}")
    stock_by_code = {x["code"]: x for x in stocks}
    industry_meta, overall_ret20 = _industry_environment(metrics_by_code, stock_by_code)
    for code, metrics in metrics_by_code.items():
        stock = stock_by_code[code]
        industry = str(stock.get("industry") or "未分类")
        ind = industry_meta.get(industry, {"industryShortScore": 50.0})
        metrics.update(ind)
        metrics["relativeStrength20"] = None if numeric(metrics.get("ret20")) is None else round(numeric(metrics.get("ret20")) - overall_ret20, 2)
        metrics["marketShortScore"] = market_score
        metrics["marketTrendLabel"] = market_meta.get("marketTrendLabel", "市场数据不足")
        metrics["shortHistorySource"] = source_by_code.get(code, "日K")
        metrics["shortDataDate"] = metrics.get("historyEnd") or updated_at
        metrics["eventRiskCoverage"] = 0
        metrics["eventRiskNote"] = "尚未自动覆盖公告、减持、解禁、停复牌与业绩披露窗口"
        metrics.update(score_short_metrics(metrics, market_score, ind.get("industryShortScore", 50.0)))
        stock.update({k: v for k, v in metrics.items() if v is not None})
        stock["trendDataLevel"] = "精确日K"
        # Keep long-term trend inputs compatible, but now use exact bars for enriched stocks.
        stock["aboveMA60"] = metrics.get("aboveMA60")
        stock["aboveMA250"] = metrics.get("aboveMA250")
        stock["ma60Slope"] = metrics.get("ma60Slope")
        stock["relativeStrength"] = metrics.get("relativeStrength20")
    for stock in stocks:
        if stock["code"] not in metrics_by_code:
            stock.setdefault("shortDataLevel", "未补充日K")
            stock.setdefault("shortDataCoverage", 0)
            stock.setdefault("shortCoverage", 0)
            stock.setdefault("shortDecision", "短线数据未覆盖")
            stock.setdefault("shortGroup", "data")
            stock.setdefault("shortAction", "该股票未进入日K补全池，不能用当前页面做短线判断。")
            stock.setdefault("eventRiskCoverage", 0)
            stock.setdefault("eventRiskNote", "尚未自动覆盖公告、减持、解禁、停复牌与业绩披露窗口")
            stock.setdefault("trendDataLevel", "行情快照趋势代理")
            if stock["code"] in errors:
                stock["shortHistoryError"] = errors[stock["code"]]
    return len(histories), market_score, market_source, histories, market_frame


def dividend_candidates(stocks: list[dict], limit: int) -> list[dict]:
    missing = [x for x in stocks if numeric(x.get("dividendYield")) is None and numeric(x.get("price")) not in (None, 0)]
    missing.sort(key=lambda x: (numeric(x.get("marketCap")) or 0, numeric(x.get("amount")) or 0), reverse=True)
    return missing[:max(0, limit)]


def supplement_dividends(stocks: list[dict]) -> int:
    candidates = dividend_candidates(stocks, DIVIDEND_DETAIL_LIMIT)
    if not candidates:
        return 0
    log(f"general dividend enrichment candidates: {len(candidates)}")
    updates: dict[str, tuple[float | None, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, DIVIDEND_DETAIL_WORKERS)) as executor:
        futures = {executor.submit(financial_dividend_yield, stock): stock["code"] for stock in candidates}
        for index, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                updates[code] = future.result()
            except Exception as exc:
                updates[code] = (None, f"股息率补全失败：{exc}")
            if index % 30 == 0:
                log(f"dividend enrichment progress: {index}/{len(candidates)}")
    count = 0
    stock_by_code = {x["code"]: x for x in stocks}
    for code, (value, method) in updates.items():
        stock = stock_by_code.get(code)
        if not stock:
            continue
        stock["dividendYieldMethod"] = method
        if value is not None:
            stock["dividendYield"] = value
            count += 1
    return count


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
    receivable_history = annual_history_by_code(a_balances, ["应收账款", "应收票据及应收账款", "应收款项"])
    inventory_history = annual_history_by_code(a_balances, ["存货"])
    share_history = annual_history_by_code(a_reports, ["总股本", "总股本-总股本"])

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
        amount = numeric(q.get("成交额"))
        volume = numeric(q.get("成交量"))
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

        revenue_cagr3 = cagr_from_history(revenue_history.get(code, []), 3)
        profit_cagr3 = cagr_from_history(profit_history.get(code, []), 3)
        receivable_cagr3 = None if financial else cagr_from_history(receivable_history.get(code, []), 3)
        inventory_cagr3 = None if financial else cagr_from_history(inventory_history.get(code, []), 3)
        share_items = share_history.get(code, [])
        share_dilution = None
        if not financial and len(share_items) >= 2:
            first_share, last_share = numeric(share_items[0][1]), numeric(share_items[-1][1])
            if first_share not in (None, 0) and last_share is not None:
                share_dilution = (last_share / first_share - 1) * 100
        receivable_risk = None if receivable_cagr3 is None or revenue_cagr3 is None else receivable_cagr3 - revenue_cagr3
        inventory_risk = None if inventory_cagr3 is None or revenue_cagr3 is None else inventory_cagr3 - revenue_cagr3

        stocks.append({
            "code": code,
            "name": str(q.get("名称") or report.get("股票简称") or ""),
            "industry": industry,
            "financialType": financial_type(industry),
            "financialMetricLevel": "基础" if is_financial(industry) else "不适用",
            "financialMetricMethod": "ROE、PB、利润趋势和股息率基础模型" if is_financial(industry) else "普通企业模型",
            "roa": None,
            "nplRatio": None,
            "provisionCoverage": None,
            "netInterestMargin": None,
            "coreTier1CapitalAdequacy": None,
            "capitalAdequacy": None,
            "solvencyRatio": None,
            "coreSolvencyRatio": None,
            "nbvGrowth": None,
            "embeddedValueGrowth": None,
            "combinedRatio": None,
            "pev": None,
            "riskCoverageRatio": None,
            "capitalLeverageRatio": None,
            "liquidityCoverageRatio": None,
            "netStableFundingRatio": None,
            "netCapital": None,
            "price": price,
            "pctChange": pct,
            "marketCap": market_cap,
            "amount": amount,
            "volume": volume,
            "volumeRatio": volume_ratio,
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
            "revenueCagr3": revenue_cagr3,
            "profitCagr3": profit_cagr3,
            "revenueGrowthQ": first_number(report, ["营业总收入-同比增长", "营业收入-同比增长"]),
            "profitGrowthQ": first_number(report, ["净利润-同比增长", "净利润同比"]),
            "epsRevision": None,
            "pe": numeric(q.get("市盈率-动态")) if numeric(q.get("市盈率-动态")) is not None else derived_pe,
            "peMethod": "动态PE" if numeric(q.get("市盈率-动态")) is not None else ("静态PE：股价/最近完整年度EPS" if derived_pe is not None else "暂无"),
            "pb": numeric(q.get("市净率")) if numeric(q.get("市净率")) is not None else derived_pb,
            "pbMethod": "行情PB" if numeric(q.get("市净率")) is not None else ("股价/最新每股净资产" if derived_pb is not None else "暂无"),
            "dividendYield": numeric(first_present(q, ["股息率", "股息率(TTM)"])),
            "dividendYieldMethod": "全市场行情股息率" if numeric(first_present(q, ["股息率", "股息率(TTM)"])) is not None else ("待金融专项补全" if financial else "行情源未提供"),
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
            "receivableRisk": receivable_risk,
            "inventoryRisk": inventory_risk,
            "dilution": share_dilution,
            "receivableRiskMethod": "应收账款3年CAGR减营收3年CAGR" if receivable_risk is not None else "未取得连续年度应收账款",
            "inventoryRiskMethod": "存货3年CAGR减营收3年CAGR" if inventory_risk is not None else "未取得连续年度存货",
            "dilutionMethod": "年度总股本首尾变化" if share_dilution is not None else "未取得连续年度总股本",
            "ocfPositive": None if ocf is None else int(ocf > 0),
            "raw60Return": ret60,
            "ytdReturn": ytd,
            "trendDataLevel": "行情快照趋势代理",
            "eventRiskCoverage": 0,
            "eventRiskNote": "尚未自动覆盖公告、减持、解禁、停复牌与业绩披露窗口",
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

    financial_candidates = [x for x in stocks if is_financial(x.get("industry", ""))]
    financial_candidates.sort(key=lambda x: (0 if financial_type(x.get("industry", "")) == "diversified" else 1, numeric(x.get("marketCap")) or 0), reverse=True)
    if FINANCIAL_DETAIL_LIMIT > 0:
        financial_candidates = financial_candidates[:FINANCIAL_DETAIL_LIMIT]
        log(f"financial industry enrichment candidates: {len(financial_candidates)}")
        financial_updates: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, FINANCIAL_DETAIL_WORKERS)) as executor:
            futures = {executor.submit(financial_metric_for_stock, stock): stock["code"] for stock in financial_candidates}
            for index, future in enumerate(as_completed(futures), 1):
                code = futures[future]
                try:
                    result_code, payload = future.result()
                    financial_updates[result_code] = payload
                except Exception as exc:
                    log(f"financial enrichment {code} failed: {exc}")
                if index % 20 == 0:
                    log(f"financial enrichment progress: {index}/{len(financial_candidates)}")
        for stock in stocks:
            if stock["code"] in financial_updates:
                update = financial_updates[stock["code"]]
                stock.update({k: v for k, v in update.items() if v is not None or k in {"financialType", "financialMetricLevel", "financialMetricMethod", "dividendYieldMethod"}})

    dividend_supplemented = supplement_dividends(stocks) if DIVIDEND_DETAIL_LIMIT > 0 else 0

    short_history_count, market_short_score, market_history_source, short_histories, market_history = enrich_short_history(stocks, updated_at)
    if short_histories:
        try:
            SHORT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with SHORT_CACHE.open("wb") as handle:
                pickle.dump({
                    "createdAt": updated_at,
                    "marketSource": market_history_source,
                    "market": market_history,
                    "stocks": short_histories,
                    "stockMeta": {x["code"]: {"name": x.get("name"), "industry": x.get("industry")} for x in stocks if x["code"] in short_histories},
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"temporary short history cache written: {SHORT_CACHE}")
        except Exception as exc:
            log(f"short history cache skipped: {exc}")

    financial_count = sum(is_financial(x.get("industry", "")) for x in stocks)
    financial_special_count = sum(x.get("financialMetricLevel") in {"专项", "部分专项"} for x in stocks)
    roic_count = sum(numeric(x.get("roic")) is not None for x in stocks)
    fcf_count = sum(numeric(x.get("fcfYield")) is not None for x in stocks)
    return {
        "schemaVersion": 7,
        "updatedAt": updated_at,
        "tradeDate": trade_date,
        "source": f"AKShare公开接口（{spot_source}；GitHub后台整理，无需数据Token）",
        "stockCount": len(stocks),
        "coverage": {
            "roe": sum(numeric(x.get("roe")) is not None for x in stocks),
            "roic": roic_count,
            "fcfYield": fcf_count,
            "exactCashFlow": exact_count,
            "financialStocks": financial_count,
            "financialSpecialMetrics": financial_special_count,
            "dividendYield": sum(numeric(x.get("dividendYield")) is not None for x in stocks),
            "dividendSupplemented": dividend_supplemented,
            "shortHistoryFetched": short_history_count,
            "shortExact250Bars": sum(x.get("shortDataLevel") == "精确日K" for x in stocks),
            "shortScored": sum(numeric(x.get("shortScore")) is not None for x in stocks),
            "eventRisk": 0,
            "epsRevision": sum(numeric(x.get("epsRevision")) is not None for x in stocks),
            "receivableRisk": sum(numeric(x.get("receivableRisk")) is not None for x in stocks),
            "inventoryRisk": sum(numeric(x.get("inventoryRisk")) is not None for x in stocks),
            "dilution": sum(numeric(x.get("dilution")) is not None for x in stocks),
        },
        "notes": [
            f"行情入口：{spot_source}。股息率优先使用全市场行情；缺失时对高流动性候选及金融股使用雪球TTM或最近已实施分红方案补全。",
            "ROIC为计算指标：全市场采用经营投入资本估算口径，优先候选在完整报表可用时采用权益+有息负债-现金口径。",
            "自由现金流：优先候选采用经营现金流减资本开支；其余非金融股采用经营现金流加投资现金流净额的代理，并在网页明确标注口径。",
            "银行、保险、券商自动切换行业模型；专项监管指标由F10主要指标接口尽力补充，接口缺失时只给基础判断，不伪造。",
            f"短线模块仅对约{SHORT_HISTORY_LIMIT}只高流动性、分行业候选补取前复权日K；市场日K来源：{market_history_source}，当前市场环境分：{market_short_score}。",
            "短线信号使用真实MA、RSI、MACD、ATR、突破及量价结构，并与长期价值判断完全分开；未进入日K补全池的股票不得据此做短线结论。",
            "短线模块尚未自动覆盖公告、减持、解禁、停复牌和财报披露窗口；页面会明确显示事件风险未覆盖。",
            "盈利预测调整、应收/存货异常和股本稀释仍可能缺失，缺失字段不应被当作利好。",
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
