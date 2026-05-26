import csv
import io
import os

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.05
ALLOWED_PERIODS = {"3mo", "6mo", "1y", "2y", "5y"}
ALLOWED_EXCHANGES = {"AUTO", "US", "NSE", "BSE"}
OPTIMISATION_SAMPLES = 15000
EXCHANGE_SUFFIXES = {
    "NSE": ".NS",
    "BSE": ".BO",
}
NSE_COLUMN_ALIASES = {"nsecode", "nse"}
BSE_COLUMN_ALIASES = {"bsecode", "bse"}
NAME_COLUMN_ALIASES = {"name", "company", "companyname"}
WEIGHT_COLUMN_ALIASES = {
    "weight",
    "weights",
    "allocation",
    "portfolioallocation",
    "portfolioweight",
    "weightage",
}


def build_error(message, status_code=400):
    return jsonify({"error": message}), status_code


def safe_float(value):
    try:
        if value in (None, "", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_get(mapping, *keys):
    for key in keys:
        try:
            if isinstance(mapping, dict):
                value = mapping.get(key)
            else:
                value = mapping[key]
        except Exception:
            value = None

        if value not in (None, "", "None"):
            return value
    return None


def safe_last(series):
    clean = pd.Series(series).dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def pct(value):
    if value is None:
        return None
    return float(value) * 100


def clip_score(value, low, high):
    if value is None:
        return None
    if high == low:
        return None
    scaled = (float(value) - low) / (high - low)
    return float(np.clip(scaled, 0, 1))


def average_score(values, fallback=0.5):
    clean = [value for value in values if value is not None]
    if not clean:
        return fallback
    return float(np.mean(clean))


def compact_number(value):
    if value is None:
        return None

    absolute = abs(value)
    if absolute >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def format_percentage(value, digits=1):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}%"


def format_ratio(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def normalise_symbol(symbol, exchange):
    cleaned = str(symbol).strip().upper()

    if not cleaned:
        return ""

    if exchange in EXCHANGE_SUFFIXES and "." not in cleaned and not cleaned.startswith("^"):
        return f"{cleaned}{EXCHANGE_SUFFIXES[exchange]}"

    return cleaned


def normalise_csv_header(header):
    return "".join(character for character in str(header).strip().lower() if character.isalnum())


def parse_csv_limit(raw_limit):
    if raw_limit in (None, ""):
        return 25

    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV import limit must be a whole number") from exc

    if limit < 1 or limit > 100:
        raise ValueError("CSV import limit must be between 1 and 100")

    return limit


def get_first_matching_value(row, aliases):
    for alias in aliases:
        value = row.get(alias, "")
        if value:
            return value
    return ""


def parse_weight_cell(raw_value):
    if raw_value in (None, ""):
        return None

    cleaned = str(raw_value).strip().replace("%", "").replace(",", "")
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def resolve_screener_symbol(row, preferred_exchange):
    nse_symbol = get_first_matching_value(row, NSE_COLUMN_ALIASES)
    bse_symbol = get_first_matching_value(row, BSE_COLUMN_ALIASES)

    if preferred_exchange == "NSE":
        return ("NSE", nse_symbol) if nse_symbol else (None, "")

    if preferred_exchange == "BSE":
        return ("BSE", bse_symbol) if bse_symbol else (None, "")

    if nse_symbol:
        return "NSE", nse_symbol

    if bse_symbol:
        return "BSE", bse_symbol

    return None, ""


def parse_screener_csv(file_storage, preferred_exchange="AUTO", limit=25):
    try:
        raw_bytes = file_storage.read()
    finally:
        file_storage.stream.seek(0)

    if not raw_bytes:
        raise ValueError("Uploaded CSV file is empty")

    csv_text = raw_bytes.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(csv_text))

    if not reader.fieldnames:
        raise ValueError("Could not read CSV headers from the uploaded file")

    imported_tickers = []
    imported_names = []
    imported_weights = []
    seen_tickers = set()
    used_exchanges = set()
    skipped_rows = 0

    for raw_row in reader:
        if len(imported_tickers) >= limit:
            break

        row = {
            normalise_csv_header(key): str(value).strip()
            for key, value in raw_row.items()
            if key is not None
        }

        if not any(row.values()):
            continue

        exchange, symbol = resolve_screener_symbol(row, preferred_exchange)
        if not exchange or not symbol:
            skipped_rows += 1
            continue

        ticker = normalise_symbol(symbol, exchange)
        if ticker in seen_tickers:
            continue

        seen_tickers.add(ticker)
        used_exchanges.add(exchange)
        imported_tickers.append(ticker)
        imported_names.append(get_first_matching_value(row, NAME_COLUMN_ALIASES) or ticker)
        imported_weights.append(parse_weight_cell(get_first_matching_value(row, WEIGHT_COLUMN_ALIASES)))

    if not imported_tickers:
        raise ValueError("Could not find any usable NSE or BSE symbols in this Screener CSV")

    warnings = []
    if any(weight is not None for weight in imported_weights):
        warnings.append(
            "This CSV contains weights, but portfolio allocations will still be chosen automatically by the optimiser."
        )

    detected_exchange = "MIXED" if len(used_exchanges) > 1 else next(iter(used_exchanges))

    return {
        "tickers": imported_tickers,
        "company_names": imported_names,
        "detected_exchange": detected_exchange,
        "rows_imported": len(imported_tickers),
        "rows_skipped": skipped_rows,
        "warnings": warnings,
    }


def parse_tickers(raw_tickers, exchange):
    if not isinstance(raw_tickers, list):
        raise ValueError("Tickers must be provided as a list")

    tickers = []
    seen = set()

    for raw in raw_tickers:
        ticker = normalise_symbol(raw, exchange)

        if not ticker:
            continue

        if ticker in seen:
            raise ValueError(f"Duplicate ticker provided: {ticker}")

        seen.add(ticker)
        tickers.append(ticker)

    if not tickers:
        raise ValueError("No tickers provided")

    return tickers


def parse_risk_free_rate(raw_rate):
    if raw_rate in (None, ""):
        return DEFAULT_RISK_FREE_RATE

    try:
        rate = float(raw_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("Risk-free rate must be numeric") from exc

    if rate >= 1:
        rate /= 100

    if rate < 0:
        raise ValueError("Risk-free rate cannot be negative")

    return rate


def clean_market_frame(frame):
    cleaned = frame.copy()

    if "Close" not in cleaned.columns:
        return pd.DataFrame()

    for column in ("Open", "High", "Low"):
        if column not in cleaned.columns:
            cleaned[column] = cleaned["Close"]

    if "Volume" not in cleaned.columns:
        cleaned["Volume"] = 0.0

    cleaned = cleaned[["Open", "High", "Low", "Close", "Volume"]]
    cleaned = cleaned.dropna(subset=["Close"])
    cleaned["Volume"] = cleaned["Volume"].fillna(0)
    return cleaned


def fetch_market_data(tickers, period="1y"):
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print("Yahoo Finance Error:", exc)
        return {}, list(tickers)

    if data.empty:
        return {}, list(tickers)

    market_data = {}
    missing_tickers = []

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            frame = pd.DataFrame()

            for level in (1, 0):
                try:
                    frame = data.xs(ticker, axis=1, level=level, drop_level=True).copy()
                    break
                except KeyError:
                    frame = pd.DataFrame()

            frame = clean_market_frame(frame)
            if frame.empty:
                missing_tickers.append(ticker)
                continue

            market_data[ticker] = frame
    else:
        frame = clean_market_frame(data.copy())
        if frame.empty:
            return {}, list(tickers)
        market_data[tickers[0]] = frame

    missing_tickers.extend([ticker for ticker in tickers if ticker not in market_data])
    return market_data, missing_tickers


def fetch_company_metadata(ticker):
    ticker_object = yf.Ticker(ticker)

    try:
        fast_info = ticker_object.fast_info
    except Exception:
        fast_info = {}

    try:
        info = ticker_object.info or {}
    except Exception:
        info = {}

    market_cap = safe_float(safe_get(fast_info, "marketCap") or safe_get(info, "marketCap"))
    beta = safe_float(safe_get(info, "beta"))
    trailing_pe = safe_float(safe_get(info, "trailingPE"))
    forward_pe = safe_float(safe_get(info, "forwardPE"))
    dividend_yield = safe_float(safe_get(info, "dividendYield"))
    profit_margins = safe_float(safe_get(info, "profitMargins"))
    return_on_equity = safe_float(safe_get(info, "returnOnEquity"))
    revenue_growth = safe_float(safe_get(info, "revenueGrowth"))
    earnings_growth = safe_float(safe_get(info, "earningsGrowth"))
    debt_to_equity = safe_float(safe_get(info, "debtToEquity"))

    return {
        "ticker": ticker,
        "name": safe_get(info, "longName", "shortName") or ticker,
        "sector": safe_get(info, "sector"),
        "industry": safe_get(info, "industry"),
        "country": safe_get(info, "country"),
        "exchange": safe_get(info, "exchange") or safe_get(fast_info, "exchange"),
        "currency": safe_get(info, "currency") or safe_get(fast_info, "currency"),
        "market_cap": market_cap,
        "market_cap_display": compact_number(market_cap) if market_cap is not None else None,
        "beta": beta,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "dividend_yield": pct(dividend_yield),
        "profit_margins": pct(profit_margins),
        "return_on_equity": pct(return_on_equity),
        "revenue_growth": pct(revenue_growth),
        "earnings_growth": pct(earnings_growth),
        "debt_to_equity": debt_to_equity,
    }


def fetch_company_metadata_batch(tickers):
    metadata = {}

    for ticker in tickers:
        try:
            metadata[ticker] = fetch_company_metadata(ticker)
        except Exception as exc:
            print(f"Metadata Error for {ticker}:", exc)
            metadata[ticker] = {
                "ticker": ticker,
                "name": ticker,
            }

    return metadata


def calc_daily_returns(prices):
    return prices.pct_change().dropna()


def calc_annualised_return(returns):
    return float(returns.mean() * TRADING_DAYS_PER_YEAR)


def calc_annualised_volatility(returns):
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_downside_volatility(returns):
    downside = returns[returns < 0]
    if downside.empty:
        return 0.0
    return float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_sharpe_ratio(returns, risk_free_rate):
    volatility = calc_annualised_volatility(returns)
    if volatility == 0:
        return 0.0
    return float((calc_annualised_return(returns) - risk_free_rate) / volatility)


def calc_sortino_ratio(returns, risk_free_rate):
    downside_volatility = calc_downside_volatility(returns)
    if downside_volatility == 0:
        return 0.0
    return float((calc_annualised_return(returns) - risk_free_rate) / downside_volatility)


def calc_max_drawdown(prices):
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def calc_calmar_ratio(annual_return, max_drawdown):
    if max_drawdown == 0:
        return 0.0
    return float(annual_return / abs(max_drawdown))


def calculate_rsi(close, period=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


def calculate_macd(close):
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_atr(frame, period=14):
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def calculate_adx(frame, period=14):
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0)


def calculate_obv(frame):
    close = frame["Close"]
    volume = frame["Volume"].fillna(0)
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def build_indicator_snapshot(frame):
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"].fillna(0)

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    rsi_14 = calculate_rsi(close, 14)
    macd, macd_signal, macd_hist = calculate_macd(close)

    bollinger_mid = close.rolling(20).mean()
    bollinger_std = close.rolling(20).std()
    bollinger_upper = bollinger_mid + 2 * bollinger_std
    bollinger_lower = bollinger_mid - 2 * bollinger_std
    bollinger_position = (close - bollinger_lower) / (bollinger_upper - bollinger_lower).replace(0, np.nan)

    atr_14 = calculate_atr(frame, 14)
    atr_14_pct = atr_14 / close.replace(0, np.nan)

    lowest_low_14 = low.rolling(14).min()
    highest_high_14 = high.rolling(14).max()
    stochastic_k = 100 * (close - lowest_low_14) / (highest_high_14 - lowest_low_14).replace(0, np.nan)
    adx_14 = calculate_adx(frame, 14)

    obv = calculate_obv(frame)
    obv_ema_20 = obv.ewm(span=20, adjust=False).mean()
    obv_signal = (obv - obv_ema_20) / obv_ema_20.abs().replace(0, np.nan)

    volume_avg_20 = volume.rolling(20).mean()
    volume_ratio = volume / volume_avg_20.replace(0, np.nan)

    rolling_high_252 = close.rolling(252, min_periods=20).max()
    rolling_low_252 = close.rolling(252, min_periods=20).min()

    momentum_21 = close.pct_change(21)
    momentum_63 = close.pct_change(63)
    momentum_126 = close.pct_change(126)
    volatility_20 = close.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "close": safe_last(close),
        "sma_20": safe_last(sma_20),
        "sma_50": safe_last(sma_50),
        "sma_200": safe_last(sma_200),
        "ema_20": safe_last(ema_20),
        "ema_12": safe_last(ema_12),
        "ema_26": safe_last(ema_26),
        "rsi_14": safe_last(rsi_14),
        "macd": safe_last(macd),
        "macd_signal": safe_last(macd_signal),
        "macd_hist": safe_last(macd_hist),
        "bollinger_upper": safe_last(bollinger_upper),
        "bollinger_lower": safe_last(bollinger_lower),
        "bollinger_position": safe_last(bollinger_position),
        "atr_14_pct": pct(safe_last(atr_14_pct)),
        "stochastic_k": safe_last(stochastic_k),
        "adx_14": safe_last(adx_14),
        "volume_ratio": safe_last(volume_ratio),
        "obv_signal": safe_last(obv_signal),
        "momentum_21d": pct(safe_last(momentum_21)),
        "momentum_63d": pct(safe_last(momentum_63)),
        "momentum_126d": pct(safe_last(momentum_126)),
        "volatility_20d": pct(safe_last(volatility_20)),
        "distance_from_52w_high": pct((safe_last(close / rolling_high_252) - 1) if safe_last(rolling_high_252) else None),
        "distance_from_52w_low": pct((safe_last(close / rolling_low_252) - 1) if safe_last(rolling_low_252) else None),
    }


def build_series_metrics(returns, prices, risk_free_rate):
    annual_return = calc_annualised_return(returns)
    annualised_volatility = calc_annualised_volatility(returns)
    max_drawdown = calc_max_drawdown(prices)

    return {
        "annualised_volatility": round(annualised_volatility * 100, 2),
        "sharpe_ratio": round(calc_sharpe_ratio(returns, risk_free_rate), 4),
        "sortino_ratio": round(calc_sortino_ratio(returns, risk_free_rate), 4),
        "calmar_ratio": round(calc_calmar_ratio(annual_return, max_drawdown), 4),
        "max_drawdown": round(max_drawdown * 100, 2),
        "annualised_return": round(annual_return * 100, 2),
        "cumulative_return": round(float(((prices.iloc[-1] / prices.iloc[0]) - 1) * 100), 2),
    }


def build_portfolio_returns(daily_returns, weights):
    return pd.Series(
        daily_returns.values @ np.array(weights),
        index=daily_returns.index,
        name="PORTFOLIO",
    )


def score_fundamentals(metadata):
    quality_components = [
        clip_score(metadata.get("return_on_equity"), 0, 20),
        clip_score(metadata.get("profit_margins"), 0, 25),
        clip_score(metadata.get("revenue_growth"), -5, 25),
        clip_score(metadata.get("earnings_growth"), -5, 25),
        clip_score(metadata.get("dividend_yield"), 0, 4),
        1 - clip_score(metadata.get("debt_to_equity"), 0, 250) if metadata.get("debt_to_equity") is not None else None,
        1 - abs((metadata.get("beta") or 1) - 1) / 2 if metadata.get("beta") is not None else None,
    ]
    score = average_score(quality_components)
    return round(score * 100, 1)


def classify_trend(indicators):
    close = indicators.get("close")
    sma_20 = indicators.get("sma_20")
    sma_50 = indicators.get("sma_50")
    sma_200 = indicators.get("sma_200")

    if close and sma_20 and sma_50 and sma_200:
        if close > sma_20 > sma_50 > sma_200:
            return "strong uptrend"
        if close > sma_20 and close > sma_50 and sma_50 > sma_200:
            return "uptrend"
        if close < sma_20 < sma_50 < sma_200:
            return "strong downtrend"
        if close < sma_20 and close < sma_50 and sma_50 < sma_200:
            return "downtrend"
    return "mixed trend"


def classify_momentum(indicators):
    rsi = indicators.get("rsi_14")
    macd_hist = indicators.get("macd_hist")
    momentum_21d = indicators.get("momentum_21d")
    momentum_63d = indicators.get("momentum_63d")

    positive_signals = sum(
        [
            1 if (rsi is not None and 50 <= rsi <= 70) else 0,
            1 if (macd_hist is not None and macd_hist > 0) else 0,
            1 if (momentum_21d is not None and momentum_21d > 0) else 0,
            1 if (momentum_63d is not None and momentum_63d > 0) else 0,
        ]
    )

    if positive_signals >= 3:
        return "strong"
    if positive_signals == 2:
        return "balanced"
    return "weak"


def classify_risk(metrics, indicators):
    annualised_volatility = metrics.get("annualised_volatility")
    max_drawdown = abs(metrics.get("max_drawdown", 0))
    atr_14_pct = indicators.get("atr_14_pct")

    if annualised_volatility is None:
        return "unknown"

    high_risk_signals = sum(
        [
            1 if annualised_volatility > 35 else 0,
            1 if max_drawdown > 25 else 0,
            1 if (atr_14_pct is not None and atr_14_pct > 4.5) else 0,
        ]
    )

    if high_risk_signals >= 2:
        return "high"
    if high_risk_signals == 1 or annualised_volatility > 22:
        return "moderate"
    return "lower"


def build_indicator_scores(indicators, metrics, metadata):
    close = indicators.get("close")
    sma_20 = indicators.get("sma_20")
    sma_50 = indicators.get("sma_50")
    sma_200 = indicators.get("sma_200")

    trend_score = average_score(
        [
            1.0 if (close is not None and sma_20 is not None and close > sma_20) else 0.0 if close and sma_20 else None,
            1.0 if (close is not None and sma_50 is not None and close > sma_50) else 0.0 if close and sma_50 else None,
            1.0 if (sma_20 is not None and sma_50 is not None and sma_20 > sma_50) else 0.0 if sma_20 and sma_50 else None,
            1.0 if (sma_50 is not None and sma_200 is not None and sma_50 > sma_200) else 0.0 if sma_50 and sma_200 else None,
        ]
    )

    rsi = indicators.get("rsi_14")
    rsi_score = None
    if rsi is not None:
        if 45 <= rsi <= 65:
            rsi_score = 1.0
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            rsi_score = 0.65
        else:
            rsi_score = 0.3

    momentum_score = average_score(
        [
            rsi_score,
            clip_score(indicators.get("momentum_21d"), -10, 15),
            clip_score(indicators.get("momentum_63d"), -15, 25),
            clip_score(indicators.get("macd_hist"), -2, 2),
            clip_score(indicators.get("stochastic_k"), 20, 80),
        ]
    )

    volume_score = average_score(
        [
            clip_score(indicators.get("volume_ratio"), 0.8, 1.5),
            clip_score(indicators.get("obv_signal"), -0.15, 0.15),
            clip_score(indicators.get("adx_14"), 15, 35),
        ]
    )

    risk_score = average_score(
        [
            1 - clip_score(metrics.get("annualised_volatility"), 12, 45)
            if metrics.get("annualised_volatility") is not None else None,
            1 - clip_score(abs(metrics.get("max_drawdown", 0)), 8, 35),
            1 - clip_score(indicators.get("atr_14_pct"), 1.5, 6) if indicators.get("atr_14_pct") is not None else None,
        ]
    )

    technical_score = round(
        (
            0.35 * trend_score +
            0.30 * momentum_score +
            0.15 * volume_score +
            0.20 * risk_score
        ) * 100,
        1,
    )

    quality_score = score_fundamentals(metadata)
    risk_adjusted_score = round(
        average_score(
            [
                clip_score(metrics.get("sharpe_ratio"), -0.5, 2.0),
                clip_score(metrics.get("sortino_ratio"), -0.5, 3.0),
                clip_score(metrics.get("annualised_return"), -15, 30),
                1 - clip_score(metrics.get("annualised_volatility"), 12, 45)
                if metrics.get("annualised_volatility") is not None else None,
            ]
        ) * 100,
        1,
    )

    selection_score = round(
        (
            0.45 * (technical_score / 100) +
            0.30 * (risk_adjusted_score / 100) +
            0.25 * (quality_score / 100)
        ) * 100,
        1,
    )

    return {
        "technical_score": technical_score,
        "quality_score": quality_score,
        "risk_adjusted_score": risk_adjusted_score,
        "selection_score": selection_score,
        "trend_label": classify_trend(indicators),
        "momentum_label": classify_momentum(indicators),
        "risk_label": classify_risk(metrics, indicators),
    }


def build_stock_summary(ticker, metadata, metrics, indicators, scores):
    name = metadata.get("name") or ticker
    trend = scores["trend_label"]
    momentum = scores["momentum_label"]
    risk = scores["risk_label"]
    rsi = indicators.get("rsi_14")
    drawdown = metrics.get("max_drawdown")
    distance_high = indicators.get("distance_from_52w_high")
    sector = metadata.get("sector")

    sector_fragment = f" in the {sector} sector" if sector else ""
    sentence_one = (
        f"{name}{sector_fragment} is currently in a {trend} with {momentum} momentum."
    )

    sentence_two = (
        f"It has returned {format_percentage(metrics.get('annualised_return'))} annualised with "
        f"{format_percentage(metrics.get('annualised_volatility'))} volatility and a "
        f"Sharpe ratio of {format_ratio(metrics.get('sharpe_ratio'))}."
    )

    sentence_three = (
        f"RSI is {format_ratio(rsi, 1)}, max drawdown is {format_percentage(drawdown)}, "
        f"and the stock is {format_percentage(abs(distance_high) if distance_high is not None else None)} "
        f"away from its 52-week high."
    )

    sentence_four = (
        f"Overall risk looks {risk}, while the model gives it a technical score of "
        f"{format_ratio(scores['technical_score'], 1)} out of 100 and an overall selection score of "
        f"{format_ratio(scores['selection_score'], 1)}."
    )

    return " ".join([sentence_one, sentence_two, sentence_three, sentence_four])


def analyse_stock(ticker, frame, metadata, risk_free_rate):
    close_prices = frame["Close"].dropna()
    returns = calc_daily_returns(close_prices.to_frame(name=ticker))[ticker]
    metrics = build_series_metrics(returns, close_prices, risk_free_rate)
    indicators = build_indicator_snapshot(frame)
    scores = build_indicator_scores(indicators, metrics, metadata)

    return {
        **metrics,
        **scores,
        "technical_snapshot": indicators,
        "profile": metadata,
        "summary": build_stock_summary(ticker, metadata, metrics, indicators, scores),
    }


def optimise_portfolio_weights(prices, stock_analyses, risk_free_rate, samples=OPTIMISATION_SAMPLES):
    daily_returns = calc_daily_returns(prices)
    ticker_count = len(prices.columns)

    if daily_returns.empty:
        weights = [1 / ticker_count] * ticker_count
        return (
            weights,
            {
                "model": "equal_weight_fallback",
                "objective": "fallback_due_to_missing_returns",
                "simulations": 0,
            },
            pd.Series(dtype=float),
        )

    if ticker_count == 1:
        portfolio_returns = daily_returns.iloc[:, 0].copy()
        portfolio_returns.name = "PORTFOLIO"
        return (
            [1.0],
            {
                "model": "single_asset",
                "objective": "only_one_valid_ticker",
                "simulations": 0,
            },
            portfolio_returns,
        )

    mean_returns = daily_returns.mean().values * TRADING_DAYS_PER_YEAR
    covariance = daily_returns.cov().values * TRADING_DAYS_PER_YEAR
    signal_strengths = np.array(
        [
            (stock_analyses[ticker]["selection_score"] or 50) / 100
            for ticker in prices.columns
        ]
    )
    preference_weights = signal_strengths / signal_strengths.sum()

    rng = np.random.default_rng(42)
    candidate_weights = rng.dirichlet(np.maximum(signal_strengths * 8, 1.2), size=samples)

    equal_weight = np.full((1, ticker_count), 1 / ticker_count)
    inverse_vol = 1 / np.maximum(daily_returns.std().values, 1e-12)
    inverse_vol = (inverse_vol / inverse_vol.sum()).reshape(1, -1)
    preference = preference_weights.reshape(1, -1)

    candidate_weights = np.vstack([equal_weight, inverse_vol, preference, candidate_weights])
    candidate_returns = candidate_weights @ mean_returns
    candidate_volatility = np.sqrt(
        np.einsum("ij,jk,ik->i", candidate_weights, covariance, candidate_weights)
    )
    candidate_sharpe = np.where(
        candidate_volatility > 0,
        (candidate_returns - risk_free_rate) / candidate_volatility,
        -np.inf,
    )
    weighted_signal = candidate_weights @ signal_strengths
    concentration_penalty = np.sum(np.square(candidate_weights), axis=1)
    objective = candidate_sharpe + (0.55 * weighted_signal) - (0.20 * concentration_penalty)

    best_index = int(np.argmax(objective))
    best_weights = candidate_weights[best_index]
    portfolio_returns = build_portfolio_returns(daily_returns, best_weights.tolist())

    return (
        best_weights.tolist(),
        {
            "model": "multi_factor_max_sharpe",
            "objective": "blend_risk_adjusted_return_with_technical_and_quality_signals",
            "simulations": int(candidate_weights.shape[0]),
            "estimated_sharpe": round(float(candidate_sharpe[best_index]), 4),
            "weighted_signal_score": round(float(weighted_signal[best_index] * 100), 1),
        },
        portfolio_returns,
    )


def calc_portfolio_metrics(prices, weights, risk_free_rate):
    daily_returns = calc_daily_returns(prices)

    if daily_returns.empty:
        return {
            "annualised_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "annualised_return": 0.0,
            "cumulative_return": 0.0,
        }

    portfolio_returns = build_portfolio_returns(daily_returns, weights)
    portfolio_price = (1 + portfolio_returns).cumprod()
    return build_series_metrics(portfolio_returns, portfolio_price, risk_free_rate)


def build_price_history(prices, portfolio_returns=None):
    price_history = {}

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            continue

        normalised = (series / series.iloc[0] * 100).round(2)
        price_history[ticker] = {
            "dates": [str(index.date()) for index in normalised.index],
            "values": normalised.tolist(),
        }

    if portfolio_returns is not None and not portfolio_returns.empty:
        portfolio_curve = ((1 + portfolio_returns).cumprod() * 100).round(2)
        price_history["PORTFOLIO"] = {
            "dates": [str(index.date()) for index in portfolio_curve.index],
            "values": portfolio_curve.tolist(),
        }

    return price_history


def build_chart_summary(price_history, weights, stock_analyses):
    stock_curves = {
        ticker: history for ticker, history in price_history.items()
        if ticker != "PORTFOLIO" and history["values"]
    }

    leader = None
    laggard = None

    if stock_curves:
        ranked = sorted(
            stock_curves.items(),
            key=lambda item: item[1]["values"][-1],
            reverse=True,
        )
        leader = ranked[0]
        laggard = ranked[-1]

    top_holding = max(weights.items(), key=lambda item: item[1]) if weights else None
    portfolio_finish = price_history.get("PORTFOLIO", {}).get("values", [100])[-1]

    bullets = [
        "Every line starts at 100, so the chart compares percentage performance rather than the raw share price.",
        (
            f"The model portfolio finished around {portfolio_finish:.1f}, which means it moved "
            f"{portfolio_finish - 100:+.1f}% over the selected period."
        ),
    ]

    if leader and laggard:
        leader_return = leader[1]["values"][-1] - 100
        laggard_return = laggard[1]["values"][-1] - 100
        bullets.append(
            f"{leader[0]} was the strongest trend at {leader_return:+.1f}%, while {laggard[0]} was the weakest at {laggard_return:+.1f}%."
        )

    if top_holding:
        bullets.append(
            f"The largest model weight is {top_holding[0]} at {top_holding[1]:.1f}% because it scored well on trend, momentum, and risk-adjusted return."
        )

    return {
        "headline": "How to read the chart",
        "overview": (
            "This chart is designed to make different stocks easy to compare at a glance. "
            "A stock priced at 50 and another priced at 3,000 both start from the same base value, so you can see relative performance clearly."
        ),
        "bullets": bullets,
    }


def build_portfolio_story(stock_analyses, weights, portfolio_metrics, allocation_model):
    weighted_technical = sum(
        stock_analyses[ticker]["technical_score"] * (weight / 100)
        for ticker, weight in weights.items()
    )
    weighted_quality = sum(
        stock_analyses[ticker]["quality_score"] * (weight / 100)
        for ticker, weight in weights.items()
    )
    weighted_rsi = sum(
        (stock_analyses[ticker]["technical_snapshot"].get("rsi_14") or 50) * (weight / 100)
        for ticker, weight in weights.items()
    )

    top_holdings = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
    top_sentence = ", ".join([f"{ticker} ({weight:.1f}%)" for ticker, weight in top_holdings])

    risk_label = classify_risk(portfolio_metrics, {"atr_14_pct": None})

    bullets = [
        f"The optimiser is using a multi-factor model that blends technical strength, quality signals, and risk-adjusted returns across {allocation_model.get('simulations', 0)} candidate portfolios.",
        f"The biggest model allocations are {top_sentence}. Those names scored best after combining momentum, trend, volatility, and available company-quality fields from yfinance.",
        f"The portfolio's blended technical score is {weighted_technical:.1f}/100, the blended quality score is {weighted_quality:.1f}/100, and the blended RSI is {weighted_rsi:.1f}.",
        f"Overall portfolio risk looks {risk_label}, with {portfolio_metrics.get('annualised_volatility', 0):.1f}% annualised volatility and a {portfolio_metrics.get('max_drawdown', 0):.1f}% max drawdown.",
    ]

    return {
        "headline": "What the model is seeing",
        "overview": (
            "This is no longer just a simple price-comparison tool. The model now looks at trend, momentum, volatility, drawdown, volume behaviour, and selected yfinance company fields before deciding how much weight to place on each stock."
        ),
        "bullets": bullets,
    }


def build_portfolio_explainers(stock_analyses, weights, portfolio_metrics, allocation_model, price_history):
    return {
        "portfolio_story": build_portfolio_story(
            stock_analyses,
            weights,
            portfolio_metrics,
            allocation_model,
        ),
        "chart_summary": build_chart_summary(
            price_history,
            weights,
            stock_analyses,
        ),
    }


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/import/screener", methods=["POST"])
def import_screener_csv():
    try:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return build_error("Please upload a Screener CSV file")

        preferred_exchange = str(request.form.get("exchange", "AUTO")).upper()
        if preferred_exchange not in {"AUTO", "NSE", "BSE"}:
            return build_error("CSV import exchange must be one of AUTO, NSE, BSE")

        limit = parse_csv_limit(request.form.get("limit"))
        imported = parse_screener_csv(upload, preferred_exchange, limit)

        return jsonify(
            {
                "source": "screener_csv",
                "exchange": preferred_exchange,
                **imported,
            }
        )
    except ValueError as exc:
        return build_error(str(exc))
    except Exception as exc:
        print("CSV Import Error:", exc)
        return build_error("Unexpected server error while importing CSV", 500)


@app.route("/api/analyse", methods=["POST"])
def analyse():
    try:
        body = request.get_json(silent=True) or {}

        exchange = str(body.get("exchange", "AUTO")).upper()
        if exchange not in ALLOWED_EXCHANGES:
            return build_error("Exchange must be one of AUTO, US, NSE, BSE")

        period = str(body.get("period", "1y"))
        if period not in ALLOWED_PERIODS:
            return build_error("Period must be one of 3mo, 6mo, 1y, 2y, 5y")

        tickers = parse_tickers(body.get("tickers", []), exchange)
        risk_free_rate = parse_risk_free_rate(body.get("risk_free_rate"))

        market_data, missing_tickers = fetch_market_data(tickers, period)
        if not market_data:
            return build_error("Failed to fetch market data from Yahoo Finance", 502)

        analysable_tickers = list(market_data.keys())
        metadata = fetch_company_metadata_batch(analysable_tickers)
        stock_analyses = {}

        for ticker in analysable_tickers:
            try:
                stock_analyses[ticker] = analyse_stock(
                    ticker,
                    market_data[ticker],
                    metadata.get(ticker, {"ticker": ticker, "name": ticker}),
                    risk_free_rate,
                )
            except Exception as exc:
                print(f"Stock Analysis Error for {ticker}:", exc)
                stock_analyses[ticker] = {"error": "No market data available"}

        valid_tickers = [
            ticker for ticker in analysable_tickers
            if "error" not in stock_analyses[ticker]
        ]

        if not valid_tickers:
            return build_error("No valid stock data available", 502)

        close_prices = pd.concat(
            [market_data[ticker]["Close"].rename(ticker) for ticker in valid_tickers],
            axis=1,
        ).dropna(how="all")
        daily_returns = calc_daily_returns(close_prices)

        optimised_weights, allocation_model, portfolio_returns = optimise_portfolio_weights(
            close_prices[valid_tickers],
            stock_analyses,
            risk_free_rate,
        )

        weights = {
            ticker: round(optimised_weights[index] * 100, 2)
            for index, ticker in enumerate(valid_tickers)
        }
        portfolio_metrics = calc_portfolio_metrics(
            close_prices[valid_tickers],
            optimised_weights,
            risk_free_rate,
        )
        price_history = build_price_history(close_prices[valid_tickers], portfolio_returns)
        explainers = build_portfolio_explainers(
            {ticker: stock_analyses[ticker] for ticker in valid_tickers},
            weights,
            portfolio_metrics,
            allocation_model,
            price_history,
        )

        return jsonify(
            {
                "requested_tickers": tickers,
                "resolved_tickers": valid_tickers,
                "invalid_tickers": missing_tickers,
                "exchange": exchange,
                "period": period,
                "risk_free_rate": round(risk_free_rate * 100, 2),
                "allocation_model": allocation_model,
                "weights": weights,
                "stock_metrics": stock_analyses,
                "portfolio_metrics": portfolio_metrics,
                "price_history": price_history,
                "correlation_matrix": daily_returns[valid_tickers].corr().round(3).fillna(0).to_dict(),
                **explainers,
            }
        )

    except ValueError as exc:
        return build_error(str(exc))
    except Exception as exc:
        print("Backend Error:", exc)
        return build_error("Unexpected server error", 500)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5001")),
        debug=True,
    )
