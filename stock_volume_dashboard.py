import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

st.set_page_config(page_title="Nifty 50 Volume Console", layout="wide")

import requests
import io

# Indices Mapping
INDICES_MAP = {
    "Nifty 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty Next 50": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "Nifty 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "Nifty 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "Nifty Smallcap 50": "https://archives.nseindia.com/content/indices/ind_niftysmallcap50list.csv",
    "Nifty Smallcap 100": "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    "Nifty Smallcap 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "Nifty Midcap 50": "https://archives.nseindia.com/content/indices/ind_niftymidcap50list.csv",
    "Nifty Midcap 100": "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    "Nifty Midcap 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "F&O Stocks": "FNO_SPECIAL"  # Special marker for F&O stocks (requires NSE API scraping)
}

st.sidebar.title("Configuration")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES_MAP.keys()))

def get_fno_tickers():
    """
    Fetch F&O (Derivatives) stocks list from NSE India.
    Uses the NSE API endpoint for "Securities in F&O" segment.
    Requires establishing a session with NSE first to get cookies.
    """
    try:
        # Create a session to maintain cookies
        session = requests.Session()

        # Headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }

        # First visit NSE homepage to establish session and get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # Use the equity-stockIndices API with F&O filter (tested and working)
        headers['Referer'] = 'https://www.nseindia.com/market-data/live-equity-market'
        fno_api_url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        response = session.get(fno_api_url, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            fno_symbols = []
            if 'data' in data:
                for item in data['data']:
                    symbol = item.get('symbol')
                    # Filter out index entries (like NIFTY 50, etc.)
                    if symbol and not symbol.startswith('NIFTY') and not symbol.startswith('BANK'):
                        fno_symbols.append(f"{symbol}.NS")

            if fno_symbols:
                return fno_symbols

        # Fallback: Scrape from HTML page if API fails
        page_url = "https://www.nseindia.com/products-services/equity-derivatives-list-underlyings-information"
        response = session.get(page_url, headers=headers, timeout=15)

        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            fno_symbols = []
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        symbol_text = cells[0].get_text(strip=True)
                        if symbol_text and symbol_text.isalpha() and symbol_text.isupper():
                            fno_symbols.append(f"{symbol_text}.NS")

            if fno_symbols:
                return list(set(fno_symbols))

        return []

    except Exception as e:
        st.warning(f"Error fetching F&O stocks: {e}")
        return []


def get_tickers(index_name):
    # Try DB first
    cached_tickers = get_tickers_from_db(index_name)
    if cached_tickers:
        return cached_tickers

    # Special handling for F&O Stocks
    if index_name == "F&O Stocks":
        tickers = get_fno_tickers()
        if tickers:
            save_tickers_to_db(index_name, tickers)
        return tickers

    # Fetch from NSE (regular indices)
    url = INDICES_MAP[index_name]
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            # Filter out dummy/invalid tickers
            tickers = [f"{symbol}.NS" for symbol in df['Symbol'].tolist()
                       if not str(symbol).startswith('DUMMY')]

            # Save to DB
            if tickers:
                save_tickers_to_db(index_name, tickers)

            return tickers
        else:
            st.error(f"Failed to fetch data from NSE. Status Code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching ticker list for {index_name}: {e}")
        return []

import sqlite3
from datetime import datetime, timedelta

# Database Functions
def init_db():
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stock_prices 
                 (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, index_group TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS cache_tracking
                 (index_group TEXT PRIMARY KEY, last_updated TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ticker_lists
                 (index_name TEXT, symbol TEXT, last_updated TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS market_breadth
                 (index_group TEXT, date TEXT, pct_above_sma20 REAL, pct_above_sma50 REAL,
                  pct_above_sma100 REAL, pct_above_sma200 REAL, total_stocks INTEGER,
                  PRIMARY KEY (index_group, date))''')
    conn.commit()
    conn.close()

def get_tickers_from_db(index_name):
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    # Check age
    c.execute("SELECT last_updated FROM ticker_lists WHERE index_name = ? LIMIT 1", (index_name,))
    row = c.fetchone()
    
    tickers = []
    if row:
        last_updated = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
        if datetime.now() - last_updated < timedelta(days=7):
            c.execute("SELECT symbol FROM ticker_lists WHERE index_name = ?", (index_name,))
            tickers = [row[0] for row in c.fetchall()]
            
    conn.close()
    return tickers

def save_tickers_to_db(index_name, tickers):
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    # Clear old list
    c.execute("DELETE FROM ticker_lists WHERE index_name = ?", (index_name,))
    
    now = datetime.now()
    for ticker in tickers:
        c.execute("INSERT INTO ticker_lists VALUES (?, ?, ?)", (index_name, ticker, now))
    
    conn.commit()
    conn.close()


def is_cache_valid(index_group):
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    c.execute("SELECT last_updated FROM cache_tracking WHERE index_group = ?", (index_group,))
    row = c.fetchone()
    conn.close()
    
    if row:
        last_updated = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
        if datetime.now() - last_updated < timedelta(minutes=15):
            return True
    return False

def save_to_db(data, index_group, tickers):
    """
    Save stock data to DB using UPSERT pattern.
    Accumulates historical data instead of replacing it.
    """
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()

    # Create unique index if not exists (for UPSERT to work)
    c.execute('''CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_prices_unique
                 ON stock_prices(ticker, date, index_group)''')

    for ticker in tickers:
        try:
            if ticker in data.columns:
                df = data[ticker].copy().dropna()
                df = df.reset_index()
                for _, row in df.iterrows():
                    # Use INSERT OR REPLACE to update existing records or insert new ones
                    c.execute("""INSERT OR REPLACE INTO stock_prices
                                 (ticker, date, open, high, low, close, volume, index_group)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                              (ticker, row['Date'].strftime('%Y-%m-%d'), row.get('Open', 0), row.get('High', 0),
                               row.get('Low', 0), row.get('Close', 0), row.get('Volume', 0), index_group))
        except Exception:
            continue

    # Update timestamp
    c.execute("INSERT OR REPLACE INTO cache_tracking VALUES (?, ?)", (index_group, datetime.now()))
    conn.commit()
    conn.close()


def get_historical_data(ticker, index_group, days=60):
    """
    Get historical data for a specific ticker from DB.
    Returns DataFrame with OHLCV data sorted by date.
    """
    conn = sqlite3.connect('stocks.db')
    query = """
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE ticker = ? AND index_group = ?
        ORDER BY date DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(ticker, index_group, days))
    conn.close()

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    return df


def compute_sma_status(df):
    """
    Compute SMA values and check if current price is above each SMA.
    Returns dict with SMA values and above/below status.
    """
    if df.empty or len(df) < 200:
        return None

    close = df['close'] if 'close' in df.columns else df['Close']
    current_price = close.iloc[-1]

    # Calculate SMAs
    sma20 = close.rolling(window=20).mean().iloc[-1]
    sma50 = close.rolling(window=50).mean().iloc[-1]
    sma100 = close.rolling(window=100).mean().iloc[-1]
    sma200 = close.rolling(window=200).mean().iloc[-1]

    return {
        'current_price': current_price,
        'sma20': sma20,
        'sma50': sma50,
        'sma100': sma100,
        'sma200': sma200,
        'above_sma20': current_price > sma20 if pd.notna(sma20) else None,
        'above_sma50': current_price > sma50 if pd.notna(sma50) else None,
        'above_sma100': current_price > sma100 if pd.notna(sma100) else None,
        'above_sma200': current_price > sma200 if pd.notna(sma200) else None,
    }


def compute_technical_indicators(df, nifty_df=None, ticker_name=None, debug=False):
    """
    Compute technical indicators for a stock DataFrame.
    Returns partial results based on available data (N/A for indicators needing more data).

    Args:
        df: DataFrame with 'close' column
        nifty_df: DataFrame with Nifty 50 'close' column for relative strength
        ticker_name: Optional ticker name for debug logging
        debug: If True, print debug info

    Returns:
        dict with RS55 (vs Nifty), EMA21, EMA55, EMA100, EMA200, RSI14
    """
    result = {'rs55': None, 'ema21': None, 'ema55': None, 'ema100': None, 'ema200': None, 'rsi14': None}

    if df.empty or len(df) < 14:
        return result

    close = df['close']
    data_len = len(df)

    # EMA calculations - only compute if enough data
    if data_len >= 21:
        ema21 = close.ewm(span=21, adjust=True).mean().iloc[-1]
        result['ema21'] = round(ema21, 2) if pd.notna(ema21) else None
    if data_len >= 55:
        ema55 = close.ewm(span=55, adjust=True).mean().iloc[-1]
        result['ema55'] = round(ema55, 2) if pd.notna(ema55) else None
    if data_len >= 100:
        ema100 = close.ewm(span=100, adjust=True).mean().iloc[-1]
        result['ema100'] = round(ema100, 2) if pd.notna(ema100) else None
    if data_len >= 200:
        ema200 = close.ewm(span=200, adjust=True).mean().iloc[-1]
        result['ema200'] = round(ema200, 2) if pd.notna(ema200) else None

    # RSI 14
    if data_len >= 14:
        rsi14_indicator = RSIIndicator(close=close, window=14)
        rsi14 = rsi14_indicator.rsi().iloc[-1]
        result['rsi14'] = round(rsi14, 2) if pd.notna(rsi14) else None

    # Relative RS55 - Stock performance vs Nifty 50
    # Formula: (baseSymbol / baseSymbol[55]) / (comparativeSymbol / comparativeSymbol[55]) - 1
    # IMPORTANT: Align dates between stock and Nifty to ensure same trading days
    if data_len >= 56 and nifty_df is not None and len(nifty_df) >= 56:
        try:
            # Prepare stock data with date index
            stock_df = df[['close']].copy()
            if 'date' in df.columns:
                stock_df['date'] = pd.to_datetime(df['date'])
                stock_df = stock_df.set_index('date')
            stock_df = stock_df.rename(columns={'close': 'stock_close'})

            # Prepare nifty data with date index
            nifty_aligned = nifty_df[['close', 'date']].copy()
            nifty_aligned['date'] = pd.to_datetime(nifty_aligned['date'])
            nifty_aligned = nifty_aligned.set_index('date')
            nifty_aligned = nifty_aligned.rename(columns={'close': 'nifty_close'})

            # Merge on date (inner join to get only matching dates)
            merged = stock_df.join(nifty_aligned, how='inner')

            if len(merged) >= 56:
                stock_now = merged['stock_close'].iloc[-1]
                stock_55_ago = merged['stock_close'].iloc[-55]
                nifty_now = merged['nifty_close'].iloc[-1]
                nifty_55_ago = merged['nifty_close'].iloc[-55]

                if nifty_55_ago > 0 and nifty_now > 0 and stock_55_ago > 0:
                    rs55 = (stock_now / stock_55_ago) / (nifty_now / nifty_55_ago) - 1
                    result['rs55'] = round(rs55, 2)

                    # Debug logging (prints to console)
                    if debug and ticker_name:
                        print(f"RS55 Debug [{ticker_name}] - Stock: {stock_now:.2f}/{stock_55_ago:.2f}, "
                              f"Nifty: {nifty_now:.2f}/{nifty_55_ago:.2f}, "
                              f"RS55={rs55:.4f}")
        except Exception as e:
            if debug and ticker_name:
                print(f"RS55 Error [{ticker_name}]: {e}")

    return result


def load_from_db(index_group):
    conn = sqlite3.connect('stocks.db')
    query = "SELECT * FROM stock_prices WHERE index_group = ?"
    df = pd.read_sql(query, conn, params=(index_group,))
    conn.close()

    if df.empty:
        return None

    # Reconstruct dictionary of DataFrames similar to yf structure
    # But since we only iterate, we can return the big DF and filter in Python, or grouped
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_latest_date_in_db(index_group):
    """
    Get the most recent date stored in DB for an index group.
    Returns None if no data exists.
    """
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    c.execute("SELECT MAX(date) FROM stock_prices WHERE index_group = ?", (index_group,))
    row = c.fetchone()
    conn.close()

    if row and row[0]:
        return datetime.strptime(row[0], '%Y-%m-%d')
    return None


def get_data_count_in_db(index_group):
    """
    Get count of unique dates in DB for an index group.
    Used to determine if we have enough historical data.
    """
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT date) FROM stock_prices WHERE index_group = ?", (index_group,))
    row = c.fetchone()
    conn.close()

    return row[0] if row else 0

# Initialize DB on start
init_db()

tickers = get_tickers(selected_index)
st.caption(f"Analyzing {len(tickers)} stocks from {selected_index}")

@st.cache_data(ttl=600)
def fetch_data(ticker_list, index_group):
    all_results = []
    
    if not ticker_list:
        return pd.DataFrame(), None

    cached_df = None
    use_cache = is_cache_valid(index_group)
    
    if use_cache:
        logging.info(f"Cache HIT for {index_group}")
        cached_df = load_from_db(index_group)
    
    if cached_df is None:
        # Cache miss or stale -> Check if we need full fetch or incremental
        latest_date = get_latest_date_in_db(index_group)
        data_count = get_data_count_in_db(index_group)

        if latest_date and data_count >= 200:
            # We have enough historical data, do incremental fetch (last 5 days to catch up)
            days_since = (datetime.now() - latest_date).days
            fetch_period = max(days_since + 2, 5)  # At least 5 days, or days since last + buffer
            logging.info(f"Incremental fetch for {index_group}: {fetch_period} days (have {data_count} days in DB)")
            data = yf.download(ticker_list, period=f"{fetch_period}d", interval="1d", group_by='ticker')
        else:
            # First time or not enough data -> Full fetch
            logging.info(f"Full fetch for {index_group}: 250 days (have {data_count} days in DB)")
            data = yf.download(ticker_list, period="250d", interval="1d", group_by='ticker')

        # Save to DB (UPSERT will handle duplicates)
        save_to_db(data, index_group, ticker_list)
        last_updated_time = datetime.now()

        # Reload from DB to get complete dataset
        cached_df = load_from_db(index_group)
    else:
        # Get timestamp from DB
        conn = sqlite3.connect('stocks.db')
        c = conn.cursor()
        c.execute("SELECT last_updated FROM cache_tracking WHERE index_group = ?", (index_group,))
        row = c.fetchone()
        conn.close()
        if row:
            last_updated_time = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
        else:
            last_updated_time = datetime.now()

    last_date = None

    # Fetch Nifty 50 data for relative strength calculation
    nifty_df = None
    try:
        nifty_data = yf.download("^NSEI", period="250d", interval="1d", progress=False)
        if not nifty_data.empty:
            # Handle MultiIndex columns from yfinance
            if isinstance(nifty_data.columns, pd.MultiIndex):
                nifty_data.columns = nifty_data.columns.get_level_values(0)
            nifty_df = nifty_data.reset_index()
            nifty_df = nifty_df.rename(columns={'Close': 'close', 'Date': 'date'})
            nifty_df = nifty_df.sort_values('date').reset_index(drop=True)
            logging.info(f"Fetched Nifty 50 data: {len(nifty_df)} days")
    except Exception as e:
        logging.warning(f"Could not fetch Nifty 50 data: {e}")

    debug_count = 0  # Counter for debug logging (first 3 tickers only)
    for ticker in ticker_list:
        try:
            df = pd.DataFrame()

            if cached_df is not None:
                # Filter from cached big table
                # Column names in DB: ticker, date, open, high, low, close, volume
                # Need to Rename to title case for compatibility with logic below
                ticker_data = cached_df[cached_df['ticker'] == ticker].copy()
                if not ticker_data.empty:
                    df = ticker_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df = df.set_index('date').sort_index()
            else:
                # Use YF data
                if ticker in data.columns:
                     df = data[ticker].copy().dropna()
            
            if df.empty:
                continue

            if len(df) >= 21:  # Need at least 21 days for volume analysis (indicators show N/A if insufficient data)
                # Current volume is the last row
                curr_vol = float(df['Volume'].iloc[-1])
                curr_date = df.index[-1].strftime('%Y-%m-%d')
                
                # Previous 20 days average (excluding today)
                # slice from -21 to -1 (last 20 rows before the final one)
                avg_vol = float(df['Volume'].iloc[-21:-1].mean())
                
                vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0

                # Price Change Calculation
                curr_close = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                price_change = ((curr_close - prev_close) / prev_close) * 100

                # Net Volume (Money Flow) Estimation
                # ( (Close - Low) - (High - Close) ) / (High - Low) * Volume
                high = float(df['High'].iloc[-1])
                low = float(df['Low'].iloc[-1])
                
                if high != low:
                    mfv_multiplier = ((curr_close - low) - (high - curr_close)) / (high - low)
                    net_vol = mfv_multiplier * curr_vol
                else:
                    net_vol = 0

                # Compute technical indicators (RS55, EMA21, EMA55, RSI14)
                # Need lowercase column names for compute_technical_indicators
                df_for_indicators = df.rename(columns={'Close': 'close'})
                should_debug = debug_count < 3  # Debug first 3 tickers only
                indicators = compute_technical_indicators(df_for_indicators, nifty_df, ticker_name=ticker, debug=should_debug)
                debug_count += 1

                # Calculate EMA proximity to price (for coloring)
                # Positive = price above EMA (bullish), Negative = price below EMA (bearish)
                ema21_prox = ((curr_close / indicators['ema21']) - 1) * 100 if indicators['ema21'] else None
                ema55_prox = ((curr_close / indicators['ema55']) - 1) * 100 if indicators['ema55'] else None
                ema100_prox = ((curr_close / indicators['ema100']) - 1) * 100 if indicators['ema100'] else None
                ema200_prox = ((curr_close / indicators['ema200']) - 1) * 100 if indicators['ema200'] else None

                all_results.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "LTP": round(curr_close, 2),
                    "Price Change %": round(price_change, 2),
                    "Current Volume": curr_vol,
                    "20D Avg Vol (Prev)": avg_vol,
                    "Multiplier": round(vol_ratio, 2),
                    "Net Vol Estimate": net_vol,
                    "RS55 vs Nifty": indicators['rs55'],
                    "EMA21": indicators['ema21'],
                    "EMA55": indicators['ema55'],
                    "EMA100": indicators['ema100'],
                    "EMA200": indicators['ema200'],
                    "EMA21 %": round(ema21_prox, 2) if ema21_prox else None,
                    "EMA55 %": round(ema55_prox, 2) if ema55_prox else None,
                    "EMA100 %": round(ema100_prox, 2) if ema100_prox else None,
                    "EMA200 %": round(ema200_prox, 2) if ema200_prox else None,
                    "RSI14": indicators['rsi14'],
                    "Last Date": curr_date
                })
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue
            
    return pd.DataFrame(all_results), last_updated_time

import logging
try:
    import streamlit_analytics2 as streamlit_analytics
except ImportError:
    import streamlit_analytics
import pytz

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def to_ist(dt_obj):
    if dt_obj is None:
        return None
    # If naive, assume UTC (since we store UTC-ish or naive in DB, usually system time)
    # Actually DB saves naive local time. Let's assume system is UTC or convert appropriately.
    # To be safe: Localize to system time then convert to IST.
    # Simpler: Just force assume it implies local system time, convert to IST.
    utc_plus_5_30 = pytz.timezone('Asia/Kolkata')
    
    if dt_obj.tzinfo is None:
        # Assume the server time is what matches the DB time.
        # Ideally we should store UTC. But for now, let's localize to naive and convert.
        # If server is GMT, we just add 5:30.
        # If we run on Streamlit Cloud (UTC), this works.
        dt_obj = pytz.utc.localize(dt_obj)
        
    return dt_obj.astimezone(utc_plus_5_30)


@st.cache_data(ttl=600)
def calculate_market_breadth(ticker_list, index_group):
    """
    Calculate market breadth - percentage of stocks above each SMA.
    Calculates historical data for line chart and stores in DB.
    Returns dict with current data, historical data, and per-stock details.
    """
    if not ticker_list:
        return None

    # Load data from DB
    cached_df = load_from_db(index_group)
    if cached_df is None:
        return None

    # Get all unique dates (need at least 200 days for SMA200)
    all_dates = sorted(cached_df['date'].unique())
    if len(all_dates) < 200:
        return None

    # Calculate breadth for each date (last 50 trading days for the chart)
    dates_to_calc = all_dates[-50:]
    historical_breadth = []

    for calc_date in dates_to_calc:
        above_sma20 = 0
        above_sma50 = 0
        above_sma100 = 0
        above_sma200 = 0
        total_valid = 0

        for ticker in ticker_list:
            try:
                ticker_data = cached_df[cached_df['ticker'] == ticker].copy()
                ticker_data = ticker_data.sort_values('date')

                # Get data up to calc_date
                ticker_data = ticker_data[ticker_data['date'] <= calc_date]
                if len(ticker_data) < 200:
                    continue

                close = ticker_data['close']
                current_price = close.iloc[-1]

                sma20 = close.rolling(window=20).mean().iloc[-1]
                sma50 = close.rolling(window=50).mean().iloc[-1]
                sma100 = close.rolling(window=100).mean().iloc[-1]
                sma200 = close.rolling(window=200).mean().iloc[-1]

                total_valid += 1

                if pd.notna(sma20) and current_price > sma20:
                    above_sma20 += 1
                if pd.notna(sma50) and current_price > sma50:
                    above_sma50 += 1
                if pd.notna(sma100) and current_price > sma100:
                    above_sma100 += 1
                if pd.notna(sma200) and current_price > sma200:
                    above_sma200 += 1

            except Exception:
                continue

        if total_valid > 0:
            historical_breadth.append({
                'date': calc_date,
                'pct_above_sma20': round((above_sma20 / total_valid) * 100, 1),
                'pct_above_sma50': round((above_sma50 / total_valid) * 100, 1),
                'pct_above_sma100': round((above_sma100 / total_valid) * 100, 1),
                'pct_above_sma200': round((above_sma200 / total_valid) * 100, 1),
                'total_stocks': total_valid
            })

    if not historical_breadth:
        return None

    # Get current (latest) data and per-stock details
    latest = historical_breadth[-1]
    breadth_details = []

    for ticker in ticker_list:
        try:
            ticker_data = cached_df[cached_df['ticker'] == ticker].copy()
            if ticker_data.empty or len(ticker_data) < 200:
                continue

            ticker_data = ticker_data.sort_values('date')
            sma_status = compute_sma_status(ticker_data)

            if sma_status is None:
                continue

            breadth_details.append({
                'Ticker': ticker.replace('.NS', ''),
                'LTP': round(sma_status['current_price'], 2),
                'SMA20': round(sma_status['sma20'], 2) if pd.notna(sma_status['sma20']) else None,
                'SMA50': round(sma_status['sma50'], 2) if pd.notna(sma_status['sma50']) else None,
                'SMA100': round(sma_status['sma100'], 2) if pd.notna(sma_status['sma100']) else None,
                'SMA200': round(sma_status['sma200'], 2) if pd.notna(sma_status['sma200']) else None,
                '>SMA20': '✓' if sma_status['above_sma20'] else '✗',
                '>SMA50': '✓' if sma_status['above_sma50'] else '✗',
                '>SMA100': '✓' if sma_status['above_sma100'] else '✗',
                '>SMA200': '✓' if sma_status['above_sma200'] else '✗',
            })

        except Exception as e:
            logging.error(f"Error calculating SMA for {ticker}: {e}")
            continue

    return {
        'pct_above_sma20': latest['pct_above_sma20'],
        'pct_above_sma50': latest['pct_above_sma50'],
        'pct_above_sma100': latest['pct_above_sma100'],
        'pct_above_sma200': latest['pct_above_sma200'],
        'total_stocks': latest['total_stocks'],
        'history': pd.DataFrame(historical_breadth),
        'details': pd.DataFrame(breadth_details)
    }


st.title(f"📊 {selected_index} Volume Dashboard")

# Refresh buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button('🔄 Refresh Market Data'):
        # Check if we can actually refresh
        conn = sqlite3.connect('stocks.db')
        c = conn.cursor()
        c.execute("SELECT last_updated FROM cache_tracking WHERE index_group = ?", (selected_index,))
        row = c.fetchone()
        conn.close()

        if row:
            last_updated_db = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
            elapsed = datetime.now() - last_updated_db
            cooldown = timedelta(minutes=15)

            if elapsed < cooldown:
                remaining = cooldown - elapsed
                mins = (remaining.seconds // 60) + 1
                st.warning(f"Market data is already fresh! Next manual refresh available in ~{mins} minutes.")
                logging.info(f"User refresh rejected: {mins} mins remaining.")
            else:
                st.cache_data.clear()
                logging.info("User requested manual cache refresh.")
                st.rerun()
        else:
            st.cache_data.clear()
            st.rerun()

with col2:
    if st.button('🗑️ Force Full Refresh (Clear Cache)'):
        # Invalidate cache completely for this index
        conn = sqlite3.connect('stocks.db')
        c = conn.cursor()
        c.execute("DELETE FROM cache_tracking WHERE index_group = ?", (selected_index,))
        conn.commit()
        conn.close()
        st.cache_data.clear()
        logging.info(f"User forced full cache invalidation for {selected_index}")
        st.toast("Cache cleared! Fetching fresh data...", icon="🗑️")
        st.rerun()

with streamlit_analytics.track():
    with st.spinner(f"Fetching data for {selected_index}..."):
        logging.info(f"Fetching data for {selected_index}...")
        # Check cache status for UI feedback before calling fetch_data
        if is_cache_valid(selected_index):
            st.toast("Loading from local cache (data < 15 mins old)...", icon="ℹ️")
        
        df, latest_update_time = fetch_data(tickers, selected_index)

    if latest_update_time:
        ist_time = to_ist(latest_update_time)
        st.caption(f"Last Data Update: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")

    if not df.empty:
        # Sort by Multiplier to see highest surges first
        df = df.sort_values(by="Multiplier", ascending=False)

        # Log success
        logging.info(f"Successfully loaded {len(df)} tickers for {selected_index}")

        # Market Health Section - Stocks Above SMA (at the top)
        st.subheader(f"📈 {selected_index} % Above Moving Averages")

        breadth_data = calculate_market_breadth(tickers, selected_index)

        if breadth_data and 'history' in breadth_data:
            history_df = breadth_data['history']

            # Current values header
            current_vals = f"**SMA20:** {breadth_data['pct_above_sma20']}  |  **SMA50:** {breadth_data['pct_above_sma50']}  |  **SMA100:** {breadth_data['pct_above_sma100']}  |  **SMA200:** {breadth_data['pct_above_sma200']}  ⊘ 80.0  50.0  20.0"
            st.markdown(current_vals)

            # Two columns: Line chart on left, Data table on right
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig_lines = go.Figure()
                fig_lines.add_trace(go.Scatter(
                    x=history_df['date'], y=history_df['pct_above_sma20'],
                    mode='lines', name='SMA 20', line=dict(color='#ff9800', width=2)
                ))
                fig_lines.add_trace(go.Scatter(
                    x=history_df['date'], y=history_df['pct_above_sma50'],
                    mode='lines', name='SMA 50', line=dict(color='#4caf50', width=2)
                ))
                fig_lines.add_trace(go.Scatter(
                    x=history_df['date'], y=history_df['pct_above_sma100'],
                    mode='lines', name='SMA 100', line=dict(color='#f44336', width=2)
                ))
                fig_lines.add_trace(go.Scatter(
                    x=history_df['date'], y=history_df['pct_above_sma200'],
                    mode='lines', name='SMA 200', line=dict(color='#000000', width=2)
                ))
                fig_lines.add_hline(y=80, line_dash="dash", line_color="gray", opacity=0.5)
                fig_lines.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
                fig_lines.add_hline(y=20, line_dash="dash", line_color="gray", opacity=0.5)
                fig_lines.update_layout(
                    yaxis_title="% of Stocks", yaxis_range=[0, 100], xaxis_title="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=300, margin=dict(l=40, r=40, t=30, b=40)
                )
                st.plotly_chart(fig_lines, use_container_width=True)

            with col_table:
                st.markdown("**% above**")
                table_data = history_df.tail(10).copy()
                table_data = table_data.iloc[::-1]
                table_data['date'] = pd.to_datetime(table_data['date']).dt.strftime('%d-%b')
                table_display = table_data[['date', 'pct_above_sma20', 'pct_above_sma50', 'pct_above_sma100', 'pct_above_sma200']].copy()
                table_display.columns = ['Date', '20MA', '50MA', '100MA', '200MA']

                def color_cell(val):
                    if isinstance(val, (int, float)):
                        return 'background-color: #c8e6c9; color: #1b5e20' if val >= 50 else 'background-color: #ffcdd2; color: #b71c1c'
                    return ''

                styled_table = table_display.style.applymap(
                    color_cell, subset=['20MA', '50MA', '100MA', '200MA']
                ).format({'20MA': '{:.1f}', '50MA': '{:.1f}', '100MA': '{:.1f}', '200MA': '{:.1f}'})
                st.dataframe(styled_table, use_container_width=True, hide_index=True, height=330)

            # Overbought/Oversold alerts
            all_pcts = [breadth_data['pct_above_sma20'], breadth_data['pct_above_sma50'],
                        breadth_data['pct_above_sma100'], breadth_data['pct_above_sma200']]
            if all(p >= 80 for p in all_pcts):
                st.warning("⚠️ **Overbought**: All MAs above 80%")
            elif all(p <= 20 for p in all_pcts):
                st.success("🚀 **Oversold**: All MAs below 20%")
        else:
            st.info("Market breadth data calculating... (requires 200+ days of data)")

        st.markdown("---")

        # Volume Section
        st.subheader("Visualizing Volume vs Average")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Ticker'], y=df['Multiplier'], name="Volume Multiplier"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="SMA 20 Baseline")
        fig.update_layout(yaxis_title="Multiplier (x times Avg)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Volume Surges")
        
        # Calculate dynamic height: ~35px per row + header buffer
        # Cap max height to avoid page getting too crazy long if 500 rows, or let it grow as requested
        # User said "show entire data", so let's let it grow.
        dynamic_height = (len(df) + 1) * 35 + 3 

        # Styling for readability
        st.dataframe(
            df.style.background_gradient(subset=['Multiplier'], cmap='RdYlGn')
                    .background_gradient(subset=['Price Change %'], cmap='RdYlGn', vmin=-2, vmax=2)
                    .background_gradient(subset=['Net Vol Estimate'], cmap='RdYlGn', vmin=-1000000, vmax=1000000)
                    .background_gradient(subset=['RS55 vs Nifty'], cmap='RdYlGn', vmin=-0.15, vmax=0.15)
                    .background_gradient(subset=['EMA21 %'], cmap='RdYlGn', vmin=-10, vmax=10)
                    .background_gradient(subset=['EMA55 %'], cmap='RdYlGn', vmin=-15, vmax=15)
                    .background_gradient(subset=['EMA100 %'], cmap='RdYlGn', vmin=-20, vmax=20)
                    .background_gradient(subset=['EMA200 %'], cmap='RdYlGn', vmin=-25, vmax=25)
                    .background_gradient(subset=['RSI14'], cmap='RdYlGn', vmin=30, vmax=70)
                    .format({
                        'LTP': "{:,.2f}",
                        'Price Change %': "{:.2f}%",
                        'Current Volume': "{:,.0f}",
                        '20D Avg Vol (Prev)': "{:,.0f}",
                        'Multiplier': "{:.2f}x",
                        'Net Vol Estimate': "{:,.0f}",
                        'RS55 vs Nifty': "{:.2f}",
                        'EMA21': "{:,.2f}",
                        'EMA55': "{:,.2f}",
                        'EMA100': "{:,.2f}",
                        'EMA200': "{:,.2f}",
                        'EMA21 %': "{:.2f}%",
                        'EMA55 %': "{:.2f}%",
                        'EMA100 %': "{:.2f}%",
                        'EMA200 %': "{:.2f}%",
                        'RSI14': "{:.1f}"
                    }),
            use_container_width=True,
            height=dynamic_height
        )

    else:
        st.error("Still no data. Try checking if you can access [finance.yahoo.com](https://finance.yahoo.com) in your browser.")