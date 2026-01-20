import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator

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
    "Nifty Midcap 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
}

st.sidebar.title("Configuration")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES_MAP.keys()))

def get_tickers(index_name):
    # Try DB first
    cached_tickers = get_tickers_from_db(index_name)
    if cached_tickers:
        return cached_tickers

    # Fetch from NSE
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
    conn = sqlite3.connect('stocks.db')
    c = conn.cursor()
    
    # Clear old data for this index group
    c.execute("DELETE FROM stock_prices WHERE index_group = ?", (index_group,))
    
    for ticker in tickers:
        try:
            if ticker in data.columns:
                 df = data[ticker].copy().dropna()
                 # Reset index to make Date a column
                 df = df.reset_index()
                 for _, row in df.iterrows():
                     c.execute("INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (ticker, row['Date'].strftime('%Y-%m-%d'), row.get('Open', 0), row.get('High', 0), 
                                row.get('Low', 0), row.get('Close', 0), row.get('Volume', 0), index_group))
        except Exception:
            continue

    # Update timestamp
    c.execute("INSERT OR REPLACE INTO cache_tracking VALUES (?, ?)", (index_group, datetime.now()))
    conn.commit()
    conn.close()

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
        # Cache miss or stale -> Download
        logging.info(f"Cache MISS or Stale for {index_group}. Downloading from Yahoo Finance...")
        data = yf.download(ticker_list, period="40d", interval="1d", group_by='ticker')
        # Save to DB for next time
        save_to_db(data, index_group, ticker_list)
        last_updated_time = datetime.now()
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

            if len(df) >= 21: # Need at least 21 days (20 prev + 1 current)
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
                
                all_results.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "Last Date": curr_date,
                    "Current Volume": curr_vol,
                    "20D Avg Vol (Prev)": avg_vol,
                    "Multiplier": round(vol_ratio, 2),
                    "Price Change %": round(price_change, 2),
                    "Net Vol Estimate": net_vol 
                })
        except Exception as e:
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

st.title(f"📊 {selected_index} Volume Dashboard")

# Simplified Refresh Logic
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
            mins = (remaining.seconds // 60) + 1 # Round up for clarity
            st.warning(f"Market data is already fresh! Next manual refresh available in ~{mins} minutes.")
            logging.info(f"User refresh rejected: {mins} mins remaining.")
        else:
            st.cache_data.clear()
            logging.info("User requested manual cache refresh.")
            st.rerun()
    else:
        st.cache_data.clear()
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

        # Layout: Chart on Top, Table Below
        
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
                    .format({
                        'Current Volume': "{:,.2f}",
                        '20D Avg Vol (Prev)': "{:,.2f}",
                        'Multiplier': "{:,.2f}",
                        'Price Change %': "{:.2f}%", 
                        'Net Vol Estimate': "{:,.2f}"
                    }),
            use_container_width=True,
            height=dynamic_height
        )
    else:
        st.error("Still no data. Try checking if you can access [finance.yahoo.com](https://finance.yahoo.com) in your browser.")