import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_engine = None

def get_engine():
    """安全獲取 SQLAlchemy 引擎"""
    global _engine
    if _engine is not None: return _engine
    db_url = os.getenv("DB_URL", "").strip()
    if not db_url: return None
    try:
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if ".render.com" in db_url and ":5432" not in db_url:
            db_url = db_url.replace("://", "://", 1)
        _engine = create_engine(db_url, connect_args={"connect_timeout": 5})
        return _engine
    except Exception as e:
        print(f"[警告] 資料庫引擎建立失敗: {e}")
        return None

def clean_dataframe_format(df, ticker_name_log=""):
    """統一在資料層將所有來源的資料清洗為完全標準的扁平化結構"""
    if df.empty: return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    
    rename_map = {'Date': 'date', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume'}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                series_vals = df[col].iloc[:, 0].values
            else:
                series_vals = df[col].values
            
            if col == 'Volume':
                df[col] = pd.Series(series_vals, index=df.index).astype(int)
            else:
                df[col] = pd.Series(series_vals, index=df.index).astype(float)
                
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_backup_data(ticker, start, end):
    """備用資料源：yfinance 網上下載"""
    print(f"[備用機制] 正在透過 yfinance 線上抓取 {ticker} 的歷史資料...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        cleaned_df = clean_dataframe_format(df, ticker)
        
        # 💡 [新增回報] 抓取並整型成功後，主動回報筆數
        if not cleaned_df.empty:
            print(f"🌐 [備用成功] 成功透過 yfinance 抓取 {ticker} 的歷史資料，共 {len(cleaned_df)} 筆。")
        else:
            print(f"⚠️ [備用無資料] yfinance 未能抓到 {ticker} 在該區間的任何資料。")
            
        return cleaned_df
    except Exception as e:
        print(f"[錯誤] 備用資料源抓取失敗: {e}")
        return pd.DataFrame()

def db_fetch_stock_data(ticker, start, end):
    """主控函式：優先讀取資料庫，無資料時走預設備用機制"""
    engine = get_engine()
    if engine is None: return fetch_backup_data(ticker, start, end)
        
    try:
        query = text("""
            SELECT date, "Open", "High", "Low", "Close", "Volume" 
            FROM stock_prices 
            WHERE ticker = :ticker 
            AND date BETWEEN :start AND :end
            ORDER BY date ASC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ticker": ticker, "start": start, "end": end})
            
        if df.empty:
            print(f"[提示] 資料庫內無 {ticker} 的資料，切換至預設備用源...")
            return fetch_backup_data(ticker, start, end)

            
        print(f"📡 [資料庫成功] 成功從資料庫撈取 {ticker} 的歷史資料，共 {len(df)} 筆。")
        return clean_dataframe_format(df, ticker)
        
    except Exception as e:
        print(f"[警告] 資料庫讀取失敗 ({e})，正在自動切換至預設備用源...")
        return fetch_backup_data(ticker, start, end)
