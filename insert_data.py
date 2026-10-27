import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. 載入環境變數並組裝連線字串
load_dotenv()
raw_url = os.getenv("DB_URL", "").strip()

if raw_url.startswith("postgresql://"):
    raw_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
if ".render.com" in raw_url and ":5432" not in raw_url:
    raw_url = raw_url.replace("://", "://", 1)

if not raw_url:
    raise ValueError("找不到 DB_URL，請確認 .env 檔案是否存在且設定正確。")

print("正在連接資料庫...")
engine = create_engine(raw_url, connect_args={"connect_timeout": 5})

# 2. 定義要追蹤的股票清單與起點
# 從環境變數讀取 TICKERS，若找不到則使用預設清單
tickers_raw = os.getenv("TICKERS", "")
if tickers_raw:
    # 依逗號切開，並自動修剪前後空格
    tickers = [tk.strip() for tk in tickers_raw.split(",") if tk.strip()]
else:
    # 備用預設清單
    tickers = ["2330.TW", "2317.TW", "2454.TW", "^TWII", "1303.TW", "1326.TW"]

DEFAULT_START = "2020-01-01"
today_str = datetime.today().strftime('%Y-%m-%d')

# 3. 智能比對：先去資料庫看「最舊的缺件起點」是哪一天
earliest_start_needed = DEFAULT_START
need_update_tickers = []

for tk in tickers:
    query_latest = text("SELECT MAX(date) FROM stock_prices WHERE ticker = :ticker")
    with engine.connect() as conn:
        latest_date = conn.execute(query_latest, {"ticker": tk}).scalar()
    
    if latest_date is not None:
        next_day = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
        if next_day <= today_str:
            need_update_tickers.append(tk)
            # 找出所有人裡面最老的起點
            if next_day < earliest_start_needed:
                earliest_start_needed = next_day
    else:
        # 有全新股票要全量下載
        need_update_tickers.append(tk)
        earliest_start_needed = DEFAULT_START

# 如果大家都是最新的，直接結束
if not need_update_tickers:
    print("✅ 所有標的的資料都已經是最新的，無需向 Yahoo Finance 發送請求！")
    exit()

# 4. 🛠️ 關鍵優化：把需要更新的股票用空格串聯，達成「只連線一次」
tickers_string = " ".join(need_update_tickers)
print(f"📥 [安全機制] 僅用 1 次連線下載群組：{need_update_tickers}")
print(f"📅 抓取區間：{earliest_start_needed} 至 {today_str}")

try:
    # 這裡只會產生「1次」網路請求，不管裡面有幾百檔股票
    all_df = yf.download(tickers_string, start=earliest_start_needed, end=today_str, group_by='ticker', progress=False)
    
    if all_df.empty:
        print("ℹ️ 線上無新交易資料，程序結束。")
        exit()

    # 5. 在記憶體裡拆包並寫入資料庫
    for tk in need_update_tickers:
        # 如果只有單檔股票，yfinance 格式會不同，做相容性提取
        if len(need_update_tickers) == 1:
            df = all_df.copy()
        else:
            if tk not in all_df.columns.levels[0]: continue
            df = all_df[tk].copy()
            
        df = df.dropna(subset=['Close']).reset_index()
        if df.empty: continue
        
        # 建立這檔股票要寫入資料庫的 DataFrame
        db_df = pd.DataFrame()
        db_df['date'] = pd.to_datetime(df['Date'])
        db_df['ticker'] = tk
        
        for col in ['Open', 'High', 'Low', 'Close']:
            db_df[col] = df[col].astype(float)
        db_df['Volume'] = df['Volume'].astype(int)
        
        # 過濾掉已經存在資料庫的歷史日子，避免重複觸發資料庫 unique 報錯（雙重防禦）
        query_exist = text("SELECT date FROM stock_prices WHERE ticker = :ticker AND date >= :start")
        with engine.connect() as conn:
            exist_dates = pd.read_sql(query_exist, conn, params={"ticker": tk, "start": earliest_start_needed})
        if not exist_dates.empty:
            exist_set = set(pd.to_datetime(exist_dates['date']))
            db_df = db_df[~db_df['date'].isin(exist_set)]
            
        if db_df.empty:
            print(f"✅ {tk} 無真正的新資料需要寫入。")
            continue
            
        print(f"🚀 正在將 {tk} 的 {len(db_df)} 筆新數據寫入 Render...")
        db_df.to_sql(name='stock_prices', con=engine, if_exists='append', index=False, method='multi', chunksize=500)
        print(f"🎉 {tk} 補件完成！")

except Exception as e:
    print(f"💥 執行大包裝下載或寫入時發生錯誤: {e}")

print("\n✨ [安全防鎖方案] 自動補件程序執行完畢！")
