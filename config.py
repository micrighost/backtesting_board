import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def setup_system():
    """初始化系統字型與環境變數"""
    # 讓 Matplotlib 優先找 Linux 的思源黑體，找不到就找 Windows 的微軟正黑體
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK TC',      # Linux 繁體中文
        'Microsoft JhengHei',    # Windows 微軟正黑體
        'sans-serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
    load_dotenv()

def load_ticker_options():
    """載入環境變數中的股票精選清單，若無則提供預設值"""
    env_tickers_raw = os.getenv("TICKERS", "")
    if env_tickers_raw:
        ticker_options = [tk.strip() for tk in env_tickers_raw.split(",") if tk.strip()]
    else:
        ticker_options = ["2330.TW", "2317.TW", "2454.TW", "NVDA"]
    
    default_selected = [tk for tk in ["2330.TW", "2317.TW", "2454.TW"] if tk in ticker_options]
    if not default_selected and ticker_options:
        default_selected = [ticker_options[0]]
        
    return ticker_options, default_selected
