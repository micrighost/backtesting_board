import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# ==================== 【解決中文字型亂碼設定】 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 步驟一：指定微軟正黑體
plt.rcParams['axes.unicode_minus'] = False                # 步驟二：解決負號 (-) 顯示為方塊的問題
# ==================================================================

# 載入 .env 檔案中的 TICKERS 精選名單
load_dotenv()
env_tickers_raw = os.getenv("TICKERS", "")
if env_tickers_raw:
    ticker_options = [tk.strip() for tk in env_tickers_raw.split(",") if tk.strip()]
else:
    ticker_options = ["2330.TW", "2317.TW", "2454.TW", "NVDA"]

default_selected = [tk for tk in ["2330.TW", "2317.TW", "2454.TW"] if tk in ticker_options]
if not default_selected and ticker_options:
    default_selected = [ticker_options[0]]

st.title("📊 全自訂多股策略競技場 (全指標掃描穩定版)")
st.write(f"💡 系統已載入 **{len(ticker_options)}** 檔精選標的。你也可以直接輸入任何其他 Yahoo Finance 代號並按 Enter 新增！")

# 1. 網頁前端選單與基礎參數設定
tickers = st.multiselect(
    "請選擇、搜尋精選標的，或直接輸入新代號：", 
    options=ticker_options, 
    default=default_selected
)

col1, col2 = st.columns(2)
start_date = col1.date_input("開始日期", pd.to_datetime("2022-01-01"))
end_date = col2.date_input("結束日期", pd.to_datetime("2026-01-01"))

st.subheader("⚙️ 步驟一：設定技術指標固定基礎參數")
col_param1, col_param2 = st.columns(2)
short_window = col_param1.slider("短期均線天數 (短天期)", 3, 20, 5)
volume_window = col_param1.slider("計算平均成交量的天數", 5, 60, 20)
rsi_window = col_param1.slider("RSI 計算天數", 5, 30, 14)
long_window = col_param2.slider("長期均線天數 (長天期)", 10, 120, 20)
volume_threshold = col_param2.slider("成交量必須是平均量的幾倍以上？", 0.5, 3.0, 1.2, step=0.1)
rsi_threshold = col_param2.slider("RSI 門檻值 (0-100)", 0, 100, 40)

st.subheader("🚀 步驟二：設定進場條件與參數掃描項目")
col_c1, col_c2 = st.columns(2)
cond_golden = col_c1.checkbox("條件 A：黃金交叉（短均線 > 長均線）", value=True)
cond_volume = col_c1.checkbox("條件 B：今日成交量 > 爆量倍數", value=False)
cond_rsi = col_c1.checkbox("條件 C：RSI 小於等於門檻值", value=False)

scan_target = col_c2.selectbox(
    "🔍 請選擇你想看期望值變化的掃描參數：",
    ["短期均線天數 (3-25天)", "長期均線天數 (10-120天)", "爆量倍數門檻 (0.5-3.0倍)", "RSI門檻值 (10-90)"]
)

if "短期均線" in scan_target: scan_range = list(range(3, 26))
elif "長期均線" in scan_target: scan_range = list(range(10, 121, 5))
elif "爆量倍數" in scan_target: scan_range = np.arange(0.5, 3.1, 0.1).tolist()
else: scan_range = list(range(10, 91, 5))

# 3. 核心回測、窮舉計算與繪圖
if st.button("🔥 開始執行完整回測與參數掃描") and tickers:
    with st.spinner("系統正在處理歷史數據與參數窮舉..."):
        tw_market = yf.download("^TWII", start=start_date, end=end_date)
        if tw_market.empty: st.stop()
        tw_market.columns = tw_market.columns.get_level_values(0)
        market_final_ret = ((1 + tw_market['Close'].squeeze().pct_change()).cumprod() - 1).iloc[-1]
        
        fig_perf, ax_perf = plt.subplots(figsize=(10, 4.5))
        fig_scan, ax_scan = plt.subplots(figsize=(10, 4.5))
        ax_perf.plot(tw_market.index, ((1 + tw_market['Close'].squeeze().pct_change()).cumprod() - 1) * 100, label="台股加權指數 (^TWII)", color="black", linestyle="--", alpha=0.7)
        
        summary_data = []

        for tk in tickers:
            df_orig = yf.download(tk, start=start_date, end=end_date)
            if df_orig.empty: continue
            df_orig.columns = df_orig.columns.get_level_values(0)
            
            # ==================== 部分 A ====================
            df_base = df_orig.copy()
            close_b, vol_b = df_base['Close'].squeeze(), df_base['Volume'].squeeze()
            df_base['MA_Short'] = close_b.rolling(short_window).mean()
            df_base['MA_Long'] = close_b.rolling(long_window).mean()
            df_base['Vol_Ratio'] = vol_b / vol_b.rolling(volume_window).mean()
            delta_b = close_b.diff()
            df_base['RSI'] = 100 - (100 / (1 + delta_b.clip(lower=0).rolling(rsi_window).mean() / delta_b.clip(upper=0).abs().rolling(rsi_window).mean()))
            
            conds_b = []
            if cond_golden: conds_b.append(df_base['MA_Short'] > df_base['MA_Long'])
            if cond_volume: conds_b.append(df_base['Vol_Ratio'] > volume_threshold)
            if cond_rsi: conds_b.append(df_base['RSI'] <= rsi_threshold)
            
            df_base['Signal'] = np.logical_and.reduce(conds_b).astype(int) if conds_b else 0
            df_base_bt = df_base.dropna().copy()
            
            if not df_base_bt.empty:
                df_base_bt['Strat_Ret'] = df_base_bt['Close'].squeeze().pct_change() * df_base_bt['Signal'].shift(1)
                cum_strat = (1 + df_base_bt['Strat_Ret'].fillna(0)).cumprod() - 1
                strat_final_ret = cum_strat.iloc[-1]
                trade_count = (df_base_bt['Signal'].diff() == 1).sum() + (1 if df_base_bt['Signal'].iloc == 1 else 0)
                active_days_b = df_base_bt[df_base_bt['Signal'].shift(1) == 1]
                win_rate = (active_days_b['Strat_Ret'] > 0).sum() / len(active_days_b) if len(active_days_b) > 0 else 0.0
                current_expected_return = active_days_b['Strat_Ret'].mean() if len(active_days_b) > 0 else 0.0
                
                summary_data.append({
                    "標的名稱": tk, "自訂策略總報酬": f"{strat_final_ret:.2%}",
                    "與大盤相差 (vs ^TWII)": f"{(strat_final_ret - market_final_ret):+.2%}",
                    "當前參數交易日期望值 🎯": f"{current_expected_return:+.4%}",
                    "策略勝率": f"{win_rate:.2%}", "總交易次數": f"{trade_count} 次"
                })
                ax_perf.plot(df_base_bt.index, cum_strat * 100, label=f"{tk} 策略", linewidth=2)

            # ==================== 部分 B ====================
            scan_x, scan_y = [], []
            for p in scan_range:
                df_scan = df_orig.copy()
                close_s, vol_s = df_scan['Close'].squeeze(), df_scan['Volume'].squeeze()
                
                s_short = p if "短期均線" in scan_target else short_window
                s_long = p if "長期均線" in scan_target else long_window
                s_vol_t = p if "爆量倍數" in scan_target else volume_threshold
                s_rsi_t = p if "RSI門檻" in scan_target else rsi_threshold
                
                df_scan['MA_Short'] = close_s.rolling(s_short).mean()
                df_scan['MA_Long'] = close_s.rolling(s_long).mean()
                df_scan['Vol_Ratio'] = vol_s / vol_s.rolling(volume_window).mean()
                delta_s = close_s.diff()
                df_scan['RSI'] = 100 - (100 / (1 + delta_s.clip(lower=0).rolling(s_rsi_win := rsi_window).mean() / delta_s.clip(upper=0).abs().rolling(s_rsi_win).mean()))
                
                conds_s = []
                if cond_golden: conds_s.append(df_scan['MA_Short'] > df_scan['MA_Long'])
                if cond_volume: conds_s.append(df_scan['Vol_Ratio'] > s_vol_t)
                if cond_rsi: conds_s.append(df_scan['RSI'] <= s_rsi_t)
                
                df_scan['Signal'] = np.logical_and.reduce(conds_s).astype(int) if conds_s else 0
                df_scan_bt = df_scan.dropna().copy()
                
                if not df_scan_bt.empty:
                    df_scan_bt['Strat_Ret'] = df_scan_bt['Close'].squeeze().pct_change() * df_scan_bt['Signal'].shift(1)
                    active_days_s = df_scan_bt[df_scan_bt['Signal'].shift(1) == 1]
                    exp_val = active_days_s['Strat_Ret'].mean() if len(active_days_s) > 0 else 0.0
                    
                    scan_x.append(p)
                    scan_y.append(exp_val * 100)
            
            if scan_x:
                ax_scan.plot(scan_x, scan_y, marker='o', label=f"{tk} 期望值趨勢", linewidth=2)

        # 4. 前端介面渲染
        if summary_data:
            st.subheader("🏆 策略回測成績單 (當前滑桿設定參數)")
            st.info(f"📅 統計區間內，大盤 (^TWII) 的總報酬率為：**{market_final_ret:.2%}**")
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            st.subheader("📈 圖一：歷史累積報酬率走勢圖 (使用目前固定參數)")
            ax_perf.set_ylabel("報酬率 (%)"); ax_perf.legend(); ax_perf.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_perf)
            
            st.subheader("🎯 圖二：參數敏感度分析")
            ax_scan.set_xlabel(f"變動參數數值 ({scan_target.split(' ')[0]})")
            ax_scan.set_ylabel("每筆交易每日平均期望值 (%)")
            ax_scan.axhline(0, color='red', linestyle=':', alpha=0.6)
            ax_scan.legend(); ax_scan.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_scan)
        else:
            st.error("沒有成功跑出任何股票的回測結果。")
