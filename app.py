import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import setup_system, load_ticker_options
from engine import calculate_indicators_and_signals, run_backtest

# 1. 系統配置與高能快取下載器
st.set_page_config(layout="wide", page_title="策略競技場", page_icon="📊")
setup_system()
st.title("📊 全自訂多股策略競技場 (多空雙向版)")

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, start, end):
    """記憶型資料下載器：強制單執行緒，快取記憶防前端閃爍"""
    df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
    if df.empty: return pd.DataFrame()
    # 【修正 1】安全移除多級索引，防範多股下載時欄位錯亂
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

if "has_run" not in st.session_state:
    st.session_state.update({"has_run": False, "summary": None, "fig_p": None, "fig_s": None, "cache_dfs": {}, "success_msgs": []})

ticker_options, default_selected = load_ticker_options()
col_tk1, col_tk2, col_tk3 = st.columns(3)
tickers = col_tk1.multiselect("請選擇標的：", options=ticker_options, default=default_selected, key="ms_tk")
start_date = pd.to_datetime(col_tk2.date_input("開始日期", pd.to_datetime("2022-01-01"), key="di_s"))
end_date = pd.to_datetime(col_tk3.date_input("結束日期", pd.to_datetime("2026-01-01"), key="di_e"))

st.write("---")
# 2. 核心設定區
col_setting1, col_setting2 = st.columns(2)
with col_setting1:
    st.markdown("### ⚙️ 步驟一：設定固定基礎參數")
    col_p1, col_p2 = st.columns(2)
    base_params = {
        'short_window': col_p1.slider("短期均線天數", 3, 20, 5), 'volume_window': col_p1.slider("計算平均成交量的天數", 5, 60, 20), 'rsi_window': col_p1.slider("RSI 計算天數", 5, 30, 14), 'macd_fast': col_p1.slider("MACD 快線天數", 5, 25, 12),
        'long_window': col_p2.slider("長期均線天數", 10, 120, 20), 'volume_threshold': col_p2.slider("成交量爆量倍數門檻", 0.5, 3.0, 1.2, step=0.1), 'rsi_threshold': col_p2.slider("RSI 門檻值 (0-100)", 0, 100, 40), 'macd_slow': col_p2.slider("MACD 慢線天數", 20, 50, 26), 'macd_signal': col_p2.slider("MACD 訊號線天數", 5, 20, 9)
    }
with col_setting2:
    st.markdown("### 🚀 步驟二：進場條件與反轉方向")
    conds_switches = {}
    c1_a, c1_b = st.columns([5, 2])
    conds_switches['golden'] = c1_a.checkbox("條件 A：均線黃金交叉（短天期均線 ＞ 長天期均線）", value=True)
    dir_a = c1_b.selectbox("方向", ["正向 (黃金交叉)", "反轉 (死亡交叉)"], key="dir_a", label_visibility="collapsed")
    c2_a, c2_b = st.columns([5, 2])
    conds_switches['volume'] = c2_a.checkbox("條件 B：量能突發性爆量（今日成交量 ＞ 歷史平均量門檻）", value=False)
    dir_b = c2_b.selectbox("方向", ["正向 (高於均量)", "反轉 (低於均量)"], key="dir_b", label_visibility="collapsed")
    c3_a, c3_b = st.columns([5, 2])
    conds_switches['rsi'] = c3_a.checkbox("條件 C：逆勢逢低佈局（RSI 指標低於超賣門檻值）", value=False)
    dir_c = c3_b.selectbox("方向", ["正向 (低於門檻)", "反轉 (高於門檻)"], key="dir_c", label_visibility="collapsed")
    c4_a, c4_b = st.columns([5, 2])
    conds_switches['macd'] = c4_a.checkbox("條件 D：動能轉折向上（MACD 柱狀體 ＞ 0 多頭黃金交叉）", value=False)
    dir_d = c4_b.selectbox("方向", ["正向 (多頭上攻)", "反轉 (空頭向下)"], key="dir_d", label_visibility="collapsed")
    conds_switches.update({'inv_a': "反轉" in dir_a, 'inv_b': "反轉" in dir_b, 'inv_c': "反轉" in dir_c, 'inv_d': "反轉" in dir_d})
    
    # 【完美歸位】依據您的設計，將下拉選單完美保留在 col_setting2 的區塊最下方
    scan_target = st.selectbox("🔍 請選擇你想看期望值變化的掃描參數：", ["短期均線天數 (3-20天)", "長期均線天數 (10-120天)", "計算平均成交量的天數 (5-60天)", "成交量爆量倍數門檻 (0.5-3.0倍)", "RSI 計算天數 (5-30天)", "RSI 門檻值 (0-100)", "MACD 快線天數 (5-25天)", "MACD 慢線天數 (20-50天)", "MACD 訊號線天數 (5-20天)"])

btn_trigger = st.button("🔥 開始執行完整回測與參數掃描", use_container_width=True)

# 3. 按鈕觸發：核心回測流程與窮舉
if btn_trigger and tickers:
    st.session_state.has_run, st.session_state.success_msgs = True, []
    with st.spinner("系統正在處理歷史數據與參數窮舉..."):
        tw_market = fetch_stock_data("^TWII", start_date, end_date)
        if tw_market.empty: st.stop()
        
        # 【修正 2】確保大盤 Close 在多股下載模式下抓取正確
        m_close = tw_market['Close'].iloc[:, 0] if isinstance(tw_market['Close'], pd.DataFrame) else tw_market['Close']
        market_cum_perf = ((1 + m_close.pct_change()).cumprod() - 1) * 100
        
        plt.close('all'); fig_perf, ax_perf = plt.subplots(figsize=(11, 4)); fig_scan, ax_scan = plt.subplots(figsize=(11, 4))
        ax_perf.plot(tw_market.index, market_cum_perf, label="台股加權指數 (^TWII)", color="black", linestyle="--", alpha=0.7)
        scan_mapping = {"短期均線": (3, 21, 1, "short_window"), "長期均線": (10, 121, 5, "long_window"), "平均成交量": (5, 61, 5, "volume_window"), "爆量倍數": (0.5, 3.1, 0.1, "volume_threshold"), "RSI 計算": (5, 31, 1, "rsi_window"), "RSI 門檻": (0, 101, 5, "rsi_threshold"), "MACD 快線": (5, 26, 1, "macd_fast"), "MACD 慢線": (20, 51, 2, "macd_slow"), "MACD 訊號線": (5, 21, 1, "macd_signal")}
        key, (start, stop, step, p_field) = next(((k, v) for k, v in scan_mapping.items() if k in scan_target), ("MACD 訊號線", (5, 21, 1, "macd_signal")))
        scan_range = np.arange(start, stop, step).tolist() if step % 1 != 0 else list(range(int(start), int(stop), int(step)))
        
        summary_data = []
        for tk in tickers:
            df_orig = fetch_stock_data(tk, start_date, end_date)
            if df_orig.empty: continue
            st.session_state.cache_dfs[tk] = df_orig
            df_base_bt = calculate_indicators_and_signals(df_orig, base_params, conds_switches)
            res_base = run_backtest(df_base_bt)
            if res_base:
                summary_data.append({"標的名稱": tk, "自訂策略總報酬": f"{res_base['final_return']:.2%}", "與大盤相差 (vs ^TWII)": f"{(res_base['final_return'] - (market_cum_perf.iloc[-1]/100)):+.2%}", "當前參數交易日期望值 🎯": f"{res_base['expected_return']:+.4%}", "策略勝率": f"{res_base['win_rate']:.2%}", "總交易次數": f"{res_base['trade_count']} 次"})
                ax_perf.plot(df_base_bt.index, res_base['cum_strat_series'], label=f"{tk} 策略", linewidth=2)
            scan_x, scan_y = [], []
            for p in scan_range:
                loop_params = base_params.copy(); loop_params[p_field] = p
                df_scan_bt = calculate_indicators_and_signals(df_orig, loop_params, conds_switches)
                res_scan = run_backtest(df_scan_bt)
                if res_scan: scan_x.append(p); scan_y.append(res_scan['expected_return'] * 100)
            if scan_x and scan_y:
                ax_scan.plot(scan_x, scan_y, marker='o', label=f"{tk} 期望值趨勢", linewidth=2, alpha=0.5)
                max_idx = np.argmax(scan_y)
                ax_scan.scatter(scan_x[max_idx], scan_y[max_idx], color='red', marker='*', s=150, zorder=5)
                st.session_state.success_msgs.append(f"🎯 **{tk}** 的 **{key}** 歷史最佳最優解為：**{scan_x[max_idx]}** (期望值: {scan_y[max_idx]/100:+.4%})")
        ax_perf.set_ylabel("報酬率 (%)"); ax_perf.legend(); ax_perf.grid(True, linestyle="--", alpha=0.5)
        ax_scan.set_xlabel(f"變動參數數值 ({key})"); ax_scan.set_ylabel("每筆交易每日平均期望值 (%)")
        ax_scan.axhline(0, color='red', linestyle=':', alpha=0.6); ax_scan.legend(); ax_scan.grid(True, linestyle="--", alpha=0.5)
        st.session_state.summary, st.session_state.fig_p, st.session_state.fig_s = summary_data, fig_perf, fig_scan
        # 【修正 3】移除原先的 st.rerun()，改用以下這行更新畫面，徹底阻斷無窮重啟死循環
        st.navigation_visibility = "visible"

# 4. 下方成果與「穿透式檢視」隔離渲染區
@st.fragment
def render_data_inspection_station():
    st.write("---")
    st.markdown("### 🔍 步驟三：策略底層數據觀測站 (逐日明細檢驗)")
    inspect_tk = st.selectbox("請選擇要檢驗明細的標的：", options=tickers, key="sb_inspect")
    if inspect_tk and inspect_tk in st.session_state.cache_dfs:
        df_detail = calculate_indicators_and_signals(st.session_state.cache_dfs[inspect_tk], base_params, conds_switches)
        display_df = pd.DataFrame(index=df_detail.index)
        
        # 安全提取收盤價與成交量，防止多股時出錯
        d_close = df_detail['Close'].iloc[:, 0] if isinstance(df_detail['Close'], pd.DataFrame) else df_detail['Close']
        d_vol = df_detail['Volume'].iloc[:, 0] if isinstance(df_detail['Volume'], pd.DataFrame) else df_detail['Volume']
        display_df['收盤價'], display_df['成交量'] = d_close.round(2), d_vol
        
        if conds_switches['golden']: display_df['短均線'], display_df['長均線'] = df_detail['MA_Short'].round(2), df_detail['MA_Long'].round(2)
        if conds_switches['volume']: display_df['量能倍數'] = df_detail['Vol_Ratio'].round(2)
        if conds_switches['rsi']: display_df['RSI'] = df_detail['RSI'].round(2)
        if conds_switches['macd']: display_df['MACD柱狀體'] = df_detail['MACD_Hist'].round(4)
        display_df['是否進場?'] = df_detail['Signal'].map({1: "🟢 持股", 0: "⚪ 空手"})
        st.dataframe(display_df.reset_index().sort_values(by='Date', ascending=False).rename(columns={'Date': '交易日期'}), use_container_width=True, hide_index=True)

if st.session_state.has_run and st.session_state.summary:
    for msg in st.session_state.success_msgs: st.success(msg)
    st.markdown("### 🏆 策略回測成績單 (當前參數)")
    st.dataframe(pd.DataFrame(st.session_state.summary), use_container_width=True, hide_index=True)
    col_graph1, col_graph2 = st.columns(2)
    col_graph1.markdown("##### 📈 圖一：歷史累積報酬率走勢")
    col_graph1.pyplot(st.session_state.fig_p)
    col_graph2.markdown("##### 🎯 圖二：大王666參數敏感度分析 (★ 為歷史最佳最優解)")
    col_graph2.pyplot(st.session_state.fig_s)

render_data_inspection_station()
