import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import setup_system, load_ticker_options
from engine import calculate_indicators_and_signals, run_backtest

st.set_page_config(layout="wide", page_title="策略競技場", page_icon="📊")
setup_system()
st.title("📊 全自訂多股策略競技場 (多空雙向版)")

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
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

config_schema = {
    'golden': {'code': 'a', 'label': '均線黃金交叉（短均 ＞ 長均）', 'info': '⚙️ 均線參數配置', 'sliders': [('short_window', '短期均線天數', 3, 20, 5, 1), ('long_window', '長期均線天數', 10, 120, 20, 1)]},
    'volume': {'code': 'b', 'label': '量能突發性爆量（成交量 ＞ 歷史平均）', 'info': '⚡ 量能參數配置', 'sliders': [('volume_window', '計算平均成交量的天數', 5, 60, 20, 1), ('volume_threshold', '成交量爆量倍數門檻', 0.5, 3.0, 1.2, 0.1)]},
    'rsi': {'code': 'c', 'label': '逆勢逢低佈局（RSI 指標低於超賣門檻）', 'info': '📈 RSI 參數配置', 'sliders': [('rsi_window', 'RSI 計算天數', 5, 30, 14, 1), ('rsi_threshold', 'RSI 門檻值 (0-100)', 0, 100, 40, 1)]},
    'macd': {'code': 'd', 'label': '動能轉折向上（MACD 柱狀體 ＞ 0）', 'info': '🎯 MACD 參數配置', 'sliders': [('macd_fast', 'MACD 快線天數', 5, 25, 12, 1), ('macd_slow', 'MACD 慢線天數', 20, 50, 26, 1), ('macd_signal', 'MACD 訊號線天數', 5, 20, 9, 1)]}
}

base_params = {p[0]: p[4] for cfg in config_schema.values() for p in cfg['sliders']}
conds_switches, available_scan = {}, []

col_setting1, col_setting2 = st.columns(2)
with col_setting1:
    st.markdown("### 📋 步驟一：勾選進場條件與反轉方向")
    for key, cfg in config_schema.items():
        c_a, c_b = st.columns(2)
        is_chk = c_a.checkbox(f"條件 {cfg['code'].upper()}：{cfg['label']}", value=(key == 'golden'))
        v_dir = c_b.selectbox("方向", ["正向", "反轉"], key=f"dir_{key}", label_visibility="collapsed")
        conds_switches[key] = is_chk
        conds_switches[f'inv_{cfg["code"]}'] = (v_dir == "反轉")

with col_setting2:
    st.markdown("### ⚙️ 步驟二：配置已啟用條件的詳細參數")
    for key, cfg in config_schema.items():
        if conds_switches[key]:
            st.caption(cfg['info']) # 修正點：改用 st.caption，字體縮小且呈現高質感的灰色小字
            cols = st.columns(len(cfg['sliders']))
            for idx, (p_id, p_lbl, p_min, p_max, p_def, p_step) in enumerate(cfg['sliders']):
                base_params[p_id] = cols[idx].slider(p_lbl, p_min, p_max, p_def, step=p_step)
                available_scan.append(p_lbl)
    st.write("---")
    scan_target = st.selectbox("🔍 請選擇掃描參數：", options=available_scan if available_scan else ["請先勾選條件"])

btn_trigger = st.button("🔥 開始執行完整回測與參數掃描", width="stretch")

if btn_trigger and tickers and available_scan:
    st.session_state.has_run, st.session_state.success_msgs = True, []
    with st.spinner("系統正在處理歷史數據與參數窮舉..."):
        tw_market = fetch_stock_data("^TWII", start_date, end_date)
        if tw_market.empty: st.stop()
        
        m_close = tw_market['Close'].iloc[:, 0] if isinstance(tw_market['Close'], pd.DataFrame) else tw_market['Close']
        market_cum_perf = ((1 + m_close.pct_change()).cumprod() - 1) * 100
        fig_perf, ax_perf = plt.subplots(figsize=(11, 4)); fig_scan, ax_scan = plt.subplots(figsize=(11, 4))
        ax_perf.plot(tw_market.index, market_cum_perf, label="台股加權指數 (^TWII)", color="black", linestyle="--", alpha=0.7)
        
        scan_mapping = {
            "短期均線": (3, 21, 1, "short_window"), "長期均線": (10, 121, 5, "long_window"), "平均成交量": (5, 61, 5, "volume_window"),
            "爆量倍數": (0.5, 3.1, 0.1, "volume_threshold"), "RSI 計算": (5, 31, 1, "rsi_window"), "RSI 門檻": (0, 101, 5, "rsi_threshold"),
            "MACD 快線": (5, 26, 1, "macd_fast"), "MACD 慢線": (20, 51, 2, "macd_slow"), "MACD 訊號線": (5, 21, 1, "macd_signal")
        }
        key, (start, stop, step, p_field) = next(((k, v) for k, v in scan_mapping.items() if k in scan_target))
        scan_range = np.arange(start, stop, step).tolist()
        
        summary_data = []
        for tk in tickers:
            df_orig = fetch_stock_data(tk, start_date, end_date)
            if df_orig.empty: continue
            st.session_state.cache_dfs[tk] = df_orig
            
            df_base_bt = calculate_indicators_and_signals(df_orig, base_params, conds_switches)
            res_base = run_backtest(df_base_bt)
            if res_base:
                summary_data.append({
                    "標的名稱": tk, "自訂策略總報酬": f"{res_base['final_return']:.2%}", "與大盤相差 (vs ^TWII)": f"{(res_base['final_return'] - (market_cum_perf.iloc[-1]/100)):+.2%}",
                    "當前參數交易日期望值 🎯": f"{res_base['expected_return']:+.4%}", "策略勝率": f"{res_base['win_rate']:.2%}", "總交易次數": f"{res_base['trade_count']} 次"
                })
                ax_perf.plot(df_base_bt.index, res_base['cum_strat_series'], label=f"{tk} 策略", linewidth=2)
                
            scan_x, scan_y = [], []
            for p in scan_range:
                loop_params = base_params.copy(); loop_params[p_field] = p
                res_scan = run_backtest(calculate_indicators_and_signals(df_orig, loop_params, conds_switches))
                if res_scan:
                    scan_x.append(p); scan_y.append(res_scan['expected_return'] * 100)
            if scan_x and scan_y:
                ax_scan.plot(scan_x, scan_y, marker='o', label=f"{tk} 期望值趨勢", linewidth=2, alpha=0.5)
                max_idx = np.argmax(scan_y)
                ax_scan.scatter(scan_x[max_idx], scan_y[max_idx], color='red', marker='*', s=150, zorder=5)
                st.session_state.success_msgs.append(f"🎯 **{tk}** 的 **{key}** 歷史最佳：**{scan_x[max_idx]:.1f}** ({scan_y[max_idx]/100:+.4%})")
                
        for ax, y_lbl in [(ax_perf, "報酬率 (%)"), (ax_scan, "每筆交易每日平均期望值 (%)")]:
            ax.set_ylabel(y_lbl); ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
        ax_scan.axhline(0, color='red', linestyle=':', alpha=0.6)
        st.session_state.summary, st.session_state.fig_p, st.session_state.fig_s = summary_data, fig_perf, fig_scan
        st.navigation_visibility = "visible"

@st.fragment
def render_data_inspection_station():
    st.write("---")
    st.markdown("### 🔍 步驟三：策略底層數據觀測站 (逐日明細檢驗)")
    inspect_tk = st.selectbox("請選擇要檢驗明細的標的：", options=tickers, key="sb_inspect")
    if inspect_tk and inspect_tk in st.session_state.cache_dfs:
        df_dt = calculate_indicators_and_signals(st.session_state.cache_dfs[inspect_tk], base_params, conds_switches)
        d_close = df_dt['Close'].iloc[:, 0] if isinstance(df_dt['Close'], pd.DataFrame) else df_dt['Close']
        d_vol = df_dt['Volume'].iloc[:, 0] if isinstance(df_dt['Volume'], pd.DataFrame) else df_dt['Volume']
        disp = pd.DataFrame({'收盤價': d_close.round(2), '成交量': d_vol}, index=df_dt.index)
        
        mapping_cols = [('golden', 'MA_Short', '短均線'), ('golden', 'MA_Long', '長均線'), ('volume', 'Vol_Ratio', '量能倍數'), ('rsi', 'RSI', 'RSI'), ('macd', 'MACD_Hist', 'MACD柱狀體')]
        for cond, src, tgt in mapping_cols:
            if conds_switches[cond] and src in df_dt.columns: disp[tgt] = df_dt[src].round(4 if "MACD" in tgt else 2)
            
        disp['是否進場?'] = df_dt['Signal'].map({1: "🟢 持股", 0: "⚪ 空手"})
        st.dataframe(disp.reset_index().sort_values(by='Date', ascending=False).rename(columns={'Date': '交易日期'}), width="stretch", hide_index=True)

if st.session_state.has_run and st.session_state.summary:
    for msg in st.session_state.success_msgs: st.success(msg)
    st.markdown("### 🏆 策略回測成績單 (當前參數)")
    st.dataframe(pd.DataFrame(st.session_state.summary), width="stretch", hide_index=True)
    c_g1, c_g2 = st.columns(2)
    c_g1.markdown("##### 📈 圖一：歷史累積報酬率走勢"); c_g1.pyplot(st.session_state.fig_p)
    c_g2.markdown("##### 🎯 圖過：參數敏感度分析"); c_g2.pyplot(st.session_state.fig_s)
    render_data_inspection_station()
