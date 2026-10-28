import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import setup_system, load_ticker_options
from engine import calculate_indicators_and_signals, run_backtest
from data_loader import db_fetch_stock_data

st.set_page_config(layout="wide", page_title="策略競技場", page_icon="📊")
setup_system()
st.title("📊 全自訂多股策略競技場 (打臉自己或是證明自己)")

@st.cache_data(show_spinner=False)
def fetch_stock_data(tk, s, e): return db_fetch_stock_data(tk, s, e)

if "has_run" not in st.session_state:
    st.session_state.update({"has_run": False, "summary": None, "fig_p": None, "fig_s": None, "cache_dfs": {}, "success_msgs": []})

tk_opts, df_sel = load_ticker_options()
c1, c2, c3 = st.columns(3)
tickers = c1.multiselect("請選擇標的：", options=tk_opts, default=df_sel, key="ms_tk")
start_date = pd.to_datetime(c2.date_input("開始日期", pd.to_datetime("2022-01-01"), key="di_s"))
end_date = pd.to_datetime(c3.date_input("結束日期", pd.to_datetime("2026-01-01"), key="di_e"))
st.write("---")

config_schema = {
    'golden': ('a', '均線黃金交叉（短均 ＞ 長均）', '⚙️ 均線參數', [('short_window', '短期均線天數', 3, 20, 5, 1), ('long_window', '長期均線天數', 10, 120, 20, 1)]),
    'volume': ('b', '量能突發性爆量（成交量 ＞ 歷史平均）', '⚡ 量能參數', [('volume_window', '計算平均成交量的天數', 5, 60, 20, 1), ('volume_threshold', '成交量爆量倍數門檻', 0.5, 3.0, 1.2, 0.1)]),
    'rsi': ('c', '逆勢逢低佈局（RSI 指標低於超賣門檻）', '📈 RSI 參數', [('rsi_window', 'RSI 計算天數', 5, 30, 14, 1), ('rsi_threshold', 'RSI 門檻值 (0-100)', 0, 100, 40, 1)]),
    'macd': ('d', '動能轉折向上（MACD 柱狀體 ＞ 0）', '🎯 MACD 參數', [('macd_fast', 'MACD 快線天數', 5, 25, 12, 1), ('macd_slow', 'MACD 慢線天數', 20, 50, 26, 1), ('macd_signal', 'MACD 訊號線天數', 5, 20, 9, 1)])
}
base_params = {p[0]: p[4] for cfg in config_schema.values() for p in cfg[3]}
conds_switches, available_scan = {}, []

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("### 📋 步驟一：勾選進場條件與反轉方向")
    for k, (code, lbl, _, _) in config_schema.items():
        ca, cb = st.columns(2)
        conds_switches[k] = ca.checkbox(f"條件 {code.upper()}：{lbl}", value=(k == 'golden'))
        conds_switches[f'inv_{code}'] = (cb.selectbox("方向", ["正向", "反轉"], key=f"dir_{k}", label_visibility="collapsed") == "反轉")

with col_s2:
    st.markdown("### ⚙️ 步驟二：配置已啟用條件的詳細參數")
    for k, (_, _, info, sliders) in config_schema.items():
        if conds_switches[k]:
            st.caption(info)
            cols = st.columns(len(sliders))
            for idx, (p_id, p_lbl, p_min, p_max, p_def, p_step) in enumerate(sliders):
                base_params[p_id] = cols[idx].slider(p_lbl, p_min, p_max, p_def, step=p_step)
                available_scan.append(p_lbl)
st.write("---")

scan_map = {"短期均線": (3, 21, 1, "short_window"), "長期均線": (10, 121, 5, "long_window"), "平均成交量": (5, 61, 5, "volume_window"), "爆量倍數": (0.5, 3.1, 0.1, "volume_threshold"), "RSI 計算": (5, 31, 1, "rsi_window"), "RSI 門檻": (0, 101, 5, "rsi_threshold"), "MACD 快線": (5, 26, 1, "macd_fast"), "MACD 慢線": (20, 51, 2, "macd_slow"), "MACD 訊號線": (5, 21, 1, "macd_signal")}
active_scans = [k for k in scan_map.keys() if any(k in label for label in available_scan)]
scan_target = st.selectbox("🔍 請選擇掃描參數：", options=active_scans if active_scans else ["請先勾選條件"])
btn_trigger = st.button("🔥 開始執行完整回測與參數掃描", width='stretch')

if btn_trigger and tickers and active_scans:
    st.session_state.has_run, st.session_state.success_msgs, summary_data = True, [], []
    with st.spinner("系統正在處理歷史數據與參數窮舉..."):
        tw_market = fetch_stock_data("^TWII", start_date, end_date)
        if not tw_market.empty:
            m_close = tw_market['Close'].iloc[:, 0] if isinstance(tw_market['Close'], pd.DataFrame) else tw_market['Close']
            m_perf = ((1 + m_close.pct_change()).cumprod() - 1) * 100
            fig_p, ax_p = plt.subplots(figsize=(11, 4))
            fig_s, ax_s = plt.subplots(figsize=(11, 4))
            ax_p.plot(tw_market.index, m_perf, label="台股加權指數 (^TWII)", color="black", linestyle="--", alpha=0.7)
            
            lbl_key, (start, stop, step, p_field) = next((k, v) for k, v in scan_map.items() if k in scan_target)
            scan_range = np.arange(start, stop, step).tolist()
            
            for tk in tickers:
                df_orig = fetch_stock_data(tk, start_date, end_date)
                if df_orig.empty: continue
                st.session_state.cache_dfs[tk] = df_orig
                
                df_base = calculate_indicators_and_signals(df_orig, base_params, conds_switches)
                res_b = run_backtest(df_base)
                if res_b:
                    summary_data.append({"標的名稱": tk, "自訂策略總報酬": f"{res_b['final_return']:.2%}", "與大盤相差 (vs ^TWII)": f"{(res_b['final_return'] - (m_perf.iloc[-1]/100)):+.2%}", "當前參數交易日期望值 🎯": f"{res_b['expected_return']:+.4%}", "策略勝率": f"{res_b['win_rate']:.2%}", "總交易次數": f"{res_b['trade_count']} 次"})
                    ax_p.plot(df_base.index, res_b['cum_strat_series'], label=f"{tk} 策略", linewidth=2)
                
                sx, sy = [], []
                for p in scan_range:
                    lp = base_params.copy(); lp[p_field] = p
                    res_s = run_backtest(calculate_indicators_and_signals(df_orig, lp, conds_switches))
                    if res_s: sx.append(p); sy.append(res_s['expected_return'] * 100)
                if sx and sy:
                    ax_s.plot(sx, sy, marker='o', label=f"{tk} 期望值", linewidth=2, alpha=0.5)
                    mi = np.argmax(sy)
                    ax_s.scatter(sx[mi], sy[mi], color='red', marker='*', s=150, zorder=5)
                    st.session_state.success_msgs.append(f"🎯 **{tk}** 的 **{lbl_key}** 最佳：**{sx[mi]:.1f}** ({sy[mi]/100:+.4%})")
            
            for ax, y_lbl in [(ax_p, "報酬率 (%)"), (ax_s, "平均期望值 (%)")]:
                ax.set_ylabel(y_lbl); ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
            ax_s.axhline(0, color='red', linestyle=':', alpha=0.6)
            st.session_state.summary, st.session_state.fig_p, st.session_state.fig_s = summary_data, fig_p, fig_s

if st.session_state.has_run:
    for m in st.session_state.success_msgs: st.success(m)
    
    st.write("---")
    st.markdown("### 📊 策略競技場回測結果與參數敏感度分析")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("#### 📈 累積報酬率對抗曲線 (vs 大盤)")
        if st.session_state.fig_p: st.pyplot(st.session_state.fig_p)
    with fig_col2:
        st.markdown("#### 🎯 參數掃描每筆交易每日平均期望值")
        if st.session_state.fig_s: st.pyplot(st.session_state.fig_s)
        
    st.write("---")
    if st.session_state.summary: st.dataframe(pd.DataFrame(st.session_state.summary), width='stretch')

@st.fragment
def render_data_inspection_station():
    st.write("---")
    st.markdown("### 🔍 步驟三：策略底層數據觀測站 (逐日明細檢驗)")
    if tickers and (inspect_tk := st.selectbox("請選擇要檢驗明細的標的：", options=tickers, key="sb_inspect")) in st.session_state.cache_dfs:
        df_dt = calculate_indicators_and_signals(st.session_state.cache_dfs[inspect_tk], base_params, conds_switches)
        dc = df_dt['Close'].iloc[:, 0] if isinstance(df_dt['Close'], pd.DataFrame) else df_dt['Close']
        dv = df_dt['Volume'].iloc[:, 0] if isinstance(df_dt['Volume'], pd.DataFrame) else df_dt['Volume']
        disp = pd.DataFrame({'收盤價': dc.round(2), '成交量': dv}, index=df_dt.index)
        for cond, src, tgt in [('golden', 'MA_Short', '短均線'), ('golden', 'MA_Long', '長均線'), ('volume', 'Vol_Ratio', '量能倍數'), ('rsi', 'RSI', 'RSI'), ('macd', 'MACD_Hist', 'MACD柱狀體')]:
            if conds_switches[cond] and src in df_dt.columns:
                disp[tgt] = df_dt[src].round(4 if "MACD" in tgt else 2)
        st.dataframe(disp, width='stretch')

if st.session_state.has_run: render_data_inspection_station()
