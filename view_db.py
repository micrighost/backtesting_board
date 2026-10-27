import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1. 初始化網頁基本配置
st.set_page_config(layout="wide", page_title="雲端資料庫獨立主控台", page_icon="🗃️")
st.title("🗃️ 雲端資料庫核心數據觀測站 (獨立管理後台)")
st.markdown("本工具獨立連線遠端 **Render PostgreSQL**，專門提供數據稽核與實體表檢視，不影響回測主程式。")
st.write("---")

# 2. 載入環境變換並延遲初始化連線引擎
load_dotenv()
raw_url = os.getenv("DB_URL", "").strip()

if raw_url.startswith("postgresql://"):
    raw_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
if ".render.com" in raw_url and ":5432" not in raw_url:
    raw_url = raw_url.replace("://", "://", 1)

if not raw_url:
    st.error("❌ 找不到 DB_URL，請確認 .env 檔案是否存在且設定正確。")
    st.stop()

@st.cache_resource
def get_view_engine():
    """建立獨立的全局資料庫引擎並加入安全與快取參數"""
    return create_engine(raw_url, connect_args={"connect_timeout": 5}, pool_pre_ping=True)

engine = get_view_engine()

# 3. 執行資料庫實體表讀取與渲染
try:
    with engine.connect() as conn:
        # 看板 A：庫存與健康度總覽
        st.markdown("### 📊 一、資料庫庫存快報 (總覽統計)")
        
        # 檢查表是否存在
        table_check = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'stock_prices'
            );
        """)).scalar()
        
        if not table_check:
            st.warning("⚠️ 警告：目前資料庫中尚未建立 'stock_prices' 資料表，請先執行 `insert_data.py` 初始化！")
            st.stop()
            
        query_stats = text("""
            SELECT ticker as "股票代號", 
                   COUNT(*) as "總總筆數", 
                   MIN(date)::date as "歷史起點", 
                   MAX(date)::date as "資料更新至" 
            FROM stock_prices 
            GROUP BY ticker 
            ORDER BY ticker ASC
        """)
        df_stats = pd.read_sql(query_stats, conn)
        
        if df_stats.empty:
            st.info("ℹ️ 資料表目前完全是空的，請先執行自動補件腳本注入資料。")
        else:
            st.dataframe(df_stats, width='stretch')
            st.write("---")
            
            # 看板 B：細部實體表檢視（Excel 抽查風格）
            st.markdown("### 🔍 二、實體資料表抽查 (Excel 視覺化風格)")
            
            # 建立篩選與控制元件
            col_ctrl1, col_ctrl2 = st.columns([1, 2])
            with col_ctrl1:
                target_tk = st.selectbox("請選擇欲抽查的股票標的：", options=df_stats["股票代號"].tolist(), key="sb_target_tk")
            with col_ctrl2:
                row_limit = st.slider("請選擇欲檢視的資料筆數 (由最新交易日往前算)：", 10, 1000, 100, step=10)
                
            # 撈取明細
            query_details = text("""
                SELECT date as "交易日期", "Open" as "開盤價", "High" as "最高價", 
                       "Low" as "最低價", "Close" as "收盤價", "Volume" as "成交量"
                FROM stock_prices 
                WHERE ticker = :ticker 
                ORDER BY date DESC 
                LIMIT :limit
            """)
            df_details = pd.read_sql(query_details, conn, params={"ticker": target_tk, "limit": row_limit})
            
            if not df_details.empty:
                # 轉成漂亮的純日期格式
                df_details["交易日期"] = pd.to_datetime(df_details["交易日期"]).dt.strftime('%Y-%m-%d')
                st.dataframe(df_details.set_index("交易日期"), width='stretch')
            else:
                st.warning(f"標的 {target_tk} 在指定範圍內無具體明細。")
                
except Exception as e:
    st.error(f"💥 讀取遠端資料庫實體表時發生錯誤: {e}")
