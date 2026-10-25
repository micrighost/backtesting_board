import pandas as pd
import numpy as np

def calculate_indicators_and_signals(df_orig, params, conds_switches):
    """計算技術指標，並從根源修正多空反轉的數學邏輯"""
    df = df_orig.copy()
    
    # 安全提取單一 Series，防範多股欄位災難
    close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    df['Close'], df['Volume'] = close, vol
    
    conds = []
    
    # 條件 A：均線（從根源修正：正向用 >，反轉用 <，相等時兩者皆為 False）
    df['MA_Short'] = close.rolling(int(params['short_window'])).mean()
    df['MA_Long'] = close.rolling(int(params['long_window'])).mean()
    if conds_switches['golden']: 
        is_active = (df['MA_Short'] < df['MA_Long']) if conds_switches['inv_a'] else (df['MA_Short'] > df['MA_Long'])
        conds.append(is_active)
    
    # 條件 B：成交量爆量（正向：大於門檻，反轉：小於或等於門檻）
    if conds_switches['volume']:
        df['Vol_Ratio'] = vol / vol.rolling(int(params['volume_window'])).mean()
        is_active = (df['Vol_Ratio'] <= params['volume_threshold']) if conds_switches['inv_b'] else (df['Vol_Ratio'] > params['volume_threshold'])
        conds.append(is_active)
    
    # 條件 C：RSI 超賣值（正向：低於門檻超賣佈局，反轉：高於門檻超買追高）
    if conds_switches['rsi']:
        delta = close.diff()
        up = delta.clip(lower=0).ewm(com=int(params['rsi_window']) - 1, adjust=False).mean()
        down = delta.clip(upper=0).abs().ewm(com=int(params['rsi_window']) - 1, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + up / (down + 1e-9)))
        is_active = (df['RSI'] > params['rsi_threshold']) if conds_switches['inv_c'] else (df['RSI'] <= params['rsi_threshold'])
        conds.append(is_active)
    
    # 條件 D：MACD 多頭交叉（正向：柱狀體 > 0，反轉：柱狀體 < 0）
    if conds_switches['macd']:
        ema_fast = close.ewm(span=int(params['macd_fast']), adjust=False).mean()
        ema_slow = close.ewm(span=int(params['macd_slow']), adjust=False).mean()
        df['MACD_Line'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=int(params['macd_signal']), adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
        is_active = (df['MACD_Hist'] < 0) if conds_switches['inv_d'] else (df['MACD_Hist'] > 0)
        conds.append(is_active)
        
    # 整合所有條件
    df['Signal'] = np.logical_and.reduce(conds).astype(int) if conds else 0
    
    # 動態追蹤需要檢驗 NaN 的欄位
    cols_to_check = ['MA_Short', 'MA_Long']
    if conds_switches['volume']: cols_to_check.append('Vol_Ratio')
    if conds_switches['rsi']: cols_to_check.append('RSI')
    if conds_switches['macd']: cols_to_check.extend(['MACD_Line', 'MACD_Signal', 'MACD_Hist'])

    # 指標準備期 (NaN) 的日子強制歸零（這屬於標準的資料準備邏輯，非物理隔離）
    df.loc[df[cols_to_check].isna().any(axis=1), 'Signal'] = 0
    
    return df.dropna(subset=cols_to_check).copy()


def run_backtest(df_bt):
    """輸入帶有進場訊號的 DataFrame，計算回測累積報酬、真正的勝率與期望值"""
    if df_bt.empty: return None
    
    close_ser = df_bt['Close'].iloc[:, 0] if isinstance(df_bt['Close'], pd.DataFrame) else df_bt['Close']
    
    # 每日策略報酬（昨天有訊號，今天吃報酬）
    df_bt['Strat_Ret'] = close_ser.pct_change() * df_bt['Signal'].shift(1)
    cum_strat = (1 + df_bt['Strat_Ret'].fillna(0)).cumprod() - 1
    
    # 精準建立交易波段 ID (最原始、乾淨的向量化切法，不加任何後端斷路器)
    is_new_trade = (df_bt['Signal'] == 1) & (df_bt['Signal'].shift(1, fill_value=0) == 0)
    df_bt['Trade_ID'] = is_new_trade.cumsum()
    
    # 只有當持有訊號為 1 時，才是真正處於交易狀態中的日子
    trade_days = df_bt[df_bt['Signal'].shift(1) == 1].copy()
    
    if not trade_days.empty:
        # 計算每筆獨立交易的總報酬
        trade_returns = trade_days.groupby('Trade_ID')['Strat_Ret'].apply(lambda x: (1 + x).prod() - 1)
        trade_count = len(trade_returns)
        win_rate = (trade_returns > 0).sum() / trade_count if trade_count > 0 else 0.0
        expected_return = trade_returns.mean() if trade_count > 0 else 0.0
    else:
        trade_count, win_rate, expected_return = 0, 0.0, 0.0
    
    return {
        "cum_strat_series": cum_strat * 100,
        "final_return": cum_strat.iloc[-1],
        "trade_count": trade_count,
        "win_rate": win_rate,
        "expected_return": expected_return
    }
