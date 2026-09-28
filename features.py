import numpy as np
import pandas as pd

def pct(s, periods=1):
    return s.pct_change(periods)

def zscore(s, window=36):
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd.replace(0, np.nan)

# 지표별 발표 지연(개월). 해당 월 값은 그 달 말에 알 수 없으므로 shift해서 미래 정보 누수를 줄인다.
PUBLICATION_LAG = {
    "cpi": 1, "core_cpi": 1, "pce": 1, "unrate": 1, "payems": 1,
    "indpro": 1, "m2": 1, "fedfunds": 1, "real_gdp": 3,
}
DEFAULT_LAG = 1  # 한국 ECOS 등 그 외 월간 지표 기본값. 일간 시장지표(10y, 2y, vix, hy_spread)는 0.
NO_LAG = {"10y", "2y", "vix", "hy_spread"}

def _apply_lag(macro_df):
    out = macro_df.copy()
    for c in out.columns:
        lag = 0 if c in NO_LAG else PUBLICATION_LAG.get(c, DEFAULT_LAG)
        if lag:
            out[c] = out[c].shift(lag)
    return out

def build_features(market_df, macro_df, target):
    macro_df = _apply_lag(macro_df)
    idx = market_df.index.union(macro_df.index).sort_values()
    m = market_df.reindex(idx).ffill()
    x = macro_df.reindex(idx).ffill()

    # Macro transformations
    out = pd.DataFrame(index=idx)
    for c in x.columns:
        out[f"{c}_chg"] = pct(x[c], 1)
        out[f"{c}_z"] = zscore(x[c])

    if "10y" in x.columns and "2y" in x.columns:
        out["curve_10y2y"] = x["10y"] - x["2y"]
    if "fedfunds" in x.columns and "10y" in x.columns:
        out["term_premium_proxy"] = x["10y"] - x["fedfunds"]
    if "cpi" in x.columns:
        out["cpi_yoy"] = x["cpi"].pct_change(12)
    if "core_cpi" in x.columns:
        out["core_cpi_yoy"] = x["core_cpi"].pct_change(12)
    if "m2" in x.columns:
        out["m2_yoy"] = x["m2"].pct_change(12)
    if "indpro" in x.columns:
        out["indpro_yoy"] = x["indpro"].pct_change(12)
    if "payems" in x.columns:
        out["payems_yoy"] = x["payems"].pct_change(12)

    px = m[target].dropna()
    out["market_ret_1m"] = px.pct_change(1)
    out["market_ret_3m"] = px.pct_change(3)
    out["market_ret_6m"] = px.pct_change(6)
    out["market_vol_3m"] = px.pct_change().rolling(3).std()
    out["market_mom_12m"] = px.pct_change(12)

    return out.replace([np.inf, -np.inf], np.nan)

def make_label(price, horizon):
    fwd = price.shift(-horizon) / price - 1
    # Neutral band scales with horizon to reduce noisy classifications.
    band = {3: 0.03, 6: 0.05, 12: 0.08}.get(horizon, 0.05)
    y = pd.Series(np.nan, index=price.index)
    y[fwd > band] = 2
    y[(fwd >= -band) & (fwd <= band)] = 1
    y[fwd < -band] = 0
    return y, fwd
