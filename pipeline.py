import numpy as np
import pandas as pd

from providers.market import load_market, MARKET_CONFIG
from providers.fred import load_us_macro
from providers.ecos import get_metric
from core.features import build_features, make_label
from models.classifier import fit_model, predict, diagnostics
from core.backtest import walk_forward, walk_forward_summary, walk_forward_report

def _load_korea_macro(key, start):
    # ECOS table discovery is deliberately used rather than hard-coding fragile table codes.
    # If an item is not found, the pipeline keeps the remaining available features.
    specs = [
        ("기준금리", ["기준금리"]),
        ("소비자물가", ["소비자물가"]),
        ("생산자물가", ["생산자물가"]),
        ("M2", ["M2"]),
    ]
    out, missing = [], []
    for name, kw in specs:
        try:
            s = get_metric(key, kw, cycle="MM", start=pd.Timestamp(start).strftime("%Y%m"))
            s.name = name
            out.append(s)
        except Exception:
            missing.append(name)
    if not out:
        raise RuntimeError("ECOS 데이터를 가져오지 못했습니다. ECOS API Key와 API 접근권한을 확인하세요.")
    df = pd.concat(out, axis=1).resample("ME").last()
    df.attrs["missing"] = missing
    return df

def _load_macro(market, key, start):
    if market == "미국":
        return load_us_macro(key, start)
    return _load_korea_macro(key, start)

def _latest_row(features, price):
    """예측 시점 = 해당 가격 시계열의 가장 최근 시점. (라벨이 없는 최신 행이어야 한다)"""
    last = price.index[-1]
    if last not in features.index:
        return None
    return features.loc[[last]]

def _stock_predictions(market, stock_prices, features, horizon, min_train, calibrate, cutoff):
    rows = []
    for name, ticker in MARKET_CONFIG[market]["stocks"].items():
        if ticker not in stock_prices.columns:
            continue
        price = stock_prices[ticker].dropna()
        if len(price) < 24:
            continue
        y, fwd = make_label(price, horizon)
        x_now = _latest_row(features, price)
        if x_now is None:
            continue
        common = features.index.intersection(y.index)
        common = common[common >= cutoff]
        X = features.reindex(common)
        yy = y.reindex(common)
        valid = yy.notna()
        X, yy = X.loc[valid], yy.loc[valid].astype(int)
        if len(X) < min_train or yy.nunique() < 3:
            continue
        model = fit_model(X, yy, calibrate=calibrate)
        p = predict(model, x_now)
        recent = fwd.dropna().tail(60)
        rows.append({
            "종목": name,
            "티커": ticker,
            "상승확률": p["상승"],
            "횡보확률": p["횡보"],
            "하락확률": p["하락"],
            "예상수익률 중앙값": float(recent.median()) if not recent.empty else np.nan,
        })
    return pd.DataFrame(rows)

def run_pipeline(market, target, horizon, train_years, settings, min_train=60, calibrate=True):
    today = pd.Timestamp.today()
    # rolling z-score(36개월) + 12개월 YoY 계산으로 앞부분이 NaN이 되므로 여유 기간을 더 불러온다.
    start = today - pd.DateOffset(years=train_years + 4)
    cutoff = (today - pd.DateOffset(years=train_years)).to_period("M").to_timestamp("M")
    macro = _load_macro(market, settings.fred_key if market == "미국" else settings.ecos_key, start)
    market_prices = load_market(market, start)

    features = build_features(market_prices, macro, target)
    price = market_prices[target].dropna()
    y, fwd = make_label(price, horizon)

    # 현재 예측 시점: 타깃 가격의 가장 최근 시점 (아직 미래 수익률이 없어 라벨은 NaN)
    x_now = _latest_row(features, price)
    if x_now is None:
        raise RuntimeError("최신 시점의 feature를 만들 수 없습니다.")

    # 학습 표본: 라벨이 확정된 행 중 학습기간(train_years) 안의 행
    idx = features.index.intersection(y.index)
    idx = idx[idx >= cutoff]
    X = features.reindex(idx)
    yy = y.reindex(idx)
    valid = yy.notna()
    X, yy = X.loc[valid], yy.loc[valid].astype(int)

    if len(X) < min_train or yy.nunique() < 3:
        raise RuntimeError(
            f"학습 데이터가 부족합니다. 현재 {len(X)}개 표본이며 최소 {min_train}개가 필요합니다. "
            "학습 기간을 늘리거나 '최소 학습 표본'을 줄여 보세요."
        )

    model = fit_model(X, yy, calibrate=calibrate)
    probs = predict(model, x_now)

    # 표본내(in-sample) 지표는 과적합으로 부풀려지므로 표본외(워크포워드) 지표를 함께 제공
    diag = {f"표본내 {k}" if k != "표본" else k: v for k, v in diagnostics(model, X, yy).items()}
    bt_full = walk_forward(X, yy, horizon=horizon, min_train=min(min_train, max(len(X) // 2, 30)), calibrate=False)
    diag.update(walk_forward_summary(bt_full))
    bt = walk_forward_report(bt_full)

    stock_pred = _stock_predictions(
        market, market_prices, features, horizon, min_train, calibrate, cutoff
    )

    missing = list(macro.attrs.get("missing", []))
    return {
        "asof": x_now.index[0],
        "n_train": len(X),
        "target_label": MARKET_CONFIG[market]["labels"].get(target, target),
        "probabilities": probs,
        "diagnostics": diag,
        "current_features": x_now.iloc[0].dropna(),
        "stock_predictions": stock_pred,
        "backtest": bt,
        "missing": missing,
        "live": not missing,
    }
