import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models.classifier import fit_model, predict

LABEL_TO_INT = {"하락": 0, "횡보": 1, "상승": 2}

def walk_forward(X, y, horizon, min_train=60, step=1, calibrate=False):
    """워크포워드 백테스트.

    시점 i의 예측에는 '시점 i에 이미 라벨이 확정된' 행만 사용한다.
    행 j의 라벨은 j+horizon 개월 뒤 가격을 쓰므로 X.index[j] <= X.index[i] - horizon개월 인 행만 학습에 쓴다(엠바고).
    """
    rows = []
    for i in range(min_train, len(X), step):
        cutoff = X.index[i] - pd.DateOffset(months=horizon)
        mask = X.index <= cutoff
        Xtr, ytr = X.loc[mask], y.loc[mask]
        if len(Xtr) < 30 or ytr.nunique() < 3:
            continue
        model = fit_model(Xtr, ytr, calibrate=calibrate)
        p = predict(model, X.iloc[[i]])
        actual = int(y.iloc[i])
        pred_label = LABEL_TO_INT[max(p, key=p.get)]
        rows.append({
            "date": X.index[i],
            "actual": actual,
            "predicted": pred_label,
            "hit": int(pred_label == actual),
            "상승확률": p["상승"],
            "횡보확률": p["횡보"],
            "하락확률": p["하락"],
        })
    return pd.DataFrame(rows)

def walk_forward_summary(df):
    """전체 워크포워드 결과의 표본외 지표."""
    if df is None or df.empty:
        return {}
    return {
        "OOS Accuracy": round(float(accuracy_score(df["actual"], df["predicted"])), 3),
        "OOS Balanced Accuracy": round(float(balanced_accuracy_score(df["actual"], df["predicted"])), 3),
        "OOS 표본": int(len(df)),
    }

def walk_forward_report(df):
    """워크포워드 결과에서 최근 24개월 표와 누적 정확도(전체 기간 기준)를 만든다."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.assign(정확도=df["hit"].expanding().mean().round(3))
    return df.tail(24)
