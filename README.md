# 데이터 구성

미국:
- FRED API
- yfinance

한국:
- ECOS API
- yfinance

한국 ECOS는 통계표 자동 검색을 사용하므로 특정 테이블 코드가 변경되더라도 검색 가능한 경우 대응할 수 있습니다.
다만 실제 운용 전에는 각 ECOS 항목이 원하는 지표인지 반드시 검증하세요.

## 프로젝트 구조
```
app.py
core/       pipeline.py, features.py, backtest.py, settings.py
models/     classifier.py
providers/  fred.py, ecos.py, market.py
```
각 폴더에는 `__init__.py`가 필요합니다. 실행: `streamlit run app.py`
