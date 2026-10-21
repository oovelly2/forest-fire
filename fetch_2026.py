"""2026-01-01 ~ 어제의 기상 일자료를 기상청 API로 받아 data/기상_2026_api.csv에 저장한다.
실행: python fetch_2026.py   (원본 CSV는 수정하지 않는다. 다시 실행하면 이 파일만 새로 쓴다.)
"""
import datetime
import tomllib
from pathlib import Path

import pandas as pd

import weather_api as wa

key = tomllib.load(open(Path(__file__).parent / ".streamlit" / "secrets.toml", "rb"))["KMA_SERVICE_KEY"]
end = datetime.date.today() - datetime.timedelta(days=1)
raw = wa.fetch_range(key, "2026-01-01", end)
raw["지점"] = raw["지점명"].map(wa.STATION_IDS)
cols = ["지점", "지점명", "일시", "평균기온(°C)", "일강수량(mm)", "평균 풍속(m/s)", "평균 상대습도(%)"]
raw = raw.drop_duplicates(["지점명", "일시"]).sort_values(["지점명", "일시"])[cols]
raw["일시"] = raw["일시"].dt.strftime("%Y-%m-%d")
out = Path(__file__).parent / "data" / "기상_2026_api.csv"
raw.to_csv(out, index=False, encoding="utf-8-sig")
print(out.name, raw.shape, raw["일시"].min(), raw["일시"].max())
print(raw.groupby("지점명").size().to_string())
print(raw.isna().sum().to_string())
