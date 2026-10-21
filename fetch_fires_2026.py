"""2026-01-01 ~ 오늘 경기도 산불 기록을 산림청 API로 받아 data/산불_2026_api.csv에 저장한다.
실행: python fetch_fires_2026.py  (원본 CSV는 수정하지 않는다. 다시 실행하면 이 파일만 새로 쓴다.)
"""
import datetime
import tomllib
from pathlib import Path

import fire_api as fa

key = tomllib.load(open(Path(__file__).parent / ".streamlit" / "secrets.toml", "rb"))["KMA_SERVICE_KEY"]
raw = fa.fetch_fires(key, "2026-01-01", datetime.date.today())
table = fa.map_causes(fa.to_fire_table(raw))
out = Path(__file__).parent / "data" / "산불_2026_api.csv"
table.assign(일시=table["일시"].dt.strftime("%Y-%m-%d")).to_csv(out, index=False, encoding="utf-8-sig")
print(out.name, table.shape, table["일시"].min().date(), "~", table["일시"].max().date())
print(table["원인"].value_counts().to_string())
