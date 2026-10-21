"""기능 2: 지역과 기상 조건을 입력하면 예상 위험도를 보여 준다."""
import pandas as pd
import streamlit as st

import core
import datetime

from weather_api import WeatherAPIError
from common import DISCLAIMER, get_api_key, get_bundle, grade_icon, live_inputs

st.set_page_config(page_title="산불 위험도 확인", page_icon="🌡️", layout="wide")
st.title("🌡️ 산불 위험도 확인")
st.info(DISCLAIMER)
b = get_bundle()
feat, rmap = b["features"], b["region_map"]

last_day = b["max_date"]  # 기상 자료가 있는 마지막 날의 다음 날
c1, c2 = st.columns(2)
region = c1.selectbox("지역", rmap["지역"])
api_key = get_api_key()
use_api = bool(api_key) and st.checkbox("🌐 기상청 API로 오늘 기준 기상 불러오기", help="오늘 날짜의 전날까지 실측 기상을 가져옵니다.")
if not api_key:
    st.caption("기상청 API 인증키가 없어 자료 기간 안에서만 선택할 수 있습니다. 설정 방법은 README 참고.")
target = c2.date_input("예측 대상일", value=last_day.date(), min_value=pd.Timestamp("2024-02-01").date(),
                       max_value=last_day.date(),
                       help="자료 마지막 날의 다음 날이 '내일 예측'입니다(2026년은 기상청 API 자료). 과거 날짜를 고르면 그날의 전날 기상으로 재현합니다.",
                       disabled=use_api)
if use_api:
    target = datetime.date.today()
cal = core.calendar_features([target]).iloc[0]
day_kind = "공휴일" if cal["공휴일"] else ("주말" if cal["주말"] else "평일")
st.caption(f"예측 대상일: {target} ({core.WEEKDAY_KO[pd.Timestamp(target).weekday()]}요일, {day_kind}) — 달력 변수는 날짜로 자동 계산됩니다.")
st_name = rmap.set_index("지역").loc[region, "관측소"]
dist = rmap.set_index("지역").loc[region, "거리_km"]
st.caption(f"기상 연결: {region} → **{st_name} 관측소** (직선거리 약 {dist}km). 아래 값은 전날까지의 실제 관측으로 채워져 있으며 직접 고칠 수 있습니다.")

if pd.Timestamp(target) > b["fire_max_date"] and not use_api:
    st.warning(f"마지막 산불 기록일은 {b['fire_max_date']:%Y-%m-%d}입니다. 그 이후의 산불은 '최근 90일 산불 건수'에 반영되지 않습니다.")
if use_api:
    try:
        base_rows = live_inputs(api_key, target)
    except WeatherAPIError as e:
        st.warning(str(e))
        st.stop()
    st.success(f"기상청 API 실측 기상({target} 전날까지)으로 채웠습니다. 최근 90일 산불 건수는 보유한 산불 기록까지만 반영되니 필요하면 직접 고치세요.")
else:
    base_rows = core.inputs_for_date(feat, target)
base = base_rows[base_rows["지역"] == region]
if base.empty:
    st.warning("선택한 날짜의 입력 자료가 없습니다. 다른 날짜를 선택해 주세요.")
    st.stop()
base = base.iloc[0]
k = f"{region}-{target}"
st.subheader("기상 조건 입력 (예측 대상일 *전날까지*의 값)")
a, b2, c = st.columns(3)
temp = a.number_input("전일 평균기온(°C)", -40.0, 50.0, float(round(base["전일_기온"], 1)), 0.5, key="t" + k)
hum = b2.number_input("전일 평균 상대습도(%)", 0.0, 100.0, float(round(base["전일_습도"], 1)), 1.0, key="h" + k)
wind = c.number_input("전일 평균 풍속(m/s)", 0.0, 40.0, float(round(base["전일_풍속"], 1)), 0.1, key="w" + k)
d, e, f = st.columns(3)
rain7 = d.number_input("최근 7일 누적 강수량(mm)", 0.0, 1000.0, float(round(base["최근7일_강수합"], 1)), 1.0, key="r" + k)
dry = e.number_input("강수 1mm 미만 연속일수", 0, 30, int(base["건조지속일수"]), 1, key="d" + k)
hist = f.number_input("최근 90일 이 지역 산불 건수", 0, 50, int(base["최근90일_산불건수"]), 1, key="f" + k)

row = base.copy()
row["월_sin"], row["월_cos"] = core.month_features(pd.Timestamp(target))
row[core.WEATHER_INPUTS] = [temp, hum, wind, rain7, dry]
row["최근90일_산불건수"] = hist
res = core.predict_risk(b, pd.DataFrame([row])).iloc[0]

st.subheader("예상 위험도")
base_rate = b["features"].loc[b["features"]["is_label"], "산불발생"].mean()
m1, m2, m3 = st.columns(3)
m1.metric("위험 등급", f"{grade_icon(res['위험등급'])} {res['위험등급']}")
m2.metric("다음 날 산불 발생 확률(모델)", f"{res['위험확률']:.1%}")
m3.metric("평소(전체 평균) 대비", f"{res['위험확률'] / base_rate:.1f}배", help=f"2024~2025년 하루 평균 발생률 {base_rate:.2%}")
st.write(f"**근거:** {res['근거']}")
th = b["thresholds"]
st.caption(f"등급 기준(학습기간 예측 확률 분포): 주의 ≥ {th['주의']:.1%} · 높음 ≥ {th['높음']:.1%} · 매우 높음 ≥ {th['매우 높음']:.1%}. "
           "확률 자체가 작은 것은 산불이 드문 사건이기 때문이며, 지역 간·날짜 간 *상대 비교*에 쓰는 것이 적절합니다.")
