"""기능 3: 가용 차량·인력을 입력하면 위험도 기반 사전 배치를 추천한다."""
import pandas as pd
import streamlit as st

import core
import datetime

from weather_api import WeatherAPIError
from common import DISCLAIMER, get_api_key, get_bundle, grade_icon, live_inputs

st.set_page_config(page_title="사전 자원 배치 추천", page_icon="🚒", layout="wide")
st.title("🚒 사전 자원 배치 추천")
st.info(DISCLAIMER)
b = get_bundle()
feat, rmap = b["features"], b["region_map"]

last_day = b["max_date"]  # 기상 자료가 있는 마지막 날의 다음 날
with st.sidebar:
    st.header("입력")
    api_key = get_api_key()
    use_api = bool(api_key) and st.checkbox("🌐 기상청 API로 오늘 기준 기상 불러오기", help="오늘 날짜의 전날까지 실측 기상을 가져옵니다.")
    if not api_key:
        st.caption("기상청 API 인증키가 없어 자료 기간(2024~2025) 안에서만 선택할 수 있습니다. 설정 방법은 README 참고.")
    target = st.date_input("배치 대상일", value=last_day.date(), min_value=pd.Timestamp("2024-02-01").date(),
                           max_value=last_day.date(), disabled=use_api)
    if use_api:
        target = datetime.date.today()
    vehicles = st.number_input("가용 진화 차량(대)", 0, 500, 20, 1)
    personnel = st.number_input("가용 진화 인력(명)", 0, 5000, 50, 1)
    n_regions = st.slider("우선 배치 지역 수", 1, len(rmap), 5, help=f"위험도 상위 지역부터 선택합니다. 최대 {len(rmap)}개(전체 지역).")
    st.caption("🧪 차량·인력은 실제 자료가 없어 **사용자가 입력한 가상 값**입니다.")

if vehicles == 0 and personnel == 0:
    st.warning("가용 차량이나 인력을 1 이상 입력해 주세요.")
    st.stop()
if pd.Timestamp(target) > b["fire_max_date"] and not use_api:
    st.warning(f"마지막 산불 기록일은 {b['fire_max_date']:%Y-%m-%d}입니다. 그 이후의 산불은 '최근 90일 산불 건수'에 반영되지 않습니다.")
if use_api:
    try:
        inputs = live_inputs(api_key, target)
    except WeatherAPIError as e:
        st.warning(str(e))
        st.stop()
    st.success(f"기상청 API 실측 기상으로 {target} 위험도를 계산합니다. (전날까지 자료)")
    st.warning("최근 90일 산불 건수는 보유한 산불 기록(마지막 기록일까지)만 반영되어 그 이후 산불은 빠져 있습니다. 모델도 2024~2025년 자료로 학습했습니다.")
else:
    inputs = core.inputs_for_date(feat, target)
if inputs.empty:
    st.warning("선택한 날짜의 입력 자료가 없습니다. 다른 날짜를 선택해 주세요.")
    st.stop()

st.subheader("① 관측소별 기상 조건 (전날까지 관측값 · 수정 가능)")
st.caption("각 시군은 가장 가까운 관측소의 날씨를 사용합니다. 예보 값을 알고 있다면 표에서 직접 고쳐 보세요.")
sw = inputs.groupby("관측소")[core.WEATHER_INPUTS].first().round(1).reset_index()
edited = st.data_editor(sw, hide_index=True, disabled=["관측소"], key=f"sw-{target}", width="stretch")
if edited[core.WEATHER_INPUTS].isna().any().any():
    st.warning("표에 빈 값이 있습니다. 모든 칸을 채워 주세요.")
    st.stop()
if not edited["전일_습도"].between(0, 100).all() or (edited[["전일_풍속", "최근7일_강수합", "건조지속일수"]] < 0).any().any():
    st.warning("습도는 0~100, 풍속·강수량·연속일수는 0 이상이어야 합니다.")
    st.stop()

risk = core.predict_risk(b, core.apply_station_inputs(inputs, edited, target))
plan = core.allocate_resources(risk, int(vehicles), int(personnel), n_regions)

st.subheader("② 사전 배치 우선지역과 배분 결과")
if (plan["차량"] == 0).any() or (plan["인력"] == 0).any():
    st.caption("가용 수량에 비해 우선 배치 지역이 많아 일부 지역은 차량 또는 인력이 0으로 배분됩니다. 지역 수를 줄이거나 수량을 늘려 보세요.")
c1, c2, c3 = st.columns(3)
c1.metric("배치 차량 / 가용", f"{plan['차량'].sum()} / {int(vehicles)}대")
c2.metric("배치 인력 / 가용", f"{plan['인력'].sum()} / {int(personnel)}명")
c3.metric("최고 위험 지역", f"{plan['지역'].iloc[0]} ({plan['위험확률'].iloc[0]:.1%})")
show = plan.assign(위험등급=plan["위험등급"].map(lambda g: f"{grade_icon(g)} {g}"))[
    ["우선순위", "지역", "관측소", "위험등급", "위험확률", "차량", "인력"]]
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"위험확률": st.column_config.NumberColumn(format="percent", help="모델이 계산한 다음 날 산불 발생 확률")})

st.subheader("③ 추천 근거")
for _, r in plan.iterrows():
    with st.expander(f"{r['우선순위']}순위 {r['지역']} — 차량 {r['차량']}대 · 인력 {r['인력']}명"):
        st.write(f"**위험 요인(예측 모델):** {r['근거']}")
        st.write(f"**배분 계산(배치 규칙):** {r['배분근거']}")

st.subheader("④ 위치")
loc = plan.merge(rmap[["지역", "위도", "경도"]], on="지역").rename(columns={"위도": "lat", "경도": "lon"}).assign(size=lambda d: 300 + d["차량"] * 400)
st.map(loc, latitude="lat", longitude="lon", size="size")
st.caption("지도의 점은 시군청 부근 *대략 좌표*이며 점 크기는 배치 차량 수에 비례합니다. 실제 대기 위치(소방서·산림청 기지)는 반영하지 않았습니다.")

with st.expander("전체 지역 위험도 표"):
    st.dataframe(risk.sort_values("위험확률", ascending=False)[["지역", "관측소", "위험등급", "위험확률", "근거"]],
                 hide_index=True, width="stretch",
                 column_config={"위험확률": st.column_config.NumberColumn(format="percent")})
st.caption("배치 규칙: 우선지역 안에서 배분량 = 가용 수량 × (지역 위험확률 ÷ 우선지역 위험확률 합), 정수는 최대잔여법으로 맞춥니다. "
           "이 규칙은 예측 모델이 아니며 입력한 가용 수량을 초과하지 않습니다.")
