"""기능 1: 산불 발생 이력 조회·분석."""
import datetime

import pandas as pd
import streamlit as st

import core
from common import get_bundle

st.set_page_config(page_title="산불 이력 분석", page_icon="📈", layout="wide")
st.title("📈 산불 발생 이력 조회·분석")
b = get_bundle()
fires = b["fires"]

today = datetime.date.today()
with st.sidebar:
    # 키(key)를 주면 화면이 다시 그려져도(연도 버튼 등) 선택한 값이 유지된다.
    regions = st.multiselect("지역 (비우면 전체)", sorted(fires["지역"].unique()), key="hist_regions")
    start = st.date_input("시작일", value=core.PERIOD_START.date(), min_value=core.PERIOD_START.date(),
                          max_value=today, key="hist_start")
    end = st.date_input("종료일", value=today, min_value=core.PERIOD_START.date(),
                        max_value=today, key="hist_end")

if start is None or end is None:  # 칸을 비운 경우
    st.warning("시작일과 종료일을 선택해 주세요.")
    st.stop()
if start > end:
    st.warning("시작일이 종료일보다 늦습니다. 기간을 다시 선택해 주세요.")
    st.stop()

if pd.Timestamp(end) > b["fire_max_date"]:
    st.warning(f"보유한 마지막 산불 기록일은 {b['fire_max_date']:%Y-%m-%d}입니다. 그 이후 구간의 0건은 '발생 없음'이 아니라 **기록이 없는 것**일 수 있습니다.")
if pd.Timestamp(end) > core.PERIOD_END:
    st.info("2026년 기록은 산림청 산불발생통계 API에서 받았고, 원인은 기존 표기에 맞춰 바꾼 값입니다(원인 미상은 API에 따로 없어 '기타'에 포함). 시·구 이름은 시 단위로 합쳤습니다.")

sel = core.filter_fires(fires, regions, start, end)
st.caption(f"기간 {start} ~ {end} · 지역 {', '.join(regions) if regions else '경기도 전체(자료에 있는 26개 시군)'}")
if sel.empty:
    st.warning("선택한 조건에 해당하는 산불 기록이 없습니다. 지역이나 기간을 넓혀 보세요.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("발생 건수", f"{len(sel)}건")
c2.metric("발생한 날 수", f"{sel['일시'].nunique()}일")
c3.metric("피해면적 합계", f"{sel['피해면적(ha)'].sum():.2f} ha")

# --- 연도 이동: 데이터가 쌓여도 그래프가 길어지지 않도록 한 해씩 보여 준다 ---
years = core.years_in_range(start, end)
if "year_view" not in st.session_state or st.session_state["year_view"] not in years:
    st.session_state["year_view"] = years[-1]  # 처음엔 가장 최근 연도
idx = years.index(st.session_state["year_view"])
nav_l, nav_c, nav_r = st.columns([1, 3, 1])
if nav_l.button("◀ 이전 해", disabled=idx == 0, width="stretch"):
    st.session_state["year_view"] = years[idx - 1]
    st.rerun()
if nav_r.button("다음 해 ▶", disabled=idx == len(years) - 1, width="stretch"):
    st.session_state["year_view"] = years[idx + 1]
    st.rerun()
year = st.session_state["year_view"]
y_start, y_end = core.year_window(start, end, year)
sel_y = core.filter_fires(fires, regions, y_start, y_end)
nav_c.markdown(f"<h3 style='text-align:center;margin:0'>{year}년</h3>", unsafe_allow_html=True)
nav_c.caption(f"<div style='text-align:center'>{y_start:%Y-%m-%d} ~ {y_end:%Y-%m-%d} · {len(sel_y)}건 "
              f"(조회 기간 {years[0]}~{years[-1]}년 중 {idx + 1}/{len(years)})</div>", unsafe_allow_html=True)

st.subheader(f"{year}년 월별 발생 추이")
monthly = core.monthly_counts(sel_y, y_start, y_end)
st.bar_chart(monthly, x="월", y="발생건수")
with st.expander("월별 표 보기"):
    st.dataframe(monthly, hide_index=True, width="stretch")

st.subheader(f"{year}년 월별 × 원인별 발생 건수")
if sel_y.empty:
    st.info(f"{year}년에는 선택한 조건의 산불 기록이 없습니다.")
else:
    cause_tbl = core.monthly_cause_table(sel_y, y_start, y_end)
    st.bar_chart(cause_tbl, x="월", y=[c for c in cause_tbl.columns if c != "월"], stack=True)
    unk = core.unknown_cause_summary(sel_y)
    st.write(f"**원인 미상 {unk['원인 미상']}건 + 기타 {unk['기타']}건 = {unk['합계']}건 ({unk['합계_비율']:.1%})** — "
             f"{year}년 {unk['전체']}건 중 원인을 알 수 없거나 분류되지 않은 비율입니다.")
    with st.expander("월 × 원인 표 보기"):
        st.dataframe(cause_tbl, hide_index=True, width="stretch")
st.caption("원인은 산불이 난 *뒤에* 기록되는 정보라 이 화면의 분석에만 쓰고, 위험 예측의 입력에는 사용하지 않습니다. "
           "위의 건수·피해면적 요약, 지역별·원인별 표는 조회 기간 전체 기준입니다.")

left, right = st.columns(2)
with left:
    st.subheader("지역별 건수")
    st.dataframe(core.region_summary(sel), hide_index=True, width="stretch")
with right:
    st.subheader("원인별 건수·비율")
    st.dataframe(core.cause_share(sel), hide_index=True, width="stretch")

with st.expander("산불 기록 원자료"):
    st.dataframe(sel[["일시", "지역", "발생장소", "원인", "피해면적(ha)"]].sort_values("일시", ascending=False)
                 .assign(일시=lambda d: d["일시"].dt.date), hide_index=True, width="stretch")
st.caption("출처: 산림청 국가산불정보시스템 경기도 산불통계(산불_기상_병합.csv). 기록이 없는 달은 0건으로 표시합니다.")
