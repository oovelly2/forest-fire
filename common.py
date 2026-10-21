"""화면 공통 도우미 (streamlit 사용 가능). 계산 로직은 core.py에 있다."""
import streamlit as st

import core

DISCLAIMER = ("⚠️ 이 위험도는 2024~2025년 자료로 학습한 통계 모델의 **참고용 예측값**입니다. "
              "실제 기상예보가 아니며, 배치 결정은 담당자의 판단을 대신하지 않습니다.")


@st.cache_resource(show_spinner="자료를 읽고 모델을 학습하는 중입니다…")
def get_bundle() -> dict:
    """CSV 로드 → 전처리 → 특성 → 모델 학습을 한 번만 실행하고 결과를 재사용한다."""
    return core.load_all()


def grade_icon(grade: str) -> str:
    """등급에 맞는 색 아이콘."""
    return {"낮음": "🟢", "주의": "🟡", "높음": "🟠", "매우 높음": "🔴"}[grade]


def get_api_key() -> str:
    """기상청 인증키를 .streamlit/secrets.toml(KMA_SERVICE_KEY) 또는 환경변수에서 읽는다. 없으면 빈 문자열."""
    import os
    try:
        key = st.secrets["KMA_SERVICE_KEY"]
    except Exception:  # secrets 파일이 없거나 키가 없음
        key = os.environ.get("KMA_SERVICE_KEY", "")
    return str(key).strip()


@st.cache_data(ttl=3600, show_spinner="기상청 API에서 최근 기상을 가져오는 중입니다…")
def live_inputs(key: str, target_date):
    """target_date 전날까지의 실측 기상으로 지역별 입력 특성을 만든다 (1시간 캐시). 오류는 WeatherAPIError."""
    import weather_api
    b = get_bundle()
    raw = weather_api.fetch_recent_weather(key, target_date)
    return weather_api.build_live_inputs(raw, target_date, b["fires"], b["region_map"])
