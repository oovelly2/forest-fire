"""기상청 ASOS 일자료 API 클라이언트 (공공데이터포털 '기상청_지상(종관, ASOS) 일자료 조회서비스').

streamlit을 쓰지 않는 순수 파이썬 모듈이다. 인증키는 코드에 쓰지 않고 호출하는 쪽에서 넘겨준다.

왜 '일자료(관측)'인가?
  우리 모델은 '전날까지의 관측'(전일 기온·습도·풍속, 최근 7일 강수합 등)을 입력으로 학습했다.
  기상 *예보*(단기예보)는 값의 성격이 달라 그대로 넣으면 학습 때와 다른 입력이 된다.
  그래서 학습 자료와 같은 종류인 ASOS 일자료를 가져와 core.add_features로 같은 방식의 입력을 만든다.
"""
from __future__ import annotations

import pandas as pd
import requests

import core

URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
# 학습 CSV의 '지점' 번호와 같다 (기상_산불_일별.csv)
STATION_IDS = {"동두천": 98, "파주": 99, "수원": 119, "양평": 202, "이천": 203}
# API 응답 항목 -> 학습 CSV 열 이름
FIELD_MAP = {"avgTa": "평균기온(°C)", "sumRn": "일강수량(mm)", "avgWs": "평균 풍속(m/s)", "avgRhm": "평균 상대습도(%)"}


class WeatherAPIError(Exception):
    """API 호출·응답 오류 (키 오류, 네트워크 오류, 자료 없음 등). 화면에서 메시지를 그대로 보여 준다."""


def parse_asos_response(payload: dict, station_name: str) -> pd.DataFrame:
    """API가 준 JSON을 학습 CSV와 같은 열 이름의 표로 바꾼다.

    - 오류 응답(키 미등록 등)은 WeatherAPIError
    - 빈 문자열은 NaN (강수 빈칸은 이후 core.clean_weather가 0으로 처리)
    """
    if "OpenAPI_ServiceResponse" in payload:  # 인증키 오류 등은 이 형태로 온다
        hdr = payload["OpenAPI_ServiceResponse"].get("cmmMsgHeader", {})
        raise WeatherAPIError(f"기상청 API 오류: {hdr.get('returnAuthMsg') or hdr.get('errMsg')} (인증키를 확인해 주세요)")
    try:
        header = payload["response"]["header"]
    except (KeyError, TypeError):
        raise WeatherAPIError("예상하지 못한 응답 형식입니다.") from None
    if header.get("resultCode") != "00":
        if header.get("resultCode") == "03":  # NO_DATA
            return pd.DataFrame(columns=["지점명", "일시", *FIELD_MAP.values()])
        raise WeatherAPIError(f"기상청 API 오류: {header.get('resultMsg')} (코드 {header.get('resultCode')})")
    items = payload["response"]["body"].get("items") or {}
    rows = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["지점명", "일시", *FIELD_MAP.values()])
    out = pd.DataFrame({"지점명": station_name, "일시": pd.to_datetime(df["tm"])})
    for api_col, col in FIELD_MAP.items():
        out[col] = pd.to_numeric(df.get(api_col), errors="coerce") if api_col in df else float("nan")
    return out.sort_values("일시").reset_index(drop=True)


def fetch_station_daily(key: str, station_name: str, start, end, session=None, timeout: int = 15) -> pd.DataFrame:
    """관측소 1곳의 start~end(양끝 포함) 일자료를 가져온다."""
    if not key:
        raise WeatherAPIError("인증키가 없습니다. .streamlit/secrets.toml에 KMA_SERVICE_KEY를 넣어 주세요.")
    params = {"serviceKey": key, "pageNo": 1, "numOfRows": 200, "dataType": "JSON", "dataCd": "ASOS", "dateCd": "DAY",
              "startDt": pd.Timestamp(start).strftime("%Y%m%d"), "endDt": pd.Timestamp(end).strftime("%Y%m%d"),
              "stnIds": STATION_IDS[station_name]}
    try:
        resp = (session or requests).get(URL, params=params, timeout=timeout)
        try:
            payload = resp.json()  # 인증키 오류(401)도 JSON 본문에 이유가 담겨 오므로 먼저 읽는다
        except ValueError as e:  # JSON이 아님(XML 오류 페이지 등)
            resp.raise_for_status()
            raise WeatherAPIError("API 응답을 해석하지 못했습니다. 인증키나 호출 한도를 확인해 주세요.") from e
    except requests.RequestException as e:
        raise WeatherAPIError(f"기상청 서버에 연결하지 못했습니다: {type(e).__name__}") from e
    return parse_asos_response(payload, station_name)


def fetch_range(key: str, start, end, session=None, chunk_days: int = 150) -> pd.DataFrame:
    """5개 관측소의 start~end 일자료. 1회 응답 한도(200건)를 넘지 않게 chunk_days씩 나눠 요청한다."""
    frames = []
    for station in STATION_IDS:
        cur = pd.Timestamp(start)
        while cur <= pd.Timestamp(end):
            stop = min(cur + pd.Timedelta(days=chunk_days - 1), pd.Timestamp(end))
            frames.append(fetch_station_daily(key, station, cur, stop, session))
            cur = stop + pd.Timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


def fetch_recent_weather(key: str, target_date, days: int = 95, session=None) -> pd.DataFrame:
    """target_date '전날'까지 최근 days일의 5개 관측소 일자료를 합쳐 돌려준다 (core.clean_weather 처리 전)."""
    end = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days - 1)
    frames = [fetch_station_daily(key, s, start, end, session) for s in STATION_IDS]
    return pd.concat(frames, ignore_index=True)


def build_live_inputs(weather_raw: pd.DataFrame, target_date, fires: pd.DataFrame, region_map: pd.DataFrame) -> pd.DataFrame:
    """API 기상 → 학습 때와 같은 방식(core.add_features)으로 target_date의 지역별 입력 특성을 만든다.

    전날 자료가 없는 관측소가 있으면 WeatherAPIError (기상청 일자료는 보통 다음 날 오전에 올라온다).
    '최근 90일 산불 건수'는 보유한 산불 이력(fires)만 반영한다.
    """
    target = pd.Timestamp(target_date)
    end = target - pd.Timedelta(days=1)
    if weather_raw.empty:
        raise WeatherAPIError("가져온 기상 자료가 없습니다.")
    have = weather_raw[weather_raw["일시"] == end]["지점명"].unique()
    missing = sorted(set(STATION_IDS) - set(have))
    if missing:
        raise WeatherAPIError(f"{end:%Y-%m-%d} 자료가 아직 없는 관측소: {', '.join(missing)}. 기상청 일자료는 보통 다음 날 오전에 올라옵니다.")
    wx, _ = core.clean_weather(weather_raw)
    start = end - pd.Timedelta(days=94)
    panel = core.build_panel(wx, fires, region_map, start=start, end=end)
    feat = core.add_features(panel)
    return core.inputs_for_date(feat, target)
