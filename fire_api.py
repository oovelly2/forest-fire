"""산림청 산불발생통계 API 클라이언트 (공공데이터포털 '산림청_산불발생통계(대국민포털)').

streamlit을 쓰지 않는 순수 파이썬 모듈. 전국 자료를 주므로 시도가 '경기'인 행만 쓴다.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd
import requests

URL = "https://apis.data.go.kr/1400000/forestStusService/getfirestatsservice"


class FireAPIError(Exception):
    """산불 API 호출·응답 오류. 화면/스크립트에서 메시지를 그대로 보여 준다."""


def parse_fire_xml(text: str) -> tuple[list[dict], int]:
    """XML 응답 -> (항목 dict 목록, 전체 건수). 오류 응답이면 FireAPIError."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise FireAPIError("산불 API 응답을 해석하지 못했습니다.") from e
    if root.tag == "OpenAPI_ServiceResponse":  # 인증키 오류 등
        msg = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg")
        raise FireAPIError(f"산불 API 오류: {msg} (이 API의 활용신청/인증키를 확인해 주세요)")
    code = root.findtext("./header/resultCode")
    if code != "00":
        raise FireAPIError(f"산불 API 오류: {root.findtext('./header/resultMsg')} (코드 {code})")
    items = [{c.tag: (c.text or "").strip() for c in it} for it in root.findall("./body/items/item")]
    return items, int(root.findtext("./body/totalCount") or 0)


def fetch_fires(key: str, start, end, session=None, rows: int = 1000, timeout: int = 30) -> pd.DataFrame:
    """start~end(발생일 기준) 전국 산불을 모두 가져온다 (페이지를 넘기며 반복)."""
    if not key:
        raise FireAPIError("인증키가 없습니다.")
    out, page = [], 1
    while True:
        params = {"serviceKey": key, "numOfRows": rows, "pageNo": page,
                  "searchStDt": pd.Timestamp(start).strftime("%Y%m%d"), "searchEdDt": pd.Timestamp(end).strftime("%Y%m%d")}
        try:
            resp = (session or requests).get(URL, params=params, timeout=timeout)
            items, total = parse_fire_xml(resp.text)
        except requests.RequestException as e:
            raise FireAPIError(f"산불 API 서버에 연결하지 못했습니다: {type(e).__name__}") from e
        out += items
        if not items or len(out) >= total:
            return pd.DataFrame(out)
        page += 1


def to_fire_table(raw: pd.DataFrame) -> pd.DataFrame:
    """API 원자료 -> 기존 산불_기상_병합.csv의 앞 열(발생장소, 원인, 피해면적(ha), 일시, 시군)과 같은 형태. 경기만."""
    if raw.empty:
        return pd.DataFrame(columns=["발생장소", "원인", "피해면적(ha)", "일시", "시군"])
    g = raw[raw["locsi"].str.startswith("경기")].copy()
    g["일시"] = pd.to_datetime(g["startyear"] + "-" + g["startmonth"] + "-" + g["startday"])
    g["시군"] = g["locgungu"]
    g["발생장소"] = ("경기 " + g["locgungu"].fillna("") + " " + g["locmenu"].fillna("")).str.strip()  # 읍면이 없으면 시군까지만
    g["원인"] = g["firecause"]
    g["피해면적(ha)"] = pd.to_numeric(g["damagearea"], errors="coerce")
    return g[["발생장소", "원인", "피해면적(ha)", "일시", "시군"]].sort_values("일시", ascending=False).reset_index(drop=True)


# API 원인 표기 -> 기존 CSV(국가산불정보시스템) 표기. '건축물화재비화'는 가장 가까운 '건축물 실화'로 대응(근사).
# API에는 '원인 미상' 항목이 따로 없어 기존의 '원인 미상'과 같은 기준으로 비교할 수 없다.
CAUSE_MAP = {"담뱃불실화": "담뱃불 실화", "쓰레기소각": "쓰레기 태우기", "입산자실화": "입산자 실화",
             "건축물화재비화": "건축물 실화", "농산부산물소각": "농산폐기물 태우기"}


def map_causes(table: pd.DataFrame) -> pd.DataFrame:
    """원인을 기존 표기로 바꾸고 API 원문은 '원인_API' 열에 남긴다."""
    out = table.copy()
    out["원인_API"] = out["원인"]
    out["원인"] = out["원인"].replace(CAUSE_MAP)
    return out
