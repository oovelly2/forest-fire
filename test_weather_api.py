"""weather_api.py 검증. 실제 API 호출 없이 가짜 응답으로 확인한다. 실행: python test_weather_api.py"""
import pandas as pd

import core
import weather_api as wa

# 가짜 응답: 공공데이터포털 ASOS 일자료 응답 형식을 따라 만든 예시 (실제 호출 결과가 아님)
def _payload(rows, code="00", msg="NORMAL_SERVICE"):
    return {"response": {"header": {"resultCode": code, "resultMsg": msg},
                         "body": {"items": {"item": rows}, "totalCount": len(rows)}}}


def _row(day, ta="5.0", rn="", ws="1.2", rhm="60.0"):
    return {"tm": day, "avgTa": ta, "sumRn": rn, "avgWs": ws, "avgRhm": rhm}


class FakeResp:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


class FakeSession:
    """station별로 target 전날까지 95일치 가짜 자료를 돌려주는 세션."""
    def __init__(self, drop_last_for=None): self.calls, self.drop = [], drop_last_for
    def get(self, url, params, timeout):
        self.calls.append(params)
        days = pd.date_range(params["startDt"], params["endDt"])
        stn = {v: k for k, v in wa.STATION_IDS.items()}[params["stnIds"]]
        if stn == self.drop:
            days = days[:-1]
        return FakeResp(_payload([_row(d.strftime("%Y-%m-%d")) for d in days]))


def test_parse_converts_blank_rain_and_columns():
    df = wa.parse_asos_response(_payload([_row("2026-10-19", rn=""), _row("2026-10-20", rn="2.5")]), "수원")
    assert list(df.columns) == ["지점명", "일시", *wa.FIELD_MAP.values()]
    assert df["일강수량(mm)"].isna().tolist() == [True, False] and df["평균기온(°C)"].iloc[0] == 5.0


def test_parse_single_item_dict_and_no_data():
    one = {"response": {"header": {"resultCode": "00"}, "body": {"items": {"item": _row("2026-10-20")}}}}
    assert len(wa.parse_asos_response(one, "수원")) == 1
    assert wa.parse_asos_response(_payload([], code="03", msg="NO_DATA"), "수원").empty


def test_parse_errors_raise_readable_error():
    bad_key = {"OpenAPI_ServiceResponse": {"cmmMsgHeader": {"errMsg": "SERVICE_KEY_IS_NOT_REGISTERED_ERROR", "returnAuthMsg": "등록되지 않은 서비스키"}}}
    for payload in (bad_key, _payload([], code="22", msg="LIMITED"), {"x": 1}):
        try:
            wa.parse_asos_response(payload, "수원")
        except wa.WeatherAPIError as e:
            assert str(e)
            continue
        raise AssertionError("WeatherAPIError가 나와야 함")


def test_fetch_requires_key():
    try:
        wa.fetch_station_daily("", "수원", "2026-10-01", "2026-10-02")
    except wa.WeatherAPIError:
        return
    raise AssertionError


def test_station_ids_match_training_csv():
    w = core.load_weather().drop_duplicates("지점명").set_index("지점명")["지점"].to_dict()
    assert wa.STATION_IDS == w


def test_live_inputs_have_all_features_and_use_previous_day():
    F, _ = core.clean_fires(core.load_fires())
    rmap = core.assign_stations(sorted(F["지역"].unique()))
    s = FakeSession()
    raw = wa.fetch_recent_weather("KEY", "2026-10-21", session=s)
    assert len(s.calls) == 5 and s.calls[0]["endDt"] == "20261020"        # 전날까지만 요청
    inp = wa.build_live_inputs(raw, "2026-10-21", F, rmap)
    assert len(inp) == len(rmap) and inp[core.FEATURES].notna().all().all()
    assert (inp["전일_기온"] == 5.0).all() and (inp["최근7일_강수합"] == 0).all()   # 강수 빈칸 -> 0
    assert (inp["건조지속일수"] == 30).all() and (inp["최근90일_산불건수"] == 0).all()


def test_missing_yesterday_gives_clear_error():
    F, _ = core.clean_fires(core.load_fires())
    rmap = core.assign_stations(sorted(F["지역"].unique()))
    raw = wa.fetch_recent_weather("KEY", "2026-10-21", session=FakeSession(drop_last_for="이천"))
    try:
        wa.build_live_inputs(raw, "2026-10-21", F, rmap)
    except wa.WeatherAPIError as e:
        assert "이천" in str(e)
        return
    raise AssertionError


def test_live_inputs_work_with_model():
    F, _ = core.clean_fires(core.load_fires())
    b = core.load_all()
    raw = wa.fetch_recent_weather("KEY", "2026-10-21", session=FakeSession())
    inp = wa.build_live_inputs(raw, "2026-10-21", F, b["region_map"])
    r = core.predict_risk(b, inp)
    assert r["위험확률"].between(0, 1).all() and len(r) == len(b["region_map"])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("통과", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("실패", name, repr(e))
    raise SystemExit(fails)
