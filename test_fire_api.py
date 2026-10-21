"""fire_api.py + 2026 산불 병합 검증 (가짜 XML 사용). 실행: python test_fire_api.py"""
import pandas as pd

import core
import fire_api as fa

def _item(si="경기", gungu="화성", menu="서신", cause="쓰레기소각", area="0.1", d=("2026", "03", "05")):
    return (f"<item><damagearea>{area}</damagearea><firecause>{cause}</firecause><locgungu>{gungu}</locgungu>"
            f"<locmenu>{menu}</locmenu><locsi>{si}</locsi><startyear>{d[0]}</startyear><startmonth>{d[1]}</startmonth>"
            f"<startday>{d[2]}</startday></item>")

def _xml(items, total=None, code="00"):
    return (f'<?xml version="1.0"?><response><header><resultCode>{code}</resultCode><resultMsg>M</resultMsg></header>'
            f'<body><items>{"".join(items)}</items><totalCount>{total if total is not None else len(items)}</totalCount></body></response>')

class R:  # 가짜 응답
    def __init__(self, text): self.text = text

class S:  # 가짜 세션: 2페이지로 나눠 준다
    def get(self, url, params, timeout):
        a, b = [_item(), _item(si="강원")], [_item(gungu="용인수지", menu="죽전")]
        return R(_xml(a if params["pageNo"] == 1 else b, total=3))

def test_paging_and_gyeonggi_filter():
    raw = fa.fetch_fires("K", "2026-01-01", "2026-12-31", session=S(), rows=2)
    assert len(raw) == 3
    t = fa.to_fire_table(raw)
    assert len(t) == 2 and set(t.columns) == {"발생장소", "원인", "피해면적(ha)", "일시", "시군"}
    assert t["일시"].iloc[0] == pd.Timestamp("2026-03-05")

def test_error_responses():
    bad = '<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>E</errMsg><returnAuthMsg>등록되지 않은 서비스키</returnAuthMsg></cmmMsgHeader></OpenAPI_ServiceResponse>'
    for text in (bad, _xml([], code="99"), "not xml"):
        try:
            fa.parse_fire_xml(text)
        except fa.FireAPIError as e:
            assert str(e); continue
        raise AssertionError

def test_empty_and_cause_mapping():
    assert fa.to_fire_table(pd.DataFrame()).empty
    t = fa.map_causes(fa.to_fire_table(fa.fetch_fires("K", "2026-01-01", "2026-12-31", session=S(), rows=2)))
    assert set(t["원인"]) == {"쓰레기 태우기"} and set(t["원인_API"]) == {"쓰레기소각"}

def test_region_normalization():
    for raw, want in [("용인수지", "용인"), ("용인처인", "용인"), ("용인 처인", "용인"), ("수원장안", "수원"), ("안산 상록", "안산"), ("안산단원", "안산"), ("가평", "가평"), ("광주", "광주")]:
        assert core.normalize_region(raw) == want, raw

def test_2026_merge_keeps_old_data_and_adds_new():
    f = core.load_fires_with_extra()
    old = core.load_fires()
    assert len(old) == 187 and len(f) == 187 + 93                     # 2024~2025 187건 + 2026 API 93건
    assert (f["일시"] <= old["일시"].max()).sum() == len(old) and f["일시"].max() > core.PERIOD_END


def test_2024_fix_rows_are_in_main_file():
    old = core.load_fires()
    assert (old["일시"].dt.year == 2024).sum() == 86 and (old["일시"].dt.year == 2025).sum() == 101
    assert not (core.DATA_DIR / "산불_2024_보정_api.csv").exists()   # 별도 보정 파일은 쓰지 않는다
    row = old[(old["일시"] == "2024-12-30") & (old["시군"] == "파주")]
    assert len(row) == 1 and row.iloc[0]["기상관측지점"] == "파주" and row.iloc[0]["지점"] == 99


def test_corrected_labels_have_ten_more_fire_days():
    b = core.load_all()
    p = b["panel"]
    assert int(p.loc[p["일시"] <= core.PERIOD_END, "산불발생"].sum()) == 181   # 보정 전 171 → 181
    assert b["train_pos"] == 83 and b["test_pos"] == 98 and b["test_n"] == 9490


def test_new_regions_not_in_prediction_targets():
    b = core.load_all()
    assert len(b["region_map"]) == 26 and not {"광명", "부천"} & set(b["region_map"]["지역"])
    assert {"광명", "부천"} <= set(b["fires"]["지역"])      # 이력 조회에는 보임
    assert b["metrics"]["표본_수"] == 9490                    # 학습·평가 표본은 그대로

def test_2026_fires_flow_into_90day_feature():
    b = core.load_all()
    f = b["features"]; d = f[f["일시"] == "2026-04-01"]
    assert d["최근90일_산불건수"].sum() > 0

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("통과", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("실패", name, repr(e))
    raise SystemExit(fails)
