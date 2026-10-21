"""core.py 검증. 실행: python -m pytest -q   (pytest가 없으면 python test_core.py)"""
import numpy as np
import pandas as pd

import core

W, F = core.load_weather(), core.load_fires()
W_CLEAN, _ = core.clean_weather(W)
F_CLEAN, _ = core.clean_fires(F)
RMAP = core.assign_stations(sorted(F_CLEAN["지역"].unique()))
PANEL = core.build_panel(W_CLEAN, F_CLEAN, RMAP)
FEAT = core.add_features(PANEL)


# --- 데이터 점검 -----------------------------------------------------------
def test_raw_shapes():
    assert W.shape == (3655, 10) and F.shape == (187, 11)


def test_fire_file_has_2024_fix_rows():
    """병합 CSV와 일별 CSV 모두 2024년 누락 10건이 반영돼 187건이고, 두 파일의 건수·발생일이 일치한다."""
    assert len(F) == 187 and int(W["산불건수"].sum()) == 187 and int(W["산불발생"].sum()) == 158
    g = F.groupby(["기상관측지점", "일시"]).size()
    d = W[W["산불건수"] > 0].set_index(["지점명", "일시"])["산불건수"]
    assert (g.astype(float).sort_index().to_numpy() == d.sort_index().to_numpy()).all()
    assert (F["일시"].dt.year == 2024).sum() == 86 and (F["일시"].dt.year == 2025).sum() == 101


def test_clean_weather_fills_rain_blank_and_ffill():
    assert W_CLEAN["일강수량(mm)"].isna().sum() == 0
    assert W_CLEAN[["평균기온(°C)", "평균 상대습도(%)"]].isna().sum().sum() == 0
    assert len(W_CLEAN) == len(W)


def test_yongin_districts_merged():
    assert "용인처인" not in set(F_CLEAN["지역"]) and (F_CLEAN["지역"] == "용인").sum() == 16


# --- 지역-관측소 연결 -------------------------------------------------------
def test_each_region_gets_nearest_station():
    for _, r in RMAP.iterrows():
        d = {s: core.haversine_km(r["위도"], r["경도"], *c) for s, c in core.STATION_COORDS.items()}
        assert r["관측소"] == min(d, key=d.get)


def test_stations_not_all_same():
    """경기도 전체에 한 관측소를 일괄 적용하지 않았는지."""
    assert RMAP["관측소"].nunique() == 5


def test_merged_csv_station_matches_our_mapping():
    """병합 CSV의 '기상관측지점'과 우리 매핑이 모두 일치."""
    m = F_CLEAN.drop_duplicates("지역").set_index("지역")["기상관측지점"]
    assert (RMAP.set_index("지역")["관측소"].reindex(m.index) == m).all()


# --- 지역 x 날짜 표 --------------------------------------------------------
def test_panel_is_full_grid_without_duplicates():
    assert len(PANEL) == len(RMAP) * 731
    assert not PANEL.duplicated(["지역", "일시"]).any()
    assert PANEL[["기온", "강수", "풍속", "습도"]].isna().sum().sum() == 0


def test_panel_label_counts():
    assert PANEL["산불건수"].sum() == 187          # 건수 보존
    assert PANEL["산불발생"].sum() == 181          # 같은 지역·날짜 2건이 6번 겹침
    assert (PANEL["산불발생"] == 0).sum() > 18000  # 기록 없는 날은 미발생


def test_weather_comes_from_region_station_same_day():
    row = PANEL[(PANEL["지역"] == "화성") & (PANEL["일시"] == "2025-03-01")].iloc[0]
    src = W_CLEAN[(W_CLEAN["지점명"] == "수원") & (W_CLEAN["일시"] == "2025-03-01")].iloc[0]
    assert row["관측소"] == "수원" and row["기온"] == src["평균기온(°C)"]


# --- 미래 정보 차단 --------------------------------------------------------
def test_features_use_previous_day_only():
    r = FEAT[(FEAT["지역"] == "가평") & (FEAT["일시"] == "2025-04-10")].iloc[0]
    prev = PANEL[(PANEL["지역"] == "가평") & (PANEL["일시"] == "2025-04-09")].iloc[0]
    assert r["전일_기온"] == prev["기온"] and r["전일_습도"] == prev["습도"]


def test_changing_today_does_not_change_features():
    """D일의 기온·습도·산불 여부를 바꿔도 D일의 입력 특성은 그대로여야 한다."""
    p2 = PANEL.copy()
    mask = (p2["지역"] == "화성") & (p2["일시"] == "2025-03-20")
    p2.loc[mask, ["기온", "습도", "강수", "풍속", "산불건수", "산불발생"]] = [99, 1, 500, 30, 5, 1]
    a = FEAT[(FEAT["지역"] == "화성") & (FEAT["일시"] == "2025-03-20")][core.FEATURES]
    b = core.add_features(p2)
    b = b[(b["지역"] == "화성") & (b["일시"] == "2025-03-20")][core.FEATURES]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float))


def test_fire_history_excludes_today():
    d = PANEL[PANEL["산불발생"] == 1].iloc[10]
    r = FEAT[(FEAT["지역"] == d["지역"]) & (FEAT["일시"] == d["일시"])].iloc[0]
    past = PANEL[(PANEL["지역"] == d["지역"]) & (PANEL["일시"] < d["일시"]) &
                 (PANEL["일시"] >= d["일시"] - pd.Timedelta(days=90))]["산불건수"].sum()
    assert r["최근90일_산불건수"] == past


def test_next_day_row_exists_and_is_not_labeled():
    nxt = FEAT[FEAT["일시"] == core.PERIOD_END + pd.Timedelta(days=1)]
    assert len(nxt) == len(RMAP) and not nxt["is_label"].any()
    assert nxt[core.FEATURES].notna().all().all()


# --- 시간 분할 · 모델 ------------------------------------------------------
def test_time_split_has_no_overlap():
    tr, te = core.time_split(core.drop_unusable(FEAT))
    assert tr["일시"].max() < te["일시"].min()
    assert te["일시"].max() == core.PERIOD_END


def test_evaluate_known_values():
    m = core.evaluate([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
    assert m["재현율"] == 0.5 and m["정밀도"] == 1.0 and m["정확도"] == 0.75
    assert abs(m["ROC-AUC"] - 0.75) < 1e-9


BUNDLE = core.fit_and_evaluate(FEAT)


def test_model_beats_season_only_baseline():
    assert BUNDLE["metrics"]["ROC-AUC"] > 0.5
    assert BUNDLE["metrics"]["ROC-AUC"] > BUNDLE["baseline"]["ROC-AUC"]


def test_predict_risk_probabilities_and_grades():
    r = core.predict_risk(BUNDLE, core.inputs_for_date(BUNDLE["features"], "2026-01-01"))
    assert len(r) == len(RMAP) and r["위험확률"].between(0, 1).all()
    assert set(r["위험등급"]) <= set(core.GRADES) and (r["근거"].str.len() > 0).all()


def test_drier_weather_means_higher_risk():
    base = core.inputs_for_date(BUNDLE["features"], "2025-03-10").head(1).copy()
    wet, dry = base.copy(), base.copy()
    wet[["전일_습도", "최근7일_강수합"]] = [90, 50]
    dry[["전일_습도", "최근7일_강수합"]] = [25, 0]
    assert core.predict_risk(BUNDLE, dry)["위험확률"].iloc[0] > core.predict_risk(BUNDLE, wet)["위험확률"].iloc[0]


def test_predict_risk_rejects_empty_or_nan():
    for bad in (FEAT.head(0), FEAT.head(1).assign(전일_기온=np.nan)):
        try:
            core.predict_risk(BUNDLE, bad)
        except ValueError:
            continue
        raise AssertionError("ValueError가 나와야 함")


# --- 자원 배치 -------------------------------------------------------------
def _risk(n=8, seed=0):
    p = np.random.default_rng(seed).uniform(0.005, 0.05, n)
    return pd.DataFrame({"지역": [f"지역{i}" for i in range(n)], "위험확률": p, "위험등급": "주의", "근거": "x"})


def test_allocation_never_exceeds_available():
    rng = np.random.default_rng(1)
    for _ in range(300):
        v, p, n = int(rng.integers(0, 30)), int(rng.integers(0, 200)), int(rng.integers(1, 12))
        a = core.allocate_resources(_risk(8, int(rng.integers(0, 99))), v, p, n)
        assert a["차량"].sum() <= v and a["인력"].sum() <= p
        assert (a["차량"] >= 0).all() and len(a) == min(n, 8)


def test_allocation_order_follows_risk():
    a = core.allocate_resources(_risk(), 10, 40, 4)
    assert a["위험확률"].is_monotonic_decreasing and a["차량"].iloc[0] >= a["차량"].iloc[-1]


def test_allocation_zero_and_invalid_inputs():
    a = core.allocate_resources(_risk(), 0, 0, 3)
    assert a["차량"].sum() == 0 and a["인력"].sum() == 0
    for args in [(-1, 5, 3), (3, 5, 0), (2.5, 5, 3)]:
        try:
            core.allocate_resources(_risk(), *args)
        except ValueError:
            continue
        raise AssertionError(f"ValueError가 나와야 함: {args}")


# --- 이력 조회 -------------------------------------------------------------
def test_monthly_counts_fills_zero_months():
    m = core.monthly_counts(core.filter_fires(F_CLEAN, ["이천"], "2024-01-01", "2025-12-31"), "2024-01-01", "2025-12-31")
    assert len(m) == 24 and m["발생건수"].sum() == 3 and (m["발생건수"] == 0).sum() >= 20


def test_filter_fires_period_and_region():
    df = core.filter_fires(F_CLEAN, ["화성", "양평"], "2024-01-01", "2024-12-31")
    assert set(df["지역"]) <= {"화성", "양평"} and df["일시"].dt.year.eq(2024).all()
    assert len(core.filter_fires(F_CLEAN, [], "2024-01-01", "2025-12-31")) == 187


# --- 달력 변수 (이번 수정) -----------------------------------------------
def test_calendar_known_dates():
    cal = core.calendar_features(["2024-05-06", "2024-05-07", "2025-01-27", "2025-03-01", "2025-03-03", "2026-02-17"])
    assert cal["공휴일"].tolist() == [1, 0, 1, 1, 1, 1]   # 어린이날 대체·평일·임시공휴일·삼일절·삼일절 대체(3/3)·설날
    assert cal["주말"].tolist() == [0, 0, 0, 1, 0, 0]       # 2025-03-01은 토요일
    assert cal.loc[1, "평일요일_화"] == 1 and cal.loc[4, ["평일요일_화", "평일요일_수", "평일요일_목", "평일요일_금"]].sum() == 0  # 월요일(3/3)은 기준


def test_calendar_works_for_future_and_leaves_no_nan():
    cal = core.calendar_features(pd.date_range("2027-01-01", "2027-12-31"))
    assert len(cal) == 365 and cal.notna().all().all() and cal["공휴일"].sum() >= 10
    assert cal["주말"].sum() == 104   # 2027년 토·일 합계


def test_calendar_columns_not_duplicated_with_month():
    assert not set(core.CALENDAR_FEATURES) & {"월_sin", "월_cos"}
    assert len(core.FEATURES) == len(set(core.FEATURES)) == len(core.BASE_FEATURES) + len(core.CALENDAR_FEATURES)


def test_calendar_features_in_panel_match_date():
    r = FEAT[(FEAT["지역"] == "화성") & (FEAT["일시"] == "2025-03-01")].iloc[0]
    assert r["주말"] == 1 and r["공휴일"] == 1


def test_cause_never_used_as_feature():
    assert not any("원인" in f for f in core.FEATURES)
    assert "원인" not in FEAT.columns


def test_comparison_uses_same_sample_and_same_threshold_rule():
    m, p = BUNDLE["metrics"], BUNDLE["metrics_prev"]
    assert m["표본_수"] == p["표본_수"] == BUNDLE["test_n"] and m["양성_수"] == p["양성_수"]
    tr, _ = core.time_split(FEAT.pipe(core.drop_unusable))
    for model, feats, met in [(BUNDLE["eval_model"], core.FEATURES, m)]:
        assert abs(np.quantile(model.predict_proba(tr[feats])[:, 1], 0.9) - met["경보_기준확률"]) < 1e-12
    assert set(["ROC-AUC", "PR-AUC", "재현율", "정밀도", "경보_비율", "경보_기준확률"]) <= set(m)


def test_calendar_effect_table_counts():
    t = core.calendar_effect_table(BUNDLE["features"])
    assert t["발생 지역일"].sum() == 181 and t["지역일수"].sum() == len(PANEL) - len(RMAP)  # 전날 정보가 없는 첫날(2024-01-01) 제외
    assert t.loc[t["구분"] == "평일", "월 보정 비율(평일=1)"].iloc[0] == 1.0


# --- 원인 분석 (이번 수정) -----------------------------------------------
def test_monthly_cause_table_sums_and_zero_months():
    t = core.monthly_cause_table(F_CLEAN, "2024-01-01", "2025-12-31")
    assert len(t) == 24 and t.drop(columns="월").to_numpy().sum() == 187
    sub = core.filter_fires(F_CLEAN, ["이천"], "2024-01-01", "2025-12-31")
    ts = core.monthly_cause_table(sub, "2024-01-01", "2025-12-31")
    assert len(ts) == 24 and ts.drop(columns="월").to_numpy().sum() == 3   # 지역 필터 적용


def test_unknown_cause_summary_from_data():
    u = core.unknown_cause_summary(F_CLEAN)
    assert (u["원인 미상"], u["기타"], u["합계"]) == (30, 46, 76) and abs(u["합계_비율"] - 76 / 187) < 1e-12
    assert core.unknown_cause_summary(F_CLEAN.head(0))["합계_비율"] == 0.0


def test_cause_table_handles_empty():
    assert len(core.monthly_cause_table(F_CLEAN.head(0), "2024-01-01", "2024-03-31")) == 3


# --- 연도 이동 ---------------------------------------------------------------
def test_years_in_range_and_window():
    assert core.years_in_range("2024-03-01", "2026-10-21") == [2024, 2025, 2026]
    lo, hi = core.year_window("2024-03-01", "2026-10-21", 2024)
    assert (lo, hi) == (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-12-31"))
    lo, hi = core.year_window("2024-03-01", "2026-10-21", 2025)
    assert (lo, hi) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    assert core.year_window("2024-03-01", "2026-10-21", 2026)[1] == pd.Timestamp("2026-10-21")
    try:
        core.year_window("2024-03-01", "2025-05-01", 2026)
    except ValueError:
        return
    raise AssertionError


def test_year_slices_add_up_to_whole_period():
    """연도별로 자른 건수의 합 = 전체 기간 건수 (빠지거나 겹치는 날이 없다)."""
    s, e = "2024-01-01", "2025-12-31"
    parts = [len(core.filter_fires(F_CLEAN, [], *core.year_window(s, e, y))) for y in core.years_in_range(s, e)]
    assert sum(parts) == len(core.filter_fires(F_CLEAN, [], s, e)) == 187 and parts == [86, 101]
    lo, hi = core.year_window(s, e, 2024)
    assert len(core.monthly_counts(core.filter_fires(F_CLEAN, [], lo, hi), lo, hi)) == 12


if __name__ == "__main__":  # pytest 없이도 실행 가능
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("통과", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("실패", name, repr(e))
    raise SystemExit(fails)
