"""경기도 산불 위험 예측 · 사전 자원 배치 추천 — 핵심 로직.

이 파일은 화면(Streamlit)을 모르는 '순수 파이썬' 함수만 담는다 (streamlit import 금지).
그래서 test_core.py로 먼저 검증한 뒤 화면에서 불러 쓸 수 있다.

실행 순서(데이터 흐름)
  1) load_weather / load_fires      CSV 읽기
  2) clean_weather / clean_fires    결측·중복 처리
  3) build_panel                    '지역 x 날짜' 표 만들기 + 가까운 관측소 기상 연결 + 발생 여부(0/1)
  4) add_features                   '전날까지' 정보만으로 입력 특성 만들기 (미래 정보 차단)
  5) time_split / train_model       시간 순서로 나눠 학습, evaluate로 평가
  6) predict_risk                   지역별 위험 확률 + 등급 + 근거
  7) allocate_resources             위험 확률 비례 규칙으로 차량·인력 배분 (모델이 아닌 '규칙')
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
WEATHER_CSV = DATA_DIR / "기상_산불_일별.csv"
FIRE_CSV = DATA_DIR / "산불_기상_병합.csv"
EXTRA_FIRE_CSV = DATA_DIR / "산불_2026_api.csv"  # fetch_fires_2026.py가 API로 받은 2026년 산불 기록
EXTRA_WEATHER_CSV = DATA_DIR / "기상_2026_api.csv"  # fetch_2026.py가 API로 받은 2026년 기상 (산불 이력은 없음)

# 자료 수집 범위: 기상 자료가 2024-01-01 ~ 2025-12-31 (5개 관측소 x 731일, 빠진 날 없음)
PERIOD_START = pd.Timestamp("2024-01-01")
PERIOD_END = pd.Timestamp("2025-12-31")

# ---------------------------------------------------------------------------
# 위치 정보 (위도, 경도). data/merge.py에 직접 적어 둔 '대략 좌표'를 그대로 사용한다.
# 공식 좌표가 아니므로 지도 표시·최근접 관측소 판단용 근사값이다.
# ---------------------------------------------------------------------------
STATION_COORDS = {
    "동두천": (37.90, 127.06), "파주": (37.76, 126.78), "수원": (37.27, 126.99),
    "양평": (37.49, 127.49), "이천": (37.27, 127.44),
}
REGION_COORDS = {
    "가평": (37.83, 127.51), "고양": (37.66, 126.83), "과천": (37.43, 126.99),
    "광주": (37.43, 127.26), "군포": (37.36, 126.94), "김포": (37.62, 126.72),
    "남양주": (37.64, 127.22), "동두천": (37.90, 127.06), "성남": (37.42, 127.13),
    "수원": (37.27, 126.99), "시흥": (37.38, 126.80), "안산": (37.32, 126.83),
    "안성": (37.01, 127.28), "양주": (37.79, 127.05), "양평": (37.49, 127.49),
    "여주": (37.30, 127.64), "연천": (38.10, 127.07), "오산": (37.15, 127.07),
    "용인": (37.24, 127.18), "의정부": (37.74, 127.03), "이천": (37.27, 127.44),
    "파주": (37.76, 126.78), "평택": (36.99, 127.11), "포천": (37.89, 127.20),
    "하남": (37.54, 127.21), "화성": (37.20, 126.83),
}

# 모델 입력 특성: 모두 '예측 대상일 전날까지'의 정보 (월은 달력이라 미리 알 수 있음)
BASE_FEATURES = ["월_sin", "월_cos", "전일_기온", "전일_습도", "전일_풍속",
                 "최근7일_강수합", "건조지속일수", "최근90일_산불건수"]  # 1차 모델(기존)
# 달력 변수: 예측 대상일의 날짜만으로 계산(미래 날짜도 가능). 사람 활동의 '간접' 지표일 뿐 실제 입산·소각을 측정한 값이 아님.
# 요일은 월요일을 기준으로 한 화~금 더미. 토·일은 '주말'이 대신하므로 정보가 겹치지 않는다.
CALENDAR_FEATURES = ["주말", "공휴일", "평일요일_화", "평일요일_수", "평일요일_목", "평일요일_금"]
FEATURES = BASE_FEATURES + CALENDAR_FEATURES
FEATURE_LABELS = {
    "월_sin": "계절(월)", "월_cos": "계절(월)", "전일_기온": "전일 평균기온(°C)",
    "전일_습도": "전일 평균 상대습도(%)", "전일_풍속": "전일 평균 풍속(m/s)",
    "최근7일_강수합": "최근 7일 누적 강수량(mm)", "건조지속일수": "강수 1mm 미만 연속일수",
    "최근90일_산불건수": "최근 90일 지역 산불 건수",
}
WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
WEATHER_INPUTS = ["전일_기온", "전일_습도", "전일_풍속", "최근7일_강수합", "건조지속일수"]
GRADES = ["낮음", "주의", "높음", "매우 높음"]


# ---------------------------------------------------------------------------
# 1) 읽기
# ---------------------------------------------------------------------------
def load_weather(path: Path = WEATHER_CSV) -> pd.DataFrame:
    """관측소 x 날짜 기상 CSV를 읽는다 (utf-8-sig, 날짜는 datetime)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["일시"] = pd.to_datetime(df["일시"])
    return df


def load_fires(path: Path = FIRE_CSV) -> pd.DataFrame:
    """산불 1건 = 1행인 CSV를 읽는다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["일시"] = pd.to_datetime(df["일시"])
    return df


# ---------------------------------------------------------------------------
# 2) 정리: 결측·중복
# ---------------------------------------------------------------------------
def clean_weather(weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """기상 결측·중복을 처리하고 처리 내역(report)을 함께 돌려준다.

    - 중복: (지점명, 일시) 중복 행 제거
    - 일강수량 빈칸: 기상청 일자료에서 빈칸은 '강수 없음'을 뜻하는 경우가 대부분이라 0으로 채움.
      (빈칸인 날은 평균 습도가 15%p가량 낮아 '맑은 날'이라는 해석과 맞음 → README 참고)
    - 기온·습도 결측(소수): 같은 관측소에서 '바로 앞날' 값으로 채움(ffill).
      뒤쪽 값을 쓰면 미래 정보가 섞이므로 앞쪽 값만 사용한다.
    """
    df = weather.drop_duplicates(["지점명", "일시"]).sort_values(["지점명", "일시"]).copy()
    report = {
        "중복_제거": int(len(weather) - len(df)),
        "강수_빈칸_0처리": int(df["일강수량(mm)"].isna().sum()),
        "기온_결측": int(df["평균기온(°C)"].isna().sum()),
        "습도_결측": int(df["평균 상대습도(%)"].isna().sum()),
        "풍속_결측": int(df["평균 풍속(m/s)"].isna().sum()),
    }
    df["일강수량(mm)"] = df["일강수량(mm)"].fillna(0.0)
    for col in ["평균기온(°C)", "평균 상대습도(%)", "평균 풍속(m/s)"]:
        df[col] = df.groupby("지점명")[col].ffill()
    report["ffill_후_남은_결측"] = int(df[["평균기온(°C)", "평균 상대습도(%)", "평균 풍속(m/s)"]].isna().sum().sum())
    return df.reset_index(drop=True), report


CITIES_WITH_DISTRICTS = ("용인", "수원", "안산", "성남", "고양", "안양", "부천")


def normalize_region(name: str) -> str:
    """'용인처인'·'수원장안'·'안산 상록'처럼 구 단위 이름은 시 이름으로 합친다 (같은 시, 같은 관측소)."""
    for city in CITIES_WITH_DISTRICTS:
        if name.startswith(city):
            return city
    return name


def clean_fires(fires: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """산불 이력의 완전 중복 행을 제거하고 지역명을 통일한다."""
    df = fires.drop_duplicates().copy()
    df["지역"] = df["시군"].map(normalize_region)
    report = {"중복_제거": int(len(fires) - len(df)), "건수": len(df)}
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# 3) 지역 -> 관측소 연결, 지역x날짜 표
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """두 좌표 사이 거리(km)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def assign_stations(regions=None) -> pd.DataFrame:
    """각 지역을 '가장 가까운' 관측소 1곳에 연결한다 (직선거리 기준).

    근거: 산불 위치(시군)와 기상 관측소가 1:1로 대응하는 자료가 없으므로,
    가장 가까운 관측소의 날씨가 그 지역 날씨와 가장 비슷하다고 가정한다.
    """
    regions = list(regions) if regions is not None else list(REGION_COORDS)
    rows = []
    for r in regions:
        lat, lon = REGION_COORDS[r]
        dist = {s: haversine_km(lat, lon, *c) for s, c in STATION_COORDS.items()}
        st = min(dist, key=dist.get)
        rows.append({"지역": r, "위도": lat, "경도": lon, "관측소": st, "거리_km": round(dist[st], 1)})
    return pd.DataFrame(rows)


def build_panel(weather: pd.DataFrame, fires: pd.DataFrame, region_map: pd.DataFrame,
                start=PERIOD_START, end=PERIOD_END) -> pd.DataFrame:
    """'지역 x 날짜' 전체 조합을 만들고 기상과 산불 발생 여부를 붙인다.

    - 기상: 지역의 관측소(region_map) + 날짜로 연결 (날짜만으로 합치지 않음)
    - 산불: 기록이 있는 (지역, 날짜)는 건수, 수집 범위 안에서 기록이 없으면 0건(미발생)
    """
    days = pd.date_range(start, end, freq="D")
    panel = pd.MultiIndex.from_product([region_map["지역"], days], names=["지역", "일시"]).to_frame(index=False)
    panel = panel.merge(region_map[["지역", "관측소"]], on="지역")
    wx = weather.rename(columns={"지점명": "관측소", "평균기온(°C)": "기온", "일강수량(mm)": "강수",
                                 "평균 풍속(m/s)": "풍속", "평균 상대습도(%)": "습도"})
    panel = panel.merge(wx[["관측소", "일시", "기온", "강수", "풍속", "습도"]], on=["관측소", "일시"], how="left")
    cnt = fires.groupby(["지역", "일시"]).size().rename("산불건수").reset_index()
    panel = panel.merge(cnt, on=["지역", "일시"], how="left")
    panel["산불건수"] = panel["산불건수"].fillna(0).astype(int)
    panel["산불발생"] = (panel["산불건수"] > 0).astype(int)
    return panel.sort_values(["지역", "일시"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) 특성: '전날까지' 정보만 사용
# ---------------------------------------------------------------------------
def _dry_streak(rain: pd.Series) -> pd.Series:
    """각 날짜까지 '강수 1mm 미만'이 연속된 일수 (그 날 포함)."""
    dry = (rain < 1.0).astype(int)
    grp = (dry == 0).cumsum()
    return dry.groupby(grp).cumsum()


def calendar_features(dates) -> pd.DataFrame:
    """날짜(들)만으로 요일·주말·한국 공휴일 변수를 만든다 (미래 날짜도 가능).

    공휴일은 holidays 패키지의 한국 달력(대체·임시공휴일 포함)을 쓴다.
    반환 열: CALENDAR_FEATURES (모두 0/1)
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).reset_index(drop=True)))
    kr = holidays.country_holidays("KR", years=sorted(set(d.year)))
    wd = d.weekday.to_numpy()
    out = pd.DataFrame({"주말": (wd >= 5).astype(int),
                        "공휴일": np.array([x.date() in kr for x in d], dtype=int)})
    for i, name in enumerate(WEEKDAY_KO[1:5], start=1):
        out[f"평일요일_{name}"] = (wd == i).astype(int)
    return out


def add_features(panel: pd.DataFrame, append_next_day: bool = True) -> pd.DataFrame:
    """예측 입력 특성을 만든다.

    입력 시점 규칙: D일의 위험을 예측할 때 D-1일 밤까지의 관측만 쓴다.
    구현: 모든 관측 값을 shift(1) 해서 D일 행에 'D-1일 값'이 들어가게 한다.
    D일 당일의 기온·습도·산불 여부는 어떤 특성에도 들어가지 않는다.

    append_next_day=True면 마지막 날의 다음 날(2026-01-01) 행을 추가한다.
    (당일 관측은 비어 있고 전날까지의 특성만 채워짐 → 실제 '내일 예측' 입력으로 사용)
    """
    df = panel.copy()
    if append_next_day:
        last = df["일시"].max()
        nxt = df.groupby("지역", as_index=False).agg(관측소=("관측소", "first"))
        nxt["일시"] = last + pd.Timedelta(days=1)
        df = pd.concat([df, nxt], ignore_index=True).sort_values(["지역", "일시"]).reset_index(drop=True)
        df["산불건수"] = df["산불건수"].fillna(0).astype(int)  # 다음 날은 아직 모름 -> 학습에서 제외(is_label=False)
        df["산불발생"] = df["산불발생"].fillna(0).astype(int)
        df["is_label"] = df["일시"] <= PERIOD_END  # 산불 이력이 없는 기간(2026~)은 학습·평가에서 제외
    else:
        df["is_label"] = df["일시"] <= PERIOD_END

    g = df.groupby("지역", group_keys=False)
    df["전일_기온"] = g["기온"].shift(1)
    df["전일_습도"] = g["습도"].shift(1)
    df["전일_풍속"] = g["풍속"].shift(1)
    rain_prev = g["강수"].shift(1)
    df["최근7일_강수합"] = rain_prev.groupby(df["지역"]).transform(lambda s: s.rolling(7, min_periods=1).sum())
    dry = g["강수"].transform(_dry_streak)
    df["건조지속일수"] = dry.groupby(df["지역"]).shift(1).clip(upper=30)
    fire_prev = g["산불건수"].shift(1).fillna(0)
    df["최근90일_산불건수"] = fire_prev.groupby(df["지역"]).transform(lambda s: s.rolling(90, min_periods=1).sum())
    ang = 2 * np.pi * (df["일시"].dt.month - 1) / 12
    df["월_sin"], df["월_cos"] = np.sin(ang), np.cos(ang)
    df[CALENDAR_FEATURES] = calendar_features(df["일시"]).to_numpy()
    return df


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """전날 정보가 없는 첫날(2024-01-01) 같은 행을 학습에서 제외한다."""
    return df.dropna(subset=FEATURES).copy()


# ---------------------------------------------------------------------------
# 5) 시간 순서 분할 · 학습 · 평가
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, cutoff="2025-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """cutoff 이전은 학습, 이후는 평가 (무작위 섞기 금지: 미래로 과거를 맞히는 착시 방지)."""
    cutoff = pd.Timestamp(cutoff)
    labeled = df[df["is_label"]]
    return labeled[labeled["일시"] < cutoff].copy(), labeled[labeled["일시"] >= cutoff].copy()


def train_model(train: pd.DataFrame, features=FEATURES) -> Pipeline:
    """표준화 + 로지스틱 회귀. 계수로 '왜 위험한지' 설명할 수 있어 소규모 자료에 적합.

    class_weight를 쓰지 않아 출력이 '실제 발생 확률'에 가깝게 유지된다.
    """
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.5, max_iter=1000))])
    return model.fit(train[features], train["산불발생"])


def evaluate(y_true, prob, threshold: float) -> dict:
    """ROC-AUC · PR-AUC · 정확도 · 정밀도 · 재현율 계산 (threshold 이상이면 '발생 예상')."""
    y = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "ROC-AUC": float(roc_auc_score(y, prob)),
        "PR-AUC": float(average_precision_score(y, prob)),
        "정확도": float((pred == y).mean()),
        "정밀도": tp / (tp + fp) if tp + fp else 0.0,
        "재현율": tp / (tp + fn) if tp + fn else 0.0,
        "양성_수": int(y.sum()), "표본_수": int(len(y)),
        "경보_비율": float(pred.mean()), "항상_미발생_정확도": float(1 - y.mean()),
    }


def bootstrap_auc_ci(y_true, prob, n: int = 300, seed: int = 0) -> tuple[float, float]:
    """평가 표본이 작아 AUC가 흔들리므로, 재표집으로 95% 구간을 어림한다."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true), np.asarray(prob)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def grade_thresholds(prob_train) -> dict:
    """등급 경계: 학습기간 예측 확률의 상위 25% / 10% / 3% 지점."""
    q = np.quantile(prob_train, [0.75, 0.90, 0.97])
    return {"주의": float(q[0]), "높음": float(q[1]), "매우 높음": float(q[2])}


def risk_grade(prob: float, thresholds: dict) -> str:
    """확률 -> 등급(낮음/주의/높음/매우 높음)."""
    if prob >= thresholds["매우 높음"]:
        return "매우 높음"
    if prob >= thresholds["높음"]:
        return "높음"
    if prob >= thresholds["주의"]:
        return "주의"
    return "낮음"


def bootstrap_auc_diff_ci(y_true, prob_a, prob_b, n: int = 300, seed: int = 0) -> tuple[float, float]:
    """같은 재표집으로 AUC(b) - AUC(a)의 95% 구간을 어림한다. 0을 포함하면 차이가 뚜렷하지 않다."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _fit_eval(train: pd.DataFrame, test: pd.DataFrame, features: list) -> tuple[Pipeline, dict, np.ndarray]:
    """한 모델을 학습하고 평가한다. 경보 기준은 모든 모델이 같은 방법(학습기간 예측 확률 상위 10%)."""
    model = train_model(train, features)
    p_train = model.predict_proba(train[features])[:, 1]
    p_test = model.predict_proba(test[features])[:, 1]
    thr = float(np.quantile(p_train, 0.90))
    m = evaluate(test["산불발생"], p_test, thr)
    m["AUC_95%구간"] = bootstrap_auc_ci(test["산불발생"], p_test)
    m["경보_기준확률"] = thr
    return model, m, p_test


def fit_and_evaluate(feat: pd.DataFrame, cutoff="2025-01-01") -> dict:
    """전체 학습 절차: (1) 시간 분할로 3개 모델 평가 (2) 서비스용 최종 모델은 전체 기간으로 재학습.

    비교 모델: 계절만 / 기존(기상+이력, BASE_FEATURES) / 달력 추가(FEATURES, 서비스에 사용).
    세 모델은 같은 학습·평가 표본과 같은 경보 기준 선정 방법을 쓴다. 평가 데이터로 조정한 것은 없다.
    """
    labeled = drop_unusable(feat[feat["is_label"]])
    usable = pd.concat([labeled, feat[~feat["is_label"]]])  # 다음 날 행은 예측 입력용으로 유지
    train, test = time_split(usable, cutoff)
    season_f = ["월_sin", "월_cos"]
    _, baseline, _ = _fit_eval(train, test, season_f)
    prev_model, metrics_prev, p_prev = _fit_eval(train, test, BASE_FEATURES)
    eval_model, metrics, p_cal = _fit_eval(train, test, FEATURES)
    metrics["AUC차이_95%구간"] = bootstrap_auc_diff_ci(test["산불발생"], p_prev, p_cal)

    final = train_model(labeled)
    thresholds = grade_thresholds(final.predict_proba(labeled[FEATURES])[:, 1])
    return {"model": final, "eval_model": eval_model, "metrics": metrics, "metrics_prev": metrics_prev,
            "baseline": baseline, "thresholds": thresholds,
            "train_range": (train["일시"].min(), train["일시"].max()),
            "test_range": (test["일시"].min(), test["일시"].max()),
            "train_pos": int(train["산불발생"].sum()), "test_pos": int(test["산불발생"].sum()),
            "train_n": len(train), "test_n": len(test), "features": usable,
            "feature_mean": labeled[FEATURES].mean().to_dict()}


def calendar_effect_table(feat: pd.DataFrame) -> pd.DataFrame:
    """평일/주말/공휴일별 실제 산불 발생률. 효과를 가정하지 않고 데이터로 확인하기 위한 표.

    '월 보정 비율'은 같은 달 평일 발생률로 기대한 발생 수 대비 실제 발생 수(평일=1).
    계절 영향(봄·가을에 산불이 많음)이 날짜 구성과 섞이는 것을 줄이려는 값이다.
    """
    df = feat[feat["is_label"]].copy()
    df["구분"] = np.where(df["공휴일"] == 1, "공휴일", np.where(df["주말"] == 1, "주말", "평일"))
    df["월"] = df["일시"].dt.month
    wd_rate = df[df["구분"] == "평일"].groupby("월")["산불발생"].mean()
    rows = []
    for g in ["평일", "주말", "공휴일"]:
        sub = df[df["구분"] == g]
        expected = float(sub["월"].map(wd_rate).sum())
        rows.append({"구분": g, "지역일수": len(sub), "발생 지역일": int(sub["산불발생"].sum()),
                     "평일 기준 기대 발생": round(expected, 1),
                     "발생률(%)": round(sub["산불발생"].mean() * 100, 2),
                     "월 보정 비율(평일=1)": round(sub["산불발생"].sum() / expected, 2) if expected else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6) 위험 예측 + 근거
# ---------------------------------------------------------------------------
def explain_row(model: Pipeline, row: pd.Series, feature_mean: dict, top: int = 3) -> str:
    """위험을 가장 많이 올린 요인 top개를 문장으로 설명한다 (로지스틱 계수 x 표준화 값)."""
    scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
    x = row[FEATURES].astype(float).to_numpy().reshape(1, -1)
    contrib = pd.Series(clf.coef_[0] * scaler.transform(pd.DataFrame(x, columns=FEATURES))[0], index=FEATURES)
    contrib["월_sin"] += contrib.pop("월_cos")  # 계절은 한 항목으로 합침
    contrib["달력"] = contrib[CALENDAR_FEATURES].sum()  # 요일·주말·공휴일도 한 항목
    contrib = contrib.drop(CALENDAR_FEATURES)
    parts = []
    for f in contrib.sort_values(ascending=False).index[:top]:
        if contrib[f] <= 0.05:
            continue
        if f == "월_sin":
            parts.append("산불이 잦은 계절")
        elif f == "달력":
            day = "공휴일" if row["공휴일"] else ("주말" if row["주말"] else "해당 요일")
            parts.append(f"달력({day}) — 사람 활동의 간접 지표")
        else:
            parts.append(f"{FEATURE_LABELS[f]} {row[f]:.1f} (평균 {feature_mean[f]:.1f})")
    return " · ".join(parts) if parts else "평균 대비 위험을 높이는 뚜렷한 요인 없음"


def predict_risk(bundle: dict, inputs: pd.DataFrame) -> pd.DataFrame:
    """지역별 입력(FEATURES 열 포함)에 대해 위험 확률·등급·근거를 계산한다."""
    out = inputs.copy()
    if out.empty:
        raise ValueError("예측할 지역이 없습니다.")
    if out[FEATURES].isna().any().any():
        raise ValueError("입력에 빈 값이 있습니다.")
    out["위험확률"] = bundle["model"].predict_proba(out[FEATURES])[:, 1]
    out["위험등급"] = out["위험확률"].map(lambda p: risk_grade(p, bundle["thresholds"]))
    out["근거"] = out.apply(lambda r: explain_row(bundle["model"], r, bundle["feature_mean"]), axis=1)
    return out


def inputs_for_date(feat: pd.DataFrame, target_date) -> pd.DataFrame:
    """target_date의 지역별 입력 특성 행들 (전날까지의 정보로 이미 계산돼 있음)."""
    rows = feat[feat["일시"] == pd.Timestamp(target_date)]
    return rows.dropna(subset=FEATURES).reset_index(drop=True)


def month_features(d: pd.Timestamp) -> tuple[float, float]:
    """월을 sin/cos 두 값으로 바꾼다 (12월과 1월이 가깝다는 것을 표현)."""
    ang = 2 * np.pi * (d.month - 1) / 12
    return float(np.sin(ang)), float(np.cos(ang))


def apply_station_inputs(inputs: pd.DataFrame, station_weather: pd.DataFrame, target_date) -> pd.DataFrame:
    """사용자가 관측소별로 고친 기상 값을 지역 입력에 덮어쓴다 (관측소 -> 연결된 지역 전체)."""
    out = inputs.copy()
    sw = station_weather.set_index("관측소")
    for col in WEATHER_INPUTS:
        out[col] = out["관측소"].map(sw[col]).astype(float)
    out["월_sin"], out["월_cos"] = month_features(pd.Timestamp(target_date))
    out[CALENDAR_FEATURES] = calendar_features([target_date]).iloc[0].to_numpy()
    return out


# ---------------------------------------------------------------------------
# 7) 자원 배치 규칙 (모델 아님)
# ---------------------------------------------------------------------------
def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """비율(weights)대로 정수 total개를 나눈다. 합계는 정확히 total (최대잔여법)."""
    raw = weights / weights.sum() * total
    base = np.floor(raw).astype(int)
    left = total - int(base.sum())
    order = np.argsort(-(raw - base), kind="stable")
    base[order[:left]] += 1
    return base


def allocate_resources(risk: pd.DataFrame, vehicles: int, personnel: int, n_regions: int = 5) -> pd.DataFrame:
    """위험 확률이 높은 상위 n_regions개 지역에 차량·인력을 확률 비례로 배분한다.

    규칙: 배분량 = 가용 수량 x (지역 위험확률 / 우선지역 위험확률 합), 정수는 최대잔여법.
    합계는 입력한 가용 수량을 넘지 않는다(정확히 같음). 예측 모델이 아니라 단순 규칙이다.
    """
    if vehicles < 0 or personnel < 0 or int(vehicles) != vehicles or int(personnel) != personnel:
        raise ValueError("차량·인력은 0 이상의 정수여야 합니다.")
    if n_regions < 1:
        raise ValueError("우선지역 수는 1 이상이어야 합니다.")
    if risk.empty:
        raise ValueError("위험도 결과가 없습니다.")
    top = risk.sort_values("위험확률", ascending=False).head(n_regions).copy()
    w = top["위험확률"].to_numpy()
    top["비중"] = w / w.sum()
    top["차량"] = _largest_remainder(w, int(vehicles))
    top["인력"] = _largest_remainder(w, int(personnel))
    top.insert(0, "우선순위", range(1, len(top) + 1))
    top["배분근거"] = top.apply(
        lambda r: f"위험확률 {r['위험확률']:.1%}, 우선지역 내 비중 {r['비중']:.0%} → 차량 {r['차량']}대·인력 {r['인력']}명", axis=1)
    assert top["차량"].sum() <= vehicles and top["인력"].sum() <= personnel
    return top.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 이력 조회
# ---------------------------------------------------------------------------
def filter_fires(fires: pd.DataFrame, regions, start, end) -> pd.DataFrame:
    """지역(목록)·기간(양끝 포함)으로 산불 이력을 거른다. regions가 비면 전체."""
    df = fires[(fires["일시"] >= pd.Timestamp(start)) & (fires["일시"] <= pd.Timestamp(end))]
    if regions:
        df = df[df["지역"].isin(regions)]
    return df


def monthly_counts(fires: pd.DataFrame, start, end) -> pd.DataFrame:
    """월별 발생 건수. 산불이 없던 달도 0으로 채워서 추이가 끊기지 않게 한다."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    cnt = fires.groupby(fires["일시"].dt.to_period("M")).size().reindex(months, fill_value=0)
    return pd.DataFrame({"월": months.astype(str), "발생건수": cnt.to_numpy()})


def years_in_range(start, end) -> list[int]:
    """조회 기간(양끝 포함)에 걸친 연도 목록 (오름차순)."""
    return list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))


def year_window(start, end, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """조회 기간을 해당 연도(1/1~12/31)로 자른 구간. 기간이 연도 일부만 걸치면 그 부분만 남는다."""
    lo = max(pd.Timestamp(start), pd.Timestamp(year=year, month=1, day=1))
    hi = min(pd.Timestamp(end), pd.Timestamp(year=year, month=12, day=31))
    if lo > hi:
        raise ValueError(f"{year}년은 조회 기간에 포함되지 않습니다.")
    return lo, hi


def monthly_cause_table(fires: pd.DataFrame, start, end) -> pd.DataFrame:
    """월 x 원인 발생 건수(빈 달은 0). 열 순서는 전체 건수가 많은 원인부터.

    원인은 산불이 난 '뒤'에 기록되는 값이므로 분석 전용이다. 예측 입력(FEATURES)에는 쓰지 않는다.
    """
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    if fires.empty:
        return pd.DataFrame({"월": months.astype(str)})
    t = pd.crosstab(fires["일시"].dt.to_period("M"), fires["원인"]).reindex(months, fill_value=0)
    t = t[t.sum().sort_values(ascending=False).index]
    t.insert(0, "월", months.astype(str))
    return t.reset_index(drop=True).rename_axis(columns=None)


def cause_share(fires: pd.DataFrame) -> pd.DataFrame:
    """원인별 건수와 비율 (많은 순)."""
    c = fires["원인"].value_counts()
    return pd.DataFrame({"원인": c.index, "건수": c.to_numpy(), "비율(%)": (c / c.sum() * 100).round(1).to_numpy()})


def unknown_cause_summary(fires: pd.DataFrame) -> dict:
    """'원인 미상'·'기타'의 건수와 비율을 실제 데이터에서 계산한다."""
    n = len(fires)
    u, e = int((fires["원인"] == "원인 미상").sum()), int((fires["원인"] == "기타").sum())
    return {"전체": n, "원인 미상": u, "기타": e, "합계": u + e, "합계_비율": (u + e) / n if n else 0.0}


def region_summary(fires: pd.DataFrame) -> pd.DataFrame:
    """지역별 건수·피해면적 합계 (많은 순)."""
    s = fires.groupby("지역").agg(발생건수=("지역", "size"), 피해면적합계_ha=("피해면적(ha)", "sum")).reset_index()
    s["피해면적합계_ha"] = s["피해면적합계_ha"].round(2)
    return s.sort_values(["발생건수", "피해면적합계_ha"], ascending=False).reset_index(drop=True)


def load_fires_with_extra() -> pd.DataFrame:
    """기본 산불 CSV(~2025)에 API로 받은 2026년 기록이 있으면 이어 붙인다 (겹치는 날짜는 기본 CSV 우선)."""
    f = load_fires()
    if EXTRA_FIRE_CSV.exists():
        extra = pd.read_csv(EXTRA_FIRE_CSV, encoding="utf-8-sig")
        extra["일시"] = pd.to_datetime(extra["일시"])
        extra = extra[extra["일시"] > f["일시"].max()]
        f = pd.concat([f, extra[[c for c in f.columns if c in extra.columns]]], ignore_index=True)
    return f


def load_weather_with_extra() -> pd.DataFrame:
    """기본 기상 CSV(2024~2025)에 API로 받은 2026년 CSV가 있으면 이어 붙인다."""
    w = load_weather()
    if EXTRA_WEATHER_CSV.exists():
        extra = pd.read_csv(EXTRA_WEATHER_CSV, encoding="utf-8-sig")
        extra["일시"] = pd.to_datetime(extra["일시"])
        w = pd.concat([w, extra[~extra["일시"].isin(w["일시"].unique())]], ignore_index=True)
    return w


def load_all() -> dict:
    """CSV 읽기부터 모델 학습까지 한 번에 수행 (화면은 이 함수 하나만 부른다)."""
    weather, w_rep = clean_weather(load_weather_with_extra())
    fires, f_rep = clean_fires(load_fires_with_extra())
    # 지역 목록은 학습 기간(~2025)에 기록이 있는 시군만. 2026에 처음 나온 시군(광명·부천 등)은 과거 이력이 없어
    # '미발생'인지 '미수집'인지 알 수 없으므로 위험도 예측 대상에서 제외하고 이력 조회에만 보여 준다.
    region_map = assign_stations(sorted(fires.loc[fires["일시"] <= PERIOD_END, "지역"].unique()))
    max_weather = weather["일시"].max()  # 기상이 있는 마지막 날 (2026 자료가 있으면 그 날짜)
    panel = build_panel(weather, fires, region_map, end=max_weather)
    feat = add_features(panel)
    bundle = fit_and_evaluate(feat)
    bundle.update(weather=weather, fires=fires, region_map=region_map, panel=panel,
                  weather_report=w_rep, fire_report=f_rep, max_date=max_weather + pd.Timedelta(days=1),
                  fire_max_date=fires["일시"].max())
    return bundle
