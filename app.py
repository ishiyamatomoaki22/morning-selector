# streamlit_app.py
# 統合版：朝イチ候補 + 夕方続行判定 + 統合ログ + バックテスト
# ✅ st.stop() を「任意入力の分岐」では使わない（他タブまで止まるため）
# ✅ 未入力のタブは “そのタブ内だけ” 情報表示して処理をスキップする

import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from zoneinfo import ZoneInfo

# ============================================================
# App Meta
# ============================================================
st.set_page_config(page_title="ジャグラー統合セレクター", layout="wide")
st.title("ジャグラー 統合セレクター（朝イチ / 夕方 / 統合ログ / バックテスト）")
st.caption("共通データ統合 → 朝イチ候補（学習ランキング）/ 夕方続行候補（当日判定）→ 実戦ログ（統合）→ バックテスト")

JST = ZoneInfo("Asia/Tokyo")
TOOL_VERSION = "v2025-12-18.2"  # ← st.stop除去版

# ============================================================
# Config
# ============================================================
BASE_HEADER = [
    "date", "weekday", "shop", "machine",
    "unit_number", "start_games", "total_start", "bb_count", "rb_count", "art_count", "max_medals",
    "bb_rate", "rb_rate", "art_rate", "gassan_rate", "prev_day_end"
]

SHOP_PRESETS = ["武蔵境", "吉祥寺", "三鷹", "国分寺", "新宿", "渋谷"]
MACHINE_PRESETS = [
    "マイジャグラーV", "ゴーゴージャグラー3", "ハッピージャグラーVIII",
    "ファンキージャグラー2KT", "ミスタージャグラー", "ジャグラーガールズSS",
    "ネオアイムジャグラーEX", "ウルトラミラクルジャグラー"
]

# 朝イチ用おすすめ（勝率寄りの目安）
RECOMMENDED_MORNING = {
    "マイジャグラーV":            {"min_games": 2500, "max_rb": 280.0, "max_gassan": 175.0},
    "ゴーゴージャグラー3":        {"min_games": 2500, "max_rb": 290.0, "max_gassan": 180.0},
    "ハッピージャグラーVIII":     {"min_games": 2800, "max_rb": 270.0, "max_gassan": 170.0},
    "ファンキージャグラー2KT":    {"min_games": 2500, "max_rb": 300.0, "max_gassan": 185.0},
    "ミスタージャグラー":         {"min_games": 2500, "max_rb": 300.0, "max_gassan": 185.0},
    "ジャグラーガールズSS":       {"min_games": 2300, "max_rb": 280.0, "max_gassan": 175.0},
    "ネオアイムジャグラーEX":      {"min_games": 2500, "max_rb": 320.0, "max_gassan": 190.0},
    "ウルトラミラクルジャグラー": {"min_games": 2800, "max_rb": 300.0, "max_gassan": 185.0},
}

# 夕方用おすすめ（続行候補を厳しめに抽出する目安）
RECOMMENDED_EVENING = {
    "マイジャグラーV":            {"min_games": 3000, "max_rb": 270.0, "max_gassan": 180.0},
    "ゴーゴージャグラー3":        {"min_games": 3000, "max_rb": 280.0, "max_gassan": 185.0},
    "ハッピージャグラーVIII":     {"min_games": 3500, "max_rb": 260.0, "max_gassan": 175.0},
    "ファンキージャグラー2KT":    {"min_games": 3000, "max_rb": 300.0, "max_gassan": 190.0},
    "ミスタージャグラー":         {"min_games": 2800, "max_rb": 300.0, "max_gassan": 190.0},
    "ジャグラーガールズSS":       {"min_games": 2500, "max_rb": 260.0, "max_gassan": 175.0},
    "ネオアイムジャグラーEX":      {"min_games": 2500, "max_rb": 330.0, "max_gassan": 200.0},
    "ウルトラミラクルジャグラー": {"min_games": 3500, "max_rb": 300.0, "max_gassan": 195.0},
}

# ============================================================
# Helpers（共通）
# ============================================================
JP_WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]

def add_weekday_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.NaT
    dt = pd.to_datetime(out["date"], errors="coerce")
    wd = dt.dt.dayofweek  # 0=Mon
    out["weekday"] = wd.map(lambda x: JP_WEEKDAYS[int(x)] if pd.notna(x) else np.nan)
    return out

def parse_rate_token(tok: str) -> float:
    """ '1/186.3' -> 186.3 , '186.3' -> 186.3 """
    if pd.isna(tok):
        return np.nan
    s = str(tok).strip().replace(",", "")
    if s == "":
        return np.nan
    m = re.match(r"^1\s*/\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    rename_map = {
        "台番": "unit_number", "台番号": "unit_number",
        "総回転": "total_start", "累計スタート": "total_start",
        "BB回数": "bb_count", "RB回数": "rb_count",
        "ART回数": "art_count", "最大持ち玉": "max_medals",
        "BB確率": "bb_rate", "RB確率": "rb_rate", "ART確率": "art_rate",
        "合成確率": "gassan_rate", "前日最終": "prev_day_end",
        "店舗": "shop", "機種": "machine", "日付": "date",
        "曜日": "weekday",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

def _zero_series(out: pd.DataFrame) -> pd.Series:
    return pd.Series([0] * len(out), index=out.index)

def compute_rates_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    - bb_rate/rb_rate/gassan_rate が空のとき、 total_start と回数から補完
    - bb_count/rb_count が欠けていても落ちないように堅牢化
    """
    out = df.copy()

    for c in ["unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals","prev_day_end"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")

    for c in ["bb_rate","rb_rate","art_rate","gassan_rate"]:
        if c not in out.columns:
            out[c] = np.nan

    out["bb_rate"] = out["bb_rate"].map(parse_rate_token)
    out["rb_rate"] = out["rb_rate"].map(parse_rate_token)
    out["art_rate"] = out["art_rate"].map(parse_rate_token)
    out["gassan_rate"] = out["gassan_rate"].map(parse_rate_token)

    if "total_start" in out.columns:
        bb_cnt = out["bb_count"] if "bb_count" in out.columns else _zero_series(out)
        rb_cnt = out["rb_count"] if "rb_count" in out.columns else _zero_series(out)

        bb_mask = out["bb_rate"].isna() & pd.to_numeric(bb_cnt, errors="coerce").fillna(0).gt(0)
        rb_mask = out["rb_rate"].isna() & pd.to_numeric(rb_cnt, errors="coerce").fillna(0).gt(0)
        gs_mask = out["gassan_rate"].isna() & (pd.to_numeric(bb_cnt, errors="coerce").fillna(0) + pd.to_numeric(rb_cnt, errors="coerce").fillna(0)).gt(0)

        if "bb_count" in out.columns:
            out.loc[bb_mask, "bb_rate"] = out.loc[bb_mask, "total_start"] / out.loc[bb_mask, "bb_count"]
        if "rb_count" in out.columns:
            out.loc[rb_mask, "rb_rate"] = out.loc[rb_mask, "total_start"] / out.loc[rb_mask, "rb_count"]
        if ("bb_count" in out.columns) and ("rb_count" in out.columns):
            denom = (out.loc[gs_mask, "bb_count"] + out.loc[gs_mask, "rb_count"]).replace(0, np.nan)
            out.loc[gs_mask, "gassan_rate"] = out.loc[gs_mask, "total_start"] / denom

    return out

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def make_filename(machine: str, suffix: str, date_str: str) -> str:
    time_part = datetime.now(JST).strftime("%H-%M-%S")
    safe_machine = str(machine).replace(" ", "").replace("/", "_").replace("\\", "_").replace(":", "-")
    return f"{date_str}_{time_part}_{safe_machine}_{suffix}.csv"

# ============================================================
# CSV Read（文字コードに強く）
# ============================================================
def read_csv_flexible(file_like) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "cp932", None]:
        try:
            if enc is None:
                return pd.read_csv(file_like)
            return pd.read_csv(file_like, encoding=enc)
        except Exception as e:
            last_err = e
            try:
                if hasattr(file_like, "seek"):
                    file_like.seek(0)
            except Exception:
                pass
    raise last_err

# ============================================================
# Island Master（共通）
# ============================================================
ISLAND_COLS = ["unit_number","island_id","side","pos","edge_type","is_end"]

def load_island_master(uploaded_file) -> pd.DataFrame:
    dfm = read_csv_flexible(uploaded_file)
    dfm = dfm.copy()
    dfm.columns = dfm.columns.astype(str).str.strip()
    missing = [c for c in ISLAND_COLS if c not in dfm.columns]
    if missing:
        raise ValueError(f"島マスタの列が不足しています: {missing}")

    dfm["unit_number"] = pd.to_numeric(dfm["unit_number"], errors="coerce").astype("Int64")
    dfm["pos"] = pd.to_numeric(dfm["pos"], errors="coerce").astype("Int64")
    dfm["is_end"] = pd.to_numeric(dfm["is_end"], errors="coerce").fillna(0).astype(int)

    dfm["island_id"] = dfm["island_id"].astype(str).str.strip()
    dfm["side"] = dfm["side"].astype(str).str.strip().str.upper()
    dfm["edge_type"] = dfm["edge_type"].astype(str).str.strip().str.lower()

    return dfm[ISLAND_COLS].copy()

def validate_island_master(dfm: pd.DataFrame) -> list[str]:
    errs = []
    if dfm["unit_number"].isna().any():
        errs.append("unit_number に欠損があります（数値に変換できない行がある）")
    if dfm["unit_number"].duplicated().any():
        errs.append("unit_number が重複しています")
    if dfm.duplicated(["island_id","side","pos"]).any():
        errs.append("(island_id, side, pos) が重複しています")

    ok_edge = {"wall","aisle","center"}
    if (~dfm["edge_type"].isin(ok_edge)).any():
        errs.append(f"edge_type は {sorted(list(ok_edge))} のみ対応です（不正値あり）")

    ok_side = {"L","R","S"}
    if (~dfm["side"].isin(ok_side)).any():
        errs.append("side は L/R/S のみ対応です（不正値あり）")

    if (~dfm["is_end"].isin([0,1])).any():
        errs.append("is_end は 0/1 のみ対応です（不正値あり）")

    dfm2 = dfm.dropna(subset=["pos"]).copy()
    for (island, side), g in dfm2.groupby(["island_id","side"]):
        poss = sorted(g["pos"].astype(int).tolist())
        if not poss:
            continue
        exp = list(range(1, max(poss)+1))
        if poss != exp:
            errs.append(f"{island}-{side}: posが連番ではありません（欠番あり）")
        maxpos = max(poss)
        end_bad = g[((g["pos"]==1) | (g["pos"]==maxpos)) & (g["is_end"]!=1)]
        if not end_bad.empty:
            errs.append(f"{island}-{side}: 端(pos=1 or {maxpos})なのに is_end!=1 の行があります")
    return errs

def attach_island_info(df_all: pd.DataFrame, island_master: pd.DataFrame) -> pd.DataFrame:
    out = df_all.copy()
    out["unit_number"] = pd.to_numeric(out["unit_number"], errors="coerce").astype("Int64")
    out = out.merge(island_master, on="unit_number", how="left")
    return out

# ============================================================
# 夕方：入力→変換（12列生データ対応）
# ============================================================
def clean_to_12_parts(line: str):
    line = line.strip()
    if not line:
        return None
    parts = re.split(r"\s+", line)

    def is_data_token(tok: str) -> bool:
        tok = tok.strip().replace(",", "")
        return bool(re.match(r"^(?:\d+(?:\.\d+)?|1/\d+(?:\.\d+)?)$", tok))

    data_parts = [p.replace(",", "") for p in parts if is_data_token(p)]
    if len(data_parts) > 12:
        data_parts = data_parts[-12:]
    if len(data_parts) != 12:
        return None
    return data_parts

def ensure_meta_columns(df: pd.DataFrame, date_str: str, shop: str, machine: str) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = date_str
    if "shop" not in out.columns:
        out["shop"] = shop
    if "machine" not in out.columns:
        out["machine"] = machine

    out["date"] = out["date"].fillna(date_str)
    out["shop"] = out["shop"].fillna(shop)
    out["machine"] = out["machine"].fillna(machine)
    return out

def align_to_base_header(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in BASE_HEADER:
        if c not in out.columns:
            out[c] = np.nan
    out = out[BASE_HEADER].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = add_weekday_column(out)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out

def parse_raw12(text: str, date_str: str, shop: str, machine: str) -> pd.DataFrame:
    rows = []
    for line_no, line in enumerate((text or "").splitlines(), start=1):
        parts12 = clean_to_12_parts(line)
        if parts12 is None:
            if line.strip() == "":
                continue
            raise ValueError(f"{line_no}行目：12列に整形できませんでした: {line}")

        (unit_number, start_games, total_start, bb_count, rb_count, art_count,
         max_medals, bb_rate, rb_rate, art_rate, gassan_rate, prev_day_end) = parts12

        rows.append({
            "date": date_str, "shop": shop, "machine": machine,
            "unit_number": unit_number, "start_games": start_games, "total_start": total_start,
            "bb_count": bb_count, "rb_count": rb_count, "art_count": art_count, "max_medals": max_medals,
            "bb_rate": bb_rate, "rb_rate": rb_rate, "art_rate": art_rate, "gassan_rate": gassan_rate,
            "prev_day_end": prev_day_end
        })

    df = pd.DataFrame(rows)
    df = normalize_columns(df)
    df = compute_rates_if_needed(df)
    df = ensure_meta_columns(df, date_str, shop, machine)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = add_weekday_column(df)
    return align_to_base_header(df)

# ============================================================
# ファイル統合（朝イチ/バックテスト用：original.csv 複数 or zip）
# ============================================================
def parse_date_machine_from_filename(filename: str):
    if not filename:
        return (None, None)
    base = str(filename).split("/")[-1]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(?:_\d{2}-\d{2}-\d{2})?_(.+?)(?:_original)?\.csv$", base)
    if not m:
        return (None, None)

    d = pd.to_datetime(m.group(1), errors="coerce")
    date_hint = d.date() if pd.notna(d) else None
    machine_hint = m.group(2)
    return (date_hint, machine_hint)

def fill_missing_meta(df: pd.DataFrame, date_hint, shop_hint, machine_hint) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns:
        out["date"] = pd.NaT
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if date_hint is not None:
        out.loc[out["date"].isna(), "date"] = pd.to_datetime(date_hint)

    if "shop" not in out.columns:
        out["shop"] = pd.NA
    if shop_hint:
        out.loc[
            out["shop"].isna()
            | (out["shop"].astype(str).str.strip() == "")
            | (out["shop"].astype(str) == "nan"),
            "shop"
        ] = shop_hint

    if "machine" not in out.columns:
        out["machine"] = pd.NA
    if machine_hint:
        out.loc[
            out["machine"].isna()
            | (out["machine"].astype(str).str.strip() == "")
            | (out["machine"].astype(str) == "nan"),
            "machine"
        ] = machine_hint

    return out

def load_many_csvs(files, default_shop: str, default_date=None, default_machine: str | None = None) -> pd.DataFrame:
    dfs = []
    for f in files:
        date_hint, machine_hint = parse_date_machine_from_filename(getattr(f, "name", ""))

        df = read_csv_flexible(f)
        df = normalize_columns(df)
        df = compute_rates_if_needed(df)

        df = fill_missing_meta(
            df,
            date_hint=date_hint or default_date,
            shop_hint=default_shop,
            machine_hint=machine_hint or default_machine,
        )

        df = align_to_base_header(df)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=BASE_HEADER)
    return pd.concat(dfs, ignore_index=True)

def load_zip_of_csv(zip_bytes: bytes, default_shop: str, default_date=None, default_machine: str | None = None) -> pd.DataFrame:
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            date_hint, machine_hint = parse_date_machine_from_filename(name)

            with z.open(name) as fp:
                df = read_csv_flexible(fp)

            df = normalize_columns(df)
            df = compute_rates_if_needed(df)

            df = fill_missing_meta(
                df,
                date_hint=date_hint or default_date,
                shop_hint=default_shop,
                machine_hint=machine_hint or default_machine,
            )
            df = align_to_base_header(df)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=BASE_HEADER)
    return pd.concat(dfs, ignore_index=True)

# ============================================================
# 朝イチ：ランキング & バックテスト（あなたの朝イチロジック）
# ============================================================
def make_threshold_df(min_games_fallback: int, max_rb_fallback: float, max_gassan_fallback: float) -> pd.DataFrame:
    rows = []
    for m in MACHINE_PRESETS:
        rec = RECOMMENDED_MORNING.get(m)
        if rec:
            rows.append({"machine": m, "min_games": int(rec["min_games"]), "max_rb": float(rec["max_rb"]), "max_gassan": float(rec["max_gassan"])})
        else:
            rows.append({"machine": m, "min_games": int(min_games_fallback), "max_rb": float(max_rb_fallback), "max_gassan": float(max_gassan_fallback)})
    rows.append({"machine": "__DEFAULT__", "min_games": int(min_games_fallback), "max_rb": float(max_rb_fallback), "max_gassan": float(max_gassan_fallback)})
    return pd.DataFrame(rows)

def add_is_good_day(df: pd.DataFrame, thr_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_start_num"] = pd.to_numeric(out["total_start"], errors="coerce")
    out["rb_rate_num"] = pd.to_numeric(out["rb_rate"], errors="coerce")
    out["gassan_rate_num"] = pd.to_numeric(out["gassan_rate"], errors="coerce")

    out = out.merge(thr_df, on="machine", how="left")
    fallback = thr_df[thr_df["machine"] == "__DEFAULT__"].iloc[0]
    for c in ["min_games","max_rb","max_gassan"]:
        out[c] = out[c].fillna(fallback[c])

    out["is_good_day"] = (
        (out["total_start_num"] >= out["min_games"]) &
        (out["rb_rate_num"] <= out["max_rb"]) &
        (out["gassan_rate_num"] <= out["max_gassan"])
    ).astype(int)
    return out

def build_ranking(
    df_all: pd.DataFrame,
    shop: str,
    machine: str,
    base_day: date,
    lookback_days: int,
    tau: int,
    min_games: int,
    max_rb: float,
    max_gassan: float,
    min_unique_days: int,
    w_unit: float,
    w_island: float,
    w_run: float,
    w_end: float,
):
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    df = df_all.copy()
    df = df[df["date"].notna()].copy()

    start_day = (pd.to_datetime(base_day) - pd.Timedelta(days=int(lookback_days))).date()
    end_day = (pd.to_datetime(base_day) - pd.Timedelta(days=1)).date()

    df = df[(df["date"] >= start_day) & (df["date"] <= end_day)].copy()
    df = df[(df["shop"] == shop) & (df["machine"] == machine)].copy()
    if df.empty:
        return pd.DataFrame()

    for c in ["island_id","side","pos","edge_type","is_end"]:
        if c not in df.columns:
            df[c] = np.nan
    df["is_end"] = pd.to_numeric(df["is_end"], errors="coerce").fillna(0).astype(int)

    df["unit_number"] = pd.to_numeric(df["unit_number"], errors="coerce")
    df = df[df["unit_number"].notna()].copy()
    df["unit_number"] = df["unit_number"].astype(int)

    df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
    df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
    df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

    df["days_ago"] = (pd.to_datetime(base_day) - pd.to_datetime(df["date"])).dt.days
    df["w"] = np.exp(-df["days_ago"] / max(int(tau), 1))

    df["is_good_day"] = (
        (df["total_start_num"] >= min_games) &
        (df["rb_rate_num"] <= max_rb) &
        (df["gassan_rate_num"] <= max_gassan)
    ).astype(int)

    gcols = ["shop", "machine", "unit_number"]
    agg_u = df.groupby(gcols, dropna=False).agg(
        samples=("date", "count"),
        unique_days=("date", "nunique"),
        w_sum=("w", "sum"),
        good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
        good_days=("is_good_day", "sum"),
        avg_rb=("rb_rate_num", "mean"),
        avg_gassan=("gassan_rate_num", "mean"),
        max_total=("total_start_num", "max"),
        island_id=("island_id", "first"),
        side=("side", "first"),
        pos=("pos", "first"),
        edge_type=("edge_type", "first"),
        is_end=("is_end", "first"),
    ).reset_index()

    agg_u = agg_u[agg_u["unique_days"] >= int(min_unique_days)].copy()
    if agg_u.empty:
        return pd.DataFrame()

    agg_u["good_rate_weighted"] = (agg_u["good_w"] / agg_u["w_sum"]).replace([np.inf, -np.inf], np.nan)
    agg_u["good_rate_simple"] = (agg_u["good_days"] / agg_u["unique_days"]).replace([np.inf, -np.inf], np.nan)

    wmax = float(agg_u["w_sum"].max() if agg_u["w_sum"].notna().any() else 0.0)
    trust = np.log1p(agg_u["w_sum"].fillna(0.0)) / np.log1p(wmax + 1e-9) if wmax > 0 else 0.0
    agg_u["unit_score"] = (agg_u["good_rate_weighted"].fillna(0.0) * 0.70 + trust * 0.30)

    out = agg_u.copy()

    # island_score
    if out["island_id"].isna().all():
        out["island_score"] = 0.0
    else:
        agg_i = df.groupby(["shop","machine","island_id"], dropna=False).agg(
            i_w_sum=("w", "sum"),
            i_good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
        ).reset_index()
        agg_i["island_score"] = (agg_i["i_good_w"] / agg_i["i_w_sum"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out = out.merge(agg_i[["shop","machine","island_id","island_score"]], on=["shop","machine","island_id"], how="left")
        out["island_score"] = out["island_score"].fillna(0.0)

    # run_score
    if out["pos"].isna().all() or out["side"].isna().all():
        out["run_score"] = 0.0
    else:
        tmp = out.sort_values(["island_id","side","pos"]).copy()
        tmp["pos"] = pd.to_numeric(tmp["pos"], errors="coerce")
        tmp["unit_score_prev"] = tmp.groupby(["island_id","side"])["unit_score"].shift(1)
        tmp["unit_score_next"] = tmp.groupby(["island_id","side"])["unit_score"].shift(-1)
        tmp["run_score"] = tmp[["unit_score_prev","unit_score_next"]].mean(axis=1, skipna=True).fillna(0.0)
        out = out.merge(tmp[["unit_number","run_score"]], on="unit_number", how="left")
        out["run_score"] = out["run_score"].fillna(0.0)

    out["end_bonus"] = (pd.to_numeric(out["is_end"], errors="coerce").fillna(0).astype(int) > 0).astype(float)

    ws = float(w_unit + w_island + w_run + w_end)
    if ws <= 0:
        ws = 1.0
    wu, wi, wr, we = (w_unit/ws, w_island/ws, w_run/ws, w_end/ws)

    out["final_score"] = (
        out["unit_score"].fillna(0.0) * wu +
        out["island_score"].fillna(0.0) * wi +
        out["run_score"].fillna(0.0) * wr +
        out["end_bonus"].fillna(0.0) * we
    )

    out["good_rate_weighted"] = (out["good_rate_weighted"] * 100).round(1)
    out["good_rate_simple"] = (out["good_rate_simple"] * 100).round(1)

    for c in ["unit_score","island_score","run_score","end_bonus","final_score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

    out["avg_rb"] = pd.to_numeric(out["avg_rb"], errors="coerce").round(1)
    out["avg_gassan"] = pd.to_numeric(out["avg_gassan"], errors="coerce").round(1)

    out = out.sort_values(
        ["final_score","island_score","run_score","unit_score","w_sum"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)
    out["train_start"] = start_day
    out["train_end"] = end_day
    out["weights"] = f"unit={wu:.2f},island={wi:.2f},run={wr:.2f},end={we:.2f}"
    return out

def backtest_precision_hit(
    df_all: pd.DataFrame,
    shop: str,
    machines: list[str],
    lookback_days: int,
    tau: int,
    min_unique_days: int,
    w_unit: float,
    w_island: float,
    w_run: float,
    w_end: float,
    min_games_fallback: int,
    max_rb_fallback: float,
    max_gassan_fallback: float,
    top_ns: list[int],
    eval_start: date | None,
    eval_end: date | None,
):
    if df_all is None or df_all.empty:
        return None, None, None

    df = df_all.copy()
    df = df[df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].notna()].copy()
    df = df[df["shop"] == shop].copy()
    if df.empty:
        return None, None, None

    all_days = sorted(df["date"].unique().tolist())
    if not all_days:
        return None, None, None

    if eval_start is None:
        eval_start = all_days[0]
    if eval_end is None:
        eval_end = all_days[-1]

    eval_days = [d for d in all_days if (d >= eval_start and d <= eval_end)]
    if not eval_days:
        return None, None, None

    thr_df = make_threshold_df(min_games_fallback, max_rb_fallback, max_gassan_fallback)
    df_labeled = add_is_good_day(df, thr_df)

    rows = []
    for day in eval_days:
        for m in machines:
            rec = RECOMMENDED_MORNING.get(m)
            mg = int(rec["min_games"]) if rec else int(min_games_fallback)
            mrb = float(rec["max_rb"]) if rec else float(max_rb_fallback)
            mgs = float(rec["max_gassan"]) if rec else float(max_gassan_fallback)

            ranking = build_ranking(
                df_all=df_all,
                shop=shop,
                machine=m,
                base_day=day,
                lookback_days=int(lookback_days),
                tau=int(tau),
                min_games=mg,
                max_rb=mrb,
                max_gassan=mgs,
                min_unique_days=int(min_unique_days),
                w_unit=float(w_unit),
                w_island=float(w_island),
                w_run=float(w_run),
                w_end=float(w_end),
            )
            if ranking is None or ranking.empty:
                continue

            df_day = df_labeled[(df_labeled["date"] == day) & (df_labeled["shop"] == shop) & (df_labeled["machine"] == m)].copy()
            if df_day.empty:
                continue

            denom_all = int(df_day["unit_number"].nunique())
            good_all = int(df_day.drop_duplicates(subset=["unit_number"])["is_good_day"].sum())
            baseline = (good_all / denom_all) if denom_all > 0 else np.nan

            rank_units = ranking["unit_number"].dropna().astype(int).tolist()

            for N in top_ns:
                chosen = rank_units[:int(N)]
                if not chosen:
                    continue

                df_ch = df_day[df_day["unit_number"].astype("Int64").isin(chosen)].drop_duplicates(subset=["unit_number"])
                denom_sel = int(len(chosen))
                good_sel = int(df_ch["is_good_day"].sum())

                precision = (good_sel / denom_sel) if denom_sel > 0 else np.nan
                hit = 1 if good_sel > 0 else 0

                rows.append({
                    "date": day,
                    "shop": shop,
                    "machine": m,
                    "topN": int(N),
                    "selected_n": denom_sel,
                    "good_in_topN": good_sel,
                    "precision_topN": precision,
                    "all_units_n": denom_all,
                    "good_all": good_all,
                    "baseline_good_rate": baseline,
                    "lift_pt": (precision - baseline) if (pd.notna(precision) and pd.notna(baseline)) else np.nan,
                    "hit_at_N": hit,
                })

    if not rows:
        return None, None, None

    detail = pd.DataFrame(rows)

    overall = []
    for N, g in detail.groupby("topN"):
        total_sel = int(g["selected_n"].sum())
        total_good_sel = int(g["good_in_topN"].sum())
        precision = (total_good_sel / total_sel) if total_sel > 0 else np.nan

        total_all = int(g["all_units_n"].sum())
        total_good_all = int(g["good_all"].sum())
        baseline = (total_good_all / total_all) if total_all > 0 else np.nan

        hit_rate = float(g["hit_at_N"].mean()) if len(g) > 0 else np.nan

        overall.append({
            "topN": int(N),
            "eval_cases": int(len(g)),
            "precision_topN": precision,
            "baseline_good_rate": baseline,
            "lift_pt": (precision - baseline) if (pd.notna(precision) and pd.notna(baseline)) else np.nan,
            "hit_rate": hit_rate,
        })
    overall_df = pd.DataFrame(overall).sort_values("topN")

    per_machine = []
    for (m, N), g in detail.groupby(["machine","topN"]):
        total_sel = int(g["selected_n"].sum())
        total_good_sel = int(g["good_in_topN"].sum())
        precision = (total_good_sel / total_sel) if total_sel > 0 else np.nan

        total_all = int(g["all_units_n"].sum())
        total_good_all = int(g["good_all"].sum())
        baseline = (total_good_all / total_all) if total_all > 0 else np.nan

        hit_rate = float(g["hit_at_N"].mean()) if len(g) > 0 else np.nan

        per_machine.append({
            "machine": m,
            "topN": int(N),
            "eval_cases": int(len(g)),
            "precision_topN": precision,
            "baseline_good_rate": baseline,
            "lift_pt": (precision - baseline) if (pd.notna(precision) and pd.notna(baseline)) else np.nan,
            "hit_rate": hit_rate,
        })
    per_machine_df = pd.DataFrame(per_machine).sort_values(["topN","lift_pt"], ascending=[True, False])

    return detail, overall_df, per_machine_df

# ============================================================
# 統合ログ（一本化）
# ============================================================
PLAYLOG_HEADER = [
    "created_at",
    "date", "weekday", "shop", "machine", "unit_number",
    "tool_phase",
    "tool_rank",
    "tool_score",
    "thr_min_games", "thr_max_rb", "thr_max_gassan",
    "tool_logic",
    "tool_version",
    "select_reason",
    "start_total_start",
    "start_bb_count",
    "start_rb_count",
    "start_rb_rate",
    "start_gassan_rate",
    "end_total_start",
    "end_bb_count",
    "end_rb_count",
    "end_rb_rate",
    "end_gassan_rate",
    "result_outcome",
    "result_hit",
    "start_time", "end_time",
    "invest_medals", "payout_medals", "profit_medals",
    "play_games",
    "stop_reason", "memo"
]

def _num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def prefill_log_from_candidate(
    row: dict,
    phase: str,
    rank: int,
    score: float,
    thr_min: float,
    thr_rb: float,
    thr_gs: float,
    tool_logic: str,
    fallback_date_str: str,
    fallback_shop: str,
    fallback_machine: str,
    select_reason: str = "ツール上位",
):
    st.session_state["log_date"] = str(row.get("date", fallback_date_str))
    st.session_state["log_shop"] = str(row.get("shop", fallback_shop))
    st.session_state["log_machine"] = str(row.get("machine", fallback_machine))
    st.session_state["log_unit"] = int(_num(row.get("unit_number", 0)) or 0)

    st.session_state["log_phase"] = phase
    st.session_state["log_rank"] = int(rank)
    st.session_state["log_score"] = float(_num(score) or 0.0)
    st.session_state["log_thr_min"] = float(_num(thr_min) or 0)
    st.session_state["log_thr_rb"] = float(_num(thr_rb) or 0)
    st.session_state["log_thr_gs"] = float(_num(thr_gs) or 0)
    st.session_state["log_logic"] = tool_logic
    st.session_state["log_reason"] = select_reason

    st.session_state["log_start_total"] = float(_num(row.get("total_start", np.nan)) or 0)
    st.session_state["log_start_bb"] = float(_num(row.get("bb_count", np.nan)) or 0)
    st.session_state["log_start_rb"] = float(_num(row.get("rb_count", np.nan)) or 0)
    st.session_state["log_start_rb_rate"] = float(_num(row.get("rb_rate", np.nan)) or 0)
    st.session_state["log_start_gassan_rate"] = float(_num(row.get("gassan_rate", np.nan)) or 0)

    for k in ["log_end_total", "log_end_bb", "log_end_rb", "log_end_rb_rate", "log_end_gassan_rate"]:
        st.session_state.setdefault(k, "")

    st.session_state.setdefault("log_outcome", "不明")
    st.session_state.setdefault("log_hit", "不明")

    st.toast("✅ 候補情報を実戦ログフォームへ反映しました（実戦ログタブを開いてください）")

def append_row_to_uploaded_csv(uploaded_bytes: bytes, new_row: dict) -> bytes:
    try:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception:
        df = pd.DataFrame(columns=PLAYLOG_HEADER)

    for c in PLAYLOG_HEADER:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PLAYLOG_HEADER]

    df2 = pd.DataFrame([new_row], columns=PLAYLOG_HEADER)
    out = pd.concat([df, df2], ignore_index=True)
    return out.to_csv(index=False).encode("utf-8-sig")

def make_log_filename(date_str: str) -> str:
    time_part = datetime.now(JST).strftime("%H-%M-%S")
    return f"{date_str}_{time_part}_playlog_unified.csv"

# ============================================================
# Sidebar（キー分離）
# ============================================================
if "morning_min_games" not in st.session_state:
    st.session_state["morning_min_games"] = 2500
if "morning_max_rb" not in st.session_state:
    st.session_state["morning_max_rb"] = 290.0
if "morning_max_gassan" not in st.session_state:
    st.session_state["morning_max_gassan"] = 195.0

if "evening_min_games" not in st.session_state:
    st.session_state["evening_min_games"] = 3000
if "evening_max_rb" not in st.session_state:
    st.session_state["evening_max_rb"] = 270.0
if "evening_max_gassan" not in st.session_state:
    st.session_state["evening_max_gassan"] = 180.0

if "evening_w_rb" not in st.session_state:
    st.session_state["evening_w_rb"] = 70.0
if "evening_w_total" not in st.session_state:
    st.session_state["evening_w_total"] = 20.0
if "evening_w_gs" not in st.session_state:
    st.session_state["evening_w_gs"] = 10.0
if "evening_run_bonus_w" not in st.session_state:
    st.session_state["evening_run_bonus_w"] = 1.5

with st.sidebar:
    st.header("基本情報（共通）")
    d = st.date_input("日付", value=date.today(), key="meta_date")
    date_str = d.strftime("%Y-%m-%d")

    shop_mode = st.radio("店名", ["選択", "手入力"], horizontal=True, key="shop_mode")
    if shop_mode == "選択":
        shop = st.selectbox("shop", SHOP_PRESETS, index=0, key="shop_select")
    else:
        shop = st.text_input("shop", value="武蔵境", key="shop_text")

    machine_mode = st.radio("機種", ["選択", "手入力"], horizontal=True, key="machine_mode")
    if machine_mode == "選択":
        machine = st.selectbox("machine", MACHINE_PRESETS, index=0, key="machine_select")
    else:
        machine = st.text_input("machine", value="マイジャグラーV", key="machine_text")

    st.caption(f"ツールVersion: {TOOL_VERSION}")

    st.divider()
    with st.expander("朝イチ設定（学習ランキング）", expanded=True):
        recm = RECOMMENDED_MORNING.get(machine)
        if recm:
            st.caption("※ 朝イチ候補の“良台日判定”目安（機種別）")
            st.write(f"- 最低総回転: **{recm['min_games']}**")
            st.write(f"- REG上限: **{recm['max_rb']}**")
            st.write(f"- 合算上限: **{recm['max_gassan']}**")
            if st.button("朝イチおすすめ値を反映", use_container_width=True, key="btn_apply_morning"):
                st.session_state["morning_min_games"] = int(recm["min_games"])
                st.session_state["morning_max_rb"] = float(recm["max_rb"])
                st.session_state["morning_max_gassan"] = float(recm["max_gassan"])
                st.rerun()
        else:
            st.caption("この機種は朝イチプリセット未登録です。手動で調整してください。")

        st.subheader("スコア重み（島マスタ前提）")
        w_island = st.slider("島（island）の重み", 0.0, 1.0, 0.45, 0.05, key="morning_w_island")
        w_run    = st.slider("並び（run）の重み",   0.0, 1.0, 0.35, 0.05, key="morning_w_run")
        w_end    = st.slider("端（end）の重み",     0.0, 1.0, 0.10, 0.05, key="morning_w_end")
        w_unit = max(0.0, 1.0 - (w_island + w_run + w_end))
        st.caption(f"台単体の重み（自動目安）: {w_unit:.2f}  ※ 合計>1でも内部で正規化")

        st.subheader("良台日判定（朝イチ）")
        morning_min_games = st.slider(
            "最低 総回転（total_start）",
            0, 10000,
            value=int(st.session_state["morning_min_games"]),
            step=100,
            key="morning_min_games_slider",
        )
        morning_max_rb = st.slider(
            "REG上限（rb_rate）",
            150.0, 600.0,
            value=float(st.session_state["morning_max_rb"]),
            step=1.0,
            key="morning_max_rb_slider",
        )
        morning_max_gassan = st.slider(
            "合算上限（gassan_rate）",
            80.0, 350.0,
            value=float(st.session_state["morning_max_gassan"]),
            step=1.0,
            key="morning_max_gassan_slider",
        )

        lookback_days = st.number_input("集計対象：過去何日（学習窓）", 1, 365, 60, 1, key="morning_lookback")
        tau = st.number_input("日付減衰 τ（小さいほど直近重視）", 1, 120, 14, 1, key="morning_tau")
        min_unique_days = st.number_input("最小サンプル日数（稼働日数）", 1, 60, 3, 1, key="morning_min_unique")
        morning_top_n = st.number_input("上位N件表示（朝イチ）", 1, 200, 30, 1, key="morning_topn")

    st.divider()
    with st.expander("夕方設定（当日フィルタ）", expanded=True):
        rece = RECOMMENDED_EVENING.get(machine)
        if rece:
            st.caption("※ 夕方続行候補を“厳しめに抽出”する目安（機種別）")
            st.write(f"- 最低総回転: **{rece['min_games']}**")
            st.write(f"- REG上限: **{rece['max_rb']}**")
            st.write(f"- 合算上限: **{rece['max_gassan']}**")
            if st.button("夕方おすすめ値を反映", use_container_width=True, key="btn_apply_evening"):
                st.session_state["evening_min_games"] = int(rece["min_games"])
                st.session_state["evening_max_rb"] = float(rece["max_rb"])
                st.session_state["evening_max_gassan"] = float(rece["max_gassan"])
                st.rerun()
        else:
            st.caption("この機種は夕方プリセット未登録です。手動で調整してください。")

        evening_min_games = st.slider(
            "最低 総回転（total_start）",
            0, 10000,
            value=int(st.session_state["evening_min_games"]),
            step=100,
            key="evening_min_games_slider",
        )
        evening_max_rb = st.slider(
            "REG上限（rb_rate）",
            150.0, 600.0,
            value=float(st.session_state["evening_max_rb"]),
            step=1.0,
            key="evening_max_rb_slider",
        )
        evening_max_gassan = st.slider(
            "合算上限（gassan_rate）",
            80.0, 350.0,
            value=float(st.session_state["evening_max_gassan"]),
            step=1.0,
            key="evening_max_gassan_slider",
        )
        evening_top_n = st.number_input("上位N件表示（夕方）", 1, 200, 30, 1, key="evening_topn")

        st.subheader("夕方スコア配分（強化）")
        st.caption("※ RB/回転/合算をどう重視するか。合計は内部で正規化します。")
        ew_rb = st.slider("RB重み", 0.0, 100.0, float(st.session_state["evening_w_rb"]), 1.0, key="evening_w_rb_slider")
        ew_total = st.slider("回転重み", 0.0, 100.0, float(st.session_state["evening_w_total"]), 1.0, key="evening_w_total_slider")
        ew_gs = st.slider("合算重み", 0.0, 100.0, float(st.session_state["evening_w_gs"]), 1.0, key="evening_w_gs_slider")
        run_bonus_w = st.slider("並びボーナス係数", 0.0, 5.0, float(st.session_state["evening_run_bonus_w"]), 0.1, key="evening_run_bonus_w_slider")

# ============================================================
# 共通UI：過去データ統合（朝イチ/バックテスト用）
# ============================================================
def upload_past_data_ui():
    st.caption("複数CSV（original.csv）または zip（CSVまとめ）をアップロードしてください。")
    st.caption("※ date/shop/machine がCSV内に無い場合：ファイル名（YYYY-MM-DD_機種...）→無ければサイドバー値で補完します。")
    st.caption("★ weekday は『統合時に date から自動生成』します（入力不要）")

    colA, colB = st.columns(2)
    with colA:
        past_files = st.file_uploader(
            "過去データCSVを複数選択（original.csv）",
            type=["csv"],
            accept_multiple_files=True,
            key="multi_csv_shared"
        )
    with colB:
        past_zip = st.file_uploader(
            "zipでまとめてアップロード（任意）",
            type=["zip"],
            accept_multiple_files=False,
            key="zip_shared"
        )

    if (not past_files) and (past_zip is None):
        st.info("ここに過去データを入れると、店×機種で候補台をランキングします。")
        return pd.DataFrame(columns=BASE_HEADER)

    default_shop = shop
    default_date = d
    default_machine = machine

    df_all = pd.DataFrame(columns=BASE_HEADER)
    if past_files:
        df_all = pd.concat(
            [df_all, load_many_csvs(past_files, default_shop=default_shop, default_date=default_date, default_machine=default_machine)],
            ignore_index=True
        )
    if past_zip is not None:
        df_all = pd.concat(
            [df_all, load_zip_of_csv(past_zip.getvalue(), default_shop=default_shop, default_date=default_date, default_machine=default_machine)],
            ignore_index=True
        )

    if df_all.empty:
        st.error("CSVが読み込めませんでした（中身が空/形式違いの可能性）。")
        return pd.DataFrame(columns=BASE_HEADER)

    do_dedup = st.checkbox(
        "統合時に重複行を除去（date+shop+machine+unit_number が同一なら最後の行を採用）",
        value=True,
        key="dedup_unified"
    )
    if do_dedup:
        df_all = df_all.drop_duplicates(subset=["date","shop","machine","unit_number"], keep="last").copy()

    st.subheader("0) ファイル統合（analysis用 unified.csv を作成）")
    st.write(f"統合結果：**{len(df_all)} 行**（列：{len(df_all.columns)}）")

    st.download_button(
        "統合CSV（unified.csv）をダウンロード",
        data=to_csv_bytes(df_all),
        file_name=f"{date_str}_unified.csv",
        mime="text/csv",
        key="dl_unified_csv"
    )

    with st.expander("統合プレビュー（先頭/末尾）", expanded=False):
        st.dataframe(df_all.head(20), use_container_width=True, hide_index=True)
        st.dataframe(df_all.tail(20), use_container_width=True, hide_index=True)

    return df_all

# ============================================================
# Main Tabs
# ============================================================
tab_common, tab_morning, tab_evening_in, tab_evening_pick, tab_log, tab_bt = st.tabs([
    "共通：データ統合 / 島マスタ",
    "朝イチ：候補ランキング",
    "夕方：入力→変換（統一CSV作成）",
    "夕方：続行候補（判定）",
    "実戦ログ（統合）",
    "バックテスト（朝イチ精度）",
])

# -------- 共通：統合データ & 島マスタ --------
with tab_common:
    st.subheader("共通：過去データアップロード（朝イチ・バックテストで使用）")
    df_all_shared = upload_past_data_ui()

    st.divider()
    st.subheader("共通：島マスタアップロード（island.csv）")
    island_file = st.file_uploader("島マスタ（island.csv）をアップロード", type=["csv"], key="island_csv_common")

    island_master = None
    if island_file is None:
        st.info("島マスタが未指定です。島/並び/端の評価は 0 として扱い、台単体のみでランキングします。")
    else:
        try:
            island_master = load_island_master(island_file)
            errs = validate_island_master(island_master)
            if errs:
                st.error("島マスタに問題があります（島マスタは無効として続行します）:\n- " + "\n- ".join(errs))
                island_master = None
            else:
                st.success(f"島マスタ読込OK：{len(island_master)}台")
        except Exception as e:
            st.error(f"島マスタの読み込みに失敗しました（島マスタは無効として続行します）: {e}")
            island_master = None

    if (island_master is not None) and (not df_all_shared.empty):
        df_all_shared = attach_island_info(df_all_shared, island_master)
        miss = df_all_shared[df_all_shared["island_id"].isna()]["unit_number"].dropna().unique().tolist()
        if miss:
            st.warning(f"島マスタに存在しない台番号が過去データに含まれています（例）: {miss[:10]}")

    st.session_state["df_all_shared"] = df_all_shared
    st.session_state["island_master"] = island_master

# -------- 朝イチ：候補ランキング --------
with tab_morning:
    st.subheader("朝イチ候補（過去の original.csv を集計してランキング）")

    df_all_shared = st.session_state.get("df_all_shared", pd.DataFrame(columns=BASE_HEADER))
    if df_all_shared.empty:
        st.info("過去データが未投入です。朝イチランキングは『共通：データ統合』で過去CSV/zipを入れると表示されます。")
    else:
        base_day = pd.to_datetime(date_str).date()

        ranking = build_ranking(
            df_all=df_all_shared,
            shop=shop,
            machine=machine,
            base_day=base_day,
            lookback_days=int(lookback_days),
            tau=int(tau),
            min_games=int(morning_min_games),
            max_rb=float(morning_max_rb),
            max_gassan=float(morning_max_gassan),
            min_unique_days=int(min_unique_days),
            w_unit=float(w_unit),
            w_island=float(w_island),
            w_run=float(w_run),
            w_end=float(w_end),
        )

        if ranking.empty:
            st.warning("指定した条件でランキングが作れません（データ不足/サンプル不足）。条件を緩めてください。")
        else:
            train_start = ranking["train_start"].iloc[0]
            train_end = ranking["train_end"].iloc[0]
            st.success(f"学習期間：{train_start}〜{train_end}（対象：{shop} / {machine}）  |  台数: {len(ranking)}")

            show_cols = [
                "rank",
                "unit_number",
                "island_id","side","pos","edge_type","is_end",
                "final_score",
                "island_score","run_score","unit_score","end_bonus",
                "good_rate_weighted","good_rate_simple",
                "unique_days","samples","w_sum",
                "avg_rb","avg_gassan","max_total",
                "weights"
            ]
            show_cols = [c for c in show_cols if c in ranking.columns]

            st.dataframe(ranking.head(int(morning_top_n))[show_cols], use_container_width=True, hide_index=True)
            st.session_state["last_morning_ranking"] = ranking

            st.divider()
            st.subheader("候補→実戦ログへ反映（朝イチ）")
            ranks = ranking["rank"].astype(int).tolist()
            sel_rank = st.selectbox("反映したいrank", options=ranks[:min(200, len(ranks))], index=0, key="morning_prefill_rank")
            if st.button("この候補を実戦ログフォームに反映", type="primary", use_container_width=True, key="btn_prefill_morning"):
                r = ranking[ranking["rank"] == int(sel_rank)].iloc[0].to_dict()
                prefill_log_from_candidate(
                    row=r,
                    phase="morning",
                    rank=int(sel_rank),
                    score=float(r.get("final_score", 0.0)),
                    thr_min=float(morning_min_games),
                    thr_rb=float(morning_max_rb),
                    thr_gs=float(morning_max_gassan),
                    tool_logic="morning_rank",
                    fallback_date_str=date_str,
                    fallback_shop=shop,
                    fallback_machine=machine,
                    select_reason="ツール上位",
                )

            filename = make_filename(machine, "morning_candidates", date_str)
            st.download_button(
                "候補台ランキングをCSVでダウンロード",
                data=to_csv_bytes(ranking),
                file_name=filename,
                mime="text/csv",
                key="tab_morning_dl_candidates"
            )

# -------- 夕方：入力→変換 --------
with tab_evening_in:
    st.subheader("夕方：入力 → 変換（統一済みCSVを作成してダウンロード）")
    st.caption("入力（CSV or 生テキスト12列）→ ヘッダー統一（weekday含む）→ 統一CSVを作成")

    input_mode = st.radio("入力", ["CSVアップロード", "生データ貼り付け（12列）"], horizontal=True, key="evening_input_mode")
    df_unified = None

    if input_mode == "CSVアップロード":
        uploaded = st.file_uploader("元CSVをアップロード（ヘッダーあり想定）", type=["csv"], key="evening_tab1_csv")
        if uploaded:
            try:
                df = read_csv_flexible(uploaded)
                df = normalize_columns(df)
                df = ensure_meta_columns(df, date_str, shop, machine)
                df = compute_rates_if_needed(df)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = add_weekday_column(df)
                df_unified = align_to_base_header(df)
            except Exception as e:
                st.error(f"CSVの読み込み/変換に失敗しました: {e}")

    else:
        sample = "478 45 3539 19 11 0 2481 1/186.3 1/321.7 0.0 118.0 449"
        raw_text = st.text_area("台データオンラインの行を貼り付け（複数行OK）", value=sample, height=220, key="evening_raw")
        if st.button("変換して統一CSVを作る", type="primary", key="evening_convert_btn"):
            try:
                df_unified = parse_raw12(raw_text, date_str, shop, machine)
            except Exception as e:
                st.error(str(e))

    if df_unified is None:
        st.info("入力を行うと、ここに統一済みデータが表示され、CSVダウンロードできます。")
    else:
        st.success(f"統一済みデータを作成しました：{len(df_unified)}行")
        st.dataframe(df_unified.head(30), use_container_width=True, hide_index=True)

        st.session_state["last_evening_unified"] = df_unified

        filename = make_filename(machine, "original", date_str)
        st.download_button(
            "統一済みCSVをダウンロード",
            data=to_csv_bytes(df_unified),
            file_name=filename,
            mime="text/csv",
            key="evening_tab1_dl_unified"
        )
        st.caption("次に「夕方：続行候補（判定）」タブで、この統一CSVをアップロードして判定できます。")

# -------- 夕方：続行候補 --------
# -------- 夕方：続行候補 --------
with tab_evening_pick:
    st.subheader("夕方：続行候補（統一済みCSVをアップロードして判定）")
    st.caption("条件：total_start / rb_rate / gassan_rate +（任意）並びボーナス で候補抽出")

    island_master = st.session_state.get("island_master", None)

    unified_file = st.file_uploader(
        "統一済みCSV（unified.csv）をアップロード",
        type=["csv"],
        key="evening_tab2_unified"
    )

    # ★ st.stop() を使わず、未入力ならこのタブだけ案内して終了（他タブは描画される）
    if unified_file is None:
        st.info("統一CSVが未指定です。ここは未入力でもOKです（他タブの実戦ログ・バックテストは使えます）。")
        df_last = st.session_state.get("last_evening_unified")
        if isinstance(df_last, pd.DataFrame) and not df_last.empty:
            with st.expander("直前に作成した統一データ（参考）", expanded=False):
                st.dataframe(df_last.head(30), use_container_width=True, hide_index=True)
    else:
        # ---- 読み込み & 正規化 ----
        df = read_csv_flexible(unified_file)
        df = normalize_columns(df)
        df = compute_rates_if_needed(df)
        df = ensure_meta_columns(df, date_str, shop, machine)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = add_weekday_column(df)
        df = align_to_base_header(df)

        # 島情報の結合（任意）
        if island_master is not None and not island_master.empty:
            df2 = df.copy()
            df2["unit_number"] = pd.to_numeric(df2["unit_number"], errors="coerce").astype("Int64")
            df2 = df2.merge(island_master, on="unit_number", how="left")
            df = df2

        df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
        df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
        df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

        cand = df[
            (df["total_start_num"] >= evening_min_games) &
            (df["rb_rate_num"] <= evening_max_rb) &
            (df["gassan_rate_num"] <= evening_max_gassan)
        ].copy()

        # 並びボーナス（当日候補内で隣接があれば加点）
        cand["pos_num"] = pd.to_numeric(cand.get("pos", np.nan), errors="coerce")
        cand["run_bonus"] = 0

        if "island_id" in cand.columns and cand["island_id"].notna().any():
            key_cols = ["island_id", "side"]
            pos_map = (
                cand.dropna(subset=["pos_num"])
                .groupby(key_cols)["pos_num"]
                .apply(lambda s: set(s.astype(int)))
                .to_dict()
            )

            def _run_bonus(row):
                if pd.isna(row["pos_num"]):
                    return 0
                k = (row.get("island_id", None), row.get("side", None))
                if k not in pos_map:
                    return 0
                p = int(row["pos_num"])
                s = pos_map[k]
                return 1 if ((p - 1 in s) or (p + 1 in s)) else 0

            cand["run_bonus"] = cand.apply(_run_bonus, axis=1).astype(int)

        if cand.empty:
            st.warning("条件に合う台がありません。閾値を緩めるか、回転数が増えてから再判定してください。")
        else:
            eps = 1e-9
            rb = cand["rb_rate_num"].replace(0, np.nan)
            gs = cand["gassan_rate_num"].replace(0, np.nan)

            cand["score"] = (
                (evening_max_rb / (rb + eps)) * 70 +
                (cand["total_start_num"] / max(evening_min_games, 1)) * 20 +
                (evening_max_gassan / (gs + eps)) * 10
            )
            cand["score"] = cand["score"] + (cand["run_bonus"] * 1.5)
            cand["score"] = cand["score"].replace([np.inf, -np.inf], np.nan).fillna(0)

            # 表示はRB優先＋同率なら回転数（従来通り）
            cand = cand.sort_values(["rb_rate_num", "total_start_num"], ascending=[True, False]).copy()
            cand["evening_rank"] = np.arange(1, len(cand) + 1)

            show_cols = [
                "evening_rank",
                "date", "weekday", "shop", "machine",
                "unit_number", "total_start", "bb_count", "rb_count",
                "bb_rate", "rb_rate", "gassan_rate",
                "run_bonus", "score"
            ]
            show_cols = [c for c in show_cols if c in cand.columns]
            show = cand[show_cols].copy()

            for c in ["bb_rate", "rb_rate", "gassan_rate", "score"]:
                if c in show.columns:
                    show[c] = pd.to_numeric(show[c], errors="coerce").round(1)

            st.dataframe(show.head(int(evening_top_n)), use_container_width=True, hide_index=True)
            st.session_state["last_evening_candidates"] = cand

            filename = make_filename(machine, "evening_candidates", date_str)
            st.download_button(
                "夕方候補台をCSVでダウンロード",
                data=to_csv_bytes(show),
                file_name=filename,
                mime="text/csv",
                key="evening_tab2_dl_candidates"
            )

# -------- 実戦ログ（統合） --------
with tab_log:
    st.subheader("実戦ログ（統合：朝イチ/夕方を1ファイルで管理）")
    st.caption("※ Streamlit Cloudではローカルファイルを直接書き換えできないため、追記した“更新版CSV”を生成してダウンロードします。")

    uploaded_log = st.file_uploader(
        "追記したいログCSVを選択（既存の playlog_unified.csv など）",
        type=["csv"],
        key="log_upload_unified"
    )

    st.divider()

    with st.form("playlog_form_unified", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            log_date = st.text_input("date（YYYY-MM-DD）", value=st.session_state.get("log_date", date_str), key="log_date")
            log_shop = st.text_input("shop", value=st.session_state.get("log_shop", shop), key="log_shop")
        with c2:
            log_machine = st.text_input("machine", value=st.session_state.get("log_machine", machine), key="log_machine")
            unit_number = st.number_input("unit_number（台番号）", min_value=0, step=1, value=int(st.session_state.get("log_unit", 0)), key="log_unit")
        with c3:
            tool_phase = st.selectbox("tool_phase", ["morning", "evening", "none"], index=["morning","evening","none"].index(st.session_state.get("log_phase","morning")), key="log_phase")
            tool_rank = st.number_input("tool_rank（候補順位）", min_value=0, step=1, value=int(st.session_state.get("log_rank", 0)), key="log_rank")
            tool_score = st.number_input("tool_score（候補スコア）", value=float(st.session_state.get("log_score", 0.0)), step=0.1, key="log_score")

        if tool_phase == "morning":
            thr_min = int(morning_min_games)
            thr_rb = float(morning_max_rb)
            thr_gs = float(morning_max_gassan)
            tool_logic_default = "morning_rank"
        elif tool_phase == "evening":
            thr_min = int(evening_min_games)
            thr_rb = float(evening_max_rb)
            thr_gs = float(evening_max_gassan)
            tool_logic_default = "evening_filter"
        else:
            thr_min, thr_rb, thr_gs = (np.nan, np.nan, np.nan)
            tool_logic_default = "none"

        c4, c5, c6 = st.columns(3)
        with c4:
            thr_min_games_in = st.number_input("thr_min_games（使用閾値）", value=float(st.session_state.get("log_thr_min", thr_min if pd.notna(thr_min) else 0.0)), step=100.0, key="log_thr_min")
        with c5:
            thr_max_rb_in = st.number_input("thr_max_rb（使用閾値）", value=float(st.session_state.get("log_thr_rb", thr_rb if pd.notna(thr_rb) else 0.0)), step=1.0, key="log_thr_rb")
        with c6:
            thr_max_gassan_in = st.number_input("thr_max_gassan（使用閾値）", value=float(st.session_state.get("log_thr_gs", thr_gs if pd.notna(thr_gs) else 0.0)), step=1.0, key="log_thr_gs")

        select_reason = st.selectbox(
            "select_reason（着席理由）",
            ["", "ツール上位", "末尾が強い", "角/角2", "並び/帯が強い", "前日挙動", "直感", "空き台都合", "その他"],
            index=0,
            key="log_reason"
        )

        with st.expander("開始/終了スナップショット（分析用・任意）", expanded=False):
            s1, s2 = st.columns(2)
            with s1:
                start_total = st.number_input("start_total_start", value=float(st.session_state.get("log_start_total", 0.0)), step=1.0, key="log_start_total")
                start_bb = st.number_input("start_bb_count", value=float(st.session_state.get("log_start_bb", 0.0)), step=1.0, key="log_start_bb")
                start_rb = st.number_input("start_rb_count", value=float(st.session_state.get("log_start_rb", 0.0)), step=1.0, key="log_start_rb")
                start_rb_rate = st.number_input("start_rb_rate", value=float(st.session_state.get("log_start_rb_rate", 0.0)), step=0.1, key="log_start_rb_rate")
                start_gs_rate = st.number_input("start_gassan_rate", value=float(st.session_state.get("log_start_gassan_rate", 0.0)), step=0.1, key="log_start_gassan_rate")
            with s2:
                end_total = st.text_input("end_total_start（未入力OK）", value=str(st.session_state.get("log_end_total", "")), key="log_end_total")
                end_bb = st.text_input("end_bb_count（未入力OK）", value=str(st.session_state.get("log_end_bb", "")), key="log_end_bb")
                end_rb = st.text_input("end_rb_count（未入力OK）", value=str(st.session_state.get("log_end_rb", "")), key="log_end_rb")
                end_rb_rate = st.text_input("end_rb_rate（未入力OK）", value=str(st.session_state.get("log_end_rb_rate", "")), key="log_end_rb_rate")
                end_gs_rate = st.text_input("end_gassan_rate（未入力OK）", value=str(st.session_state.get("log_end_gassan_rate", "")), key="log_end_gassan_rate")

            rr1, rr2 = st.columns(2)
            with rr1:
                result_outcome = st.selectbox("result_outcome", ["不明", "勝ち", "負け", "トントン"], key="log_outcome")
            with rr2:
                result_hit = st.selectbox("result_hit", ["不明", "当たり", "外れ"], key="log_hit")

        c7, c8, c9 = st.columns(3)
        with c7:
            start_time = st.text_input("start_time（例 09:05 / 18:10）", value="", key="log_start")
        with c8:
            end_time = st.text_input("end_time（例 11:20 / 20:45）", value="", key="log_end")
        with c9:
            stop_reason = st.selectbox(
                "stop_reason（ヤメ理由）",
                ["", "様子見終了", "REG悪化", "合算悪化", "資金切れ", "他に移動", "閉店", "その他"],
                key="log_stop"
            )

        c10, c11, c12 = st.columns(3)
        with c10:
            invest = st.number_input("invest_medals（投資枚）", min_value=0, step=50, value=0, key="log_invest")
        with c11:
            payout = st.number_input("payout_medals（回収枚）", min_value=0, step=50, value=0, key="log_payout")
        with c12:
            profit = int(payout - invest)
            st.metric("profit_medals（収支枚）", profit)

        auto_games = st.checkbox("play_games を（end_total - start_total）で自動計算（入力がある場合）", value=True, key="log_auto_games")
        play_games_in = st.number_input("play_games（自分が回したG数）", min_value=0, step=10, value=0, key="log_games")

        memo = st.text_area("memo（任意）", value="", height=120, key="log_memo")

        submit = st.form_submit_button("この内容で追記用データを作成", type="primary")

    if submit:
        if uploaded_log is None:
            st.error("先に「追記したいログCSV」を選択してください。")
        else:
            dt = pd.to_datetime(log_date, errors="coerce")
            wd = JP_WEEKDAYS[int(dt.dayofweek)] if pd.notna(dt) else ""

            tool_logic = st.session_state.get("log_logic", tool_logic_default)

            end_total_f = _num(end_total)
            end_bb_f = _num(end_bb)
            end_rb_f = _num(end_rb)
            end_rb_rate_f = _num(end_rb_rate)
            end_gs_rate_f = _num(end_gs_rate)

            if auto_games and pd.notna(end_total_f) and pd.notna(start_total):
                play_games = int(max(end_total_f - float(start_total), 0))
            else:
                play_games = int(play_games_in)

            new_row = {
                "created_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
                "date": log_date,
                "weekday": wd,
                "shop": log_shop,
                "machine": log_machine,
                "unit_number": int(unit_number),

                "tool_phase": tool_phase,
                "tool_rank": int(tool_rank),
                "tool_score": float(tool_score),
                "thr_min_games": int(thr_min_games_in) if thr_min_games_in is not None else np.nan,
                "thr_max_rb": float(thr_max_rb_in) if thr_max_rb_in is not None else np.nan,
                "thr_max_gassan": float(thr_max_gassan_in) if thr_max_gassan_in is not None else np.nan,
                "tool_logic": tool_logic,
                "tool_version": TOOL_VERSION,
                "select_reason": select_reason,

                "start_total_start": float(start_total) if start_total is not None else np.nan,
                "start_bb_count": float(start_bb) if start_bb is not None else np.nan,
                "start_rb_count": float(start_rb) if start_rb is not None else np.nan,
                "start_rb_rate": float(start_rb_rate) if start_rb_rate is not None else np.nan,
                "start_gassan_rate": float(start_gs_rate) if start_gs_rate is not None else np.nan,

                "end_total_start": end_total_f,
                "end_bb_count": end_bb_f,
                "end_rb_count": end_rb_f,
                "end_rb_rate": end_rb_rate_f,
                "end_gassan_rate": end_gs_rate_f,

                "result_outcome": st.session_state.get("log_outcome", "不明"),
                "result_hit": st.session_state.get("log_hit", "不明"),

                "start_time": start_time,
                "end_time": end_time,
                "invest_medals": int(invest),
                "payout_medals": int(payout),
                "profit_medals": int(profit),
                "play_games": int(play_games),
                "stop_reason": stop_reason,
                "memo": memo,
            }

            out_bytes = append_row_to_uploaded_csv(uploaded_log.getvalue(), new_row)
            out_name = make_log_filename(log_date)

            st.success("追記済みの更新版CSVを作成しました。下のボタンからダウンロードしてください。")
            st.download_button(
                "追記済みログCSVをダウンロード（更新版）",
                data=out_bytes,
                file_name=out_name,
                mime="text/csv",
                key="log_download_unified"
            )

            st.divider()
            st.markdown("#### 追記後プレビュー（末尾5行）")
            preview_df = pd.read_csv(io.BytesIO(out_bytes))
            st.dataframe(preview_df.tail(5), use_container_width=True, hide_index=True)

# -------- バックテスト（朝イチ精度） --------
with tab_bt:
    st.subheader("バックテスト（朝イチ精度検証：上位Nの良台率 / lift / Hit@N）")
    st.caption("※ その日を予測する際、学習には前日までのデータのみ使用（リーク防止）。良台判定は朝イチRECOMMENDED（無い機種はサイドバー値）を使用。")

    df_all_shared = st.session_state.get("df_all_shared", pd.DataFrame(columns=BASE_HEADER))
    if df_all_shared.empty:
        st.info("過去データが未投入です。バックテストは『共通：データ統合』で過去CSV/zipを入れると表示されます。")
    else:
        if "bt_detail" not in st.session_state:
            st.session_state["bt_detail"] = None
            st.session_state["bt_overall"] = None
            st.session_state["bt_per_machine"] = None
            st.session_state["bt_sig"] = None

        with st.expander("指標の説明（クリックで開く）", expanded=False):
            st.markdown("""
- **precision_topN**：TopNの良台率  
- **baseline_good_rate**：その日その機種の全体良台率（店の地合い）  
- **lift_pt**：TopNが平均よりどれだけ有利か（ポイント差）  
- **hit_rate**：TopNの中に当たりが1台でもある確率  
""")

        df_tmp = df_all_shared.copy()
        df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce").dt.date
        df_tmp = df_tmp[df_tmp["date"].notna()].copy()
        df_tmp = df_tmp[df_tmp["shop"] == shop].copy()

        if df_tmp.empty:
            st.warning("この店名(shop)に一致するデータがありません。shop表記ゆれ（例：武蔵境/メッセ武蔵境）を確認してください。")
        else:
            all_days = sorted(df_tmp["date"].unique().tolist())
            min_day, max_day = all_days[0], all_days[-1]

            cA, cB, cC = st.columns(3)
            with cA:
                eval_start = st.date_input("評価開始日", value=min_day, min_value=min_day, max_value=max_day, key="bt_eval_start")
            with cB:
                eval_end = st.date_input("評価終了日", value=max_day, min_value=min_day, max_value=max_day, key="bt_eval_end")
            with cC:
                topN_text = st.text_input("TopN（カンマ区切り）", value="5,10,20,30", key="bt_topn")

            try:
                top_ns = [int(x.strip()) for x in topN_text.split(",") if x.strip()]
                top_ns = sorted(list(set([n for n in top_ns if n > 0])))
            except Exception:
                top_ns = [5, 10, 20, 30]

            machines_available = sorted(df_tmp["machine"].dropna().unique().tolist())
            target_mode = st.radio("対象機種", ["全機種", "選択"], horizontal=True, key="bt_machine_mode")
            if target_mode == "全機種":
                machines_bt = machines_available
            else:
                machines_bt = st.multiselect(
                    "対象機種を選択",
                    options=machines_available,
                    default=[machine] if machine in machines_available else machines_available[:1],
                    key="bt_machines"
                )

            st.divider()

            c1, c2 = st.columns([3, 1])
            with c1:
                run_bt = st.button("バックテストを実行", type="primary", use_container_width=True, key="bt_run")
            with c2:
                clear_bt = st.button("結果をクリア", use_container_width=True, key="bt_clear")

            if clear_bt:
                st.session_state["bt_detail"] = None
                st.session_state["bt_overall"] = None
                st.session_state["bt_per_machine"] = None
                st.session_state["bt_sig"] = None
                st.rerun()

            def _make_bt_signature():
                try:
                    top_ns_sig = tuple(int(x) for x in top_ns)
                except Exception:
                    top_ns_sig = tuple()
                try:
                    machines_sig = tuple(sorted([str(x) for x in machines_bt]))
                except Exception:
                    machines_sig = tuple()
                return (
                    str(shop),
                    str(eval_start), str(eval_end),
                    top_ns_sig,
                    machines_sig,
                    int(lookback_days), int(tau), int(min_unique_days),
                    float(w_unit), float(w_island), float(w_run), float(w_end),
                    int(morning_min_games), float(morning_max_rb), float(morning_max_gassan),
                )

            if run_bt:
                if not machines_bt:
                    st.warning("対象機種が未選択です。")
                else:
                    detail, overall_df, per_machine_df = backtest_precision_hit(
                        df_all=df_all_shared,
                        shop=shop,
                        machines=machines_bt,
                        lookback_days=int(lookback_days),
                        tau=int(tau),
                        min_unique_days=int(min_unique_days),
                        w_unit=float(w_unit),
                        w_island=float(w_island),
                        w_run=float(w_run),
                        w_end=float(w_end),
                        min_games_fallback=int(morning_min_games),
                        max_rb_fallback=float(morning_max_rb),
                        max_gassan_fallback=float(morning_max_gassan),
                        top_ns=top_ns,
                        eval_start=eval_start,
                        eval_end=eval_end,
                    )
                    if detail is None:
                        st.error("バックテスト結果が生成できませんでした（評価期間が短い / サンプル不足 / データ欠損の可能性）。")
                    else:
                        st.session_state["bt_detail"] = detail
                        st.session_state["bt_overall"] = overall_df
                        st.session_state["bt_per_machine"] = per_machine_df
                        st.session_state["bt_sig"] = _make_bt_signature()

            detail = st.session_state["bt_detail"]
            overall_df = st.session_state["bt_overall"]
            per_machine_df = st.session_state["bt_per_machine"]

            if detail is None:
                st.info("まだバックテストが未実行です。上の『バックテストを実行』を押してください。")
            else:
                cur_sig = _make_bt_signature()
                if st.session_state["bt_sig"] is not None and st.session_state["bt_sig"] != cur_sig:
                    st.warning("⚠️ 計算条件が変更されています。表示中の結果は『前回実行時の条件』のものです。必要なら再度『バックテストを実行』してください。")

                overall_show = overall_df.copy()
                for c in ["precision_topN", "baseline_good_rate", "lift_pt", "hit_rate"]:
                    overall_show[c] = pd.to_numeric(overall_show[c], errors="coerce")

                overall_show["precision_topN(%)"] = (overall_show["precision_topN"] * 100).round(1)
                overall_show["baseline_good_rate(%)"] = (overall_show["baseline_good_rate"] * 100).round(1)
                overall_show["lift_pt(%pt)"] = (overall_show["lift_pt"] * 100).round(1)
                overall_show["hit_rate(%)"] = (overall_show["hit_rate"] * 100).round(1)

                st.subheader("結果サマリ（全体）")
                st.dataframe(
                    overall_show[["topN", "eval_cases", "precision_topN(%)", "baseline_good_rate(%)", "lift_pt(%pt)", "hit_rate(%)"]],
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("結果サマリ（機種別）")
                pm = per_machine_df.copy()
                for c in ["precision_topN", "baseline_good_rate", "lift_pt", "hit_rate"]:
                    pm[c] = pd.to_numeric(pm[c], errors="coerce")
                pm["precision_topN(%)"] = (pm["precision_topN"] * 100).round(1)
                pm["baseline_good_rate(%)"] = (pm["baseline_good_rate"] * 100).round(1)
                pm["lift_pt(%pt)"] = (pm["lift_pt"] * 100).round(1)
                pm["hit_rate(%)"] = (pm["hit_rate"] * 100).round(1)

                st.dataframe(
                    pm[["machine", "topN", "eval_cases", "precision_topN(%)", "baseline_good_rate(%)", "lift_pt(%pt)", "hit_rate(%)"]],
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("詳細（day × machine × topN）")
                det = detail.copy()
                det["date"] = pd.to_datetime(det["date"], errors="coerce").dt.date
                det["precision_topN(%)"] = (pd.to_numeric(det["precision_topN"], errors="coerce") * 100).round(1)
                det["baseline_good_rate(%)"] = (pd.to_numeric(det["baseline_good_rate"], errors="coerce") * 100).round(1)
                det["lift_pt(%pt)"] = (pd.to_numeric(det["lift_pt"], errors="coerce") * 100).round(1)

                st.dataframe(
                    det[["date", "machine", "topN", "selected_n", "good_in_topN", "precision_topN(%)", "baseline_good_rate(%)", "lift_pt(%pt)", "hit_at_N"]],
                    use_container_width=True,
                    hide_index=True
                )
