import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from zoneinfo import ZoneInfo

# ========= Config =========
HEADER = [
    "date","shop","machine",
    "unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals",
    "bb_rate","rb_rate","art_rate","gassan_rate","prev_day_end",
    # ---- date context (NEW) ----
    "dow_num","dow","is_weekend","special_flag","special_name"
]

SHOP_PRESETS = ["武蔵境", "吉祥寺", "三鷹", "国分寺", "新宿", "渋谷"]
MACHINE_PRESETS = [
    "マイジャグラーV", "ゴーゴージャグラー3", "ハッピージャグラーVIII",
    "ファンキージャグラー2KT", "ミスタージャグラー", "ジャグラーガールズSS",
    "ネオアイムジャグラーEX", "ウルトラミラクルジャグラー"
]

st.set_page_config(page_title="ジャグラー朝イチセレクター", layout="wide")
st.title("ジャグラー 朝イチ台セレクター（過去データ集計 → 候補ランキング）")
st.caption("過去の original.csv（複数日）をアップロード → 店×機種ごとに集計 → 朝イチ候補を順位付け（島マスタ対応）")

JST = ZoneInfo("Asia/Tokyo")

# ========= 朝イチ用おすすめ設定（勝率重視の目安） =========
RECOMMENDED = {
    "マイジャグラーV":            {"min_games": 2500, "max_rb": 280.0, "max_gassan": 175.0},
    "ゴーゴージャグラー3":        {"min_games": 2500, "max_rb": 290.0, "max_gassan": 180.0},
    "ハッピージャグラーVIII":     {"min_games": 2800, "max_rb": 270.0, "max_gassan": 170.0},
    "ファンキージャグラー2KT":    {"min_games": 2500, "max_rb": 300.0, "max_gassan": 185.0},
    "ミスタージャグラー":         {"min_games": 2500, "max_rb": 300.0, "max_gassan": 185.0},
    "ジャグラーガールズSS":       {"min_games": 2300, "max_rb": 280.0, "max_gassan": 175.0},
    "ネオアイムジャグラーEX":      {"min_games": 2500, "max_rb": 320.0, "max_gassan": 190.0},
    "ウルトラミラクルジャグラー": {"min_games": 2800, "max_rb": 300.0, "max_gassan": 185.0},
}

# ========= Date Context (NEW) =========
DOW_LABEL = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
SPECIAL_COLS = ["date", "special_flag", "special_name"]

def load_special_days(uploaded_file) -> pd.DataFrame:
    """
    special_days.csv を読む
    必須列: date
    任意列: special_flag (0/1), special_name (文字)
    """
    df = pd.read_csv(uploaded_file)
    if "date" not in df.columns:
        raise ValueError("特定日マスタには date 列が必要です（YYYY-MM-DD）")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).copy()

    if "special_flag" not in df.columns:
        df["special_flag"] = 1
    df["special_flag"] = pd.to_numeric(df["special_flag"], errors="coerce").fillna(1).astype(int).clip(0, 1)

    if "special_name" not in df.columns:
        df["special_name"] = ""
    df["special_name"] = df["special_name"].astype(str).fillna("").str.strip()

    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df[SPECIAL_COLS].copy()

def add_date_context(df: pd.DataFrame, special_days: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    dfに曜日などを付与する。既存列があっても上書きして最新化する。
    special_days: ["date","special_flag","special_name"]
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date", pd.NaT), errors="coerce")
    out["date"] = out["date"].dt.date

    out["dow_num"] = pd.to_datetime(out["date"], errors="coerce").dt.weekday
    out["dow"] = out["dow_num"].map(DOW_LABEL)
    out["is_weekend"] = out["dow_num"].isin([5, 6]).astype(int)

    out["special_flag"] = 0
    out["special_name"] = ""

    if special_days is not None and not special_days.empty:
        sd = special_days.copy()
        sd["date"] = pd.to_datetime(sd["date"], errors="coerce").dt.date
        sd = sd.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        sd["special_flag"] = pd.to_numeric(sd["special_flag"], errors="coerce").fillna(1).astype(int).clip(0, 1)
        sd["special_name"] = sd["special_name"].astype(str).fillna("").str.strip()

        out = out.merge(sd, on="date", how="left", suffixes=("", "_m"))
        out["special_flag"] = out["special_flag_m"].fillna(out["special_flag"]).astype(int)
        out["special_name"] = out["special_name_m"].fillna(out["special_name"]).astype(str)
        out = out.drop(columns=["special_flag_m", "special_name_m"], errors="ignore")

    return out

def special_days_template_bytes() -> bytes:
    tmp = pd.DataFrame([
        {"date":"2025-12-07", "special_flag":1, "special_name":"例：イベント/取材/周年"},
        {"date":"2025-12-08", "special_flag":1, "special_name":"例：ゾロ目"},
    ])
    return tmp.to_csv(index=False).encode("utf-8-sig")

# ========= Helpers =========
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
    df.columns = df.columns.astype(str).str.strip()
    rename_map = {
        "台番": "unit_number", "台番号": "unit_number",
        "総回転": "total_start", "累計スタート": "total_start",
        "BB回数": "bb_count", "RB回数": "rb_count",
        "ART回数": "art_count", "最大持ち玉": "max_medals",
        "BB確率": "bb_rate", "RB確率": "rb_rate", "ART確率": "art_rate",
        "合成確率": "gassan_rate", "前日最終": "prev_day_end",
        "店舗": "shop", "機種": "machine", "日付": "date",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

def compute_rates_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals","prev_day_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    for c in ["bb_rate","rb_rate","art_rate","gassan_rate"]:
        if c not in df.columns:
            df[c] = np.nan

    df["bb_rate"] = df["bb_rate"].map(parse_rate_token)
    df["rb_rate"] = df["rb_rate"].map(parse_rate_token)
    df["art_rate"] = df["art_rate"].map(parse_rate_token)
    df["gassan_rate"] = df["gassan_rate"].map(parse_rate_token)

    if "total_start" in df.columns:
        bb_mask = df["bb_rate"].isna() & df["bb_count"].gt(0)
        rb_mask = df["rb_rate"].isna() & df["rb_count"].gt(0)
        gs_mask = df["gassan_rate"].isna() & (df["bb_count"].add(df["rb_count"]).gt(0))

        df.loc[bb_mask, "bb_rate"] = df.loc[bb_mask, "total_start"] / df.loc[bb_mask, "bb_count"]
        df.loc[rb_mask, "rb_rate"] = df.loc[rb_mask, "total_start"] / df.loc[rb_mask, "rb_count"]
        df.loc[gs_mask, "gassan_rate"] = df.loc[gs_mask, "total_start"] / (df.loc[gs_mask, "bb_count"] + df.loc[gs_mask, "rb_count"])

    return df

# ========= 統合用：ファイル名から date / machine を推定 =========
def parse_date_machine_from_filename(filename: str):
    """
    例：
      2025-12-16_マイジャグラーV_original.csv
      2025-12-16_08-32-45_マイジャグラーV_original.csv
    """
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
    """date/shop/machine が欠損なら補完（行単位で欠損だけ埋める）"""
    out = df.copy()

    # date
    if "date" not in out.columns:
        out["date"] = pd.NaT
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if date_hint is not None:
        out.loc[out["date"].isna(), "date"] = pd.to_datetime(date_hint)

    # shop
    if "shop" not in out.columns:
        out["shop"] = pd.NA
    if shop_hint:
        out.loc[
            out["shop"].isna()
            | (out["shop"].astype(str).str.strip() == "")
            | (out["shop"].astype(str) == "nan"),
            "shop"
        ] = shop_hint

    # machine
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

def load_many_csvs(files, default_shop: str, default_date: date | None = None, default_machine: str | None = None) -> pd.DataFrame:
    dfs = []
    for f in files:
        date_hint, machine_hint = parse_date_machine_from_filename(getattr(f, "name", ""))

        df = pd.read_csv(f)
        df = normalize_columns(df)
        df = compute_rates_if_needed(df)

        for c in HEADER:
            if c not in df.columns:
                df[c] = np.nan

        df = fill_missing_meta(
            df,
            date_hint=date_hint or default_date,
            shop_hint=default_shop,
            machine_hint=machine_hint or default_machine,
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df[HEADER].copy()
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=HEADER)
    return pd.concat(dfs, ignore_index=True)

def load_zip_of_csv(zip_bytes: bytes, default_shop: str, default_date: date | None = None, default_machine: str | None = None) -> pd.DataFrame:
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue

            date_hint, machine_hint = parse_date_machine_from_filename(name)

            with z.open(name) as fp:
                df = pd.read_csv(fp)

            df = normalize_columns(df)
            df = compute_rates_if_needed(df)

            for c in HEADER:
                if c not in df.columns:
                    df[c] = np.nan

            df = fill_missing_meta(
                df,
                date_hint=date_hint or default_date,
                shop_hint=default_shop,
                machine_hint=machine_hint or default_machine,
            )

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df = df[HEADER].copy()
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=HEADER)
    return pd.concat(dfs, ignore_index=True)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def make_filename(machine: str, suffix: str, date_str: str) -> str:
    time_part = datetime.now(JST).strftime("%H-%M-%S")
    safe_machine = str(machine).replace(" ", "").replace("/", "_").replace("\\", "_").replace(":", "-")
    return f"{date_str}_{time_part}_{safe_machine}_{suffix}.csv"

# ========= Island Master (island.csv) =========
ISLAND_COLS = ["unit_number","island_id","side","pos","edge_type","is_end"]

def load_island_master(uploaded_file) -> pd.DataFrame:
    dfm = pd.read_csv(uploaded_file)
    missing = [c for c in ISLAND_COLS if c not in dfm.columns]
    if missing:
        raise ValueError(f"島マスタの列が不足しています: {missing}")

    dfm = dfm.copy()
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

# ========= Play Log (append to uploaded CSV) =========
PLAYLOG_HEADER = [
    "created_at",
    "date","shop","machine","unit_number",
    "tool_rank",
    "select_reason",
    "start_time","end_time",
    "invest_medals","payout_medals","profit_medals",
    "play_games",
    "stop_reason","memo"
]

def append_row_to_uploaded_csv(uploaded_bytes: bytes, new_row: dict) -> bytes:
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
    for c in PLAYLOG_HEADER:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PLAYLOG_HEADER]
    df2 = pd.DataFrame([new_row], columns=PLAYLOG_HEADER)
    out = pd.concat([df, df2], ignore_index=True)
    return out.to_csv(index=False).encode("utf-8-sig")

def make_log_filename(date_str: str) -> str:
    time_part = datetime.now(JST).strftime("%H-%M-%S")
    return f"{date_str}_{time_part}_playlog.csv"

# ========= Backtest helpers =========
def make_threshold_df(min_games_fallback: int, max_rb_fallback: float, max_gassan_fallback: float) -> pd.DataFrame:
    rows = []
    for m in MACHINE_PRESETS:
        rec = RECOMMENDED.get(m)
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
    if df_all.empty:
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
    if "is_end" in df.columns:
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

    agg_u["unit_score"] = (
        (agg_u["good_rate_weighted"].fillna(0.0) * 1.0) * 0.70 +
        (trust * 1.0) * 0.30
    )

    out = agg_u.copy()

    if out["island_id"].isna().all():
        out["island_score"] = 0.0
    else:
        agg_i = df.groupby(["shop","machine","island_id"], dropna=False).agg(
            i_w_sum=("w", "sum"),
            i_good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
        ).reset_index()
        agg_i["island_score"] = (agg_i["i_good_w"] / agg_i["i_w_sum"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out = out.merge(
            agg_i[["shop","machine","island_id","island_score"]],
            on=["shop","machine","island_id"],
            how="left"
        )
        out["island_score"] = out["island_score"].fillna(0.0)

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
    if df_all.empty:
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
            rec = RECOMMENDED.get(m)
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

# ========= Sidebar =========
if "min_games" not in st.session_state:
    st.session_state["min_games"] = 2500
if "max_rb" not in st.session_state:
    st.session_state["max_rb"] = 290.0
if "max_gassan" not in st.session_state:
    st.session_state["max_gassan"] = 195.0

def apply_recommended(machine_name: str):
    rec = RECOMMENDED.get(machine_name)
    if not rec:
        return
    st.session_state["min_games"] = int(rec["min_games"])
    st.session_state["max_rb"] = float(rec["max_rb"])
    st.session_state["max_gassan"] = float(rec["max_gassan"])

with st.sidebar:
    st.header("補完情報（date / shop / machine）")
    d = st.date_input("基準日（今日）", value=date.today())
    date_str = d.strftime("%Y-%m-%d")

    shop_mode = st.radio("店名", ["選択", "手入力"], horizontal=True)
    if shop_mode == "選択":
        shop = st.selectbox("shop", SHOP_PRESETS, index=0)
    else:
        shop = st.text_input("shop", value="武蔵境")

    machine_mode = st.radio("機種", ["選択", "手入力"], horizontal=True)
    if machine_mode == "選択":
        machine = st.selectbox(
            "machine",
            MACHINE_PRESETS,
            index=0,
            key="machine_select",
            on_change=lambda: apply_recommended(st.session_state["machine_select"]),
        )
        machine = st.session_state["machine_select"]
    else:
        machine = st.text_input("machine", value="マイジャグラーV")

    rec = RECOMMENDED.get(machine)
    with st.expander("おすすめ設定値（朝イチ用・補足）", expanded=False):
        if rec:
            st.caption("※ 朝イチ候補の“良台日判定”に使う目安です。候補が少なければ緩めてください。")
            st.write(f"- 最低総回転: **{rec['min_games']}**")
            st.write(f"- REG上限: **{rec['max_rb']}**")
            st.write(f"- 合算上限: **{rec['max_gassan']}**")
            if st.button("おすすめ値をスライダーに反映", use_container_width=True):
                apply_recommended(machine)
                st.rerun()
        else:
            st.caption("この機種はプリセット未登録です。手動でスライダーを調整してください。")

    st.divider()
    st.header("朝イチ候補スコア（島マスタ前提）")

    w_island = st.slider("島（island）の重み", 0.0, 1.0, 0.45, 0.05)
    w_run    = st.slider("並び（run）の重み",   0.0, 1.0, 0.35, 0.05)
    w_end    = st.slider("端（end）の重み",     0.0, 1.0, 0.10, 0.05)

    w_unit = max(0.0, 1.0 - (w_island + w_run + w_end))
    st.caption(f"台単体の重み（自動の目安）: {w_unit:.2f}  ※ 合計が1を超えても内部で正規化します")

    st.divider()
    st.header("良台日判定（スライダー）")

    min_games = st.slider(
        "良台日判定：最低 総回転（total_start）",
        0, 10000,
        value=int(st.session_state["min_games"]),
        step=100,
        key="min_games",
    )
    max_rb = st.slider(
        "良台日判定：REG上限（rb_rate）",
        150.0, 600.0,
        value=float(st.session_state["max_rb"]),
        step=1.0,
        key="max_rb",
    )
    max_gassan = st.slider(
        "良台日判定：合算上限（gassan_rate）",
        80.0, 350.0,
        value=float(st.session_state["max_gassan"]),
        step=1.0,
        key="max_gassan",
    )

    lookback_days = st.number_input("集計対象：過去何日（学習窓）", 1, 365, 60, 1)
    tau = st.number_input("日付減衰 τ（小さいほど直近重視）", 1, 120, 14, 1)
    min_unique_days = st.number_input("最小サンプル日数（稼働日数）", 1, 60, 3, 1)
    top_n = st.number_input("上位N件表示（候補テーブル）", 1, 200, 30, 1)

# --------- 共通：過去データの投入（＋統合DL機能） ---------
def upload_past_data_ui():
    st.caption("複数CSV（original.csv）または zip（CSVをまとめたもの）をアップロードしてください。")
    st.caption("※ date/shop/machine がCSV内に無い場合：ファイル名（YYYY-MM-DD_機種...）→無ければサイドバー値で補完します。")

    st.subheader("補完情報（日付コンテキスト：曜日/週末/特定日）")
    special_file = st.file_uploader("特定日マスタ（任意：special_days.csv）", type=["csv"], key="special_days_csv_morning")
    st.download_button(
        "特定日マスタのテンプレCSVをダウンロード",
        data=special_days_template_bytes(),
        file_name="special_days_template.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_special_template_morning"
    )
    special_df = None
    if special_file is not None:
        try:
            special_df = load_special_days(special_file)
            st.success(f"特定日マスタ読込OK：{len(special_df)}日")
            st.dataframe(special_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"特定日マスタの読み込みに失敗しました: {e}")
            st.stop()

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
        return pd.DataFrame(columns=HEADER)

    default_shop = shop
    default_date = d
    default_machine = machine

    df_all = pd.DataFrame(columns=HEADER)
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
        return pd.DataFrame(columns=HEADER)

    # ★日付コンテキスト付与（NEW）
    df_all = add_date_context(df_all, special_df)

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
        data=to_csv_bytes(df_all[HEADER].copy()),
        file_name=f"{date_str}_unified.csv",
        mime="text/csv",
        key="dl_unified_csv"
    )

    with st.expander("統合プレビュー（先頭/末尾）", expanded=False):
        st.dataframe(df_all.head(20), use_container_width=True, hide_index=True)
        st.dataframe(df_all.tail(20), use_container_width=True, hide_index=True)

    return df_all[HEADER].copy()

# ========= Main UI =========
st.divider()
st.subheader("共通：過去データアップロード（朝イチ候補で使用）")
df_all_shared = upload_past_data_ui()

st.subheader("共通：島マスタアップロード（island.csv）")
island_file = st.file_uploader("島マスタ（island.csv）をアップロード", type=["csv"], key="island_csv")

island_master = None
if island_file is None:
    st.info("島マスタが未指定です。島/並び/端の評価は 0 として扱い、台単体のみでランキングします。")
else:
    try:
        island_master = load_island_master(island_file)
        errs = validate_island_master(island_master)
        if errs:
            st.error("島マスタに問題があります:\n- " + "\n- ".join(errs))
            st.stop()
        st.success(f"島マスタ読込OK：{len(island_master)}台")
    except Exception as e:
        st.error(f"島マスタの読み込みに失敗しました: {e}")
        st.stop()

if (island_master is not None) and (not df_all_shared.empty):
    df_all_shared = attach_island_info(df_all_shared, island_master)
    miss = df_all_shared[df_all_shared["island_id"].isna()]["unit_number"].dropna().unique().tolist()
    if miss:
        st.warning(f"島マスタに存在しない台番号が過去データに含まれています（例）: {miss[:10]}")

tab1, tab2, tab3 = st.tabs([
    "朝イチ候補（過去データ集計）",
    "実戦ログ（CSVに追記して更新版DL）",
    "バックテスト（ツール精度検証）"
])

with tab1:
    st.subheader("① 朝イチ候補（過去の original.csv を集計してランキング）")

    if df_all_shared.empty:
        st.info("上の『共通：過去データアップロード』にCSV/zipを入れてください。")
    else:
        base_day = pd.to_datetime(date_str).date()

        ranking = build_ranking(
            df_all=df_all_shared,
            shop=shop,
            machine=machine,
            base_day=base_day,
            lookback_days=int(lookback_days),
            tau=int(tau),
            min_games=int(min_games),
            max_rb=float(max_rb),
            max_gassan=float(max_gassan),
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

            st.dataframe(ranking.head(int(top_n))[show_cols], use_container_width=True, hide_index=True)

            filename = make_filename(machine, "morning_candidates", date_str)
            st.download_button(
                "候補台ランキングをCSVでダウンロード",
                data=to_csv_bytes(ranking),
                file_name=filename,
                mime="text/csv",
                key="tab1_dl_candidates"
            )

with tab2:
    st.subheader("② 実戦ログ（ローカルCSVに追記 → 更新版をダウンロード）")
    st.caption("※ Streamlit Cloudではローカルファイルを直接書き換えできないため、追記した“更新版CSV”を生成してダウンロードします。")

    uploaded_log = st.file_uploader(
        "追記したいログCSVを選択（既存のplay_log.csvなど）",
        type=["csv"],
        key="tab2_log_upload"
    )

    st.divider()

    with st.form("playlog_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            log_date = st.text_input("date（YYYY-MM-DD）", value=date_str)
            log_shop = st.text_input("shop", value=shop)
        with col2:
            log_machine = st.text_input("machine", value=machine)
            unit_number = st.number_input("unit_number（台番号）", min_value=0, step=1, value=0)
            tool_rank = st.number_input("tool_rank（ツール候補順位）", min_value=0, step=1, value=0)

            select_reason = st.selectbox(
                "select_reason（着席理由）",
                ["", "ツール上位", "末尾が強い", "角/角2", "並び/帯が強い", "前日挙動", "直感", "空き台都合", "その他"]
            )
        with col3:
            play_games = st.number_input("play_games（自分が回したG数）", min_value=0, step=10, value=0)

        col4, col5, col6 = st.columns(3)
        with col4:
            start_time = st.text_input("start_time（例 09:05）", value="")
        with col5:
            end_time = st.text_input("end_time（例 11:20）", value="")
        with col6:
            stop_reason = st.selectbox(
                "stop_reason（ヤメ理由）",
                ["", "様子見終了", "REG悪化", "合算悪化", "資金切れ", "他に移動", "閉店", "その他"]
            )

        col7, col8, col9 = st.columns(3)
        with col7:
            invest = st.number_input("invest_medals（投資枚）", min_value=0, step=50, value=0)
        with col8:
            payout = st.number_input("payout_medals（回収枚）", min_value=0, step=50, value=0)
        with col9:
            profit = int(payout - invest)
            st.metric("profit_medals（収支枚）", profit)

        memo = st.text_area("memo（任意）", value="", height=100)
        submit = st.form_submit_button("この内容で追記用データを作成", type="primary")

    if submit:
        if uploaded_log is None:
            st.error("先に「追記したいログCSV」を選択してください。")
            st.stop()

        new_row = {
            "created_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
            "date": log_date,
            "shop": log_shop,
            "machine": log_machine,
            "unit_number": int(unit_number),
            "tool_rank": int(tool_rank),
            "select_reason": select_reason,
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
            key="tab2_log_download"
        )

        st.divider()
        st.markdown("#### 追記後プレビュー（末尾5行）")
        preview_df = pd.read_csv(io.BytesIO(out_bytes))
        st.dataframe(preview_df.tail(5), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("③ バックテスト（ツール精度検証：上位Nの良台率 / lift / Hit@N）")
    st.caption("※ その日を予測する際、学習には前日までのデータのみ使用（リーク防止）。良台判定は機種別RECOMMENDED（無い機種はサイドバー値）を使用。")

    if "bt_detail" not in st.session_state:
        st.session_state["bt_detail"] = None
        st.session_state["bt_overall"] = None
        st.session_state["bt_per_machine"] = None
        st.session_state["bt_sig"] = None

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
            int(min_games), float(max_rb), float(max_gassan),
        )

    with st.expander("指標の説明（クリックで開く）", expanded=False):
        st.markdown("""
### このバックテストがやっていること
- ある日 **D** を評価したい  
- 学習は **Dより前の日だけ** を使う（当日データは使わない＝リーク防止）  
- 学習データでランキングを作り、当日Dの結果（良台だったか）で採点します

### 良台日（当たり）の定義
**min_games / max_rb / max_gassan** を満たす台を **良台（=1）** として扱います。

### 各指標の意味
- **topN**：ランキング上位から「何台を見るか」（Top10なら上位10台）
- **selected_n**：その評価ケースで実際にTopNとして扱えた台数
- **good_in_topN**：TopNの中に「良台（当たり）」が何台あったか
- **precision_topN**：TopNの良台率  `good_in_topN / selected_n`
- **baseline_good_rate**：その日その機種の全体良台率（店の地合い） `good_all / all_units_n`
- **lift_pt**：TopNが平均よりどれだけ有利か（ポイント差） `precision_topN - baseline_good_rate`
- **hit_at_N**：TopNの中に当たりが1台でもあれば1、なければ0
- **hit_rate**：hit_at_Nの平均
""")

    if df_all_shared.empty:
        st.info("まずは『共通：過去データアップロード』に統合データを投入してください。")
        st.stop()

    df_tmp = df_all_shared.copy()
    df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce").dt.date
    df_tmp = df_tmp[df_tmp["date"].notna()].copy()
    df_tmp = df_tmp[df_tmp["shop"] == shop].copy()

    if df_tmp.empty:
        st.warning("この店名(shop)に一致するデータがありません。shop表記ゆれ（例：武蔵境/メッセ武蔵境）を確認してください。")
        st.stop()

    all_days = sorted(df_tmp["date"].unique().tolist())
    min_day, max_day = all_days[0], all_days[-1]

    colA, colB, colC = st.columns(3)
    with colA:
        eval_start = st.date_input("評価開始日", value=min_day, min_value=min_day, max_value=max_day, key="bt_eval_start")
    with colB:
        eval_end = st.date_input("評価終了日", value=max_day, min_value=min_day, max_value=max_day, key="bt_eval_end")
    with colC:
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
        if not machines_bt:
            st.warning("対象機種が未選択です。")
            st.stop()

    st.divider()

    c1, c2 = st.columns([3, 1])
    with c1:
        run_bt = st.button("バックテストを実行", type="primary", use_container_width=True)
    with c2:
        clear_bt = st.button("結果をクリア", use_container_width=True)

    if clear_bt:
        st.session_state["bt_detail"] = None
        st.session_state["bt_overall"] = None
        st.session_state["bt_per_machine"] = None
        st.session_state["bt_sig"] = None
        st.rerun()

    if run_bt:
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
            min_games_fallback=int(min_games),
            max_rb_fallback=float(max_rb),
            max_gassan_fallback=float(max_gassan),
            top_ns=top_ns,
            eval_start=eval_start,
            eval_end=eval_end,
        )

        if detail is None:
            st.error("バックテスト結果が生成できませんでした（評価期間が短い / サンプル不足 / データ欠損の可能性）。")
            st.stop()

        st.session_state["bt_detail"] = detail
        st.session_state["bt_overall"] = overall_df
        st.session_state["bt_per_machine"] = per_machine_df
        st.session_state["bt_sig"] = _make_bt_signature()

    detail = st.session_state["bt_detail"]
    overall_df = st.session_state["bt_overall"]
    per_machine_df = st.session_state["bt_per_machine"]

    if detail is None:
        st.info("まだバックテストが未実行です。上の『バックテストを実行』を押してください。")
        st.stop()

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

    best = overall_show.sort_values("lift_pt", ascending=False).head(1)
    if not best.empty:
        bN = int(best["topN"].iloc[0])
        st.success(f"liftが最大のTopN：**Top{bN}**（lift={best['lift_pt(%pt)'].iloc[0]}%pt / Hit={best['hit_rate(%)'].iloc[0]}%）")

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

    st.divider()
    st.subheader("外れ日分析（特定の日に外れやすいかを見る）")

    topN_options = sorted(det["topN"].dropna().unique().astype(int).tolist())
    if not topN_options:
        st.info("TopNの結果がありません。")
        st.stop()

    N_focus = st.selectbox(
        "外れ日ランキングの対象TopN",
        options=topN_options,
        index=0,
        key="bt_focus_topN_view"
    )

    detN = det[det["topN"] == int(N_focus)].copy()

    def _daily_agg(g: pd.DataFrame) -> pd.Series:
        total_sel = int(pd.to_numeric(g["selected_n"], errors="coerce").fillna(0).sum())
        total_good_sel = int(pd.to_numeric(g["good_in_topN"], errors="coerce").fillna(0).sum())
        precision = (total_good_sel / total_sel) if total_sel > 0 else np.nan

        total_all = int(pd.to_numeric(g["all_units_n"], errors="coerce").fillna(0).sum())
        total_good_all = int(pd.to_numeric(g["good_all"], errors="coerce").fillna(0).sum())
        baseline = (total_good_all / total_all) if total_all > 0 else np.nan

        hit_rate = float(pd.to_numeric(g["hit_at_N"], errors="coerce").mean()) if len(g) > 0 else np.nan
        return pd.Series({
            "eval_cases(machine数)": int(len(g)),
            "selected_n_total": total_sel,
            "good_in_topN_total": total_good_sel,
            "precision_topN": precision,
            "baseline_good_rate": baseline,
            "lift_pt": (precision - baseline) if (pd.notna(precision) and pd.notna(baseline)) else np.nan,
            "hit_rate": hit_rate,
        })

    day_sum = detN.groupby("date", dropna=False).apply(_daily_agg).reset_index()

    day_show = day_sum.copy()
    day_show["precision_topN(%)"] = (pd.to_numeric(day_show["precision_topN"], errors="coerce") * 100).round(1)
    day_show["baseline_good_rate(%)"] = (pd.to_numeric(day_show["baseline_good_rate"], errors="coerce") * 100).round(1)
    day_show["lift_pt(%pt)"] = (pd.to_numeric(day_show["lift_pt"], errors="coerce") * 100).round(1)
    day_show["hit_rate(%)"] = (pd.to_numeric(day_show["hit_rate"], errors="coerce") * 100).round(1)

    colX, colY = st.columns([1, 1])
    with colX:
        k = st.number_input("外れ日ランキング表示件数（liftが低い順）", min_value=5, max_value=200, value=20, step=5, key="bt_bad_days_k_view")
    with colY:
        only_really_bad = st.checkbox("liftがマイナスの日だけに絞る", value=True, key="bt_only_bad_view")

    bad_days = day_show.sort_values("lift_pt", ascending=True).copy()
    if only_really_bad:
        bad_days = bad_days[pd.to_numeric(bad_days["lift_pt"], errors="coerce") < 0].copy()

    st.markdown(f"#### 外れ日ランキング（Top{int(N_focus)} / liftが低い順）")
    if bad_days.empty:
        st.info("条件に一致する外れ日がありません（lift<0 が無い、またはデータ不足）。")
    else:
        st.dataframe(
            bad_days.head(int(k))[[
                "date",
                "eval_cases(machine数)",
                "selected_n_total",
                "good_in_topN_total",
                "precision_topN(%)",
                "baseline_good_rate(%)",
                "lift_pt(%pt)",
                "hit_rate(%)",
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "外れ日ランキング（CSV）をダウンロード",
            data=bad_days.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{date_str}_bad_days_top{int(N_focus)}.csv",
            mime="text/csv",
            use_container_width=True,
            key="bt_dl_bad_days"
        )

    st.divider()
    st.subheader("日別ヒートマップ（date × topN の lift）")
    st.caption("lift(%pt)=（ツール上位の良台率）−（全体良台率）。緑ほど良く、赤ほど悪い日です。")

    det2 = det.copy()
    det2["selected_n"] = pd.to_numeric(det2["selected_n"], errors="coerce").fillna(0).astype(int)
    det2["good_in_topN"] = pd.to_numeric(det2["good_in_topN"], errors="coerce").fillna(0).astype(int)
    det2["all_units_n"] = pd.to_numeric(det2["all_units_n"], errors="coerce").fillna(0).astype(int)
    det2["good_all"] = pd.to_numeric(det2["good_all"], errors="coerce").fillna(0).astype(int)

    def _day_topN_agg(g: pd.DataFrame) -> pd.Series:
        sel = int(g["selected_n"].sum())
        good_sel = int(g["good_in_topN"].sum())
        precision = (good_sel / sel) if sel > 0 else np.nan

        alln = int(g["all_units_n"].sum())
        good_all = int(g["good_all"].sum())
        baseline = (good_all / alln) if alln > 0 else np.nan

        lift = (precision - baseline) if (pd.notna(precision) and pd.notna(baseline)) else np.nan
        return pd.Series({"lift_pt": lift})

    day_topN = det2.groupby(["date", "topN"], dropna=False).apply(_day_topN_agg).reset_index()
    day_topN["lift_pt(%pt)"] = (pd.to_numeric(day_topN["lift_pt"], errors="coerce") * 100).round(1)

    pivot = day_topN.pivot_table(index="date", columns="topN", values="lift_pt(%pt)", aggfunc="mean")

    show_last = st.number_input("直近何日を表示する？（0なら全日）", min_value=0, max_value=9999, value=60, step=10, key="bt_heatmap_last_days_view")
    pivot2 = pivot.copy()
    if int(show_last) > 0 and len(pivot2) > int(show_last):
        pivot2 = pivot2.tail(int(show_last))

    st.markdown("#### ヒートマップ（テーブル色付け）")
    try:
        st.dataframe(
            pivot2.style.format("{:.1f}").background_gradient(axis=None, cmap="RdYlGn"),
            use_container_width=True
        )
    except Exception:
        st.dataframe(pivot2, use_container_width=True)

    st.download_button(
        "日別ヒートマップ（pivot CSV）をダウンロード",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"{date_str}_lift_heatmap_pivot.csv",
        mime="text/csv",
        use_container_width=True,
        key="bt_dl_heatmap"
    )

    st.divider()
    st.subheader("バックテスト結果のダウンロード")
    dl_zip = st.checkbox("詳細・サマリをzipでまとめてDL", value=True, key="bt_dl_zip")
    if dl_zip:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{date_str}_bt_overall.csv", overall_df.to_csv(index=False).encode("utf-8-sig"))
            z.writestr(f"{date_str}_bt_per_machine.csv", per_machine_df.to_csv(index=False).encode("utf-8-sig"))
            z.writestr(f"{date_str}_bt_detail.csv", detail.to_csv(index=False).encode("utf-8-sig"))
            z.writestr(f"{date_str}_bt_bad_days_top{int(N_focus)}.csv", bad_days.to_csv(index=False).encode("utf-8-sig") if not bad_days.empty else b"")
            z.writestr(f"{date_str}_bt_lift_heatmap_pivot.csv", pivot.to_csv().encode("utf-8-sig"))
        st.download_button(
            "バックテスト結果（zip）をダウンロード",
            data=mem.getvalue(),
            file_name=f"{date_str}_backtest_results.zip",
            mime="application/zip",
            use_container_width=True,
            key="bt_dl_zip_btn"
        )
    else:
        st.download_button(
            "overallをCSVでDL",
            data=overall_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{date_str}_bt_overall.csv",
            mime="text/csv",
            use_container_width=True,
            key="bt_dl_overall"
        )
        st.download_button(
            "per_machineをCSVでDL",
            data=per_machine_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{date_str}_bt_per_machine.csv",
            mime="text/csv",
            use_container_width=True,
            key="bt_dl_pm"
        )
        st.download_button(
            "detailをCSVでDL",
            data=detail.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{date_str}_bt_detail.csv",
            mime="text/csv",
            use_container_width=True,
            key="bt_dl_detail"
        )
