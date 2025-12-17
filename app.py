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
    "bb_rate","rb_rate","art_rate","gassan_rate","prev_day_end"
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
        df = df[HEADER].copy()

        df = fill_missing_meta(
            df,
            date_hint=date_hint or default_date,
            shop_hint=default_shop,
            machine_hint=machine_hint or default_machine,
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
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
            df = df[HEADER].copy()

            df = fill_missing_meta(
                df,
                date_hint=date_hint or default_date,
                shop_hint=default_shop,
                machine_hint=machine_hint or default_machine,
            )

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
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

    # pos連番 + is_end
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

# ========= Core ranking =========
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
    # weights
    w_unit: float,
    w_island: float,
    w_run: float,
    w_end: float,
):
    """
    base_day を基準に、(base_day-lookback_days)〜(base_day-1) のデータだけでランキングを作る
    ※ base_day 当日のデータは学習に使わない（リーク防止）
    ※ 台番号のクセ（tail/block）は一切使わない（方式A）
    """
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

    # 島マスタ前提（必須）
    required_island_cols = ["island_id","side","pos","edge_type","is_end"]
    if not all(c in df.columns for c in required_island_cols):
        raise ValueError("島マスタ列が不足しています。過去データに島マスタをJOINしてください。")
    if df["island_id"].isna().all():
        raise ValueError("島マスタがJOINされていません（island_idが全て欠損）。")

    # 数値化
    df["unit_number"] = pd.to_numeric(df["unit_number"], errors="coerce")
    df = df[df["unit_number"].notna()].copy()
    df["unit_number"] = df["unit_number"].astype(int)

    df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
    df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
    df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

    # 減衰重み（直近重視）
    df["days_ago"] = (pd.to_datetime(base_day) - pd.to_datetime(df["date"])).dt.days
    df["w"] = np.exp(-df["days_ago"] / max(int(tau), 1))

    # 良台日判定（正解の定義）
    df["is_good_day"] = (
        (df["total_start_num"] >= min_games) &
        (df["rb_rate_num"] <= max_rb) &
        (df["gassan_rate_num"] <= max_gassan)
    ).astype(int)

    # -------------------------
    # ① 台単体（unit）集計
    # -------------------------
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

    # ブレ対策（サンプル不足排除）
    agg_u = agg_u[agg_u["unique_days"] >= int(min_unique_days)].copy()
    if agg_u.empty:
        return pd.DataFrame()

    agg_u["good_rate_weighted"] = (agg_u["good_w"] / agg_u["w_sum"]).replace([np.inf, -np.inf], np.nan)
    agg_u["good_rate_simple"] = (agg_u["good_days"] / agg_u["unique_days"]).replace([np.inf, -np.inf], np.nan)

    # 信頼度（重み合計が大きいほど信頼）
    wmax = float(agg_u["w_sum"].max() if agg_u["w_sum"].notna().any() else 0.0)
    trust = np.log1p(agg_u["w_sum"].fillna(0.0)) / np.log1p(wmax + 1e-9) if wmax > 0 else 0.0

    # 台単体スコア（0〜1）
    agg_u["unit_score"] = (
        (agg_u["good_rate_weighted"].fillna(0.0) * 1.0) * 0.70 +
        (trust * 1.0) * 0.30
    )

    # -------------------------
    # ② 島スコア（island_score）
    # -------------------------
    agg_i = df.groupby(["shop","machine","island_id"], dropna=False).agg(
        i_w_sum=("w", "sum"),
        i_good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
    ).reset_index()
    agg_i["island_score"] = (agg_i["i_good_w"] / agg_i["i_w_sum"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    out = agg_u.merge(
        agg_i[["shop","machine","island_id","island_score"]],
        on=["shop","machine","island_id"],
        how="left"
    )
    out["island_score"] = out["island_score"].fillna(0.0)

    # -------------------------
    # ③ 並びスコア（run_score）
    #     同じ島×同じ列(side)で pos±1 の unit_score 平均
    # -------------------------
    tmp = out.sort_values(["island_id","side","pos"]).copy()
    tmp["pos"] = pd.to_numeric(tmp["pos"], errors="coerce")
    tmp["unit_score_prev"] = tmp.groupby(["island_id","side"])["unit_score"].shift(1)
    tmp["unit_score_next"] = tmp.groupby(["island_id","side"])["unit_score"].shift(-1)
    tmp["run_score"] = tmp[["unit_score_prev","unit_score_next"]].mean(axis=1, skipna=True).fillna(0.0)
    out = out.merge(tmp[["unit_number","run_score"]], on="unit_number", how="left")
    out["run_score"] = out["run_score"].fillna(0.0)

    # -------------------------
    # ④ 端ボーナス（end_bonus）
    # -------------------------
    out["end_bonus"] = (pd.to_numeric(out["is_end"], errors="coerce").fillna(0).astype(int) > 0).astype(float)

    # -------------------------
    # ⑤ 最終スコア（正規化して事故防止）
    # -------------------------
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

    # 表示整形
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
    top_n = st.number_input("上位N件表示", 1, 200, 30, 1)

# --------- 共通：過去データの投入（＋統合DL機能） ---------
def upload_past_data_ui():
    st.caption("複数CSV（original.csv）または zip（CSVをまとめたもの）をアップロードしてください。")
    st.caption("※ date/shop/machine がCSV内に無い場合：ファイル名（YYYY-MM-DD_機種...）→無ければサイドバー値で補完します。")

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

    # サイドバー値（補完用）
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

    # 統合時の重複除去
    do_dedup = st.checkbox(
        "統合時に重複行を除去（date+shop+machine+unit_number が同一なら最後の行を採用）",
        value=True,
        key="dedup_unified"
    )
    if do_dedup:
        df_all = df_all.drop_duplicates(subset=["date","shop","machine","unit_number"], keep="last").copy()

    # ---- ファイル統合機能 ----
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

# ========= Main UI =========
st.divider()
st.subheader("共通：過去データアップロード（朝イチ候補で使用）")
df_all_shared = upload_past_data_ui()

st.subheader("共通：島マスタアップロード（island.csv）")
island_file = st.file_uploader("島マスタ（island.csv）をアップロード", type=["csv"], key="island_csv")

island_master = None
if island_file is None:
    st.info("島マスタが未指定です。島/並び/端の評価は使われません（末尾・台番帯中心のランキング）。")
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

tab1, tab2 = st.tabs([
    "朝イチ候補（過去データ集計）",
    "実戦ログ（CSVに追記して更新版DL）"
])

with tab1:
    with st.expander("候補テーブルの見方（用語説明）", expanded=False):
        st.markdown("""
- **rank**：最終順位（final_scoreの高い順）
- **unit_number**：台番号

- **island_id / side / pos**：島マスタによる実配置（島 / 列 / 列内位置）
- **edge_type**：wall / aisle / center
- **is_end**：列の端（1=端、0=端以外）

- **final_score**：最終スコア（朝イチ優先度の結論）
- 島マスタあり：島/並び/端 + 台単体 + 末尾 + 台番帯 を合成（重みはサイドバー）
- 島マスタなし：台単体 + 末尾 + 台番帯 を合成

- **unit_score**：その台“単体”の強さ（良台率＋信頼度）
- **island_score**：その島が強いか（島全体の良台率）
- **run_score**：同じ列(side)で両隣(pos±1)が強いか（並び）
- **end_bonus**：端ボーナス（端なら1.0）

- **tail / tail_score**：末尾（unit_number%10）とその強さ

- **good_rate_weighted(%)**：直近重視の良台率（新しい日ほど重く評価）
- **good_rate_simple(%)**：単純な良台率（期間を均等に扱う）
- **unique_days**：データに登場した日数（少ないとブレる）
- **w_sum**：減衰込みの学習量（大きいほど信頼）
- **avg_rb / avg_gassan / max_total**：参考値
""")

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
