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
st.caption("過去の original.csv（複数日）をアップロード → 店×機種ごとに重み付き集計 → 朝イチ候補を順位付け ＆ 精度検証")

JST = ZoneInfo("Asia/Tokyo")

# ========= 朝イチ用おすすめ設定（# 機種ごとのおすすめ設定（朝イチ向け・勝率重視の目安)） =========
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

def load_many_csvs(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = normalize_columns(df)
        df = compute_rates_if_needed(df)
        for c in HEADER:
            if c not in df.columns:
                df[c] = np.nan
        df = df[HEADER].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=HEADER)
    return pd.concat(dfs, ignore_index=True)

def load_zip_of_csv(zip_bytes: bytes) -> pd.DataFrame:
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as fp:
                df = pd.read_csv(fp)
            df = normalize_columns(df)
            df = compute_rates_if_needed(df)
            for c in HEADER:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[HEADER].copy()
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

# ======== Play Log (append to uploaded CSV) ========
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

# ========= Core ranking (for candidates + backtest) =========
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
    block_size: int,
    w_unit: float,
    w_tail: float,
    w_block: float,
):
    """
    base_day を基準に、(base_day-lookback_days)〜(base_day-1) のデータだけでランキングを作る
    ※ base_day 当日のデータは学習に使わない（リーク防止）
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

    # 数値化
    df["unit_number"] = pd.to_numeric(df["unit_number"], errors="coerce")
    df = df[df["unit_number"].notna()].copy()
    df["unit_number"] = df["unit_number"].astype(int)

    df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
    df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
    df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

    # 末尾・台番帯
    df["tail"] = df["unit_number"] % 10
    bs = max(int(block_size), 1)
    df["block"] = df["unit_number"] // bs

    # 減衰重み
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
        tail=("tail", "first"),
        block=("block", "first"),
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

    # 台単体スコア：良台率（重み付き）＋信頼度
    agg_u["unit_score"] = (
        (agg_u["good_rate_weighted"].fillna(0) * 100.0) * 0.70 +
        (trust * 100.0) * 0.30
    ) / 100.0

    # -------------------------
    # ② 末尾（tail）集計
    # -------------------------
    agg_t = df.groupby(["shop", "machine", "tail"], dropna=False).agg(
        t_unique_days=("date", "nunique"),
        t_w_sum=("w", "sum"),
        t_good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
    ).reset_index()
    agg_t["tail_score"] = (agg_t["t_good_w"] / agg_t["t_w_sum"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # -------------------------
    # ③ 台番帯（block）集計
    # -------------------------
    agg_b = df.groupby(["shop", "machine", "block"], dropna=False).agg(
        b_unique_days=("date", "nunique"),
        b_w_sum=("w", "sum"),
        b_good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
    ).reset_index()
    agg_b["block_score"] = (agg_b["b_good_w"] / agg_b["b_w_sum"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 台単体にJOIN
    out = agg_u.merge(agg_t[["shop","machine","tail","tail_score"]], on=["shop","machine","tail"], how="left")
    out = out.merge(agg_b[["shop","machine","block","block_score"]], on=["shop","machine","block"], how="left")
    out["tail_score"] = out["tail_score"].fillna(0.0)
    out["block_score"] = out["block_score"].fillna(0.0)

    # 最終スコア（勝率寄り＝場所重視）
    out["final_score"] = (
        out["unit_score"].fillna(0.0) * float(w_unit) +
        out["tail_score"].fillna(0.0) * float(w_tail) +
        out["block_score"].fillna(0.0) * float(w_block)
    )

    # 表示整形
    out["good_rate_weighted"] = (out["good_rate_weighted"] * 100).round(1)
    out["good_rate_simple"] = (out["good_rate_simple"] * 100).round(1)
    out["unit_score"] = pd.to_numeric(out["unit_score"], errors="coerce").round(3)
    out["tail_score"] = pd.to_numeric(out["tail_score"], errors="coerce").round(3)
    out["block_score"] = pd.to_numeric(out["block_score"], errors="coerce").round(3)
    out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce").round(3)
    out["avg_rb"] = pd.to_numeric(out["avg_rb"], errors="coerce").round(1)
    out["avg_gassan"] = pd.to_numeric(out["avg_gassan"], errors="coerce").round(1)

    out = out.sort_values(
        ["final_score", "tail_score", "block_score", "unit_score", "w_sum"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)
    out["train_start"] = start_day
    out["train_end"] = end_day
    out["block_size"] = int(bs)

    return out

def truth_good_units(
    df_all: pd.DataFrame,
    shop: str,
    machine: str,
    target_day: date,
    min_games: int,
    max_rb: float,
    max_gassan: float,
):
    """
    target_day 当日のデータから「正解台（良台日判定を満たす台）」を返す
    """
    df = df_all.copy()
    df = df[df["date"] == target_day].copy()
    df = df[(df["shop"] == shop) & (df["machine"] == machine)].copy()
    if df.empty:
        return set(), 0

    df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
    df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
    df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

    good = df[
        (df["total_start_num"] >= min_games) &
        (df["rb_rate_num"] <= max_rb) &
        (df["gassan_rate_num"] <= max_gassan)
    ]["unit_number"].dropna().astype(int).tolist()

    total_units = int(df["unit_number"].dropna().nunique())
    return set(good), total_units


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
            on_change=lambda: apply_recommended(st.session_state["machine_select"]),
            key="machine_select",
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
    st.header("朝イチ候補スコア（場所重視）")

    block_size = st.number_input("台番帯の幅（block_size）", 5, 50, 10, 1)
    w_tail = st.slider("末尾（tail）の重み", 0.0, 1.0, 0.5, 0.05)
    w_block = st.slider("台番帯（block）の重み", 0.0, 1.0, 0.3, 0.05)

    # 台単体の重みは残り
    w_unit = max(0.0, 1.0 - (w_tail + w_block))
    st.caption(f"台単体の重み（自動）: {w_unit:.2f}  ※ w_tail + w_block が 1 を超えると台単体は 0 になります")

    st.divider()
    st.header("朝イチ判定（スライダー）")

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

# --------- 共通：過去データの投入（tab1で使う） ---------
def upload_past_data_ui():
    st.caption("複数CSV（original.csv）または zip（CSVをまとめたもの）をアップロードしてください。")
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

    df_all = pd.DataFrame(columns=HEADER)
    if past_files:
        df_all = pd.concat([df_all, load_many_csvs(past_files)], ignore_index=True)
    if past_zip is not None:
        df_all = pd.concat([df_all, load_zip_of_csv(past_zip.getvalue())], ignore_index=True)

    if df_all.empty:
        st.error("CSVが読み込めませんでした（中身が空/形式違いの可能性）。")
        return pd.DataFrame(columns=HEADER)

    return df_all


# ========= Main UI =========
st.divider()
st.subheader("共通：過去データアップロード（朝イチ候補 / 精度検証で使用）")
df_all_shared = upload_past_data_ui()

tab1, tab2 = st.tabs([
    "朝イチ候補（過去データ集計）",
    "実戦ログ（CSVに追記して更新版DL）"
])

with tab1:
    with st.expander("候補テーブルの見方（用語説明）", expanded=False):
        st.markdown("""
    - **rank**：最終順位（final_scoreの高い順）
    - **unit_number**：台番号
    - **machine**：機種名（全機種表示にする場合に重要）
    - **tail**：末尾（unit_number % 10）
    - **block**：台番帯（unit_number // block_size）※近い台＝同じ島付近の近似

    - **final_score**：最終スコア（朝イチ優先度の結論）
    - `final_score = w_unit*unit_score + w_tail*tail_score + w_block*block_score`

    - **unit_score**：その台“単体”の強さ（良台率＋信頼度）
    - **tail_score**：その末尾が強いか（店のクセ）
    - **block_score**：その帯が強いか（並び・島のクセ）

    - **good_rate_weighted(%)**：直近重視の良台率（新しい日ほど重く評価）
    - **good_rate_simple(%)**：単純な良台率（期間を均等に扱う）
    - **unique_days**：データに登場した日数（少ないとブレる）
    - **w_sum**：減衰込みの学習量（大きいほど信頼）
    - **avg_rb / avg_gassan**：平均RB/合算（参考）
    - **max_total**：最大総回転（参考）
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
            block_size=int(block_size),
            w_unit=float(w_unit),
            w_tail=float(w_tail),
            w_block=float(w_block),
        )


        if ranking.empty:
            st.warning("指定した条件でランキングが作れません（データ不足/サンプル不足）。条件を緩めてください。")
        else:
            train_start = ranking["train_start"].iloc[0]
            train_end = ranking["train_end"].iloc[0]
            st.success(f"学習期間：{train_start}〜{train_end}（対象：{shop} / {machine}）  |  台数: {len(ranking)}")
            show_cols = [
                "rank",
                "unit_number","tail","block",
                "final_score","unit_score","tail_score","block_score",
                "good_rate_weighted","good_rate_simple",
                "unique_days","samples","w_sum",
                "avg_rb","avg_gassan","max_total",
            ]
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