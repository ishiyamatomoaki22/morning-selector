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

# ========= 朝イチ用おすすめ設定（候補を狭めすぎない） =========
RECOMMENDED = {
    "マイジャグラーV":           {"min_games": 2500, "max_rb": 290.0, "max_gassan": 195.0},
    "ゴーゴージャグラー3":       {"min_games": 2500, "max_rb": 300.0, "max_gassan": 200.0},
    "ハッピージャグラーVIII":    {"min_games": 3000, "max_rb": 285.0, "max_gassan": 190.0},
    "ファンキージャグラー2KT":   {"min_games": 2500, "max_rb": 320.0, "max_gassan": 210.0},
    "ミスタージャグラー":        {"min_games": 2300, "max_rb": 320.0, "max_gassan": 210.0},
    "ジャグラーガールズSS":      {"min_games": 2300, "max_rb": 285.0, "max_gassan": 190.0},
    "ネオアイムジャグラーEX":    {"min_games": 2300, "max_rb": 350.0, "max_gassan": 220.0},
    "ウルトラミラクルジャグラー":{"min_games": 3000, "max_rb": 320.0, "max_gassan": 210.0},
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
    agg = df.groupby(gcols, dropna=False).agg(
        samples=("date", "count"),
        unique_days=("date", "nunique"),
        w_sum=("w", "sum"),
        good_w=("is_good_day", lambda s: float(np.sum(s.values * df.loc[s.index, "w"].values))),
        good_days=("is_good_day", "sum"),
        avg_rb=("rb_rate_num", "mean"),
        avg_gassan=("gassan_rate_num", "mean"),
        max_total=("total_start_num", "max"),
    ).reset_index()

    # ブレ対策（サンプル不足排除）
    agg = agg[agg["unique_days"] >= int(min_unique_days)].copy()
    if agg.empty:
        return pd.DataFrame()

    agg["good_rate_weighted"] = (agg["good_w"] / agg["w_sum"]).replace([np.inf, -np.inf], np.nan)
    agg["good_rate_simple"] = (agg["good_days"] / agg["unique_days"]).replace([np.inf, -np.inf], np.nan)

    # 信頼度補正（重み合計が大きいほど加点）
    wmax = float(agg["w_sum"].max() if agg["w_sum"].notna().any() else 0.0)
    trust = np.log1p(agg["w_sum"].fillna(0.0)) / np.log1p(wmax + 1e-9) if wmax > 0 else 0.0

    agg["score"] = (
        (agg["good_rate_weighted"].fillna(0) * 100.0) * 0.70 +
        (trust * 100.0) * 0.30
    ) / 100.0

    # 表示整形
    out = agg.copy()
    out["good_rate_weighted"] = (out["good_rate_weighted"] * 100).round(1)
    out["good_rate_simple"] = (out["good_rate_simple"] * 100).round(1)
    out["avg_rb"] = pd.to_numeric(out["avg_rb"], errors="coerce").round(1)
    out["avg_gassan"] = pd.to_numeric(out["avg_gassan"], errors="coerce").round(1)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").round(3)

    out = out.sort_values(["score", "good_rate_weighted", "w_sum"], ascending=[False, False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["train_start"] = start_day
    out["train_end"] = end_day
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

# --------- 共通：過去データの投入（tab1/tab3で使う） ---------
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

tab1, tab2, tab3 = st.tabs([
    "朝イチ候補（過去データ集計）",
    "実戦ログ（CSVに追記して更新版DL）",
    "精度検証（バックテスト）"
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
        )

        if ranking.empty:
            st.warning("指定した条件でランキングが作れません（データ不足/サンプル不足）。条件を緩めてください。")
        else:
            train_start = ranking["train_start"].iloc[0]
            train_end = ranking["train_end"].iloc[0]
            st.success(f"学習期間：{train_start}〜{train_end}（対象：{shop} / {machine}）  |  台数: {len(ranking)}")
            st.dataframe(
                ranking.head(int(top_n))[[
                    "shop","machine","unit_number",
                    "rank","score",
                    "good_rate_weighted","good_rate_simple",
                    "unique_days","samples","w_sum",
                    "avg_rb","avg_gassan","max_total"
                ]],
                use_container_width=True,
                hide_index=True
            )

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
    st.subheader("③ 精度検証（バックテスト：対象日 vs 直前N日）")
    st.caption("対象日を選び、その前N日だけで作ったランキングが、対象日の“良台日”をどれだけ当てられたかを評価します（リークなし）。")

    if df_all_shared.empty:
        st.info("上の『共通：過去データアップロード』にCSV/zipを入れてください。")
    else:
        df_all = df_all_shared.copy()
        df_all = df_all[df_all["date"].notna()].copy()
        df_all = df_all[(df_all["shop"] == shop) & (df_all["machine"] == machine)].copy()

        if df_all.empty:
            st.warning("この店×機種のデータがありません。店/機種を変えるか、データを追加してください。")
        else:
            # 対象日の候補範囲（データに存在する日だけ）
            all_days = sorted(pd.Series(df_all["date"].dropna().unique()).tolist())
            if not all_days:
                st.warning("日付データが取得できませんでした（date列の形式を確認してください）。")
            else:
                min_day, max_day = all_days[0], all_days[-1]

                st.markdown("#### 対象日の指定")
                mode = st.radio("検証モード", ["単日", "期間（まとめて）"], horizontal=True)
                K_list = st.multiselect("評価するK（上位K台）", [3,5,10,15,20,30], default=[10,20])

                if mode == "単日":
                    target_day = st.date_input("対象日", value=max_day, min_value=min_day, max_value=max_day)
                    target_days = [pd.to_datetime(target_day).date()]
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        target_from = st.date_input("対象期間 From", value=min_day, min_value=min_day, max_value=max_day)
                    with c2:
                        target_to = st.date_input("対象期間 To", value=max_day, min_value=min_day, max_value=max_day)

                    target_from = pd.to_datetime(target_from).date()
                    target_to = pd.to_datetime(target_to).date()
                    if target_from > target_to:
                        st.error("From と To の順序が逆です。")
                        target_days = []
                    else:
                        target_days = [d for d in all_days if (d >= target_from and d <= target_to)]

                st.divider()
                st.markdown("#### 検証実行")

                if st.button("この条件でバックテストする", type="primary", use_container_width=True):
                    rows = []
                    for td in target_days:
                        start_train = (pd.to_datetime(td) - pd.Timedelta(days=int(lookback_days))).date()
                        if start_train < min_day:
                            continue

                        ranking = build_ranking(
                            df_all=df_all,
                            shop=shop,
                            machine=machine,
                            base_day=td,
                            lookback_days=int(lookback_days),
                            tau=int(tau),
                            min_games=int(min_games),
                            max_rb=float(max_rb),
                            max_gassan=float(max_gassan),
                            min_unique_days=int(min_unique_days),
                        )

                        truth_set, total_units = truth_good_units(
                            df_all=df_all,
                            shop=shop,
                            machine=machine,
                            target_day=td,
                            min_games=int(min_games),
                            max_rb=float(max_rb),
                            max_gassan=float(max_gassan),
                        )

                        if ranking.empty:
                            rows.append({
                                "target_day": td,
                                "train_ok": False,
                                "truth_good_units": len(truth_set),
                                "total_units": total_units,
                                "note": "学習側がデータ不足/サンプル不足",
                            })
                            continue

                        first_hit_rank = None
                        if truth_set:
                            for _, r in ranking.iterrows():
                                u = int(r["unit_number"]) if pd.notna(r["unit_number"]) else None
                                if u in truth_set:
                                    first_hit_rank = int(r["rank"])
                                    break

                        base = {
                            "target_day": td,
                            "train_ok": True,
                            "truth_good_units": len(truth_set),
                            "total_units": total_units,
                            "first_hit_rank": first_hit_rank if first_hit_rank is not None else np.nan,
                            "note": "" if truth_set else "対象日に正解台が0（閾値が厳しすぎる可能性）",
                        }

                        for K in K_list:
                            topK_units = set(pd.to_numeric(ranking.head(int(K))["unit_number"], errors="coerce").dropna().astype(int).tolist())
                            hits = len(topK_units & truth_set)
                            base[f"hits@{K}"] = hits
                            base[f"Hit@{K}"] = 1 if hits > 0 else 0
                            base[f"P@{K}"] = round((hits / int(K)) if K > 0 else 0.0, 3)

                        rows.append(base)

                    res = pd.DataFrame(rows)

                    if res.empty:
                        st.warning("検証できる対象日がありません（過去何日が大きすぎる/データ期間が短い等）。")
                    else:
                        res = res.sort_values("target_day", ascending=False)
                        st.success(f"検証完了：{len(res)}日（店: {shop} / 機種: {machine}）")
                        st.dataframe(res, use_container_width=True, hide_index=True)

                        # ===== 見える化（単日モードのみ）=====
                        if mode == "単日":
                            st.divider()
                            st.markdown("#### 見える化（予測 vs 当たり）")

                            # 単日の対象日
                            td = target_days[0]

                            # その日の予測ランキング（リークなし：直前N日で作る）
                            ranking_single = build_ranking(
                                df_all=df_all,
                                shop=shop,
                                machine=machine,
                                base_day=td,
                                lookback_days=int(lookback_days),
                                tau=int(tau),
                                min_games=int(min_games),
                                max_rb=float(max_rb),
                                max_gassan=float(max_gassan),
                                min_unique_days=int(min_unique_days),
                            )

                            # 対象日の正解（当たり台）
                            truth_set, _ = truth_good_units(
                                df_all=df_all,
                                shop=shop,
                                machine=machine,
                                target_day=td,
                                min_games=int(min_games),
                                max_rb=float(max_rb),
                                max_gassan=float(max_gassan),
                            )

                            if ranking_single.empty:
                                st.warning("この対象日は予測ランキングを作れません（学習側データ不足/サンプル不足）。")
                            else:
                                truth_list = sorted([int(x) for x in truth_set]) if truth_set else []
                                st.caption(f"対象日: {td} / 当たり台（正解）: {len(truth_list)}台")
                                st.write(truth_list if truth_list else "（当たり台が0台：閾値が厳しすぎる可能性）")

                                for K in K_list:
                                    st.markdown(f"##### K={K}")

                                    pred_units = pd.to_numeric(
                                        ranking_single.head(int(K))["unit_number"],
                                        errors="coerce"
                                    ).dropna().astype(int).tolist()

                                    pred_set = set(pred_units)
                                    inter = sorted(list(pred_set & truth_set))

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown("**予測 上位K台（台番）**")
                                        st.write(pred_units if pred_units else "（なし）")
                                    with col2:
                                        st.markdown("**対象日の当たり台（台番）**")
                                        st.write(truth_list if truth_list else "（なし）")
                                    with col3:
                                        st.markdown("**一致（台番）**")
                                        st.write(inter if inter else "（一致なし）")

                                    st.metric("hits（一致台数）", len(inter))


                        st.divider()
                        st.markdown("#### 集計（平均）")
                        summary = {}
                        valid = res[res["train_ok"] == True].copy()
                        summary["days_tested"] = int(len(res))
                        summary["days_train_ok"] = int(len(valid))
                        summary["avg_truth_good_units"] = float(valid["truth_good_units"].mean()) if len(valid) else np.nan
                        summary["avg_first_hit_rank"] = float(valid["first_hit_rank"].mean()) if len(valid) else np.nan

                        for K in K_list:
                            if f"Hit@{K}" in valid.columns:
                                summary[f"Hit@{K}_rate"] = float(valid[f"Hit@{K}"].mean())
                            if f"P@{K}" in valid.columns:
                                summary[f"P@{K}_avg"] = float(valid[f"P@{K}"].mean())

                        st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

                        out_name = make_filename(machine, "backtest_result", date_str)
                        st.download_button(
                            "バックテスト結果をCSVでダウンロード",
                            data=to_csv_bytes(res),
                            file_name=out_name,
                            mime="text/csv",
                            key="tab3_dl_backtest"
                        )
