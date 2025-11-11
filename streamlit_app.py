import streamlit as st
import pandas as pd
import numpy as np
import json, re, ast
from typing import Any, Dict, List, Tuple

# -------------------- Page --------------------
st.set_page_config(page_title="Sectional CSV Builder — Gallop + Gmax", layout="wide")
st.title("Sectional CSV Builder")
st.caption("Race Edge CSV generator with fixed steps: Gallop = 100 m • Gmax/TPD = 200 m. Includes live edits and a single, always-on Horse Weight (text) column.")

# -------------------- Utils --------------------
def parse_time_to_seconds(x):
    if pd.isna(x) or x == "":
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def build_markers(distance_m:int, step:int)->List[int]:
    start = distance_m - step
    markers = list(range(start, -1, -step))
    if markers[-1] != 0:
        markers.append(0)
    return markers

def ordered_columns(distance_m:int, step:int)->Tuple[List[str], List[int]]:
    cols = ["Draw", "Horse", "Horse Weight", "Weight Allocated"]
    markers = build_markers(distance_m, step)
    for m in markers[:-1]:
        cols.append(f"{m}_Time"); cols.append(f"{m}_Pos")
    cols += ["Finish_Time", "Finish_Pos", "Race Time", "800-400", "400-Finish"]
    return cols, markers

def normalize_time_formats(df, time_cols):
    for c in time_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: round(parse_time_to_seconds(v), 2) if (pd.notna(v) and str(v)!="") else v)
    if "Finish_Time" in df.columns:
        df["Finish_Time"] = df["Finish_Time"].apply(lambda v: round(parse_time_to_seconds(v), 2) if (pd.notna(v) and str(v)!="") else v)
    return df

def compute_derived_segments(df, distance_m, step, markers):
    split_cols = [f"{m}_Time" for m in markers[:-1] if f"{m}_Time" in df.columns]

    def total_time(row):
        tot = 0.0; seen=False
        for c in split_cols:
            val = row.get(c, "")
            if str(val) not in ("", "nan") and pd.notna(val):
                tot += parse_time_to_seconds(val); seen=True
        if "Finish_Time" in df.columns:
            val = row.get("Finish_Time", "")
            if str(val) not in ("", "nan") and pd.notna(val):
                tot += parse_time_to_seconds(val); seen=True
        return round(tot,2) if seen else np.nan

    df["Race Time"] = df.apply(total_time, axis=1)

    # 800-400
    if "800_Time" in df.columns and "400_Time" in df.columns:
        if step == 200:
            for i,row in df.iterrows():
                vals=[]
                for m in [800,600]:
                    c=f"{m}_Time"
                    if c in df.columns and pd.notna(row.get(c,"")) and str(row.get(c,""))!="":
                        vals.append(parse_time_to_seconds(row[c]))
                df.at[i,"800-400"]= round(sum(vals),2) if vals else np.nan
        else:
            for i,row in df.iterrows():
                vals=[]
                for m in [800,700,600,500,400]:
                    c=f"{m}_Time"
                    if c in df.columns and pd.notna(row.get(c,"")) and str(row.get(c,""))!="":
                        vals.append(parse_time_to_seconds(row[c]))
                df.at[i,"800-400"]= round(sum(vals),2) if vals else np.nan

    # 400-Finish
    if "400_Time" in df.columns:
        for i,row in df.iterrows():
            vals=[]
            if step == 200:
                for m in [400,200]:
                    c=f"{m}_Time"
                    if c in df.columns and pd.notna(row.get(c,"")) and str(row.get(c,""))!="":
                        vals.append(parse_time_to_seconds(row[c]))
            else:
                for m in [400,300,200,100]:
                    c=f"{m}_Time"
                    if c in df.columns and pd.notna(row.get(c,"")) and str(row.get(c,""))!="":
                        vals.append(parse_time_to_seconds(row[c]))
            if "Finish_Time" in df.columns and pd.notna(row.get("Finish_Time","")) and str(row.get("Finish_Time",""))!="":
                vals.append(parse_time_to_seconds(row["Finish_Time"]))
            df.at[i,"400-Finish"]= round(sum(vals),2) if vals else np.nan

    return df

def enforce_finish_pos_by_row(df):
    if "Finish_Pos" in df.columns:
        df["Finish_Pos"] = [i+1 for i in range(len(df))]
    return df

def reorder_columns(df, distance_m, step):
    desired, _ = ordered_columns(distance_m, step)
    # ensure single Horse Weight col, normalized naming
    if "Horse Weight" not in df.columns:
        df["Horse Weight"] = ""
    df.columns = [c.strip() for c in df.columns]
    extra = [c for c in df.columns if c not in desired]
    return df.reindex(columns=desired + extra)

def detect_distance_from_runners(runners: List[Dict[str,Any]], default_step:int)->int:
    """Distance = largest positive end + step; fallback to 1000."""
    ends = [s["end"] for r in runners for s in r.get("sections",[]) if isinstance(s.get("end"), int) and s["end"]>0]
    if not ends:
        return 1000
    return int(max(ends) + default_step)

# -------------------- Gallop adapter (100 m fixed) --------------------
def fetch_json(url: str):
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def normalize_gallop_payload(payload:Any)->List[Dict[str,Any]]:
    runners=[]
    if isinstance(payload, dict) and isinstance(payload.get("sectionals"), list):
        blocks = payload["sectionals"]
    elif isinstance(payload, list):
        blocks = payload
    else:
        blocks = []

    for obj in blocks:
        horse = obj.get("horse") or obj.get("runner") or obj.get("name") or obj.get("Horse") or ""
        secs=[]
        for s in obj.get("sections", []):
            try:
                end = int(float(str(s.get("end","0")).replace("m","").strip()))
            except Exception:
                continue
            t = s.get("timeSec")
            try: t = float(t) if t is not None else None
            except: t=None
            try: rnk = int(str(s.get("rankSec","")).strip()) if s.get("rankSec") not in (None,"") else None
            except: rnk=None
            secs.append({"end": end, "timeSec": t, "rankSec": rnk})
        runners.append({"horse": horse, "sections": secs, "meta": {}})
    return runners

# -------------------- Gmax / TPD adapter (200 m fixed) --------------------
def fetch_text(url: str):
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def parse_gmax_dataset(raw_text: str)->Tuple[List[str], List[List[Any]]]:
    """
    Accepts bodies like:
      "new Ajax.Web.DataSet([new Ajax.Web.DataTable(...)]);/*"
    (may be quoted and backslash-escaped)
    Returns (columns, rows).
    """
    if not raw_text:
        return [], []

    s = raw_text.strip()

    # If the whole payload is quoted, remove outer quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # Unescape \" → "
    s = s.replace(r'\"', '"')

    # Strip wrapper
    if s.startswith("new Ajax.Web.DataSet("):
        s = s[len("new Ajax.Web.DataSet("):]
    # drop trailing ');' and optional comment
    s = re.sub(r"\);\s*/\*.*$", "", s)
    s = s.rstrip(");").strip()

    # Normalize constructor & tokens to Python
    s = s.replace("new Ajax.Web.DataTable(", "(")
    s = re.sub(r"new Date\(Date\.UTC\([^\)]*\)\)", "None", s)
    s = s.replace("true","True").replace("false","False").replace("null","None")

    try:
        data = ast.literal_eval(s)
    except Exception:
        return [], []

    # Find first (columns, rows) pair
    def find_table(obj):
        if isinstance(obj,(list,tuple)) and len(obj)>=2:
            a,b = obj[0], obj[1]
            if isinstance(a,(list,tuple)) and a and isinstance(a[0],(list,tuple)) and len(a[0])==2:
                return a,b
        if isinstance(obj,(list,tuple)):
            for x in obj:
                r = find_table(x)
                if r: return r
        return None

    res = find_table(data)
    if not res: return [], []
    col_defs, rows = res
    cols = [c[0] for c in col_defs]
    return cols, rows

def normalize_gmax_rows(cols: List[str], rows: List[List[Any]]) -> List[Dict[str,Any]]:
    """
    Convert Gmax dataset rows into normalized runners with sections (200 m gates).
    - Horse name: prefer TrackName, fallback to ObjectName (cloth number)
    - Draw: TrackNumber (int) or TrackCode (string)
    """
    if not cols or not rows:
        return []
    pivot_ids=[]
    for c in cols:
        m = re.match(r"Pivot(\d+)_GateName", c)
        if m: pivot_ids.append(int(m.group(1)))
    pivot_ids = sorted(set(pivot_ids))

    out=[]
    for row in rows:
        rec = {cols[i]: (row[i] if i < len(row) else None) for i in range(len(cols))}
        # ---- Names & draw
        horse_name = (rec.get("TrackName") or rec.get("ObjectName") or "").strip()
        draw = rec.get("TrackNumber")
        if draw in (None, "", -99):
            draw = rec.get("TrackCode")

        sections=[]
        for pid in pivot_ids:
            gate = rec.get(f"Pivot{pid}_GateName")
            if not gate:
                continue
            g = str(gate)
            if g.lower() == "finish":
                end_marker = 0
            else:
                try:
                    end_marker = int(g.lower().replace("m","").strip())
                except:
                    continue

            t = rec.get(f"Pivot{pid}_SecTime")
            try: t = float(t) if t not in (None,"",-99) else None
            except: t = None
            pos = rec.get(f"Pivot{pid}_Position")
            try: pos = int(pos) if pos not in (None,"",-99) else None
            except: pos = None

            sections.append({"end": end_marker, "timeSec": t, "rankSec": pos})

        meta = {
            "FinishPosition": rec.get("FinishPosition"),
            "DidNotFinish": rec.get("DidNotFinish"),
            "NotRun": rec.get("NotRun"),
            "Draw": draw,
        }
        out.append({"horse": horse_name, "sections": sections, "meta": meta})
    return out

# -------------------- Build DF from normalized runners --------------------
def build_df_from_runners(runners:List[Dict[str,Any]], distance_m:int, step:int)->pd.DataFrame:
    cols, markers = ordered_columns(distance_m, step)
    df = pd.DataFrame([{c:"" for c in cols} for _ in range(len(runners))])
    for i, r in enumerate(runners):
        df.at[i,"Horse"] = (r.get("horse") or "").strip()

        # Draw from meta if provided (Gmax)
        draw = (r.get("meta") or {}).get("Draw")
        if draw not in (None, "", -99):
            df.at[i,"Draw"] = draw

        by_end = {s["end"]: s for s in r.get("sections",[]) if isinstance(s.get("end"), int)}
        for m in markers[:-1]:
            if m in by_end:
                seg = by_end[m]
                if seg.get("timeSec") is not None:
                    df.at[i, f"{m}_Time"] = round(float(seg["timeSec"]),2)
                if seg.get("rankSec") is not None:
                    df.at[i, f"{m}_Pos"] = seg["rankSec"]
        if 0 in by_end and by_end[0].get("timeSec") is not None:
            df.at[i,"Finish_Time"] = round(float(by_end[0]["timeSec"]),2)

        # Finish_Pos: prefer source when present
        fp = (r.get("meta") or {}).get("FinishPosition")
        try:
            fp_val = int(fp) if fp not in (None,"",-99) else None
        except:
            fp_val = None
        if fp_val is not None:
            df.at[i,"Finish_Pos"] = fp_val

    # Guarantee a single Horse Weight column, as text (stable on iPad)
    if "Horse Weight" not in df.columns:
        df["Horse Weight"] = ""
    df["Horse Weight"] = df["Horse Weight"].astype("object")

    return df

# -------------------- Session --------------------
if "raw_snapshot" not in st.session_state:
    st.session_state.raw_snapshot = None  # (df, distance, step)
if "edited_df" not in st.session_state:
    st.session_state.edited_df = None

# -------------------- UI --------------------
tab1, tab2, tab3 = st.tabs(["1) Source", "2) Edit", "3) Export"])

with tab1:
    source = st.radio("Source", ["Gallop JSON (100 m)", "Gmax / TPD (200 m)"], horizontal=True)

    if source.startswith("Gallop"):
        # Fixed step for Gallop
        fixed_step = 100
        url = st.text_input("Gallop JSON URL")
        raw = st.text_area("...or paste raw JSON here", height=160)
        st.caption("📏 Step size fixed: 100 m")

        if st.button("Fetch Gallop"):
            payload=None
            if url.strip():
                try:
                    payload = fetch_json(url.strip())
                except Exception as e:
                    st.error(f"Fetch error: {e}")
            elif raw.strip():
                try:
                    payload = json.loads(raw.strip())
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
            else:
                st.warning("Provide a URL or paste raw JSON.")

            if payload is not None:
                runners = normalize_gallop_payload(payload)
                if not runners:
                    st.warning("Parsed JSON but couldn't find 'sectionals'/'sections'.")
                else:
                    distance_m = detect_distance_from_runners(runners, fixed_step)
                    df = build_df_from_runners(runners, distance_m, fixed_step)
                    desired, markers = ordered_columns(distance_m, fixed_step)
                    tcols = [f"{m}_Time" for m in markers[:-1]]
                    df = normalize_time_formats(df, tcols + (["Finish_Time"] if "Finish_Time" in df.columns else []))
                    df = reorder_columns(df, distance_m, fixed_step)
                    st.session_state.raw_snapshot = (df.copy(), distance_m, fixed_step)
                    st.session_state.edited_df = df.copy()
                    st.success(f"Gallop loaded. Distance inferred: {distance_m} m. Switch to Edit.")
                    st.dataframe(df, use_container_width=True)

    else:
        # Fixed step for Gmax
        fixed_step = 200
        url = st.text_input("Gmax/TPD XHR URL (optional)")
        raw = st.text_area("...or paste the full Response body (starts with 'new Ajax.Web.DataSet')", height=220)
        st.caption("📏 Step size fixed: 200 m")

        if st.button("Fetch Gmax"):
            text=None
            if url.strip():
                try:
                    text = fetch_text(url.strip())
                except Exception as e:
                    st.error(f"Fetch error: {e}")
            elif raw.strip():
                text = raw.strip()
            else:
                st.warning("Provide a URL or paste the response body.")

            if text:
                cols, rows = parse_gmax_dataset(text)
                if not cols or not rows:
                    st.error("Could not parse the Ajax DataSet payload. Paste the exact XHR response.")
                else:
                    runners = normalize_gmax_rows(cols, rows)
                    if not runners:
                        st.warning("Parsed payload but found no runners.")
                    else:
                        distance_m = detect_distance_from_runners(runners, fixed_step)
                        df = build_df_from_runners(runners, distance_m, fixed_step)
                        desired, markers = ordered_columns(distance_m, fixed_step)
                        tcols = [f"{m}_Time" for m in markers[:-1]]
                        df = normalize_time_formats(df, tcols + (["Finish_Time"] if "Finish_Time" in df.columns else []))
                        df = reorder_columns(df, distance_m, fixed_step)
                        st.session_state.raw_snapshot = (df.copy(), distance_m, fixed_step)
                        st.session_state.edited_df = df.copy()
                        st.success(f"Gmax loaded. Distance inferred: {distance_m} m. Switch to Edit.")
                        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Edit your sheet (Horse Weight always editable)")
    if st.session_state.raw_snapshot is None:
        st.info("Load a source first in tab 1.")
    else:
        raw_df, dist, step = st.session_state.raw_snapshot

        if st.session_state.edited_df is None:
            st.session_state.edited_df = raw_df.copy()

        st.caption("Tip: paste into cells. Click **Save edits** to commit. Use **Reset to source** to discard.")
        # Keep Horse Weight as a text field (prevents iPad first-keystroke loss)
        colcfg = {
            "Horse Weight": st.column_config.TextColumn(help="Enter as text (e.g., 494 or 494kg)")
        }

        # Wrap the editor in a FORM to avoid rerun wiping inputs on iPad Safari
        with st.form("edit_form", clear_on_submit=False):
            edited_df = st.data_editor(
                st.session_state.edited_df,
                num_rows="dynamic",
                use_container_width=True,
                key="grid",
                column_config=colcfg,
            )
            c1, c2 = st.columns(2)
            save = c1.form_submit_button("💾 Save edits", use_container_width=True)
            reset = c2.form_submit_button("↩️ Reset to source", use_container_width=True)

        if save:
            st.session_state.edited_df = edited_df.copy()
            st.success("Edits saved.")
        if reset:
            st.session_state.edited_df = raw_df.copy()
            st.info("Reverted to source.")

with tab3:
    st.subheader("Process & export")
    if st.session_state.raw_snapshot is None or st.session_state.edited_df is None:
        st.info("Nothing to export yet.")
    else:
        _, dist, step = st.session_state.raw_snapshot
        df_out = st.session_state.edited_df.copy()

        # Finish_Pos fallback to row order if empty
        if "Finish_Pos" in df_out.columns:
            try:
                empty = df_out["Finish_Pos"].isna().all() or (df_out["Finish_Pos"]=="").all()
            except Exception:
                empty = False
            if empty:
                df_out = enforce_finish_pos_by_row(df_out)

        desired, markers = ordered_columns(int(dist), int(step))
        tcols = [f"{m}_Time" for m in markers[:-1]]
        df_out = normalize_time_formats(df_out, tcols + (["Finish_Time"] if "Finish_Time" in df_out.columns else []))
        df_out = compute_derived_segments(df_out, int(dist), int(step), markers)
        df_out = reorder_columns(df_out, int(dist), int(step))

                # --- Auto naming ---
        # Winner name
        winner_name = None
        if "Finish_Pos" in df_out.columns and "Horse" in df_out.columns:
            try:
                win_row = df_out[df_out["Finish_Pos"].astype(str) == "1"]
                if not win_row.empty:
                    winner_name = str(win_row.iloc[0]["Horse"]).strip()
            except Exception:
                pass
        if not winner_name:
            winner_name = "WinnerUnknown"
        safe_winner = re.sub(r"[^A-Za-z0-9_]+", "_", winner_name.replace(" ", "_")).strip("_")

        # Race date (priority order):
        # 1) Gallop payload field (date/raceDate/meetingDate)
        # 2) YYYYMMDD inside URL or raw text (covers Sect-XGD-20251107-2)
        # 3) Gmax ShareCode=VAYYYYMMDDHHMM pattern in URL
        # 4) Today
        from datetime import datetime
        race_date = None
        ctx = st.session_state

        # 1) Gallop payload explicit date
        payload = ctx.get("source_payload")
        if isinstance(payload, dict):
            for k in ("date", "raceDate", "meetingDate"):
                if k in payload and payload[k]:
                    race_date = str(payload[k])[:10].replace("-", "")
                    # if it's already YYYYMMDD leave as is; else try to parse ISO
                    if not re.match(r"^20\d{6}$", race_date):
                        try:
                            race_date = datetime.fromisoformat(str(payload[k])[:10]).strftime("%Y%m%d")
                        except Exception:
                            race_date = None
                    break

        # 2 & 3) Any YYYYMMDD in URL/raw (covers Sect code & Gmax ShareCode)
        if not race_date:
            src_url = ctx.get("source_url") or ""
            src_raw = ctx.get("source_raw") or ""
            race_date = extract_date_from_text(src_url) or extract_date_from_text(src_raw)

        # 4) Fallback to today
        if not race_date:
            race_date = datetime.now().strftime("%Y%m%d")

        # Build clean filename
        file_name = f"{safe_winner}_{race_date}_{dist}m_{step}splits.csv"

        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            f"⬇️ Download CSV — {winner_name} ({race_date})",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
        )
