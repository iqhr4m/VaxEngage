
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

st.set_page_config(
    page_title="VaxEngage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{
      --navy:#0E2B4C; --teal:#0B8C87; --blue:#2563EB;
      --green:#15805D; --amber:#B97808; --red:#B53A3A;
      --ink:#172033; --muted:#667085; --line:#E2E8F0; --soft:#F5F8FB;
    }
    .stApp{background:#F5F8FB;color:var(--ink);}
    .block-container{max-width:1320px;padding-top:1.1rem;padding-bottom:3rem;}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#0E2B4C 0%,#163B61 100%);}
    [data-testid="stSidebar"] *{color:#F8FAFC;}
    .hero{
      background:
        radial-gradient(circle at 90% 15%, rgba(55,214,197,.24), transparent 25%),
        linear-gradient(120deg,#0E2B4C 0%,#17486E 64%,#0B8C87 135%);
      border-radius:22px;padding:28px 30px;color:white;
      box-shadow:0 16px 42px rgba(14,43,76,.16);margin-bottom:1rem;
    }
    .hero h1{margin:0;font-size:2.2rem;letter-spacing:-.03em;}
    .hero p{margin:.55rem 0 0;color:#DCEAF2;max-width:920px;line-height:1.55;}
    .eyebrow{display:inline-block;font-size:.73rem;font-weight:800;letter-spacing:.09em;
      text-transform:uppercase;color:#A7F3D0;margin-bottom:.5rem;}
    .card{
      background:white;border:1px solid var(--line);border-radius:18px;
      padding:18px 20px;box-shadow:0 7px 22px rgba(14,43,76,.05);margin-bottom:1rem;
    }
    .chip{display:inline-block;padding:6px 11px;border-radius:999px;font-size:.76rem;
      font-weight:800;background:#E8F4FF;color:#1459A3;border:1px solid #CCE3F7;}
    .subtle{color:var(--muted);font-size:.86rem;line-height:1.45;}
    .big-status{font-size:1.9rem;font-weight:850;margin:.2rem 0;}
    .signal{
      border-left:4px solid #0B8C87;background:#F0FAF9;border-radius:10px;
      padding:10px 12px;margin:.42rem 0;color:#294154;
    }
    .context-note{
      border-left:4px solid #2563EB;background:#F1F6FF;border-radius:10px;
      padding:10px 12px;margin:.42rem 0;color:#294154;
    }
    .action-low,.action-moderate,.action-high{
      border-radius:14px;padding:15px 17px;font-weight:650;margin-top:.45rem;line-height:1.45;
    }
    .action-low{background:#EAF8F1;border:1px solid #BCE6D1;color:#126B49;}
    .action-moderate{background:#FFF6E3;border:1px solid #EBCF8D;color:#805500;}
    .action-high{background:#FDECEC;border:1px solid #EDB9B9;color:#922F2F;}
    .timeline{
      display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin:.65rem 0 1rem;
    }
    .timebox{
      border:1px solid var(--line);border-radius:13px;text-align:center;padding:10px 5px;min-height:88px;
    }
    .timebox .age{font-size:.68rem;color:var(--muted);font-weight:750;}
    .timebox .mark{font-size:1.35rem;margin:.18rem 0;}
    .timebox .txt{font-size:.70rem;line-height:1.18;color:#344054;}
    div[data-testid="stMetric"]{
      background:white;border:1px solid var(--line);padding:10px 13px;border-radius:14px;
    }
    button[kind="primary"]{border-radius:12px!important;min-height:2.75rem;font-weight:750!important;}
    @media(max-width:850px){
      .timeline{grid-template-columns:repeat(2,minmax(0,1fr));}
      .hero h1{font-size:1.65rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Geography and questionnaire mappings
# -----------------------------------------------------------------------------
PROVINCES = {
    "11":"Aceh","12":"Sumatera Utara","13":"Sumatera Barat","14":"Riau",
    "15":"Jambi","16":"Sumatera Selatan","17":"Bengkulu","18":"Lampung",
    "19":"Kepulauan Bangka Belitung","21":"Kepulauan Riau","31":"DKI Jakarta",
    "32":"Jawa Barat","33":"Jawa Tengah","34":"DI Yogyakarta","35":"Jawa Timur",
    "36":"Banten","51":"Bali","52":"Nusa Tenggara Barat","53":"Nusa Tenggara Timur",
    "61":"Kalimantan Barat","62":"Kalimantan Tengah","63":"Kalimantan Selatan",
    "64":"Kalimantan Timur","65":"Kalimantan Utara","71":"Sulawesi Utara",
    "72":"Sulawesi Tengah","73":"Sulawesi Selatan","74":"Sulawesi Tenggara",
    "75":"Gorontalo","76":"Sulawesi Barat","81":"Maluku","82":"Maluku Utara",
    "91":"Papua Barat","92":"Papua Barat Daya","94":"Papua","95":"Papua Selatan",
    "96":"Papua Tengah","97":"Papua Pegunungan"
}

I30_WINDOWS = [
    ("I30A","29 days–3 months"),
    ("I30B","3–6 months"),
    ("I30C","6–9 months"),
    ("I30D","9–12 months"),
    ("I30E","12–18 months"),
    ("I30F","18–24 months"),
]

I30_OPTIONS = {
    "Monitored":1,
    "Not monitored":2,
    "Record unavailable / unknown":3,
}

CONTACT = {"Yes":1, "No":2, "Don't know":8}
VITA = {"Yes, once":1, "Yes, twice":2, "Never":3}

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    s1 = joblib.load(MODEL_DIR / "stage1_model.joblib")
    s2 = joblib.load(MODEL_DIR / "stage2_model.joblib")
    with open(MODEL_DIR / "stage1_metadata.json", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(MODEL_DIR / "stage2_metadata.json", encoding="utf-8") as f:
        m2 = json.load(f)
    return s1, m1, s2, m2

try:
    S1_MODEL, S1_META, S2_MODEL, S2_META = load_models()
except Exception as exc:
    st.error(
        "The saved models could not be loaded. Run this dashboard with the package's "
        "pinned scikit-learn version (1.9.0)."
    )
    st.exception(exc)
    st.stop()

# -----------------------------------------------------------------------------
# Feature engineering – must mirror the final derivation script exactly
# -----------------------------------------------------------------------------
def derive_features(record):
    vals = [record.get(var, np.nan) for var, _ in I30_WINDOWS]

    known = [int(v) for v in vals if pd.notna(v) and int(v) in (1,2)]
    record["dev_known_windows"] = len(known)
    record["dev_engaged_windows"] = sum(v == 1 for v in known)
    record["dev_missed_windows"] = sum(v == 2 for v in known)
    record["dev_engagement_ratio"] = (
        record["dev_engaged_windows"] / record["dev_known_windows"]
        if record["dev_known_windows"] > 0 else np.nan
    )
    record["dev_recent_known_status"] = known[-1] if known else np.nan

    # Strict consecutive misses:
    # unknown / code 3 / code 7 / missing BREAKS the streak.
    current = 0
    maximum = 0
    for v in vals:
        if pd.isna(v):
            current = 0
            continue
        v = int(v)
        if v == 2:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    record["dev_max_consecutive_missed"] = maximum

    # Frequencies are valid only when the corresponding monitoring occurred.
    if record.get("I23") != 1:
        record["I24"] = np.nan
    if record.get("I26") != 1:
        record["I27"] = np.nan

    return record


def model_frame(record, meta):
    needed = meta["categorical_features"] + meta["numeric_features"]
    row = record.copy()
    for col in needed:
        if col not in row:
            row[col] = np.nan
    return pd.DataFrame([{c: row[c] for c in needed}])


def score_child(record, stage):
    record = derive_features(record.copy())
    if stage == 1:
        model, meta = S1_MODEL, S1_META
    else:
        model, meta = S2_MODEL, S2_META

    p = float(model.predict_proba(model_frame(record, meta))[0,1])
    low = float(meta["prototype_low_to_moderate_threshold"])
    high = float(meta["prototype_moderate_to_high_threshold"])
    level = "High" if p >= high else ("Moderate" if p >= low else "Low")
    return record, p, level, meta


def service_signals(record):
    signals = []

    if record.get("dev_max_consecutive_missed", 0) >= 2:
        n = int(record["dev_max_consecutive_missed"])
        signals.append(f"{n} consecutive developmental-monitoring windows were missed.")

    missed = int(record.get("dev_missed_windows", 0) or 0)
    if missed > 0 and record.get("dev_max_consecutive_missed", 0) < 2:
        signals.append(f"{missed} developmental-monitoring window(s) were reported as missed.")

    if record.get("I23") == 2:
        signals.append("No weight monitoring was reported in the previous 12 months.")
    elif record.get("I23") == 8:
        signals.append("Weight-monitoring status is unknown.")

    if record.get("I26") == 2:
        signals.append("No length/height monitoring was reported in the previous 12 months.")
    elif record.get("I26") == 8:
        signals.append("Length/height-monitoring status is unknown.")

    if record.get("I29") == 3:
        signals.append("No vitamin A receipt was reported in the previous 12 months.")

    unavailable = sum(record.get(v) == 3 for v, _ in I30_WINDOWS)
    if unavailable:
        signals.append(
            f"{unavailable} developmental-monitoring record(s) were unavailable; "
            "these are treated as unknown, not as missed monitoring."
        )

    if not signals:
        signals.append("No major disengagement signal was observed in the entered service-contact history.")
    return signals


def action_text(level, stage):
    if stage == 1:
        if level == "High":
            return (
                "Verify vaccination status promptly. If vaccination cannot be confirmed, "
                "prioritize active outreach or a home visit."
            )
        if level == "Moderate":
            return (
                "Verify vaccination status at the next contact and closely monitor subsequent "
                "child-health service engagement."
            )
        return (
            "Continue routine monitoring and verify vaccination status during scheduled child-health contacts."
        )

    # Stage 2
    if level == "High":
        return (
            "Vaccination has already been initiated. Verify outstanding doses, arrange catch-up where indicated, "
            "and assess practical or acceptance barriers to completion."
        )
    if level == "Moderate":
        return (
            "Review the vaccination record and reinforce the next scheduled dose or catch-up appointment."
        )
    return "Continue routine vaccination follow-up and scheduled child-health services."


def gauge(score, meta):
    low = float(meta["prototype_low_to_moderate_threshold"]) * 100
    high = float(meta["prototype_moderate_to_high_threshold"]) * 100
    value = score * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font":{"size":42}},
        title={"text":"Concern score", "font":{"size":17}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#0E2B4C"},
            "bgcolor":"white",
            "steps":[
                {"range":[0,low],"color":"#EAF8F1"},
                {"range":[low,high],"color":"#FFF3D8"},
                {"range":[high,100],"color":"#FBE4E4"},
            ],
        }
    ))
    fig.update_layout(
        height=275, margin=dict(l=30,r=30,t=45,b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family":"Arial, sans-serif","color":"#172033"}
    )
    return fig


def timeline_html(record, age):
    status = {
        1:("✓","Monitored","#15805D","#EAF8F1"),
        2:("×","Missed","#B53A3A","#FDECEC"),
        3:("?","Unknown","#B97808","#FFF6E3"),
        7:("–","Not yet applicable","#667085","#F1F5F9"),
    }
    parts = []
    for var, label in I30_WINDOWS:
        v = record.get(var, 7)
        mark, txt, color, bg = status.get(v, ("?","Unknown","#667085","#F1F5F9"))
        parts.append(
            f"""<div class="timebox" style="background:{bg}">
            <div class="age">{label}</div><div class="mark" style="color:{color}">{mark}</div>
            <div class="txt">{txt}</div></div>"""
        )
    return '<div class="timeline">' + "".join(parts) + '</div>'


def empty_record(age, province):
    return {
        "province_code": province,
        "B4K7BLN": age,
        "I23": np.nan, "I24": np.nan,
        "I26": np.nan, "I27": np.nan,
        "I29": np.nan,
        "I30A": np.nan, "I30B": np.nan, "I30C": np.nan,
        "I30D": np.nan, "I30E": np.nan, "I30F": 7 if age < 18 else np.nan,
    }


def batch_score(data):
    rows = []
    for idx, r in data.iterrows():
        raw = r.to_dict()
        stage = int(raw.get("screening_stage", 1) or 1)
        stage = 2 if stage == 2 else 1
        raw, p, level, _ = score_child(raw, stage)
        rows.append({
            "child_id": raw.get("child_id", idx),
            "screening_stage": stage,
            "concern_score": p,
            "concern_level": level,
            "service_signals": " | ".join(service_signals(raw)),
            "suggested_action": action_text(level, stage),
        })
    out = pd.DataFrame(rows)
    order = pd.Categorical(out["concern_level"], ["High","Moderate","Low"], ordered=True)
    out = out.assign(_order=order).sort_values(["_order","concern_score"], ascending=[True,False]).drop(columns="_order")
    return out

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ VaxEngage")
    st.caption("Child-health engagement early warning")
    st.markdown("---")
    page = st.radio("Workspace", ["Single child", "Outreach queue", "Model information"])
    st.markdown("---")
    st.caption("Minimal deployment model")
    st.success("Stage 1 · Primary alarm")
    st.info("Stage 2 · Completion follow-up")

# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <span class="eyebrow">Context + routine child-health engagement</span>
      <h1>VaxEngage</h1>
      <p>
        A proof-of-concept early-warning dashboard for vaccination disengagement.
        Geographic context and child age are combined with routine growth, vitamin A,
        and developmental-monitoring contacts to prioritize vaccination-status verification and outreach.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Single child
# -----------------------------------------------------------------------------
if page == "Single child":
    st.markdown("### Screening mode")
    screen_mode = st.segmented_control(
        "Choose the appropriate use",
        [
            "Primary screening: possible failure to initiate vaccination",
            "Completion follow-up: vaccination already initiated"
        ],
        default="Primary screening: possible failure to initiate vaccination"
    )
    stage = 1 if screen_mode.startswith("Primary") else 2

    if stage == 1:
        st.caption("Use this as the main early-warning screen. Vaccination status is not entered into the model.")
    else:
        st.caption("Use only when at least one vaccine is already known to have been received.")

    left, right = st.columns([1.08,.92], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### 1 · Automatic context")
            c1,c2 = st.columns(2)
            with c1:
                province_name = st.selectbox("Province", list(PROVINCES.values()), index=0)
                province_code = next(k for k,v in PROVINCES.items() if v == province_name)
            with c2:
                child_age = st.slider("Child age (months)", 12, 23, 18)

            st.markdown(
                '<div class="context-note">Geography and age provide background context. '
                'They are not interpreted as reasons for disengagement.</div>',
                unsafe_allow_html=True
            )

        with st.container(border=True):
            st.markdown("#### 2 · Routine child-health engagement")

            c1,c2,c3 = st.columns(3)
            with c1:
                weight_label = st.selectbox(
                    "Weight monitoring in previous 12 months",
                    ["Yes","No","Don't know"]
                )
                i23 = CONTACT[weight_label]
                if weight_label == "Yes":
                    i24 = st.number_input(
                        "Number of weight measurements",
                        min_value=1, max_value=87, value=10, step=1
                    )
                else:
                    i24 = np.nan

            with c2:
                height_label = st.selectbox(
                    "Length/height monitoring in previous 12 months",
                    ["Yes","No","Don't know"]
                )
                i26 = CONTACT[height_label]
                if height_label == "Yes":
                    i27 = st.number_input(
                        "Number of length/height measurements",
                        min_value=1, max_value=87, value=10, step=1
                    )
                else:
                    i27 = np.nan

            with c3:
                vita_label = st.selectbox(
                    "Vitamin A in previous 12 months",
                    ["Yes, once","Yes, twice","Never"]
                )
                i29 = VITA[vita_label]

            st.markdown("##### Developmental monitoring across age windows")
            st.caption(
                "Record unavailable is treated as unknown, not as missed monitoring. "
                "Future age windows are marked not applicable automatically."
            )

            dev = {}
            cols = st.columns(3)

            # A-E are applicable for the 12-23 month cohort.
            for i,(var,label) in enumerate(I30_WINDOWS[:5]):
                with cols[i % 3]:
                    chosen = st.selectbox(label, list(I30_OPTIONS.keys()), key=f"dev_{var}")
                    dev[var] = I30_OPTIONS[chosen]

            # I30F applies from 18 months onward.
            if child_age >= 18:
                with cols[5 % 3]:
                    chosen = st.selectbox("18–24 months", list(I30_OPTIONS.keys()), key="dev_I30F")
                    dev["I30F"] = I30_OPTIONS[chosen]
            else:
                dev["I30F"] = 7
                with cols[5 % 3]:
                    st.text_input("18–24 months", value="Not yet applicable", disabled=True)

        run = st.button("Assess concern", type="primary", use_container_width=True)

    with right:
        st.markdown("### Early-warning output")

        if not run:
            st.markdown(
                """
                <div class="card">
                  <span class="chip">Ready</span>
                  <h3>Enter the current service-contact history</h3>
                  <p class="subtle">
                    The dashboard derives the longitudinal engagement features automatically
                    and assigns a Low, Moderate, or High concern band.
                  </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            record = empty_record(child_age, province_code)
            record.update({
                "I23":i23, "I24":i24,
                "I26":i26, "I27":i27,
                "I29":i29,
                **dev
            })

            record, score, level, meta = score_child(record, stage)
            color = {"Low":"#15805D","Moderate":"#B97808","High":"#B53A3A"}[level]

            st.markdown(
                f"""
                <div class="card">
                  <span class="chip">{'Stage 1 · Initiation' if stage == 1 else 'Stage 2 · Completion'}</span>
                  <div class="big-status" style="color:{color}">{level.upper()} CONCERN</div>
                  <div class="subtle">
                    The numeric output is a model-based concern score, not an absolute clinical probability.
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.plotly_chart(gauge(score, meta), use_container_width=True, config={"displayModeBar":False})

            st.markdown("#### Developmental-monitoring timeline")
            st.markdown(timeline_html(record, child_age), unsafe_allow_html=True)

            a,b,c = st.columns(3)
            a.metric("Known windows", int(record["dev_known_windows"]))
            b.metric("Monitored windows", int(record["dev_engaged_windows"]))
            c.metric("Longest confirmed missed run", int(record["dev_max_consecutive_missed"]))

            st.markdown("#### Service-contact signals")
            for s in service_signals(record):
                st.markdown(f'<div class="signal">{s}</div>', unsafe_allow_html=True)

            st.markdown("#### Suggested action")
            css = {"Low":"action-low","Moderate":"action-moderate","High":"action-high"}[level]
            st.markdown(f'<div class="{css}">{action_text(level, stage)}</div>', unsafe_allow_html=True)

            with st.expander("Technical details"):
                st.json({
                    "stage": stage,
                    "model": meta["model"],
                    "concern_score": round(score,4),
                    "low_to_moderate_threshold": round(float(meta["prototype_low_to_moderate_threshold"]),4),
                    "moderate_to_high_threshold": round(float(meta["prototype_moderate_to_high_threshold"]),4),
                    "derived_features": {
                        "dev_known_windows": int(record["dev_known_windows"]),
                        "dev_engaged_windows": int(record["dev_engaged_windows"]),
                        "dev_missed_windows": int(record["dev_missed_windows"]),
                        "dev_engagement_ratio": (
                            None if pd.isna(record["dev_engagement_ratio"])
                            else round(float(record["dev_engagement_ratio"]),3)
                        ),
                        "dev_recent_known_status": (
                            None if pd.isna(record["dev_recent_known_status"])
                            else int(record["dev_recent_known_status"])
                        ),
                        "dev_max_consecutive_missed": int(record["dev_max_consecutive_missed"])
                    }
                })
                st.caption(meta["note"])

# -----------------------------------------------------------------------------
# Outreach queue / batch
# -----------------------------------------------------------------------------
elif page == "Outreach queue":
    st.markdown("### Outreach queue")
    st.write(
        "Upload a CSV exported from a child-health registry. The result is sorted with High-concern "
        "children first to support vaccination-status verification and outreach prioritization."
    )

    template = APP_DIR / "VaxEngage_batch_template.csv"
    if template.exists():
        st.download_button(
            "Download CSV template",
            data=template.read_bytes(),
            file_name="VaxEngage_batch_template.csv",
            mime="text/csv"
        )

    uploaded = st.file_uploader("Upload child list", type=["csv"])

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head(30), use_container_width=True)

        if st.button("Generate outreach queue", type="primary"):
            try:
                result = batch_score(data)
                st.success(f"Screened {len(result):,} children.")

                summary = (
                    result["concern_level"]
                    .value_counts()
                    .reindex(["High","Moderate","Low"], fill_value=0)
                    .rename_axis("Concern")
                    .reset_index(name="Children")
                )
                x,y,z = st.columns(3)
                x.metric("High concern", int(summary.loc[summary.Concern=="High","Children"].iloc[0]))
                y.metric("Moderate concern", int(summary.loc[summary.Concern=="Moderate","Children"].iloc[0]))
                z.metric("Low concern", int(summary.loc[summary.Concern=="Low","Children"].iloc[0]))

                st.dataframe(
                    result.style.format({"concern_score":"{:.3f}"}),
                    use_container_width=True,
                    height=520
                )

                st.download_button(
                    "Download outreach queue",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="VaxEngage_outreach_queue.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as exc:
                st.error("The child list could not be scored. Check the template field names.")
                st.exception(exc)

# -----------------------------------------------------------------------------
# Model information
# -----------------------------------------------------------------------------
else:
    st.markdown("### Model information")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="card">
              <span class="chip">Primary early-warning model</span>
              <h3>Stage 1 · Vaccination initiation</h3>
              <p><b>Outcome:</b> never vaccinated vs ever vaccinated</p>
              <p><b>Algorithm:</b> {S1_META['model']}</p>
              <p><b>Held-out ROC-AUC:</b> 0.873</p>
              <p class="subtle">Engagement alone: ROC-AUC 0.840 · context alone: 0.729.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
              <span class="chip">Secondary follow-up model</span>
              <h3>Stage 2 · Vaccination completion</h3>
              <p><b>Outcome:</b> incomplete vs complete after initiation</p>
              <p><b>Algorithm:</b> {S2_META['model']}</p>
              <p><b>Held-out ROC-AUC:</b> 0.687</p>
              <p class="subtle">This model has lower discrimination and should be interpreted as a follow-up aid.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Inputs used by the deployment model")
    st.code(
        """
AUTOMATIC CONTEXT
• Province
• Child age

ROUTINE CHILD-HEALTH ENGAGEMENT
• Weight monitoring
• Frequency of weight monitoring
• Length/height monitoring
• Frequency of length/height monitoring
• Vitamin A
• Developmental monitoring across age windows

DERIVED AUTOMATICALLY
• Number of known developmental-monitoring windows
• Number monitored
• Number missed
• Engagement ratio
• Most recent known status
• Longest uninterrupted run of confirmed missed monitoring
        """,
        language="text"
    )

    st.warning(
        "Proof-of-concept only. The models were derived from cross-sectional SKI 2023 data. "
        "The concern score is not a prospectively validated risk probability and should not replace "
        "vaccination records, clinical judgment, or local immunization protocols."
    )

st.markdown("---")
st.caption(
    "VaxEngage v3 · Minimal context + service-engagement model · "
    "Designed for vaccination-status verification and outreach prioritization."
)
