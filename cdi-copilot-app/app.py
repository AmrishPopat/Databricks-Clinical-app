import os
import json
import time
import streamlit as st
import psycopg2
import psycopg2.extras
from databricks.sdk.core import Config
from databricks import sql

# ──────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit command)
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CDI Co-pilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
CATALOG = "external-ai-build-day"
SCHEMA_PIPELINE = "cdi_copilot"
SCHEMA_SOURCE = "clinical_coding"
DISCLAIMER = "⚠️ AI-generated — verify before submission"

RISK_COLORS = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
RISK_TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


# ──────────────────────────────────────────────────────────────────────
# Connections
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=300)
def get_sql_connection():
    """SQL warehouse connection for reading Unity Catalog tables."""
    cfg = Config()
    wh_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{wh_id}",
        credentials_provider=lambda: cfg.authenticate,
    )


def get_pg_connection():
    """Lakebase connection for transactional state."""
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        database=os.getenv("PGDATABASE", "databricks_postgres"),
        user=os.getenv("PGUSER", ""),
        password=os.getenv("PGPASSWORD", ""),
        port=os.getenv("PGPORT", "5432"),
    )


# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_worklist():
    """Load encounter worklist with AI risk scores from the coding review view."""
    conn = get_sql_connection()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                e.encounter_id,
                e.service_line,
                e.encounter_type,
                e.clinician_id,
                e.signature_timestamp,
                n.payer,
                COUNT(DISTINCT cr.code) AS n_codes,
                ROUND(AVG(cr.confidence_score), 1) AS avg_confidence,
                MIN(cr.confidence_tier) AS worst_tier
            FROM `{CATALOG}`.`{SCHEMA_PIPELINE}`.silver_notes_masked n
            JOIN `{CATALOG}`.`{SCHEMA_SOURCE}`.encounters e
                ON n.encounter_id = e.encounter_id
            LEFT JOIN `{CATALOG}`.`{SCHEMA_SOURCE}`.encounter_coding_review cr
                ON e.encounter_id = cr.encounter_id
            GROUP BY 1, 2, 3, 4, 5, 6
            ORDER BY avg_confidence ASC NULLS FIRST
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


@st.cache_data(ttl=60)
def load_encounter_codes(encounter_id: str):
    """Load AI-suggested codes for a specific encounter."""
    conn = get_sql_connection()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                cr.code,
                cr.code_system,
                cr.code_description,
                cr.avg_reimbursement,
                cr.confidence_score,
                cr.confidence_tier,
                cr.risk_flags,
                cr.evidence_snippet,
                dp.denial_rate,
                dp.top_denial_reason
            FROM `{CATALOG}`.`{SCHEMA_SOURCE}`.encounter_coding_review cr
            LEFT JOIN `{CATALOG}`.`{SCHEMA_PIPELINE}`.gold_code_denial_priors dp
                ON cr.code = dp.code
                AND (
                    SELECT n.payer
                    FROM `{CATALOG}`.`{SCHEMA_PIPELINE}`.silver_notes_masked n
                    WHERE n.encounter_id = cr.encounter_id
                    LIMIT 1
                ) = dp.payer
            WHERE cr.encounter_id = '{encounter_id}'
            ORDER BY cr.confidence_score ASC
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


@st.cache_data(ttl=60)
def load_masked_note(encounter_id: str):
    """Load PII-masked clinical note text."""
    conn = get_sql_connection()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT masked_text, payer, encounter_type, service_line, signature_timestamp
            FROM `{CATALOG}`.`{SCHEMA_PIPELINE}`.silver_notes_masked
            WHERE encounter_id = '{encounter_id}'
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    return None


@st.cache_data(ttl=60)
def load_note_labels(encounter_id: str):
    """Load gold-standard labels for evaluation."""
    conn = get_sql_connection()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT gold_codes, gold_cdi_gap
            FROM `{CATALOG}`.`{SCHEMA_PIPELINE}`.note_labels
            WHERE encounter_id = '{encounter_id}'
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    return None


# ──────────────────────────────────────────────────────────────────────
# Lakebase write helpers
# ──────────────────────────────────────────────────────────────────────
def commit_decision(encounter_id, note_id, coder_id, code_data, decision, final_code,
                    override_reason, payer):
    """Write a coding decision to Lakebase."""
    pg = get_pg_connection()
    pg.autocommit = True
    with pg.cursor() as cur:
        cur.execute("""
            INSERT INTO cdi.coding_decisions
            (encounter_id, note_id, coder_id, ai_suggested_code, ai_code_system,
             ai_confidence, ai_risk_tier, ai_denial_risk, ai_evidence, ai_cdi_gap,
             decision, final_code, override_reason, payer)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING decision_id;
        """, (
            encounter_id, note_id, coder_id,
            code_data.get("code", ""),
            code_data.get("code_system", "ICD-10"),
            code_data.get("confidence_score"),
            code_data.get("confidence_tier", ""),
            code_data.get("denial_rate"),
            code_data.get("evidence_snippet", ""),
            code_data.get("risk_flags", ""),
            decision, final_code, override_reason, payer,
        ))
        decision_id = cur.fetchone()[0]

    # Audit log
    with pg.cursor() as cur:
        cur.execute("""
            INSERT INTO cdi.coding_audit_log
            (decision_id, encounter_id, action, actor, detail)
            VALUES (%s, %s, %s, %s, %s);
        """, (
            decision_id, encounter_id, decision, coder_id,
            json.dumps({"code": final_code, "ai_code": code_data.get("code", "")}),
        ))
    pg.close()
    return decision_id


def send_cdi_query(encounter_id, question, related_code):
    """Send a CDI query to a physician."""
    pg = get_pg_connection()
    pg.autocommit = True
    with pg.cursor() as cur:
        cur.execute("""
            INSERT INTO cdi.cdi_queries
            (encounter_id, question, related_code)
            VALUES (%s, %s, %s)
            RETURNING query_id;
        """, (encounter_id, question, related_code))
        qid = cur.fetchone()[0]
    pg.close()
    return qid


def load_decisions(encounter_id):
    """Read back committed decisions from Lakebase."""
    try:
        pg = get_pg_connection()
        with pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            t0 = time.time()
            cur.execute("""
                SELECT decision_id, ai_suggested_code, final_code, decision,
                       override_reason, decided_at, ai_disclaimer
                FROM cdi.coding_decisions
                WHERE encounter_id = %s
                ORDER BY decided_at DESC;
            """, (encounter_id,))
            rows = cur.fetchall()
            latency_ms = (time.time() - t0) * 1000
        pg.close()
        return rows, latency_ms
    except Exception:
        return [], 0


# ──────────────────────────────────────────────────────────────────────
# UI: Sidebar
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 CDI Co-pilot")
st.sidebar.caption("Clinical Coding & CDI Co-pilot")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["📋 Worklist", "🔍 Coding Review", "📊 Dashboard"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.warning(DISCLAIMER)
st.sidebar.caption("Lakebase: `cdi-copilot-db`")


# ──────────────────────────────────────────────────────────────────────
# PAGE 1: Encounter Worklist
# ──────────────────────────────────────────────────────────────────────
if page == "📋 Worklist":
    st.title("📋 Coder Worklist")
    st.caption("Encounters sorted by AI confidence (lowest first = most attention needed)")
    st.info(DISCLAIMER)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        svc_filter = st.selectbox("Service Line", ["All"] + [
            "general_medicine", "cardiology", "pulmonology", "oncology",
            "obstetrics", "orthopedics", "neurology", "gastroenterology",
        ])
    with col_f2:
        risk_filter = st.selectbox("Risk Tier", ["All", "LOW", "MEDIUM", "HIGH"])
    with col_f3:
        st.text("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()

    worklist = load_worklist()

    # Apply filters
    if svc_filter != "All":
        worklist = [w for w in worklist if w.get("service_line") == svc_filter]
    if risk_filter != "All":
        worklist = [w for w in worklist if w.get("worst_tier") == risk_filter]

    if not worklist:
        st.warning("No encounters match the current filters.")
    else:
        for enc in worklist:
            tier = enc.get("worst_tier", "MEDIUM") or "MEDIUM"
            icon = RISK_COLORS.get(tier, "🟡")
            conf = enc.get("avg_confidence") or 0
            n_codes = enc.get("n_codes", 0)

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1.5, 1, 1.5])
                with c1:
                    st.markdown(f"**{enc['encounter_id']}**")
                    st.caption(f"{enc.get('service_line', '')} · {enc.get('encounter_type', '')}")
                with c2:
                    st.caption(f"Payer: **{enc.get('payer', 'N/A')}**")
                    st.caption(f"Clinician: {enc.get('clinician_id', '')}")
                with c3:
                    st.metric("Avg Confidence", f"{conf}%")
                with c4:
                    st.markdown(f"{icon} **{tier}**")
                    st.caption(f"{n_codes} codes")
                with c5:
                    if st.button("Review ▶", key=f"review_{enc['encounter_id']}",
                                 use_container_width=True, type="primary"):
                        st.session_state["selected_encounter"] = enc["encounter_id"]
                        st.session_state["page"] = "review"
                        st.rerun()


# ──────────────────────────────────────────────────────────────────────
# PAGE 2: Coding Review
# ──────────────────────────────────────────────────────────────────────
elif page == "🔍 Coding Review":
    st.title("🔍 Coding Review")

    enc_id = st.session_state.get("selected_encounter", "")
    enc_input = st.text_input("Encounter ID", value=enc_id, placeholder="ENC-000000")
    if enc_input:
        enc_id = enc_input
        st.session_state["selected_encounter"] = enc_id

    if not enc_id:
        st.info("Select an encounter from the Worklist or enter an Encounter ID above.")
    else:
        note_data = load_masked_note(enc_id)
        codes = load_encounter_codes(enc_id)
        labels = load_note_labels(enc_id)

        if not note_data:
            st.error(f"No note found for {enc_id}")
        else:
            # Header
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Encounter", enc_id)
            h2.metric("Payer", note_data.get("payer", "N/A"))
            h3.metric("Service", note_data.get("service_line", "N/A"))
            h4.metric("Type", note_data.get("encounter_type", "N/A"))

            st.divider()
            st.info(DISCLAIMER)

            # Split view: Note | Codes
            col_note, col_codes = st.columns([1, 1])

            # LEFT: Clinical Note (masked)
            with col_note:
                st.subheader("📄 Clinical Note (PII Masked)")
                note_text = note_data.get("masked_text", "")

                # Highlight evidence snippets in the note
                highlighted = note_text
                if codes:
                    for cd in codes:
                        snippet = cd.get("evidence_snippet", "")
                        if snippet and len(snippet) > 10:
                            # Bold the first 60 chars of snippet for highlighting
                            short = snippet[:60]
                            if short in highlighted:
                                highlighted = highlighted.replace(
                                    short, f"**🔍 {short}**", 1
                                )

                st.markdown(
                    f'<div style="background:#f8f9fa; padding:16px; border-radius:8px; '
                    f'max-height:500px; overflow-y:auto; font-size:14px; line-height:1.6;">'
                    f'{highlighted}</div>',
                    unsafe_allow_html=True,
                )

                # Gold-standard labels (for evaluation)
                if labels:
                    with st.expander("🏷️ Gold-Standard Labels (evaluation)"):
                        st.write(f"**Expected codes:** {labels.get('gold_codes', 'N/A')}")
                        gap = labels.get("gold_cdi_gap")
                        if gap:
                            st.warning(f"**CDI Gap:** {gap}")

            # RIGHT: AI Suggested Codes
            with col_codes:
                st.subheader("🤖 AI Suggested Codes")

                if not codes:
                    st.warning("No AI-suggested codes for this encounter.")
                else:
                    # Overall risk score
                    avg_conf = sum(c.get("confidence_score", 0) or 0 for c in codes) / max(len(codes), 1)
                    worst = min(codes, key=lambda c: c.get("confidence_score", 100) or 100)
                    tier = worst.get("confidence_tier", "MEDIUM")
                    st.metric(
                        "Overall Risk Score",
                        f"{avg_conf:.0f}%",
                        delta=f"{RISK_COLORS.get(tier, '🟡')} {tier}",
                        delta_color="off",
                    )

                    st.divider()

                    # Per-code cards
                    for i, cd in enumerate(codes):
                        code = cd.get("code", "")
                        system = cd.get("code_system", "")
                        desc = cd.get("code_description", "")
                        conf = cd.get("confidence_score", 0) or 0
                        c_tier = cd.get("confidence_tier", "MEDIUM")
                        denial = cd.get("denial_rate")
                        denial_reason = cd.get("top_denial_reason", "")
                        snippet = cd.get("evidence_snippet", "")
                        flags = cd.get("risk_flags", "")
                        reimb = cd.get("avg_reimbursement", 0) or 0

                        c_icon = RISK_COLORS.get(c_tier, "🟡")

                        with st.container(border=True):
                            st.markdown(f"### {c_icon} `{code}` — {desc}")
                            st.caption(f"{system} · Avg reimbursement: ${reimb:,.0f}")

                            m1, m2 = st.columns(2)
                            m1.metric("Confidence", f"{conf}%")
                            if denial is not None:
                                m2.metric(
                                    "Denial Risk",
                                    f"{denial * 100:.0f}%",
                                    delta=denial_reason or None,
                                    delta_color="inverse",
                                )

                            if flags:
                                st.warning(f"⚠ {flags}")

                            if snippet:
                                with st.expander("📎 Evidence from note"):
                                    st.info(snippet)

                            # Action buttons
                            b1, b2, b3 = st.columns(3)
                            with b1:
                                if st.button("✅ Accept", key=f"accept_{enc_id}_{code}_{i}",
                                             use_container_width=True, type="primary"):
                                    did = commit_decision(
                                        enc_id, f"NOTE-{enc_id.split('-')[1]}",
                                        "CODER-001", cd, "ACCEPT", code, None,
                                        note_data.get("payer"),
                                    )
                                    st.success(f"✅ Accepted {code} → Decision {did}")
                            with b2:
                                if st.button("✏️ Override", key=f"override_{enc_id}_{code}_{i}",
                                             use_container_width=True):
                                    st.session_state[f"show_override_{code}_{i}"] = True
                            with b3:
                                if st.button("❌ Remove", key=f"remove_{enc_id}_{code}_{i}",
                                             use_container_width=True):
                                    did = commit_decision(
                                        enc_id, f"NOTE-{enc_id.split('-')[1]}",
                                        "CODER-001", cd, "REMOVE", code, "Coder removed",
                                        note_data.get("payer"),
                                    )
                                    st.warning(f"❌ Removed {code} → Decision {did}")

                            # Override form
                            if st.session_state.get(f"show_override_{code}_{i}", False):
                                with st.form(key=f"override_form_{code}_{i}"):
                                    new_code = st.text_input("Override code", value=code)
                                    reason = st.text_area("Reason for override")
                                    if st.form_submit_button("Submit Override"):
                                        did = commit_decision(
                                            enc_id, f"NOTE-{enc_id.split('-')[1]}",
                                            "CODER-001", cd, "OVERRIDE", new_code,
                                            reason, note_data.get("payer"),
                                        )
                                        st.success(
                                            f"✏️ Overridden {code} → {new_code} (Decision {did})"
                                        )
                                        st.session_state[f"show_override_{code}_{i}"] = False

            st.divider()

            # CDI Query Section
            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.subheader("📝 Send CDI Query")
                with st.form("cdi_query_form"):
                    q_text = st.text_area(
                        "Query to physician",
                        placeholder="e.g. Please document antibiotic administration start time",
                    )
                    q_code = st.text_input("Related code", placeholder="A41.9")
                    if st.form_submit_button("Send CDI Query", type="primary"):
                        if q_text:
                            qid = send_cdi_query(enc_id, q_text, q_code)
                            st.success(f"📝 CDI Query sent (ID: {qid})")
                        else:
                            st.error("Please enter a query.")

            with gc2:
                st.subheader("📂 Committed Decisions")
                decisions, latency = load_decisions(enc_id)
                if decisions:
                    st.caption(f"Read-back latency: {latency:.1f}ms")
                    for d in decisions:
                        emoji = {"ACCEPT": "✅", "OVERRIDE": "✏️", "REMOVE": "❌",
                                 "ESCALATE": "🚨"}.get(d["decision"], "📌")
                        st.markdown(
                            f"{emoji} **{d['final_code']}** — {d['decision']} "
                            f"({d['decided_at'].strftime('%H:%M:%S') if d.get('decided_at') else ''})"
                        )
                        if d.get("override_reason"):
                            st.caption(f"  Reason: {d['override_reason']}")
                        st.caption(f"  {d.get('ai_disclaimer', DISCLAIMER)}")
                else:
                    st.caption("No decisions committed yet for this encounter.")


# ──────────────────────────────────────────────────────────────────────
# PAGE 3: Dashboard Summary
# ──────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 CDI Analytics Dashboard")
    st.info(DISCLAIMER)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)

    conn = get_sql_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM `{CATALOG}`.`{SCHEMA_PIPELINE}`.silver_notes_masked")
        n_notes = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM `{CATALOG}`.`{SCHEMA_SOURCE}`.encounters")
        n_enc = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM `{CATALOG}`.`{SCHEMA_SOURCE}`.claims_denials WHERE status = 'denied'")
        n_denied = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM `{CATALOG}`.`{SCHEMA_SOURCE}`.claims_denials")
        n_claims = cur.fetchone()[0]

    c1.metric("Clinical Notes", n_notes)
    c2.metric("Encounters", f"{n_enc:,}")
    c3.metric("Total Claims", f"{n_claims:,}")
    c4.metric("Denied Claims", f"{n_denied:,}", delta=f"{n_denied/max(n_claims,1)*100:.1f}%",
              delta_color="inverse")

    st.divider()

    # Denial rate by payer
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Denial Rate by Payer")
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT payer,
                       count(*) as total,
                       sum(CASE WHEN status = 'denied' THEN 1 ELSE 0 END) as denied,
                       round(sum(CASE WHEN status = 'denied' THEN 1 ELSE 0 END) * 100.0 / count(*), 1) as denial_pct
                FROM `{CATALOG}`.`{SCHEMA_SOURCE}`.claims_denials
                GROUP BY payer ORDER BY denial_pct DESC
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        import pandas as pd
        df = pd.DataFrame(rows, columns=cols)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with d2:
        st.subheader("Top Denial Reasons")
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT denial_reason, count(*) as count,
                       round(avg(billed_amount), 2) as avg_billed
                FROM `{CATALOG}`.`{SCHEMA_SOURCE}`.claims_denials
                WHERE denial_reason IS NOT NULL
                GROUP BY denial_reason ORDER BY count DESC
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        df2 = pd.DataFrame(rows, columns=cols)
        st.dataframe(df2, use_container_width=True, hide_index=True)

    st.divider()

    # High-risk codes
    st.subheader("Highest Denial Risk Codes (by Payer)")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT code, payer, n_claims, denial_rate, top_denial_reason,
                   round(avg_billed_amount, 2) as avg_billed
            FROM `{CATALOG}`.`{SCHEMA_PIPELINE}`.gold_code_denial_priors
            ORDER BY denial_rate DESC
            LIMIT 15
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df3 = pd.DataFrame(rows, columns=cols)
    st.dataframe(df3, use_container_width=True, hide_index=True)
