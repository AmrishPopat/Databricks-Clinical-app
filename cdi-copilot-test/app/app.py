import os, json, uuid
from datetime import datetime, timezone

import dash
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import pandas as pd
from flask import request
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

import agent

# ── Config ──────────────────────────────────────────────────────────────────
CATALOG   = "`external-ai-build-day`"
SCHEMA_CC = f"{CATALOG}.clinical_coding"
SCHEMA_CDI = f"{CATALOG}.cdi_copilot"
WAREHOUSE = os.environ.get("DATABRICKS_WAREHOUSE_ID", "22998c886e21ee6c")
GENIE_URL = os.environ.get("GENIE_SPACE_URL", "")
# Overridable so a test deployment doesn't write into the live decisions table
DECISIONS_TABLE = f"{SCHEMA_CDI}.{os.environ.get('CDI_DECISIONS_TABLE', 'coder_decisions')}"

# SDK WorkspaceClient auto-authenticates as the App service principal
w = WorkspaceClient()

def _params(params: dict | None):
    out = []
    for k, v in (params or {}).items():
        if isinstance(v, tuple):  # (value, SQL type)
            v, t = v
        else:
            t = "STRING"
        out.append(StatementParameterListItem(name=k, value=None if v is None else str(v), type=t))
    return out

def run_query(query: str, params: dict | None = None) -> pd.DataFrame:
    """Execute SQL via Statement Execution API (App SP auth), with named :params."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE,
        statement=query,
        parameters=_params(params),
        wait_timeout="50s",
    )
    if resp.status.state.value != "SUCCEEDED":
        err = resp.status.error if resp.status.error else "Unknown"
        raise Exception(f"Query failed: {err}")
    cols = [c.name for c in resp.manifest.schema.columns]
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return pd.DataFrame(data, columns=cols)

def run_stmt(stmt: str, params: dict | None = None) -> None:
    """Execute a write statement via Statement Execution API."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE,
        statement=stmt,
        parameters=_params(params),
        wait_timeout="50s",
    )
    if resp.status.state.value != "SUCCEEDED":
        err = resp.status.error if resp.status.error else "Unknown"
        raise Exception(f"Write failed: {err}")

def ensure_tables():
    """Idempotently create the agent output + decisions tables (same DDL as notebooks/batch_code_notes)."""
    for ddl in [
        f"""CREATE TABLE IF NOT EXISTS {SCHEMA_CDI}.agent_runs (
              run_id STRING, note_id STRING, encounter_id STRING, coding_risk_score DOUBLE,
              risk_components STRING, cdi_gaps STRING, entities STRING, rejected_codes STRING,
              model_endpoint STRING, prompt_version STRING, latency_ms INT, input_tokens INT,
              output_tokens INT, trace_id STRING, created_at TIMESTAMP)""",
        f"""CREATE TABLE IF NOT EXISTS {SCHEMA_CDI}.agent_code_suggestions (
              suggestion_id STRING, run_id STRING, note_id STRING, encounter_id STRING, code STRING,
              code_system STRING, code_description STRING, avg_reimbursement BIGINT, confidence DOUBLE,
              principal BOOLEAN, rationale STRING, evidence_quote STRING, evidence_start INT, evidence_end INT,
              evidence_verified BOOLEAN, created_at TIMESTAMP)""",
        f"CREATE TABLE IF NOT EXISTS {DECISIONS_TABLE} LIKE {SCHEMA_CDI}.coder_decisions",
    ]:
        try:
            run_stmt(ddl)
        except Exception as e:
            print(f"[startup] table setup failed: {e}")

# ── Data loaders ────────────────────────────────────────────────────────────
_codebook = None

def get_codebook():
    global _codebook
    if _codebook is None:
        df = run_query(f"SELECT code, code_system, description, avg_reimbursement FROM {SCHEMA_CDI}.codebook ORDER BY code")
        _codebook = df.to_dict("records")
    return _codebook

LATEST_RUNS = f"""
    SELECT * FROM {SCHEMA_CDI}.agent_runs
    QUALIFY row_number() OVER (PARTITION BY note_id ORDER BY created_at DESC) = 1
"""

def load_note_queue():
    return run_query(f"""
        WITH latest AS ({LATEST_RUNS}),
        sugg AS (
            SELECT run_id, COUNT(*) AS num_codes, SUM(avg_reimbursement) AS total_reimbursement
            FROM {SCHEMA_CDI}.agent_code_suggestions GROUP BY run_id
        ),
        decided AS (
            SELECT DISTINCT note_id FROM {DECISIONS_TABLE} WHERE note_id IS NOT NULL
        )
        SELECT
            s.note_id, s.encounter_id, s.service_line, s.encounter_type, s.payer,
            s.signature_timestamp AS signed_at,
            s.baseline_days_to_final_code AS days_to_code,
            r.run_id, r.coding_risk_score,
            coalesce(g.num_codes, 0) AS num_codes,
            coalesce(g.total_reimbursement, 0) AS total_reimbursement,
            d.note_id IS NOT NULL AS committed
        FROM {SCHEMA_CDI}.silver_notes_masked s
        LEFT JOIN latest r ON s.note_id = r.note_id
        LEFT JOIN sugg g ON r.run_id = g.run_id
        LEFT JOIN decided d ON s.note_id = d.note_id
        ORDER BY committed, r.run_id IS NULL, r.coding_risk_score DESC, s.signature_timestamp DESC
    """)

def load_note(note_id: str):
    df = run_query(f"""
        SELECT note_id, encounter_id, masked_text, service_line, encounter_type, payer, signature_timestamp
        FROM {SCHEMA_CDI}.silver_notes_masked WHERE note_id = :note_id
    """, {"note_id": note_id})
    return None if df.empty else df.iloc[0]

def load_latest_run(note_id: str):
    """Latest agent run for a note + its code suggestions with payer denial priors."""
    runs = run_query(f"SELECT * FROM ({LATEST_RUNS}) WHERE note_id = :note_id", {"note_id": note_id})
    if runs.empty:
        return None, pd.DataFrame()
    run = runs.iloc[0]
    sugg = run_query(f"""
        SELECT a.*, dp.denial_rate, dp.top_denial_reason
        FROM {SCHEMA_CDI}.agent_code_suggestions a
        JOIN {SCHEMA_CDI}.silver_notes_masked s ON a.note_id = s.note_id
        LEFT JOIN {SCHEMA_CDI}.gold_code_denial_priors dp ON a.code = dp.code AND s.payer = dp.payer
        WHERE a.run_id = :run_id
        ORDER BY a.principal DESC, a.avg_reimbursement DESC
    """, {"run_id": run["run_id"]})
    return run, sugg

def load_denial_priors(payer: str) -> dict:
    df = run_query(f"SELECT code, denial_rate FROM {SCHEMA_CDI}.gold_code_denial_priors WHERE payer = :payer",
                   {"payer": payer})
    return {r["code"]: float(r["denial_rate"]) for _, r in df.iterrows() if r["denial_rate"] is not None}

def load_kpis():
    return run_query(f"""
        SELECT
            COUNT(DISTINCT e.encounter_id) AS total_encounters,
            ROUND(AVG(e.days_to_final_code),1) AS avg_days_to_code,
            SUM(CASE WHEN cd.status='denied' THEN 1 ELSE 0 END) AS total_denials,
            COUNT(cd.claim_id) AS total_claims,
            ROUND(SUM(CASE WHEN cd.status='denied' THEN 1 ELSE 0 END)*100.0
                / NULLIF(COUNT(cd.claim_id),0),1) AS denial_rate_pct,
            ROUND(SUM(CASE WHEN cd.status='denied' THEN cd.billed_amount ELSE 0 END),0)
                AS total_denied_dollars
        FROM {SCHEMA_CC}.encounters e
        LEFT JOIN {SCHEMA_CC}.claims_denials cd ON e.encounter_id = cd.encounter_id
    """)

def load_decisions():
    return run_query(f"SELECT * FROM {DECISIONS_TABLE} ORDER BY decision_timestamp DESC LIMIT 200")

def save_agent_result(result: dict) -> None:
    run, suggestions = agent.to_rows(result)
    run_stmt(f"""
        INSERT INTO {SCHEMA_CDI}.agent_runs ({", ".join(agent.RUNS_COLUMNS)})
        VALUES (:run_id, :note_id, :encounter_id, :coding_risk_score, :risk_components, :cdi_gaps,
                :entities, :rejected_codes, :model_endpoint, :prompt_version, :latency_ms,
                :input_tokens, :output_tokens, :trace_id, :created_at)
    """, {**run,
          "coding_risk_score": (run["coding_risk_score"], "DOUBLE"),
          "latency_ms": (run["latency_ms"], "INT"),
          "input_tokens": (run["input_tokens"], "INT"),
          "output_tokens": (run["output_tokens"], "INT"),
          "created_at": (run["created_at"], "TIMESTAMP")})
    if not suggestions:
        return
    types = {"avg_reimbursement": "BIGINT", "confidence": "DOUBLE", "principal": "BOOLEAN",
             "evidence_start": "INT", "evidence_end": "INT", "evidence_verified": "BOOLEAN",
             "created_at": "TIMESTAMP"}
    values, params = [], {}
    for i, s in enumerate(suggestions):
        values.append("(" + ", ".join(f":{c}_{i}" for c in agent.SUGGESTION_COLUMNS) + ")")
        for c in agent.SUGGESTION_COLUMNS:
            params[f"{c}_{i}"] = (s.get(c), types.get(c, "STRING"))
    run_stmt(f"""
        INSERT INTO {SCHEMA_CDI}.agent_code_suggestions ({", ".join(agent.SUGGESTION_COLUMNS)})
        VALUES {", ".join(values)}
    """, params)

def current_user() -> str:
    try:
        return request.headers.get("X-Forwarded-Email") or request.headers.get("X-Forwarded-User") or "demo-coder-01"
    except RuntimeError:
        return "demo-coder-01"

# ── Dash app ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="CDI Coding Co-pilot",
)
server = app.server

# ── Colour helpers ──────────────────────────────────────────────────────────
RISK_COLORS = {"HIGH RISK": "danger", "MEDIUM RISK": "warning", "LOW RISK": "success", "AWAITING AI": "secondary"}
TIER_COLORS = {"LOW": "danger", "MEDIUM": "warning", "HIGH": "success"}

def risk_level(score):
    if score is None or pd.isna(score):
        return "AWAITING AI"
    score = float(score)
    return "HIGH RISK" if score >= 50 else "MEDIUM RISK" if score >= 30 else "LOW RISK"

def confidence_tier(conf: float) -> str:
    return "HIGH" if conf >= 0.8 else "MEDIUM" if conf >= 0.6 else "LOW"

def _is_true(v) -> bool:
    return str(v).lower() == "true"

def _int_or_none(v):
    return None if v is None or pd.isna(v) else int(float(v))

AI_DISCLAIMER = dbc.Badge("AI-generated — verify against source note", color="light", text_color="dark",
                          className="border ms-2", style={"fontSize": "0.65rem"})

def kpi_card(title, value, sub="", color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H3(value, className=f"text-{color} mb-0", style={"fontWeight": "700"}),
            html.Small(sub, className="text-muted") if sub else None,
        ], className="py-2 px-3")
    ], className="shadow-sm h-100")

def highlighted_note(text: str, spans: list):
    """Render the masked note with <mark> anchors. spans: [(start, end, anchor_id, label, css_class)]."""
    children, pos = [], 0
    for start, end, anchor, label, cls in sorted((s for s in spans if s[0] is not None), key=lambda s: s[0]):
        if start < pos:  # overlapping evidence: the earlier span already highlights this text
            continue
        children.append(text[pos:start])
        children.append(html.Mark([html.Sup(label, className="me-1 fw-bold"), text[start:end]],
                                  id=anchor, className=cls))
        pos = end
    children.append(text[pos:])
    return html.Div(children, className="note-text")

# ── Layout ──────────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    dcc.Store(id="selected-note", data=None),
    dcc.Store(id="agent-run", data=None),
    dcc.Store(id="refresh-trigger", data=0),

    # Header
    dbc.Navbar(
        dbc.Container([
            html.Span("\U0001F3E5", style={"fontSize": "1.5rem", "marginRight": "10px"}),
            dbc.NavbarBrand("Clinical Coding & CDI Co-pilot",
                           className="fw-bold", style={"fontSize": "1.2rem"}),
            html.Span(f"AI coding agent: {agent.MODEL_ENDPOINT} · prompt {agent.PROMPT_VERSION}",
                      className="text-light ms-3", style={"fontSize": "0.85rem", "opacity": 0.8}),
            dbc.Button("\U0001F4AC Ask Genie", href=GENIE_URL, target="_blank", color="light",
                       size="sm", className="ms-auto") if GENIE_URL else None,
        ], fluid=True),
        color="dark", dark=True, className="mb-3"
    ),

    # KPI bar
    dbc.Row(id="kpi-bar", className="mb-3 g-2"),

    # Main 3-panel layout
    dbc.Row([
        # Left: Note queue
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("\U0001F4CB Note Queue", className="mb-0 fw-bold"),
                    html.Small("Sorted by AI Coding Risk Score", className="text-muted")
                ]),
                dbc.CardBody(id="note-queue-body", style={
                    "maxHeight": "70vh", "overflowY": "auto", "padding": "0"
                })
            ], className="shadow-sm")
        ], width=3),

        # Center: Code review
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("\U0001F50D Code Review", className="mb-0 fw-bold")),
                dbc.CardBody(dcc.Loading(html.Div(id="code-review-body"), type="circle"),
                             style={"maxHeight": "75vh", "overflowY": "auto"})
            ], className="shadow-sm")
        ], width=6),

        # Right: CDI Query + Commit
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("⚠️ CDI & Commit", className="mb-0 fw-bold")),
                dbc.CardBody(id="cdi-panel-body", style={"maxHeight": "75vh", "overflowY": "auto"})
            ], className="shadow-sm")
        ], width=3),
    ]),

    # Toast for commit feedback
    dbc.Toast(id="commit-toast", header="Decision Recorded", is_open=False,
              duration=4000, icon="success",
              style={"position": "fixed", "top": 10, "right": 10, "zIndex": 9999}),

], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# ── Callbacks ───────────────────────────────────────────────────────────────

@callback(Output("kpi-bar", "children"), Input("refresh-trigger", "data"))
def render_kpis(_):
    try:
        kdf = load_kpis()
        k = kdf.iloc[0]
        return [
            dbc.Col(kpi_card("Avg Days to Code",
                             f"{k['avg_days_to_code']}d",
                             "vs <1 day target", "danger"), width=3),
            dbc.Col(kpi_card("Denial Rate",
                             f"{k['denial_rate_pct']}%",
                             f"{int(k['total_denials'])}/{int(k['total_claims'])} claims", "warning"), width=3),
            dbc.Col(kpi_card("$ Denied (Revenue at Risk)",
                             f"${int(float(k['total_denied_dollars'])):,}",
                             "recoverable with CDI", "danger"), width=3),
            dbc.Col(kpi_card("Encounters to Review",
                             str(int(k['total_encounters'])),
                             "awaiting coder action", "info"), width=3),
        ]
    except Exception as e:
        return [dbc.Col(dbc.Alert(f"KPI load error: {e}", color="danger"))]


@callback(Output("note-queue-body", "children"),
          Input("refresh-trigger", "data"), Input("agent-run", "data"))
def render_note_queue(*_):
    try:
        nq = load_note_queue()
        items = []
        for _, r in nq.iterrows():
            risk = risk_level(r["coding_risk_score"])
            coded = r["run_id"] is not None
            items.append(
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Badge(risk, color=RISK_COLORS[risk], className="me-2",
                                          style={"fontSize": "0.65rem"}),
                                dbc.Badge("COMMITTED", color="primary", className="me-2",
                                          style={"fontSize": "0.65rem"}) if _is_true(r["committed"]) else None,
                                html.Strong(r["encounter_id"], style={"fontSize": "0.85rem"}),
                            ]),
                            html.Div([
                                html.Small(f"{r['service_line']} | {r['encounter_type']}",
                                           className="text-muted"),
                            ]),
                            html.Div([
                                html.Small(f"{r['payer']} | {int(r['num_codes'])} codes" if coded
                                           else f"{r['payer']} | signed, not yet coded",
                                           className="text-muted"),
                            ]),
                        ], width=8),
                        dbc.Col([
                            html.Div(f"Risk {float(r['coding_risk_score']):.0f}" if coded else "—",
                                     className="fw-bold text-end", style={"fontSize": "0.85rem"}),
                            html.Small(f"${int(float(r['total_reimbursement'])):,}" if coded else "",
                                       className="text-muted d-block text-end"),
                            html.Small(f"{r['days_to_code']}d baseline",
                                       className="text-muted d-block text-end"),
                        ], width=4),
                    ])
                ], id={"type": "queue-item", "index": r["note_id"]},
                   action=True, style={"cursor": "pointer", "padding": "8px 12px"})
            )
        return dbc.ListGroup(items, flush=True)
    except Exception as e:
        return dbc.Alert(f"Queue load error: {e}", color="danger")


@callback(
    Output("selected-note", "data"),
    Input({"type": "queue-item", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_note(clicks):
    if not ctx.triggered_id or not any(clicks):
        return no_update
    return ctx.triggered_id["index"]


@callback(
    Output("agent-run", "data"),
    Output("commit-toast", "is_open", allow_duplicate=True),
    Output("commit-toast", "children", allow_duplicate=True),
    Input("sign-btn", "n_clicks"),
    State("selected-note", "data"),
    prevent_initial_call=True,
)
def sign_and_code(n, note_id):
    """Simulated clinician signature -> agent codes the masked note -> persisted."""
    if not n or not note_id:
        return no_update, no_update, no_update
    try:
        note = load_note(note_id)
        result = agent.code_note(w, note_id, note["encounter_id"], note["masked_text"],
                                 get_codebook(), load_denial_priors(note["payer"]))
        save_agent_result(result)
        return result["run_id"], True, (f"AI coded {note['encounter_id']} in {result['latency_ms']/1000:.1f}s: "
                                        f"{len(result['codes'])} codes, risk {result['coding_risk_score']:.0f}")
    except Exception as e:
        return no_update, True, f"Agent error: {e}"


@callback(
    Output("code-review-body", "children"),
    Input("selected-note", "data"),
    Input("agent-run", "data"),
)
def render_code_review(note_id, _run):
    if not note_id:
        return html.Div([
            html.I(className="bi bi-arrow-left-circle", style={"fontSize": "2rem"}),
            html.P("Select a note from the queue to begin review.",
                   className="text-muted mt-2")
        ], className="text-center py-5")

    try:
        note = load_note(note_id)
        if note is None:
            return dbc.Alert("Note not found.", color="warning")
        run, df = load_latest_run(note_id)
        text = note["masked_text"] or ""

        header = html.Div([
            html.Div([
                html.H5(note["encounter_id"], className="mb-0 fw-bold"),
                html.Small(f"{note['service_line']} | {note['encounter_type']} | {note['payer']} | note {note_id}",
                           className="text-muted"),
            ]),
            dbc.Button("↻ Re-run AI coding" if run is not None else "✍️ Sign note & run AI coding",
                       id="sign-btn", color="secondary" if run is not None else "primary",
                       size="sm", className="ms-auto"),
        ], className="d-flex align-items-start mb-3")

        if run is None:
            return html.Div([
                header,
                dbc.Alert("Signed note not yet coded. Click “Sign note & run AI coding”: the agent "
                          "reads the PII-masked note, maps entities to ICD-10/CPT and cites evidence for every code.",
                          color="info", style={"fontSize": "0.85rem"}),
                dbc.Card([dbc.CardHeader(html.Small("Masked Clinical Note", className="fw-bold")),
                          dbc.CardBody(highlighted_note(text, []))], className="mb-3"),
            ])

        gaps = json.loads(run["cdi_gaps"] or "[]")
        spans = [(_int_or_none(r["evidence_start"]), _int_or_none(r["evidence_end"]),
                  f"ev-{i}", r["code"], "ev-code") for i, (_, r) in enumerate(df.iterrows())]
        spans += [(g.get("evidence_start"), g.get("evidence_end"), f"gap-{j}", f"CDI {j+1}", "ev-gap")
                  for j, g in enumerate(gaps)]

        note_section = dbc.Card([
            dbc.CardHeader(html.Small(["Masked Clinical Note — click a code's evidence to jump to it"],
                                      className="fw-bold")),
            dbc.CardBody(highlighted_note(text, spans))
        ], className="mb-3", style={"border": "1px solid #dee2e6"})

        code_rows = []
        for i, (_, r) in enumerate(df.iterrows()):
            conf = float(r["confidence"])
            tier = confidence_tier(conf)
            verified = _is_true(r["evidence_verified"])
            denial_rate = r.get("denial_rate")
            denial = float(denial_rate) if denial_rate is not None and pd.notna(denial_rate) else None
            denial_reason = r.get("top_denial_reason") or ""
            flags = [f for f, on in [("Principal Dx", _is_true(r["principal"])),
                                     ("Unverified evidence", not verified),
                                     ("Unspecified code", "unspecified" in str(r["code_description"]).lower()),
                                     (f"Payer denial {denial*100:.0f}%" if denial else "", bool(denial and denial > 0.2))] if on]

            code_rows.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Code(r["code"], style={"fontSize": "1rem", "fontWeight": "bold"}),
                                    html.Span(f" ({r['code_system']})", className="text-muted ms-1",
                                              style={"fontSize": "0.75rem"}),
                                    dbc.Badge(f"Confidence: {conf*100:.0f}%", color=TIER_COLORS[tier],
                                              className="ms-2", style={"fontSize": "0.65rem"}),
                                ]),
                                html.Div(r["code_description"], style={"fontSize": "0.8rem"}, className="mt-1"),
                            ], width=6),
                            dbc.Col([
                                html.Div(f"${int(float(r['avg_reimbursement'])):,}",
                                         className="fw-bold", style={"fontSize": "0.9rem"}),
                                html.Small(f"Denial risk: {denial*100:.0f}%" if denial is not None else "Denial risk: N/A",
                                           className="text-danger" if denial and denial > 0.2 else "text-muted"),
                                html.Br(),
                                html.Small(denial_reason.replace("_", " ").title(), className="text-muted"),
                            ], width=3, className="text-end"),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("✓", color="success", size="sm",
                                               id={"type": "accept-btn", "index": f"{note_id}|{r['code']}"},
                                               title="Accept"),
                                    dbc.Button("✎", color="warning", size="sm",
                                               id={"type": "override-btn", "index": f"{note_id}|{r['code']}"},
                                               title="Override"),
                                    dbc.Button("✗", color="danger", size="sm",
                                               id={"type": "reject-btn", "index": f"{note_id}|{r['code']}"},
                                               title="Reject"),
                                ], size="sm")
                            ], width=3, className="text-end"),
                        ]),
                        html.Div([
                            dbc.Badge(f, color="dark", className="me-1", style={"fontSize": "0.6rem"})
                            for f in flags
                        ], className="mt-1") if flags else None,
                        # Evidence snippet: anchor link scrolls to + highlights the span in the note
                        html.Div([
                            html.Small("Evidence: ", className="fw-bold text-muted"),
                            html.A(f"“{r['evidence_quote']}”", href=f"#ev-{i}",
                                   className="text-dark fst-italic") if verified else
                            html.Small(f"“{r['evidence_quote']}” (not found verbatim in note)",
                                       className="text-danger fst-italic"),
                        ], className="mt-1",
                           style={"backgroundColor": "#fff3cd", "padding": "4px 8px",
                                  "borderRadius": "4px", "fontSize": "0.75rem"}),
                        html.Div(html.Small(f"Why: {r['rationale']}", className="text-muted"),
                                 className="mt-1") if r["rationale"] else None,
                    ], className="py-2")
                ], className="mb-2", style={"border": "1px solid #dee2e6"})
            )

        components = json.loads(run["risk_components"] or "{}")
        score = float(run["coding_risk_score"])
        risk = risk_level(score)
        score_card = dbc.Alert([
            html.Div([
                html.Strong(f"Coding Risk Score: {score:.0f}/100"),
                dbc.Badge(risk, color=RISK_COLORS[risk], className="ms-2"),
                AI_DISCLAIMER,
            ]),
            html.Small(" · ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in components.items()),
                       className="text-muted d-block mt-1"),
            html.Small(f"{run['model_endpoint']} · {run['prompt_version']} · {int(float(run['latency_ms']))/1000:.1f}s"
                       + (f" · rejected out-of-codebook: {', '.join(json.loads(run['rejected_codes']))}"
                          if json.loads(run["rejected_codes"] or "[]") else ""),
                       className="text-muted d-block"),
        ], color=RISK_COLORS[risk], className="py-2 mb-3")

        return html.Div([
            header,
            score_card,
            note_section,
            html.H6([f"AI-Suggested Codes ({len(df)})", AI_DISCLAIMER], className="fw-bold mb-2"),
            html.Div(code_rows),
        ])
    except Exception as e:
        return dbc.Alert(f"Error loading note: {e}", color="danger")


@callback(
    Output("cdi-panel-body", "children"),
    Input("selected-note", "data"),
    Input("agent-run", "data"),
)
def render_cdi_panel(note_id, _run):
    if not note_id:
        return html.P("Select a note to see CDI details.", className="text-muted")

    try:
        runs = run_query(f"SELECT cdi_gaps FROM ({LATEST_RUNS}) WHERE note_id = :note_id", {"note_id": note_id})
        if runs.empty:
            return html.P("Run AI coding to identify documentation gaps.", className="text-muted")
        gaps = json.loads(runs.iloc[0]["cdi_gaps"] or "[]")
        query_text = "\n\n".join(g["physician_query"] for g in gaps if g.get("physician_query"))

        gap_cards = [
            dbc.Alert([
                html.H6([f"CDI {j+1}: ", html.A("view in note", href=f"#gap-{j}", className="small")],
                        className="fw-bold mb-1"),
                html.P(g.get("gap", ""), className="mb-1", style={"fontSize": "0.8rem"}),
                html.Small(f"Impact: {g.get('impact', '')}", className="text-muted"),
            ], color="warning", className="mb-2 py-2")
            for j, g in enumerate(gaps)
        ] or [dbc.Alert("No documentation gaps identified by the agent.", color="success")]

        return html.Div([
            html.H6(["CDI Documentation Gaps", AI_DISCLAIMER], className="fw-bold mb-2"),
            *gap_cards,

            html.Hr(),

            # CDI Query composer (pre-filled with the agent's non-leading physician queries)
            html.H6("Compose CDI Query to Physician", className="fw-bold mb-2"),
            dbc.Textarea(
                id="cdi-query-text",
                value=query_text,
                placeholder="Type a CDI query for the attending physician...",
                style={"fontSize": "0.8rem", "minHeight": "140px"},
                className="mb-2",
            ),

            html.Hr(),

            # Override section
            html.H6("Override Code (optional)", className="fw-bold mb-2"),
            dbc.Input(id="override-code", placeholder="e.g. R65.21",
                      size="sm", className="mb-1"),
            dbc.Input(id="override-reason", placeholder="Override reason...",
                      size="sm", className="mb-3"),

            # Commit button
            dbc.Button(
                "✅ Commit All Decisions",
                id="commit-btn", color="primary", size="lg",
                className="w-100 fw-bold",
            ),

            html.Hr(),

            # Recent decisions
            html.H6("Recent Decisions", className="fw-bold mb-2"),
            html.Div(id="recent-decisions"),
        ])
    except Exception as e:
        return dbc.Alert(f"CDI panel error: {e}", color="danger")


@callback(
    [Output("commit-toast", "is_open"), Output("commit-toast", "children"),
     Output("recent-decisions", "children"), Output("refresh-trigger", "data")],
    Input("commit-btn", "n_clicks"),
    [State("selected-note", "data"),
     State("cdi-query-text", "value"),
     State("override-code", "value"),
     State("override-reason", "value"),
     State("refresh-trigger", "data")],
    prevent_initial_call=True,
)
def commit_decisions(n, note_id, cdi_text, override_code, override_reason, refresh):
    if not note_id:
        return False, "", no_update, no_update

    try:
        run, detail_df = load_latest_run(note_id)
        if run is None:
            return True, "Run AI coding before committing.", no_update, no_update
        decision_type = "override" if override_code else "accept"
        override_code = (override_code or "").strip().upper()
        book = {c["code"]: c for c in get_codebook()}
        override_desc = book.get(override_code, {}).get("description", "")
        ts = datetime.now(timezone.utc).isoformat()
        coder_id = current_user()

        for _, r in detail_df.iterrows():
            conf = float(r["confidence"])
            delta = ""
            if override_code:
                d = int(book.get(override_code, {}).get("avg_reimbursement") or 0) - int(float(r["avg_reimbursement"]))
                delta = f"{r['code']}->{override_code} ({d:+,} USD)"
            run_stmt(f"""
                INSERT INTO {DECISIONS_TABLE} VALUES (
                    :decision_id, :encounter_id, :note_id, :code, :code_system, :code_description,
                    :confidence, :tier, :flags, :evidence, :reimb, :decision, :override_code,
                    :override_desc, :override_reason, :cdi_text, :coder_id, :ts, :delta)
            """, {
                "decision_id": str(uuid.uuid4())[:12], "encounter_id": r["encounter_id"], "note_id": note_id,
                "code": r["code"], "code_system": r["code_system"], "code_description": r["code_description"],
                "confidence": (round(conf * 100, 1), "DOUBLE"), "tier": confidence_tier(conf),
                "flags": "" if _is_true(r["evidence_verified"]) else "Unverified evidence",
                "evidence": str(r["evidence_quote"] or "")[:500],
                "reimb": (int(float(r["avg_reimbursement"])), "BIGINT"),
                "decision": decision_type, "override_code": override_code, "override_desc": override_desc,
                "override_reason": override_reason or "", "cdi_text": cdi_text or "",
                "coder_id": coder_id, "ts": (ts, "TIMESTAMP"), "delta": delta,
            })

        # Load recent decisions
        recent = load_decisions()
        rows = []
        for _, d in recent.head(5).iterrows():
            rows.append(html.Div([
                html.Small(f"{d['encounter_id']} | {d['ai_suggested_code']} | {d['coder_decision']}",
                           className="text-muted"),
            ], className="mb-1"))

        msg = f"Committed {len(detail_df)} code decisions for {run['encounter_id']}"
        if cdi_text:
            msg += " + CDI query recorded"
        return True, msg, html.Div(rows) if rows else html.Small("None yet"), (refresh or 0) + 1
    except Exception as e:
        return True, f"Error: {e}", no_update, no_update


# Accept/Override/Reject individual button callbacks (pattern-matching)
@callback(
    Output("commit-toast", "is_open", allow_duplicate=True),
    Output("commit-toast", "children", allow_duplicate=True),
    Input({"type": "accept-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def handle_accept(clicks):
    if not any(clicks):
        return no_update, no_update
    triggered = ctx.triggered_id
    if triggered:
        parts = triggered["index"].split("|")
        note_id, code = parts[0], parts[1]
        return True, f"Accepted {code} for {note_id}"
    return no_update, no_update


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_tables()
    port = int(os.environ.get("PORT", os.environ.get("DATABRICKS_APP_PORT", 8050)))
    app.run(host="0.0.0.0", port=port, debug=False)
