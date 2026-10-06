"""CDI coding agent.

Takes a PII-masked clinical note, asks a Foundation Model endpoint to extract
clinical entities and map them to codes from the codebook, then grounds every
suggestion deterministically:
  * codes outside the codebook are rejected (no hallucinated codes reach a coder)
  * every evidence quote is located verbatim in the note -> char offsets for highlighting
  * a Coding Risk Score is computed in code (explainable), not by the LLM

Shared by the Dash app (live "sign note") and the batch backfill notebook.
"""
import json, os, re, time, uuid
from datetime import datetime, timezone

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

try:
    import mlflow
    from mlflow.entities import SpanType
except ImportError:  # tracing is optional
    mlflow = None

MODEL_ENDPOINT = os.environ.get("CDI_AGENT_ENDPOINT", "databricks-claude-sonnet-5")
PROMPT_VERSION = "cdi-coder-v1"
UNVERIFIED_CONFIDENCE_CAP = 0.5

SYSTEM_PROMPT = """You are an expert clinical coder and CDI (Clinical Documentation Integrity) specialist.
You review a signed clinical note whose PHI has been masked (tokens like [PATIENT], [CLINICIAN], <PERSON>).

Tasks:
1. Extract the clinical entities: diagnoses, symptoms, medications, procedures.
2. Assign codes, using ONLY codes from the CODEBOOK provided. Never invent codes.
   - Code only what the clinician documented; do not code from suspicion alone.
   - Prefer the most specific code the documentation supports.
   - Include CPT codes for documented procedures / E&M services when present in the codebook.
3. For EVERY code give `evidence_quote`: an EXACT, verbatim, contiguous substring copied from the note
   (5-25 words) that justifies the code. Do not paraphrase, do not stitch fragments.
4. Give `confidence` 0.0-1.0 that the code is correct AND fully supported by the documentation.
5. Identify CDI documentation gaps: places where vague, missing or conflicting documentation
   prevents a more specific / higher-acuity code or creates denial risk (e.g. sepsis without
   organ dysfunction documented, missing antibiotic start time, heart failure type/acuity unspecified).
   For each gap write a compliant, NON-LEADING physician query (AHIMA/ACDIS style: present the
   clinical indicators, offer options including "other" and "unable to determine").

Respond with JSON only, no prose, matching exactly:
{
  "entities": [{"type": "diagnosis|symptom|medication|procedure", "text": "..."}],
  "codes": [{"code": "...", "code_system": "ICD-10|CPT", "evidence_quote": "...",
             "rationale": "one sentence", "confidence": 0.0, "principal": false}],
  "cdi_gaps": [{"gap": "...", "impact": "coding / reimbursement / denial impact",
                "evidence_quote": "...", "physician_query": "..."}]
}"""


# ── Tracing ─────────────────────────────────────────────────────────────────
_tracing_ready = False

def init_tracing() -> bool:
    """Send traces to the MLflow experiment in MLFLOW_EXPERIMENT_ID (bound to a UC trace location)."""
    global _tracing_ready
    if _tracing_ready or mlflow is None or not os.environ.get("MLFLOW_EXPERIMENT_ID"):
        return _tracing_ready
    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"])
        mlflow.config.enable_async_logging(True)
        _tracing_ready = True
    except Exception as e:
        print(f"[agent] MLflow tracing disabled: {e}")
    return _tracing_ready


class _NoSpan:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def set_inputs(self, *a, **k): pass
    def set_outputs(self, *a, **k): pass
    def set_attributes(self, *a, **k): pass

def _span(name, span_type):
    if _tracing_ready:
        return mlflow.start_span(name=name, span_type=span_type)
    return _NoSpan()


# ── Grounding helpers ───────────────────────────────────────────────────────
def locate_quote(quote: str, text: str):
    """Return (start, end) of `quote` in `text`, tolerant to case and whitespace; None if absent."""
    if not quote or not text:
        return None
    q = quote.strip().strip('"\'').strip()
    if not q:
        return None
    i = text.find(q)
    if i >= 0:
        return i, i + len(q)
    words = [re.escape(t) for t in q.split()]
    m = re.search(r"\s+".join(words), text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    # LLMs sometimes drop trailing punctuation or a leading word; retry on the longest inner run
    if len(words) > 6:
        m = re.search(r"\s+".join(words[1:-1]), text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None


def _parse_json(content: str) -> dict:
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1)
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start:end + 1])


def coding_risk_score(codes, gaps, denial_priors=None):
    """0-100, higher = chart needs coder attention first. Returns (score, components)."""
    denial_priors = denial_priors or {}
    if codes:
        weights = [max(c["avg_reimbursement"] or 0, 1) for c in codes]
        conf = sum(w * c["confidence"] for w, c in zip(weights, codes)) / sum(weights)
        denial = max(float(denial_priors.get(c["code"], 0) or 0) for c in codes)
        unverified = sum(not c["evidence_verified"] for c in codes) / len(codes)
    else:
        conf, denial, unverified = 0.0, 0.0, 1.0
    components = {
        "low_confidence": round((1 - conf) * 40, 1),
        "cdi_gaps": round(min(len(gaps), 3) / 3 * 25, 1),
        "payer_denial_prior": round(min(denial / 0.5, 1) * 25, 1),
        "unverified_evidence": round(unverified * 10, 1),
    }
    return round(sum(components.values()), 1), components


# ── Agent ───────────────────────────────────────────────────────────────────
def _call_llm(w: WorkspaceClient, note_text: str, codebook: list):
    codebook_txt = "\n".join(f"{c['code']} | {c['code_system']} | {c['description']}" for c in codebook)
    user = f"CODEBOOK (code | system | description):\n{codebook_txt}\n\nSIGNED CLINICAL NOTE (PHI masked):\n<<<\n{note_text}\n>>>"
    with _span("llm_extract_and_code", SpanType.CHAT_MODEL if mlflow else None) as span:
        span.set_inputs({"endpoint": MODEL_ENDPOINT, "prompt_version": PROMPT_VERSION, "note_chars": len(note_text)})
        resp = w.serving_endpoints.query(
            name=MODEL_ENDPOINT,
            messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT),
                      ChatMessage(role=ChatMessageRole.USER, content=user)],
            max_tokens=4000,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):  # reasoning models return [{type: reasoning}, {type: text}]
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        usage = resp.usage.as_dict() if resp.usage else {}
        span.set_outputs({"content": content, "usage": usage})
    return content, usage


def code_note(w: WorkspaceClient, note_id: str, encounter_id: str, note_text: str,
              codebook: list, denial_priors: dict | None = None) -> dict:
    """Run the agent on one masked note. `codebook`: list of {code, code_system, description, avg_reimbursement}.
    `denial_priors`: {code: denial_rate} for this encounter's payer."""
    init_tracing()
    book = {c["code"]: c for c in codebook}
    t0 = time.time()
    with _span("cdi_code_note", SpanType.AGENT if mlflow else None) as root:
        root.set_inputs({"note_id": note_id, "encounter_id": encounter_id})
        content, usage = _call_llm(w, note_text, codebook)
        try:
            raw = _parse_json(content)
        except Exception as e:
            raise RuntimeError(f"Agent returned non-JSON output: {e}: {content[:300]}")

        with _span("ground_and_score", SpanType.PARSER if mlflow else None) as gspan:
            codes, rejected, seen = [], [], set()
            for c in raw.get("codes", []):
                code = str(c.get("code", "")).strip().upper()
                if code not in book:
                    rejected.append(code)
                    continue
                if code in seen:
                    continue
                seen.add(code)
                span_pos = locate_quote(c.get("evidence_quote", ""), note_text)
                conf = max(0.0, min(1.0, float(c.get("confidence", 0) or 0)))
                if span_pos is None:
                    conf = min(conf, UNVERIFIED_CONFIDENCE_CAP)
                codes.append({
                    "code": code,
                    "code_system": book[code]["code_system"],
                    "code_description": book[code]["description"],
                    "avg_reimbursement": int(book[code]["avg_reimbursement"] or 0),
                    "confidence": round(conf, 3),
                    "principal": bool(c.get("principal", False)),
                    "rationale": c.get("rationale", ""),
                    "evidence_quote": note_text[span_pos[0]:span_pos[1]] if span_pos else c.get("evidence_quote", ""),
                    "evidence_start": span_pos[0] if span_pos else None,
                    "evidence_end": span_pos[1] if span_pos else None,
                    "evidence_verified": span_pos is not None,
                })
            codes.sort(key=lambda c: (not c["principal"], -c["avg_reimbursement"]))

            gaps = []
            for g in raw.get("cdi_gaps", []):
                pos = locate_quote(g.get("evidence_quote", ""), note_text)
                gaps.append({
                    "gap": g.get("gap", ""), "impact": g.get("impact", ""),
                    "physician_query": g.get("physician_query", ""),
                    "evidence_quote": g.get("evidence_quote", ""),
                    "evidence_start": pos[0] if pos else None,
                    "evidence_end": pos[1] if pos else None,
                })

            score, components = coding_risk_score(codes, gaps, denial_priors)
            gspan.set_outputs({"n_codes": len(codes), "rejected_codes": rejected, "risk_score": score})

        result = {
            "run_id": str(uuid.uuid4()),
            "note_id": note_id,
            "encounter_id": encounter_id,
            "codes": codes,
            "cdi_gaps": gaps,
            "entities": raw.get("entities", []),
            "rejected_codes": rejected,
            "coding_risk_score": score,
            "risk_components": components,
            "model_endpoint": MODEL_ENDPOINT,
            "prompt_version": PROMPT_VERSION,
            "latency_ms": int((time.time() - t0) * 1000),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if _tracing_ready:
            result["trace_id"] = mlflow.get_active_span().trace_id if mlflow.get_active_span() else None
        root.set_outputs({k: result[k] for k in ("coding_risk_score", "risk_components", "rejected_codes", "latency_ms")})
    return result


# ── Persistence rows (same shape for app + batch notebook) ──────────────────
RUNS_COLUMNS = ["run_id", "note_id", "encounter_id", "coding_risk_score", "risk_components",
                "cdi_gaps", "entities", "rejected_codes", "model_endpoint", "prompt_version",
                "latency_ms", "input_tokens", "output_tokens", "trace_id", "created_at"]
SUGGESTION_COLUMNS = ["suggestion_id", "run_id", "note_id", "encounter_id", "code", "code_system",
                      "code_description", "avg_reimbursement", "confidence", "principal", "rationale",
                      "evidence_quote", "evidence_start", "evidence_end", "evidence_verified", "created_at"]

def to_rows(result: dict):
    run = {k: result.get(k) for k in RUNS_COLUMNS}
    for k in ("risk_components", "cdi_gaps", "entities", "rejected_codes"):
        run[k] = json.dumps(result.get(k))
    sugg = [{"suggestion_id": str(uuid.uuid4()), "run_id": result["run_id"], "note_id": result["note_id"],
             "encounter_id": result["encounter_id"], "created_at": result["created_at"],
             **{k: c[k] for k in SUGGESTION_COLUMNS if k in c}} for c in result["codes"]]
    return run, sugg
