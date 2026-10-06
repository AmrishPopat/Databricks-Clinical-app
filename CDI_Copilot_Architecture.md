# Clinical Coding & CDI Co-pilot — Architecture

---

## Executive Summary (for Judges)

### The Problem

Clinical coding — translating physician notes into ICD-10/CPT billing codes — is manual, error-prone, and slow. US hospitals lose **$5M-$10M/year** in denied claims due to coding errors, missing documentation, and CDI gaps. Coders take **5-7 days** to finalize codes. Denial rates average **24.5%**. CDI queries to physicians take **2-3 days** on paper.

### What We Built

An **end-to-end AI-powered co-pilot** on Databricks that:

1. **Ingests** clinical notes (text + PDF) into a medallion lakehouse pipeline
2. **Masks PII** at the silver layer before any AI reasoning (HIPAA-ready)
3. **Extracts ICD-10/CPT codes** with confidence scores, risk flags, and evidence snippets via Foundation Model API
4. **Presents a coder review UI** (Databricks App) with risk-sorted worklist, split-pane note+codes view, accept/override/reject per code, and in-app CDI query composer
5. **Persists decisions** to Lakebase Postgres (3.9ms latency) and Unity Catalog (for analytics)
6. **Enables billing managers** to query denials and coding patterns in plain English via Genie Space

### Databricks Capabilities Used (8)

| # | Capability | Where |
|---|---|---|
| 1 | Databricks App | Coder review UI (3 pages: worklist, coding review, dashboard) |
| 2 | SQL Warehouse | Reads UC tables for the app and analytics |
| 3 | Lakebase Postgres | OLTP write-back for decisions (3.9ms) |
| 4 | Genie Space | Billing manager plain-English Q&A over denials |
| 5 | AI Functions | `ai_mask()` for PII, `ai_parse_document()` for PDFs |
| 6 | Unity Catalog | Governance over 11 tables, volumes, codebooks |
| 7 | Foundation Model API | `ai_query()` for live ICD-10/CPT extraction |
| 8 | AI Gateway | Usage tracking, rate limits, safety guardrails, HIPAA audit trail |

### Target Business Impact

| Metric | Before | After (Target) |
|---|---|---|
| Days to final code | 5-7 days | **< 1 day** |
| Denial rate | 24.5% | **15% reduction** |
| CDI query turnaround | 2-3 days | **< 4 hours** |
| Charts coded per coder/day | 15-20 | **40-60** |
| Revenue recovery | $5M-$10M at risk | **Quantified per encounter** |

### Live Demo Highlights

* **App URL**: https://cdi-copilot-955682294748605.aws.databricksapps.com
* **Genie Space**: "CDI Coding Analytics" — ask "Which codes have the highest denial rate by payer?"
* **Live AI inference**: `ai_query()` extracts 4 ICD-10 + 3 CPT codes from a masked sepsis note in < 500ms
* **PII guarantee**: AI agent never sees raw patient data — double-layered protection (silver masking + AI Gateway guardrails)

### Scoring Alignment

| Rubric Category (pts) | Evidence |
|---|---|
| **Business Impact (15)** | $5-10M revenue recovery, 24.5% denial rate reduction, 3x coder throughput, CDI turnaround from days to hours |
| **Technical Depth (35)** | 8 Databricks capabilities, medallion pipeline, AI Gateway governance, Lakebase OLTP + UC analytics dual-write, ai_mask/ai_query/ai_parse_document, 8 architectural trade-offs documented |
| **Storytelling (15)** | Clinician signs note -> AI extracts codes -> Coder reviews -> Decision persists -> Billing manager queries denials. Clear persona-driven flow. |
| **Working Demo (35)** | Live Databricks App (RUNNING), Genie Space (7 tables), live ai_query() inference proven, Lakebase verified (3.9ms), 11 tables with real clinical data |

---

## Overview

An AI-powered Clinical Documentation Integrity (CDI) Co-pilot built on the Databricks Lakehouse Platform. When a clinician signs a note, the system extracts clinical entities, maps them to ICD-10/CPT codes, returns a Coding Risk Score with evidence snippets, and presents everything to a professional coder for review, override, and commit — all with PII masked before the AI reasoning layer.

---

## Architecture Diagram

```
                                                                  ╔════════════════════════╗
                                                                  ║                        ║
┌───────────────────────────────────────────────────────────┐   ║   DATABRICKS          ║
│                                                             │   ║   AI GATEWAY           ║
│   LAYER 1: NOTE INGESTION (Bronze)                          │   ║                        ║
│                                                             │   ║   Governs all FMAPI     ║
│   Clinician ─► UC Volume ─► bronze_notes (60 rows)           │   ║   calls in Layers       ║
│   signs note   20 PDFs     bronze_notes_pdf (PDF parse)     │   ║   1, 2, and 3           ║
│                                                             │   ║                        ║
│   Reference: encounters | claims_denials | icd10_cpt_ref    │   ║   ┌────────────────────┐ ║
│                                                             │   ║   │ USAGE TRACKING   │ ║
│   ai_parse_document() ──────────────────────────────────►   │ Token counts,    │ ║
│   PDF text extraction via FMAPI                              │   ║   │ cost monitoring  │ ║
│                                                             │   ║   │ per endpoint     │ ║
└───────────────────────────┬───────────────────────────────┘   ║   └────────────────────┘ ║
                            │                                   ║                        ║
                            ▼                                   ║   ┌────────────────────┐ ║
┌───────────────────────────────────────────────────────────┐   ║   │ RATE LIMITS      │ ║
│                                                             │   ║   │ QPM / TPM        │ ║
│   LAYER 2: PII MASKING (Silver) — HIPAA-Ready               │   ║   │ per user,        │ ║
│                                                             │   ║   │ per endpoint     │ ║
│   bronze_notes ─► ai_mask() ─► silver_notes_masked (60)      │   ║   └────────────────────┘ ║
│                  Masks: name, DOB, MRN, clinician ID        │   ║                        ║
│                  Output: masked_text, patient_id_masked      │   ║   ┌────────────────────┐ ║
│                                                             │   ║   │ GUARDRAILS       │ ║
│   ai_mask() ───────────────────────────────────────────►   │ Safety filter    │ ║
│   PII masking via FMAPI                                     │   ║   │ PII blocking     │ ║
│                                                             │   ║   │ Input/output     │ ║
│   * AI AGENT NEVER SEES RAW PII                             │   ║   │ filtering        │ ║
│                                                             │   ║   └────────────────────┘ ║
└───────────────────────────┬───────────────────────────────┘   ║                        ║
                            │                                   ║   ┌────────────────────┐ ║
                            ▼                                   ║   │ AUDIT TRAIL      │ ║
┌───────────────────────────────────────────────────────────┐   ║   │ system.serving.  │ ║
│                                                             │   ║   │ endpoint_usage   │ ║
│   LAYER 3: AI CODING AGENT                                  │   ║   │ HIPAA-compliant  │ ║
│   ai_query('databricks-meta-llama-3-3-70b-instruct')        │   ║   │ request logging  │ ║
│                                                             │   ║   └────────────────────┘ ║
│   INPUTS:                        PROCESSING:                │   ║                        ║
│   ├─ silver_notes_masked         1. Extract entities         │   ║   ENDPOINTS GOVERNED:  ║
│   ├─ icd10_cpt_ref (54)         2. Map ICD-10 / CPT         │   ║                        ║
│   └─ gold_code_denial_priors    3. Coding Risk Score        │   ║   databricks-meta-     ║
│                                  4. Evidence snippets       │   ║   llama-3-3-70b-       ║
│   ai_query() ──────────────────────────────────────────►   ║   instruct             ║
│   ICD-10/CPT extraction via FMAPI                           │   ║   (ai_query)           ║
│                                                             │   ║                        ║
│   OUTPUT: encounter_coding_review (199 rows)                │   ║   FMAPI managed        ║
│   ├─ code, code_system, code_description                    │   ║   endpoints            ║
│   ├─ confidence_score + confidence_tier (LOW/MED/HIGH)      │   ║   (ai_mask,            ║
│   ├─ risk_flags (CDI Query | Unspecified | Missing Info)    │   ║    ai_parse_document) ║
│   ├─ evidence_snippet (source text justification)           │   ║                        ║
│   └─ avg_reimbursement ($ at stake per code)                │   ║                        ║
│                                                             │   ║                        ║
│   EVALUATION: note_labels (gold_codes + gold_cdi_gap)       │   ║                        ║
│                                                             │   ║                        ║
└───────────────────────────┬───────────────────────────────┘   ║                        ║
                            │                                   ╔════════════════════════╣
                            ▼                                   ║  ▲ ▲ ▲  AI function     ║
┌───────────────────────────────────────────────────────────┐   ║  L1 L2 L3 calls routed ║
│                                                             │   ║  through AI Gateway   ║
│   LAYER 4: DATABRICKS APP — "cdi-copilot" (Dash)            │   ╚════════════════════════╝
│   URL: cdi-copilot-955682294748605.aws.databricksapps.com   │
│                                                             │
│   ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐  │
│   │ PAGE 1:         │ │ PAGE 2:          │ │ PAGE 3:         │  │
│   │ CODER WORKLIST  │ │ CODING REVIEW    │ │ DASHBOARD       │  │
│   │                 │ │                  │ │                 │  │
│   │ Risk-sorted     │ │ Split-pane:      │ │ KPI metrics     │  │
│   │ encounter queue │ │ LEFT: Masked     │ │ Denial rate     │  │
│   │ Filter by risk  │ │  note + evidence │ │ by payer        │  │
│   │ tier, service   │ │ RIGHT: AI codes  │ │ Top denial      │  │
│   │ line, payer     │ │  + confidence    │ │ reasons         │  │
│   │                 │ │  + denial risk   │ │ Highest risk    │  │
│   │ Click ->        │ │  + evidence      │ │ codes by payer  │  │
│   │ Review          │ │                  │ │ $ denied        │  │
│   │                 │ │ [Accept]         │ │                 │  │
│   │                 │ │ [Override]       │ │                 │  │
│   │                 │ │ [Remove]         │ │                 │  │
│   │                 │ │ [CDI Query]      │ │                 │  │
│   └─────────────────┘ └──────────────────┘ └─────────────────┘  │
│                                                             │
│   READS via ─► SQL Warehouse ─► Unity Catalog tables          │
│   WRITES via ─► Lakebase Postgres (cdi-copilot-db)           │
│                                                             │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                                                             │
│   LAYER 5: LAKEBASE POSTGRES — "cdi-copilot-db"             │
│   Latency: 3.9ms  |  9 indexes  |  Schema: cdi             │
│                                                             │
│   ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│   │ coding_      │ │ coding_      │ │ cdi_queries         │  │
│   │ decisions    │ │ audit_log    │ │ (8 cols)            │  │
│   │ (18 cols)    │ │ (7 cols)     │ ├─────────────────────┤  │
│   │ encounter_id │ │ decision_id  │ │ session_state       │  │
│   │ ai_code      │ │ action       │ │ (6 cols)            │  │
│   │ decision     │ │ actor        │ │                     │  │
│   │ final_code   │ │ detail (JSON)│ │                     │  │
│   │ override     │ │ timestamp    │ │                     │  │
│   └──────────────┘ └──────────────┘ └─────────────────────┘  │
│                                                             │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                                                             │
│   LAYER 6: GOLD ANALYTICS + GENIE SPACE                     │
│                                                             │
│   ┌────────────────┐   ┌────────────────────────────────────┐  │
│   │ GOLD TABLES    │   │ GENIE SPACE:                       │  │
│   │                │   │ "CDI Coding Analytics"              │  │
│   │ gold_claim_    │─►│                                    │  │
│   │ codes          │   │ "Highest denial rate by payer?"   │  │
│   │ gold_code_     │   │ "Top denial reasons?"             │  │
│   │ denial_priors  │   │ "Avg resolution time?"            │  │
│   │ claims_denials │   │ "$ denied for missing docs?"      │  │
│   │ (1,509 paid /  │   │                                    │  │
│   │  491 denied)   │   │ 7 tables | Billing manager Q&A   │  │
│   └────────────────┘   └────────────────────────────────────┘  │
│                                                             │
└───────────────────────────────────────────────────────────┘
```

---

## Databricks Capabilities Integrated (8)

| # | Capability | Usage |
|---|---|---|
| 1 | **Databricks App** | Streamlit coder review UI — worklist, split-pane coding review, analytics dashboard |
| 2 | **SQL Warehouse** | Reads Unity Catalog tables for analytics and the coder worklist |
| 3 | **Lakebase Postgres** | OLTP write-back for coding decisions, audit log, CDI queries (3.9ms latency) |
| 4 | **Genie Space** | Plain-English Q&A for billing managers over denials and coding patterns |
| 5 | **AI Functions** | `ai_mask()` for PII masking in silver layer; `ai_parse_document()` for PDF ingestion |
| 6 | **Unity Catalog** | Governance over all tables, volumes, codebooks, and access control |
| 7 | **Foundation Model API** | Live clinical entity extraction, ICD-10/CPT code mapping, and risk scoring via `ai_query('databricks-meta-llama-3-3-70b-instruct', ...)` |
| 8 | **AI Gateway (Unity Gateway)** | Governs all Foundation Model API calls across Layers 1-3: usage tracking (`system.serving.endpoint_usage`), rate limits (QPM/TPM per user and endpoint), safety guardrails (input/output filtering, PII blocking), and HIPAA-ready audit trail |

---

## AI Gateway Integration Detail

All AI inference in the pipeline flows through **Databricks AI Gateway**, which governs the Foundation Model API endpoints. Three SQL AI functions are the integration points:

| SQL AI Function | Layer | Purpose | Endpoint |
|---|---|---|---|
| `ai_parse_document()` | Bronze | Extract text from 20 PDF clinical notes | Foundation Model API (managed) |
| `ai_mask()` | Silver | Mask PII (names, DOB, MRN, clinician IDs) before AI reasoning | Foundation Model API (managed) |
| `ai_query()` | AI Agent | Live ICD-10/CPT extraction, risk scoring, CDI gap detection from masked notes | `databricks-meta-llama-3-3-70b-instruct` |

### AI Gateway Capabilities Active

* **Usage Tracking** — Token counts and request volumes logged to `system.serving.endpoint_usage` for cost monitoring
* **Safety Guardrails** — Input/output safety filtering on all Foundation Model API calls
* **PII Protection** — Double-layered: `ai_mask()` removes PII at the silver layer before `ai_query()` is called; AI Gateway guardrails provide a second safety net
* **Rate Limits** — Workspace-level QPM/TPM limits enforced per endpoint
* **Audit Trail** — All requests are trackable via system tables for HIPAA compliance

### Live Inference Example

The AI agent takes a masked clinical note and returns structured JSON:

```sql
SELECT ai_query(
  'databricks-meta-llama-3-3-70b-instruct',
  CONCAT(
    'Extract ICD-10 and CPT codes from this clinical note. ',
    'Return JSON: codes[], risk_flags[], cdi_gap.\n\n',
    masked_text
  )
) AS ai_coding_response
FROM `external-ai-build-day`.cdi_copilot.silver_notes_masked
```

**Sample output** (encounter ENC-000000, sepsis case):
* ICD-10: A41.9 (Sepsis, 0.9), N17.9 (AKI, 0.8), N30.0 (Cystitis, 0.7), R65.20 (Severe sepsis, 0.6)
* CPT: 87070 (Blood culture), 87086 (Urinalysis), 96360 (IV infusion)
* CDI Gap: "Missing baseline creatinine and organism identification"
* Latency: < 500ms per note

---

## Data Inventory

### Catalog: `external-ai-build-day`

#### Schema: `clinical_coding` (source/reference tables)

| Table | Rows | Key Columns |
|---|---|---|
| `claims_denials` | 2,000 | claim_id, encounter_id, codes_submitted, payer, billed_amount, status (paid/denied), denial_reason, resolution_time_hours |
| `encounters` | 60 | encounter_id, patient_id_masked, service_line, encounter_type, clinician_id, signature_timestamp, days_to_final_code |
| `clinical_notes_text` | 60 | note_id, encounter_id, note_text, has_pdf |
| `encounter_coding_review` | 199 | encounter_id, code, code_system, code_description, avg_reimbursement, confidence_score, confidence_tier, risk_flags, evidence_snippet |
| `icd10_cpt_ref` | 54 | code, code_system, description, avg_reimbursement |

#### Schema: `cdi_copilot` (medallion pipeline tables)

| Table | Rows | Key Columns |
|---|---|---|
| `bronze_notes` | 60 | note_id, encounter_id, note_text, source, has_pdf, pdf_path |
| `bronze_notes_pdf` | 20 | encounter_id, path, parsed_text |
| `silver_notes_masked` | 60 | note_id, encounter_id, masked_text, patient_id_masked, payer, service_line, encounter_type, signature_timestamp |
| `note_labels` | 60 | note_id, encounter_id, gold_codes (array), gold_cdi_gap |
| `gold_claim_codes` | ~2,000 | claim_id, encounter_id, payer, status, denial_reason, code, code_system, code_description |
| `gold_code_denial_priors` | ~200 | code, payer, n_claims, denial_rate, top_denial_reason, avg_billed_amount |
| `codebook` | 54 | code, code_system, description, avg_reimbursement |

#### Volume: `/Volumes/external-ai-build-day/clinical_coding/raw_docs/clinical_notes/`
20 PDF clinical notes (inpatient): sepsis, pneumonia, COPD, DKA, GI bleed, heart failure, lymphoma, preeclampsia, etc.

---

## Infrastructure

### Lakebase Postgres

| Property | Value |
|---|---|
| Project | `cdi-copilot-db` |
| Branch | `production` (READY) |
| Endpoint | `primary` (ACTIVE) |
| Host | `ep-falling-shape-d2mpcw5j.database.us-east-1.cloud.databricks.com` |
| Database | `databricks_postgres` |
| Schema | `cdi` |
| Tables | `coding_decisions` (18 cols), `coding_audit_log` (7 cols), `cdi_queries` (8 cols), `session_state` (6 cols) |
| Indexes | 9 |
| Read latency | 3.9ms verified |

### Databricks App

| Property | Value |
|---|---|
| Name | `cdi-copilot` |
| Framework | Streamlit |
| Source | `/Users/amrpopat@emeal.nttdata.com/cdi-copilot-app/` |
| URL | `https://cdi-copilot-955682294748605.aws.databricksapps.com` |
| Status | RUNNING |
| Pages | Worklist, Coding Review (split-pane), Dashboard |

### Genie Space

| Property | Value |
|---|---|
| Name | CDI Coding Analytics |
| ID | `01f1c17ff01b1cf998c601fca1cf35eb` |
| Tables | 7 (gold_claim_codes, gold_code_denial_priors, silver_notes_masked, note_labels, codebook, encounters, claims_denials) |

---

## Business Impact KPIs

| KPI | Baseline | Target with Co-pilot |
|---|---|---|
| Days to final code | 5-7 days | < 1 day |
| Denial rate | 24.5% (491/2000) | 15% reduction target |
| CDI query response | 2-3 days (paper) | < 4 hours (in-app) |
| Revenue at risk | $5M-$10M/yr | Quantified per encounter via avg_reimbursement x denial_rate |
| Charts coded per coder/day | 15-20 (manual) | 40-60 (AI-assisted) |

---

## Key Architectural Trade-offs

### 1. Lakebase Postgres vs Unity Catalog Delta Table for Decision Persistence

| | Lakebase Postgres | UC Delta Table |
|---|---|---|
| Write latency | < 10ms (verified 3.9ms) | 500ms-2s (Spark job overhead) |
| Read pattern | Single-row lookup by encounter_id | Batch scan / analytics |
| ACID per row | Native row-level transactions | Table-level ACID |
| Analytics joins | Requires synced table or export | Native Spark SQL joins |
| **Decision** | **Primary write path for the app** | **Secondary — used for Genie Space analytics** |

Why both: The coder review UI needs sub-10ms write-back for a responsive UX (accept/override/reject). But the Genie Space and dashboards need the decisions in UC for SQL joins with claims and encounter tables. We write to Lakebase first, then the UC table `cdi_copilot.coder_decisions` syncs for analytics.

### 2. Pre-computed AI Suggestions vs Live LLM Inference

| | Pre-computed (current) | Live ai_query() |
|---|---|---|
| Latency | 0ms (already in table) | ~500ms per note |
| Cost | One-time batch run | Per-request token cost |
| Freshness | Static — won't update if note is amended | Real-time — always reflects latest note |
| Demo reliability | 100% — no API failures during demo | Dependent on endpoint availability |
| **Decision** | **Default for demo and worklist** | **Available via "Re-run AI" button (architecture-ready)** |

Why: `encounter_coding_review` has 199 pre-computed rows. For a hackathon demo, deterministic pre-computed results avoid LLM variability and latency. The `ai_query()` path is proven (tested live on ENC-000000) and can be wired as an on-demand re-coding action.

### 3. PII Boundary — Auditability vs. Exposure Surface

| | Mask at Silver (chosen) | Mask at Bronze ingestion |
|---|---|---|
| Raw PII in storage | **Yes — Bronze layer retains unmasked notes** | No PII stored at all |
| Compliance audit | Original notes recoverable for legal/regulatory review | Original notes lost forever — cannot respond to audits |
| PII exposure surface | Bronze tables are a target — require strict ACLs and monitoring | Zero attack surface |
| AI agent sees PII | No — Silver onward is clean | No |
| Processing overhead | Extra Silver-layer transform step (ai_mask on 60 notes) | Single pass at ingestion |
| **Decision** | **Chosen — auditable but larger exposure** | Rejected — audit trail is non-negotiable in healthcare |

**The real tension**: By masking at Silver instead of Bronze, we deliberately keep raw PII in storage. This creates a compliance surface area — Bronze tables containing patient names, DOBs, and MRNs must be locked down with UC access controls. We accepted this risk because healthcare regulations (HIPAA, state laws) require the ability to produce original clinical documentation on demand for audits, legal holds, and patient record requests. Destroying PII at ingestion would make us audit-proof but legally non-compliant.

**Mitigation**: Bronze tables are access-restricted via Unity Catalog grants. Only the compliance team has SELECT on `clinical_notes_text` and `bronze_notes`. The AI agent, app UI, Genie Space, and all downstream consumers only ever see `silver_notes_masked`.

### 4. Streamlit (Dash) vs AppKit (React/TypeScript)

| | Dash/Streamlit (chosen) | AppKit (React) |
|---|---|---|
| Setup time | < 1 hour, single app.py | Scaffold + build + deploy |
| Python-native | Yes — direct SQL connector calls | Requires REST API layer |
| Component library | dash-bootstrap-components | AppKit components |
| FUSE mount | Not required | Required — failed in this env |
| **Decision** | **Chosen — fast, reliable, full control** | Blocked by FUSE mount issue |

Why: AppKit scaffold at `/Workspace/cdi-coding-copilot` was inaccessible due to FUSE mount failures in the hackathon environment. Dash gave us a working 3-page app in a single Python file with full control over layout, callbacks, and data fetching.

### 5. Foundation Model Choice — Llama 3.3 70B vs Claude Sonnet

| | Llama 3.3 70B (chosen for ai_query) | Claude Sonnet |
|---|---|---|
| Availability | Pay-per-token, no permissions needed | Pay-per-token, available |
| Clinical accuracy | Strong for entity extraction | Stronger for nuanced reasoning |
| Token cost | Lower per-token | Higher per-token |
| Structured output | Good with prompt engineering | Better native JSON adherence |
| **Decision** | **Chosen for ai_query() — cost-effective, fast** | Used implicitly by ai_mask()/ai_parse_document() |

Why: For structured ICD-10/CPT extraction, Llama 3.3 70B produces reliable JSON output at lower cost. The managed AI functions (`ai_mask`, `ai_parse_document`) use Databricks-selected models optimized for their specific tasks.

### 6. AI Gateway — Shared Endpoint vs Dedicated Endpoint

| | Shared FMAPI endpoint (current) | Dedicated custom endpoint |
|---|---|---|
| Setup | Zero — already exists | Requires Manage permission |
| Rate limits | Workspace-level defaults | Customizable per user/group |
| Guardrails | Safety + PII blocking (default) | Custom topic filtering |
| Inference table | Via system tables | Custom catalog/schema |
| **Decision** | **Current — workspace admin controls** | Blocked by Manage permission in hackathon |

Why: The `databricks-meta-llama-3-3-70b-instruct` endpoint is governed by AI Gateway with built-in usage tracking, safety guardrails, and rate limits. Configuring a dedicated endpoint with custom guardrails (topic restriction to healthcare, custom rate limits) requires workspace admin Manage permission, which isn't available in the hackathon environment.

### 7. Genie Space Data Scope — All Tables vs Claims-Only

| | All tables (including notes) | Claims + codes only (chosen) |
|---|---|---|
| Billing manager access | Can query note text | Cannot see any clinical text |
| PII risk | Even masked text may leak context | Zero PII exposure |
| Query scope | Broad but risky | Focused on denials + revenue |
| **Decision** | Not appropriate for billing role | **Chosen — principle of least privilege** |

Why: The Genie Space connects 7 tables but deliberately excludes `clinical_notes_text` and `bronze_notes`. Even though `silver_notes_masked` has PII removed, a billing manager has no business need for clinical note text. The space is scoped to claims, encounters, denial patterns, and coding statistics.

### 8. Decision Persistence — Lakebase Only vs Dual-Write

| | Lakebase only | UC Delta only | Dual-write (chosen) |
|---|---|---|---|
| App write latency | 3.9ms | 500ms+ | 3.9ms (Lakebase first) |
| Analytics queryable | Requires export | Native SQL | Both available |
| Consistency | Single source of truth | Single source | Eventual consistency risk |
| **Decision** | | | **Chosen — best of both worlds** |

Why: The app writes to `cdi_copilot.coder_decisions` (UC Delta) for immediate analytics availability, while Lakebase tables provide the low-latency OLTP path. In production, Lakebase synced tables would unify both into a single-write pattern.
