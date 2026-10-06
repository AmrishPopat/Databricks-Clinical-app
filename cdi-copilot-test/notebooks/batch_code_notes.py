# Databricks notebook source
# MAGIC %md
# MAGIC # CDI coding agent — setup, backfill & evaluation
# MAGIC Runs the **same** `agent.py` the App uses for live "sign note" coding, so batch and live results are identical in shape.
# MAGIC 1. Create agent tables (`agent_runs`, `agent_code_suggestions`)
# MAGIC 2. Create the MLflow experiment with traces stored in Unity Catalog
# MAGIC 3. Backfill the coder queue
# MAGIC 4. Evaluate vs. gold codes / CDI gaps (`note_labels`) — the agent only ever sees masked text with gold codes stripped

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.11.1" "databricks-sdk>=0.40.0"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("max_notes", "50", "Max notes to code")
dbutils.widgets.text("endpoint", "databricks-claude-sonnet-5", "Model endpoint")
dbutils.widgets.text("workers", "8", "Parallel workers")

import os, sys, json
os.environ["CDI_AGENT_ENDPOINT"] = dbutils.widgets.get("endpoint")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.path.append(os.path.abspath("../app"))

CAT, SCH = "external-ai-build-day", "cdi_copilot"
T = f"`{CAT}`.{SCH}"
WAREHOUSE_ID = "22998c886e21ee6c"
EXPERIMENT_NAME = "/Shared/Team1-HealthCare-Solution/cdi-copilot-test/cdi-copilot-agent"

# COMMAND ----------

# MAGIC %md ## 1. Agent tables

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T}.agent_runs (
  run_id STRING, note_id STRING, encounter_id STRING, coding_risk_score DOUBLE,
  risk_components STRING, cdi_gaps STRING, entities STRING, rejected_codes STRING,
  model_endpoint STRING, prompt_version STRING, latency_ms INT, input_tokens INT,
  output_tokens INT, trace_id STRING, created_at TIMESTAMP)
COMMENT 'One row per CDI agent run over a masked note (risk score, CDI gaps, model/prompt lineage)'
""")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T}.agent_code_suggestions (
  suggestion_id STRING, run_id STRING, note_id STRING, encounter_id STRING, code STRING,
  code_system STRING, code_description STRING, avg_reimbursement BIGINT, confidence DOUBLE,
  principal BOOLEAN, rationale STRING, evidence_quote STRING, evidence_start INT, evidence_end INT,
  evidence_verified BOOLEAN, created_at TIMESTAMP)
COMMENT 'AI-suggested codes, each grounded to a verbatim evidence span (char offsets) in the masked note'
""")
# isolated decisions table for the test app (CDI_DECISIONS_TABLE in app.yaml)
spark.sql(f"CREATE TABLE IF NOT EXISTS {T}.coder_decisions_test LIKE {T}.coder_decisions")

# COMMAND ----------

# MAGIC %md ## 2. MLflow experiment with Unity Catalog trace storage
# MAGIC Put the printed experiment id into `app.yaml` (`MLFLOW_EXPERIMENT_ID`) and grant the App service principal CAN_EDIT on it.

# COMMAND ----------

import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WAREHOUSE_ID
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(name=EXPERIMENT_NAME,
                                      trace_location=UnityCatalog(catalog_name=CAT, schema_name=SCH))
else:
    exp_id = exp.experiment_id
    print("existing trace location:", exp.trace_location)
os.environ["MLFLOW_EXPERIMENT_ID"] = exp_id
print("MLFLOW_EXPERIMENT_ID =", exp_id)

# COMMAND ----------

# MAGIC %md ## 3. Backfill

# COMMAND ----------

import agent
from concurrent.futures import ThreadPoolExecutor, as_completed
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
agent.init_tracing()
codebook = [r.asDict() for r in spark.sql(
    f"SELECT code, code_system, description, avg_reimbursement FROM {T}.codebook").collect()]
priors = {}
for r in spark.sql(f"SELECT payer, code, denial_rate FROM {T}.gold_code_denial_priors").collect():
    priors.setdefault(r.payer, {})[r.code] = float(r.denial_rate or 0)

todo = spark.sql(f"""
  SELECT s.note_id, s.encounter_id, s.masked_text, s.payer
  FROM {T}.silver_notes_masked s
  LEFT ANTI JOIN {T}.agent_runs r ON s.note_id = r.note_id
  ORDER BY s.signature_timestamp DESC
  LIMIT {int(dbutils.widgets.get('max_notes'))}
""").collect()
print(f"coding {len(todo)} notes with {agent.MODEL_ENDPOINT}")

results, errors = [], []
with ThreadPoolExecutor(max_workers=int(dbutils.widgets.get("workers"))) as pool:
    futs = {pool.submit(agent.code_note, w, n.note_id, n.encounter_id, n.masked_text,
                        codebook, priors.get(n.payer, {})): n.note_id for n in todo}
    for f in as_completed(futs):
        try:
            results.append(f.result())
        except Exception as e:
            errors.append((futs[f], str(e)[:200]))
print(f"ok={len(results)} errors={len(errors)}", errors[:5])

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

if results:
    runs, suggs = [], []
    for r in results:
        run, s = agent.to_rows(r)
        runs.append(run); suggs.extend(s)
    (spark.createDataFrame(pd.DataFrame(runs, columns=agent.RUNS_COLUMNS).astype(object).where(lambda d: d.notna(), None))
          .withColumn("coding_risk_score", F.col("coding_risk_score").cast("double"))
          .withColumn("latency_ms", F.col("latency_ms").cast("int"))
          .withColumn("input_tokens", F.col("input_tokens").cast("int"))
          .withColumn("output_tokens", F.col("output_tokens").cast("int"))
          .withColumn("trace_id", F.col("trace_id").cast("string"))
          .withColumn("created_at", F.to_timestamp("created_at"))
          .write.mode("append").saveAsTable(f"{CAT}.{SCH}.agent_runs"))
    if suggs:
        (spark.createDataFrame(pd.DataFrame(suggs, columns=agent.SUGGESTION_COLUMNS).astype(object).where(lambda d: d.notna(), None))
              .withColumn("avg_reimbursement", F.col("avg_reimbursement").cast("bigint"))
              .withColumn("confidence", F.col("confidence").cast("double"))
              .withColumn("principal", F.col("principal").cast("boolean"))
              .withColumn("evidence_start", F.col("evidence_start").cast("int"))
              .withColumn("evidence_end", F.col("evidence_end").cast("int"))
              .withColumn("evidence_verified", F.col("evidence_verified").cast("boolean"))
              .withColumn("created_at", F.to_timestamp("created_at"))
              .write.mode("append").saveAsTable(f"{CAT}.{SCH}.agent_code_suggestions"))
mlflow.flush_trace_async_logging()

# COMMAND ----------

# MAGIC %md ## 4. Evaluation vs. gold labels
# MAGIC Gold codes are embedded in the raw notes and stripped by `silver_notes_masked`, so this is a fair held-out check.

# COMMAND ----------

display(spark.sql(f"""
WITH latest AS (
  SELECT * FROM {T}.agent_runs QUALIFY row_number() OVER (PARTITION BY note_id ORDER BY created_at DESC) = 1
),
pred AS (
  SELECT l.note_id, collect_set(a.code) AS pred_codes, max(json_array_length(l.cdi_gaps)) AS n_gaps
  FROM latest l LEFT JOIN {T}.agent_code_suggestions a ON a.run_id = l.run_id
  GROUP BY l.note_id
),
j AS (
  SELECT p.note_id, p.pred_codes, array_distinct(g.gold_codes) AS gold_codes, p.n_gaps, g.gold_cdi_gap
  FROM pred p JOIN {T}.note_labels g ON p.note_id = g.note_id
)
SELECT
  count(*) AS notes_evaluated,
  round(sum(size(array_intersect(pred_codes, gold_codes))) / sum(size(pred_codes)), 3) AS code_precision,
  round(sum(size(array_intersect(pred_codes, gold_codes))) / sum(size(gold_codes)), 3) AS code_recall,
  round(avg(CASE WHEN size(array_except(gold_codes, pred_codes)) = 0 THEN 1.0 ELSE 0.0 END), 3) AS notes_all_gold_codes_found,
  round(avg(CASE WHEN gold_cdi_gap IS NOT NULL THEN CASE WHEN n_gaps > 0 THEN 1.0 ELSE 0.0 END END), 3) AS cdi_gap_recall
FROM j
"""))

# COMMAND ----------

display(spark.sql(f"""
SELECT
  round(avg(CASE WHEN evidence_verified THEN 1.0 ELSE 0.0 END), 3) AS pct_codes_with_verbatim_evidence,
  (SELECT round(avg(latency_ms)/1000, 1) FROM {T}.agent_runs) AS avg_latency_s,
  (SELECT round(avg(coding_risk_score), 1) FROM {T}.agent_runs) AS avg_risk_score
FROM {T}.agent_code_suggestions
"""))
