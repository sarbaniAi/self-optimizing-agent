# Databricks notebook source
# MAGIC %md
# MAGIC # Continuous Monitoring Setup
# MAGIC Schedule periodic evaluations with drift detection and auto-optimization.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.1 openai pyyaml --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "classic_stable_4a2ohn_azure"
SCHEMA = "self_optimizing_agent"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"

import mlflow
mlflow.set_tracking_uri("databricks")
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/self-optimizing-agent"
mlflow.set_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Set Baseline

# COMMAND ----------

baseline_scores = {
    "overall_pass_rate": 0.87,
    "security_pass_rate": 1.0,
    "per_scorer": {
        "pii_leakage": 1.0,
        "prompt_injection_detection": 1.0,
        "domain_guidelines": 0.70,
        "response_correctness": 0.90,
    }
}
print(f"Baseline set: {baseline_scores['overall_pass_rate']:.0%} overall, {baseline_scores['security_pass_rate']:.0%} security")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Run Monitoring Cycle

# COMMAND ----------

DRIFT_THRESHOLD = 0.05

# Simulate current eval scores (in production, this runs the actual evaluation)
current_scores = {
    "overall_pass_rate": 0.83,
    "security_pass_rate": 1.0,
    "per_scorer": {
        "pii_leakage": 1.0,
        "prompt_injection_detection": 1.0,
        "domain_guidelines": 0.65,
        "response_correctness": 0.85,
    }
}

# Drift detection
drift = baseline_scores["overall_pass_rate"] - current_scores["overall_pass_rate"]
drift_detected = drift > DRIFT_THRESHOLD

print(f"Current pass rate: {current_scores['overall_pass_rate']:.0%}")
print(f"Baseline pass rate: {baseline_scores['overall_pass_rate']:.0%}")
print(f"Drift: {drift:+.1%}")
print(f"Threshold: {DRIFT_THRESHOLD:.0%}")
print(f"Drift detected: {drift_detected}")

if drift_detected:
    print("\n=== ACTION REQUIRED ===")
    print("Quality has drifted beyond threshold. Recommended actions:")
    print("  1. Run judge alignment (02_judge_alignment.py)")
    print("  2. Run prompt optimization (03_prompt_optimization.py)")
    print("  3. Review domain_guidelines scorer (dropped from 70% to 65%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Log to Monitoring Table

# COMMAND ----------

import uuid
from datetime import datetime

run_id = str(uuid.uuid4())[:8]
action = "Re-optimization triggered" if drift_detected else "No action needed"

spark.sql(f"""
INSERT INTO {CATALOG}.{SCHEMA}.monitoring_runs
VALUES ('{run_id}', {current_scores['overall_pass_rate']}, {baseline_scores['overall_pass_rate']},
        {str(drift_detected).lower()}, {drift}, '{action}', current_timestamp())
""")
print(f"Logged monitoring run: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: View History

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.monitoring_runs ORDER BY timestamp DESC LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Schedule as Job
# MAGIC
# MAGIC Deploy this notebook as a scheduled Databricks Job:
# MAGIC ```yaml
# MAGIC # In databricks.yml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     continuous-monitoring:
# MAGIC       name: self-optimizing-agent-monitoring
# MAGIC       tasks:
# MAGIC         - task_key: run_monitoring
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./notebooks/04_monitoring_setup.py
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 6 * * ?"
# MAGIC         timezone_id: Asia/Kolkata
# MAGIC ```
