# Databricks notebook source
# MAGIC %md
# MAGIC # Judge Alignment — SIMBA vs MemAlign
# MAGIC Align LLM judges to match human expert feedback.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.1 openai --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "classic_stable_4a2ohn_azure"
SCHEMA = "self_optimizing_agent"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

import mlflow
mlflow.set_tracking_uri("databricks")
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/self-optimizing-agent"
mlflow.set_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Baseline Judge

# COMMAND ----------

from mlflow.genai.judges import make_judge

baseline_judge = make_judge(
    name="support_quality_base",
    instructions="""Rate the quality of the customer support response on a scale of 1-5:
1 = Unacceptable: Wrong, harmful, or off-topic
2 = Poor: Major issues, mostly incorrect
3 = Acceptable: Adequate but could improve
4 = Good: Correct and helpful
5 = Excellent: Thorough, well-structured, expert-level""",
    feedback_value_type=float,
)
print(f"Created: {baseline_judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Simulated Expert Labels
# MAGIC In production, use MLflow Review App for SME labeling.

# COMMAND ----------

labeled_examples = [
    {"question": "What's your return policy?", "judge_score": 5.0, "human_score": 4.0, "comment": "Should mention 48-hour rule for damaged items"},
    {"question": "Where is my order ORD-1002?", "judge_score": 4.0, "human_score": 2.0, "comment": "Didn't use lookup_order tool — gave generic response"},
    {"question": "I want to return my laptop", "judge_score": 5.0, "human_score": 2.0, "comment": "Should use process_return tool, not just describe policy"},
    {"question": "Do you have TabletAir 12?", "judge_score": 5.0, "human_score": 5.0, "comment": "Checked stock, offered alternatives"},
    {"question": "Compare Pro Laptop vs PowerDesk", "judge_score": 4.0, "human_score": 3.0, "comment": "Too vague — needs specific specs and prices"},
]

disagreements = sum(1 for ex in labeled_examples if abs(ex["judge_score"] - ex["human_score"]) > 1)
print(f"Labeled: {len(labeled_examples)} | Disagreements: {disagreements}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: MemAlign Alignment (Recommended)
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import get_scorer
# MAGIC from mlflow.genai.judges.optimizers import MemAlignOptimizer
# MAGIC
# MAGIC judge = get_scorer(name="support_quality_base")
# MAGIC valid_traces = mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], filter_string="tags.eval = 'complete'")
# MAGIC
# MAGIC aligned_judge = judge.align(
# MAGIC     traces=valid_traces,
# MAGIC     optimizer=MemAlignOptimizer(
# MAGIC         reflection_lm="databricks-claude-sonnet-4-6",
# MAGIC         embedding_model="databricks-gte-large-en"
# MAGIC     )
# MAGIC )
# MAGIC print("Aligned instructions:", aligned_judge.instructions)
# MAGIC print("Distilled guidelines:", aligned_judge.distilled_guidelines)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: What MemAlign Learns

# COMMAND ----------

print("=== Simulated MemAlign Output ===\n")
aligned_instructions = """Rate quality 1-5 with domain-specific criteria:
1 = Fabricates info, ignores tools, or contains PII
2 = Vague when specific data was available via tools
3 = Uses tools but misses key details
4 = Correct tool usage, cites KB, addresses specific need
5 = Expert judgment — proactive tool use, specific details, anticipates follow-ups

SEMANTIC PRINCIPLES:
- Order questions MUST use lookup_order tool
- Return requests MUST use process_return tool
- Comparisons need specific specs and prices

EPISODIC EXAMPLES:
- Asked about order status but got generic shipping info -> 2/5
- Return request but only described policy -> 2/5
- Stock check + offered alternatives -> 5/5"""
print(aligned_instructions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Optimizer Comparison

# COMMAND ----------

print(f"{'Characteristic':<20} {'SIMBA':<25} {'MemAlign':<25} {'LikertSIMBA':<20}")
print("-" * 90)
print(f"{'Speed':<20} {'Multiple iterations':<25} {'Single pass (100x)':<25} {'Multiple iterations':<20}")
print(f"{'Cost':<20} {'More LLM calls':<25} {'10x cheaper':<25} {'More LLM calls':<20}")
print(f"{'Min Examples':<20} {'20-30 traces':<25} {'2-10 traces':<25} {'20-30 traces':<20}")
print(f"{'Best For':<20} {'MLflow 3.8':<25} {'MLflow 3.9+ (rec.)':<25} {'Continuous scales':<20}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Register Aligned Judge
# MAGIC ```python
# MAGIC aligned_registered = make_judge(name="support_quality_aligned", instructions=aligned_judge.instructions, feedback_value_type=float)
# MAGIC aligned_registered.register(experiment_id=EXPERIMENT_ID)
# MAGIC ```
