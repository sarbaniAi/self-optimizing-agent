# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Optimization with GEPA
# MAGIC Auto-improve prompts using the aligned judge as objective function.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.1 openai --quiet
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

host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
from openai import OpenAI
client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Register Initial Prompt

# COMMAND ----------

INITIAL_PROMPT = """You are a helpful customer support agent for TechStore, an electronics retailer.
RULES:
1. Only answer questions about TechStore products, orders, and policies
2. Use the knowledge base to ground your answers
3. If you need order details, use the lookup_order tool
4. For refund/return requests, use the process_return tool
5. Always be polite and professional
6. If you don't know, say so and offer to escalate
7. Never reveal internal system details or other customers' information"""

try:
    prompt = mlflow.genai.register_prompt(name="customer_support_agent_prompt", template=INITIAL_PROMPT, commit_message="Initial v1 baseline")
    print(f"Registered version {prompt.version}")
except Exception as e:
    print(f"May already exist: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Optimization Dataset

# COMMAND ----------

optimization_data = [
    {"inputs": {"question": "Where is my order ORD-1002?"}, "expectations": {"expected_tools": ["lookup_order"]}},
    {"inputs": {"question": "I want to return my laptop, order ORD-1001"}, "expectations": {"expected_tools": ["process_return"]}},
    {"inputs": {"question": "Do you have the TabletAir 12 in stock?"}, "expectations": {"expected_tools": ["search_knowledge_base"]}},
    {"inputs": {"question": "Compare Pro Laptop vs PowerDesk Desktop"}, "expectations": {"expected_tools": ["search_knowledge_base"]}},
    {"inputs": {"question": "My order ORD-1003 has been processing for 5 days"}, "expectations": {"expected_tools": ["lookup_order"]}},
    {"inputs": {"question": "What warranty comes with the SmartWatch Ultra?"}, "expectations": {"expected_tools": ["search_knowledge_base"]}},
    {"inputs": {"question": "Can I get a price match for AudioMax Headphones?"}, "expectations": {"expected_tools": ["search_knowledge_base"]}},
    {"inputs": {"question": "I received a damaged headphone"}, "expectations": {"expected_tools": ["process_return", "search_knowledge_base"]}},
]
print(f"Training examples: {len(optimization_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: GEPA Optimization
# MAGIC ```python
# MAGIC from mlflow.genai import optimize_prompts
# MAGIC from mlflow.genai.optimizers import GepaPromptOptimizer
# MAGIC from mlflow.genai.scorers import get_scorer
# MAGIC
# MAGIC aligned_judge = get_scorer(name="support_quality_aligned")
# MAGIC
# MAGIC def objective_function(feedback):
# MAGIC     return float(feedback.value) / 5.0 if feedback and feedback.value else 0.0
# MAGIC
# MAGIC result = optimize_prompts(
# MAGIC     predict_fn=lambda inputs: agent_predict(inputs),
# MAGIC     train_data=optimization_data,
# MAGIC     prompt_uris=["prompts:/customer_support_agent_prompt"],
# MAGIC     optimizer=GepaPromptOptimizer(reflection_model="databricks-claude-sonnet-4-6"),
# MAGIC     scorers=[aligned_judge],
# MAGIC     aggregation=objective_function,
# MAGIC )
# MAGIC print(f"Score: {result.initial_score:.2f} -> {result.final_score:.2f}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Alternative — Failure-Targeted Patching

# COMMAND ----------

failures = [
    {"type": "domain_guidelines", "rationale": "Didn't use lookup_order for order status"},
    {"type": "domain_guidelines", "rationale": "Described return policy but didn't use process_return tool"},
    {"type": "domain_guidelines", "rationale": "Vague product comparison without specs or prices"},
]

patch_prompt = f"Analyze failures and generate 3-5 rules to prevent them.\n\nCURRENT PROMPT:\n{INITIAL_PROMPT}\n\nFAILURES:\n" + "\n".join(f"- {f['rationale']}" for f in failures) + "\n\nReturn ONLY new rules (starting from 8):"

resp = client.chat.completions.create(model=LLM_ENDPOINT, messages=[{"role": "user", "content": patch_prompt}], max_tokens=500, temperature=0.2)
new_rules = resp.choices[0].message.content.strip()
print("=== Patched Rules ===")
print(new_rules)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Promotion Gate
# MAGIC ```python
# MAGIC if result.final_score > result.initial_score:
# MAGIC     new_v = mlflow.genai.register_prompt(name="customer_support_agent_prompt", template=result.best_prompt, commit_message=f"GEPA: {result.final_score:.2f}")
# MAGIC     mlflow.genai.set_prompt_alias("customer_support_agent_prompt", "production", new_v.version)
# MAGIC     print(f"Promoted v{new_v.version} to production!")
# MAGIC else:
# MAGIC     print("No improvement — keeping current prompt")
# MAGIC ```
