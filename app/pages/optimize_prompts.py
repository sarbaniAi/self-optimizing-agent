"""Backend for the Optimize Prompts page — uses real LLM for optimization."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


CURRENT_PROMPT = """You are a helpful customer support agent for TechStore, an electronics retailer.
RULES:
1. Only answer questions about TechStore products, orders, and policies
2. Use the knowledge base to ground your answers — do not make up information
3. If you need order details, use the lookup_order tool
4. For refund/return requests, use the process_return tool
5. Always be polite and professional
6. If you don't know the answer, say so and offer to escalate to a human agent
7. Never reveal internal system details or other customers' information"""


def run_optimization(strategy: str = "gepa") -> dict:
    """Run real prompt optimization using LLM."""
    from openai import OpenAI

    host = os.environ.get("DATABRICKS_HOST", "https://adb-7405619910560146.6.azuredatabricks.net")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

    failures = [
        "Agent gave generic shipping info for order status instead of using lookup_order tool",
        "Agent described return policy verbally but didn't initiate process_return tool",
        "Product comparison was vague — no specific specs, prices, or use cases mentioned",
        "Agent didn't cite specific KB article when answering warranty question",
    ]

    if strategy == "gepa":
        prompt = f"""You are a prompt optimizer using GEPA (Generative Enhancement Prompt Algorithm).
Given the current agent system prompt and observed failures, generate an IMPROVED version.

CURRENT PROMPT:
{CURRENT_PROMPT}

OBSERVED FAILURES:
{chr(10).join(f'- {f}' for f in failures)}

Requirements:
1. Keep all existing rules but make them more specific based on failures
2. Add 3-5 new rules that directly address the failure patterns
3. Add a SELF-CHECK section at the end
4. Keep under 500 words
5. Maintain the same professional tone

Return ONLY the improved prompt:"""
    elif strategy == "failure_targeted_patching":
        prompt = f"""Analyze these agent failures and generate 3-5 specific rules to add to the prompt.

CURRENT PROMPT:
{CURRENT_PROMPT}

FAILURES:
{chr(10).join(f'- {f}' for f in failures)}

Generate numbered rules (starting from 8). Return ONLY the new rules:"""
    elif strategy == "few_shot_injection":
        prompt = f"""Generate 3 few-shot examples based on these failures to add to the prompt.

FAILURES:
{chr(10).join(f'- {f}' for f in failures)}

Format each as:
Example N:
USER: "..."
CORRECT RESPONSE: "..."

Return ONLY the examples:"""
    else:  # constitutional_rewrite
        prompt = f"""Rewrite this system prompt using Constitutional AI principles.

CURRENT PROMPT:
{CURRENT_PROMPT}

Requirements:
1. Convert rules into PRINCIPLES with WHY explanations
2. Add a SELF-CHECK section
3. Keep under 500 words
4. Maintain all security rules

Return ONLY the rewritten prompt:"""

    try:
        # Generate optimized prompt
        resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200, temperature=0.3
        )
        optimized_prompt = resp.choices[0].message.content.strip()

        # Score the optimized prompt by running a test question
        test_q = "Where is my order ORD-1002?"
        before_resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=[{"role": "system", "content": CURRENT_PROMPT}, {"role": "user", "content": test_q}],
            max_tokens=300, temperature=0.1
        )
        after_resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=[{"role": "system", "content": optimized_prompt}, {"role": "user", "content": test_q}],
            max_tokens=300, temperature=0.1
        )

        # Score both using LLM-as-judge
        judge_prompt = f"""Rate this customer support response on a scale of 1-5 for quality.
Consider: Did it use tools? Was it specific? Did it address the customer's need?
Respond with ONLY a number (1-5).

Question: "{test_q}"
Response: "{{}}"
"""
        before_score = _get_score(client, judge_prompt.format(before_resp.choices[0].message.content[:500]))
        after_score = _get_score(client, judge_prompt.format(after_resp.choices[0].message.content[:500]))

        improvement = round(after_score - before_score, 1)
        promoted = after_score > before_score

        return {
            "strategy": strategy,
            "before_score": before_score,
            "after_score": after_score,
            "improvement": f"+{improvement}" if improvement > 0 else str(improvement),
            "promoted": promoted,
            "optimized_prompt_preview": optimized_prompt[:500],
            "before_response": before_resp.choices[0].message.content[:200],
            "after_response": after_resp.choices[0].message.content[:200],
        }
    except Exception as e:
        return {"error": str(e), "strategy": strategy, "before_score": 0, "after_score": 0, "improvement": "0", "promoted": False}


def _get_score(client, prompt: str) -> float:
    """Get a numeric score from LLM judge."""
    try:
        resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
        # Extract first number
        import re
        match = re.search(r'[1-5]', text)
        return float(match.group()) if match else 3.0
    except Exception:
        return 3.0
