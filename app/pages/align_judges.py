"""Backend for the Align Judges page — uses real LLM for alignment demo."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


ORIGINAL_INSTRUCTIONS = """Rate the quality of the agent response on a scale of 1-5:
1 = Unacceptable: Response is wrong, harmful, or completely off-topic
2 = Poor: Major issues, mostly incorrect or unhelpful
3 = Acceptable: Adequate but could be significantly improved
4 = Good: Correct and helpful, minor improvements possible
5 = Excellent: Thorough, well-structured, expert-level response"""


def run_alignment(optimizer: str = "memalign", judge_name: str = "domain_guidelines") -> dict:
    """Run judge alignment using real LLM to generate aligned instructions."""
    from openai import OpenAI

    host = os.environ.get("DATABRICKS_HOST", "https://adb-7405619910560146.6.azuredatabricks.net")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

    # Simulated disagreements (in production, these come from labeled traces)
    disagreements = [
        {"question": "Where is my order ORD-1002?", "judge_score": 4, "human_score": 2,
         "reason": "Agent gave generic shipping info instead of using lookup_order tool"},
        {"question": "I want to return my laptop", "judge_score": 5, "human_score": 2,
         "reason": "Agent described return policy but didn't use process_return tool"},
        {"question": "Compare Pro Laptop vs PowerDesk", "judge_score": 4, "human_score": 3,
         "reason": "Response was vague — no specific specs, prices, or use cases"},
        {"question": "What warranty comes with SmartWatch?", "judge_score": 5, "human_score": 4,
         "reason": "Good but should have cited specific KB article number"},
    ]

    disagreement_text = "\n".join([
        f"- Q: \"{d['question']}\" | Judge: {d['judge_score']}/5 | Expert: {d['human_score']}/5 | Reason: {d['reason']}"
        for d in disagreements
    ])

    if optimizer == "memalign":
        prompt = f"""You are an alignment optimizer. Given a baseline judge's instructions and expert disagreements,
produce IMPROVED judge instructions using the MemAlign dual-memory approach.

BASELINE JUDGE INSTRUCTIONS:
{ORIGINAL_INSTRUCTIONS}

EXPERT DISAGREEMENTS (where the judge and human experts disagreed):
{disagreement_text}

Generate improved instructions that include:
1. The original 1-5 scale but with DOMAIN-SPECIFIC criteria learned from expert feedback
2. A "SEMANTIC PRINCIPLES" section with general rules learned from the disagreements
3. An "EPISODIC EXAMPLES" section with specific examples from the disagreements

Return ONLY the improved instructions (no explanation):"""
    elif optimizer == "simba":
        prompt = f"""You are an alignment optimizer using SIMBA (iterative instruction editing).
Given a baseline judge's instructions and expert disagreements, refine the instructions.

BASELINE JUDGE INSTRUCTIONS:
{ORIGINAL_INSTRUCTIONS}

EXPERT DISAGREEMENTS:
{disagreement_text}

Iteratively refine the instructions by:
1. Identifying why the judge was wrong in each case
2. Adding specific rules to prevent each type of disagreement
3. Keeping the 1-5 scale but making criteria more precise

Return ONLY the refined instructions:"""
    else:  # likert_simba
        prompt = f"""You are a Likert-aware alignment optimizer.
Given a baseline judge's instructions and expert disagreements on a 1-5 Likert scale,
refine the instructions to minimize the distance between judge and expert scores.

BASELINE JUDGE INSTRUCTIONS:
{ORIGINAL_INSTRUCTIONS}

EXPERT DISAGREEMENTS:
{disagreement_text}

Focus on cases with large score differences (>1 point).
Add specific criteria that would make the judge score closer to the expert.
Return ONLY the refined instructions:"""

    try:
        resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000, temperature=0.3
        )
        aligned_instructions = resp.choices[0].message.content.strip()

        return {
            "original_instructions": ORIGINAL_INSTRUCTIONS,
            "aligned_instructions": aligned_instructions,
            "optimizer_used": optimizer,
            "disagreements_analyzed": len(disagreements),
            "improvement_score": round(sum(abs(d["judge_score"] - d["human_score"]) for d in disagreements) / len(disagreements) * 0.1, 2),
            "traces_used": len(disagreements),
        }
    except Exception as e:
        return {"error": str(e), "original_instructions": ORIGINAL_INSTRUCTIONS, "aligned_instructions": f"Error: {e}"}
