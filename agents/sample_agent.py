"""
Sample Agent Template -- Replace with your own agent.
This agent uses Databricks Model Serving + tool calling.
"""
import os
from openai import OpenAI


class SampleAgent:
    """Minimal agent template for use with the self-optimizing harness."""

    def __init__(self, system_prompt: str, endpoint: str = "databricks-claude-sonnet-4-6"):
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN", ""),
            base_url=os.environ.get("DATABRICKS_HOST", "") + "/serving-endpoints",
        )
        self.endpoint = endpoint

    def predict(self, inputs: dict) -> dict:
        """Run the agent on a single input. Returns dict with 'response' key."""
        question = inputs.get("question", str(inputs))
        response = self.client.chat.completions.create(
            model=self.endpoint,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        return {"response": response.choices[0].message.content}
