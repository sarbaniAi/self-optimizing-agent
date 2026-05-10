"""
Prompt optimizer for the self-optimizing agent harness.

Provides five optimization strategies:
    1. Failure-targeted patching
    2. Few-shot injection
    3. Constitutional rewrite
    4. Routing rule refinement
    5. GEPA (Generalized Efficient Prompt Alignment) via mlflow.genai

Integrates with the MLflow Prompt Registry for version control.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

import mlflow
import pandas as pd

from harness.eval_config import EvalHarnessConfig, OptimizationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Registry Manager
# ---------------------------------------------------------------------------

class PromptRegistryManager:
    """Manages prompt versions in the MLflow Prompt Registry.

    Tracks prompt evolution across optimization iterations and aliases
    the best-performing version to production.
    """

    def __init__(self, config: EvalHarnessConfig) -> None:
        self.prompt_name = config.prompt_registry.prompt_name
        self.production_alias = config.prompt_registry.production_alias
        self.catalog = config.catalog
        self._current_version: Optional[int] = None

    def register_prompt(self, prompt_text: str, commit_message: str = "") -> int:
        """Register a new prompt version and return its version number.

        Args:
            prompt_text: The full prompt template text.
            commit_message: Description of what changed.

        Returns:
            The new version number.
        """
        prompt = mlflow.genai.register_prompt(
            name=self.prompt_name,
            template=prompt_text,
            commit_message=commit_message or "Auto-optimization update",
        )
        self._current_version = prompt.version
        logger.info(
            "Registered prompt '%s' version %d: %s",
            self.prompt_name, prompt.version, commit_message,
        )
        return prompt.version

    def set_production_alias(self, version: Optional[int] = None) -> None:
        """Alias a prompt version as production.

        Args:
            version: Specific version to alias.  Defaults to the last
                registered version.
        """
        ver = version or self._current_version
        if ver is None:
            raise ValueError("No version available to alias.")
        mlflow.genai.set_prompt_alias(
            name=self.prompt_name,
            alias=self.production_alias,
            version=ver,
        )
        logger.info(
            "Set alias '%s' -> %s version %d",
            self.production_alias, self.prompt_name, ver,
        )

    def load_production_prompt(self) -> str:
        """Load the current production prompt text."""
        prompt = mlflow.genai.load_prompt(
            name=self.prompt_name,
            alias=self.production_alias,
        )
        return prompt.template

    def load_prompt_version(self, version: int) -> str:
        """Load a specific prompt version."""
        prompt = mlflow.genai.load_prompt(
            name=self.prompt_name,
            version=version,
        )
        return prompt.template


# ---------------------------------------------------------------------------
# Prompt Optimizer
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """Applies optimization strategies to improve agent system prompts.

    Args:
        config: Evaluation harness configuration.
    """

    def __init__(self, config: EvalHarnessConfig) -> None:
        self.config = config
        self.registry = PromptRegistryManager(config)
        self.llm_endpoint = config.llm_endpoint

    def optimize(
        self,
        prompt: str,
        eval_results: Dict[str, Any],
        strategy: OptimizationStrategy,
        eval_data: Optional[pd.DataFrame] = None,
    ) -> str:
        """Optimize a prompt using the given strategy.

        Args:
            prompt: Current system prompt text.
            eval_results: Dict from EvalRunner.run() containing ``metrics``
                and optionally ``results_table``.
            strategy: Which optimization strategy to apply.
            eval_data: Original evaluation dataset (required for GEPA).

        Returns:
            The optimized prompt text.
        """
        strategy_map = {
            OptimizationStrategy.FAILURE_TARGETED_PATCHING: self._failure_targeted_patching,
            OptimizationStrategy.FEW_SHOT_INJECTION: self._few_shot_injection,
            OptimizationStrategy.CONSTITUTIONAL_REWRITE: self._constitutional_rewrite,
            OptimizationStrategy.ROUTING_RULE_REFINEMENT: self._routing_rule_refinement,
            OptimizationStrategy.GEPA: self._gepa_optimization,
        }

        fn = strategy_map.get(strategy)
        if fn is None:
            raise ValueError(f"Unknown optimization strategy: {strategy}")

        logger.info("Applying strategy: %s", strategy.value)

        if strategy == OptimizationStrategy.GEPA:
            optimized = fn(prompt, eval_results, eval_data=eval_data)
        else:
            optimized = fn(prompt, eval_results)

        # Register the new prompt version
        self.registry.register_prompt(
            prompt_text=optimized,
            commit_message=f"Optimized via {strategy.value}",
        )

        return optimized

    # ------------------------------------------------------------------
    # Strategy 1: Failure-targeted patching
    # ------------------------------------------------------------------

    def _failure_targeted_patching(
        self,
        prompt: str,
        eval_results: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Analyze failures and patch the prompt with targeted instructions.

        Identifies the worst-performing metrics and adds explicit guidance
        to the system prompt to address those failure modes.
        """
        metrics = eval_results.get("metrics", {})
        failed_metrics = {
            k: v for k, v in metrics.items()
            if v < self._get_threshold(k)
        }

        if not failed_metrics:
            logger.info("No failed metrics; returning prompt unchanged.")
            return prompt

        failure_descriptions = "\n".join(
            f"- {name}: scored {value:.3f} (threshold: {self._get_threshold(name):.3f})"
            for name, value in sorted(failed_metrics.items(), key=lambda x: x[1])
        )

        patch_instruction = self._call_llm(
            f"""You are an expert prompt engineer. The following agent system prompt
is failing on these evaluation metrics:

{failure_descriptions}

Current system prompt:
---
{prompt}
---

Generate an improved version of the system prompt that specifically addresses
each failing metric. Add explicit instructions, constraints, or examples that
would improve the agent's performance on those dimensions. Return ONLY the
improved prompt, no commentary."""
        )

        return patch_instruction.strip()

    # ------------------------------------------------------------------
    # Strategy 2: Few-shot injection
    # ------------------------------------------------------------------

    def _few_shot_injection(
        self,
        prompt: str,
        eval_results: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Inject few-shot examples from high-scoring traces into the prompt.

        Selects the best-performing examples from the evaluation results
        and embeds them as demonstrations in the system prompt.
        """
        results_table = eval_results.get("results_table")
        if results_table is None or results_table.empty:
            logger.warning("No results table available for few-shot injection.")
            return prompt

        # Find high-scoring examples
        score_cols = [c for c in results_table.columns if c.endswith("_score") or c.startswith("assessment/")]
        if not score_cols:
            return prompt

        results_table["_avg_score"] = results_table[score_cols].mean(axis=1, skipna=True)
        top_examples = results_table.nlargest(3, "_avg_score")

        examples_text = ""
        for idx, row in top_examples.iterrows():
            req = row.get("request", "")
            resp = row.get("response", row.get("output", ""))
            examples_text += f"\n<example>\nUser: {req}\nAssistant: {resp}\n</example>\n"

        if not examples_text.strip():
            return prompt

        enhanced = f"""{prompt}

## Few-Shot Examples
The following are high-quality example interactions. Follow these patterns:
{examples_text}"""

        return enhanced

    # ------------------------------------------------------------------
    # Strategy 3: Constitutional rewrite
    # ------------------------------------------------------------------

    def _constitutional_rewrite(
        self,
        prompt: str,
        eval_results: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Rewrite the prompt using constitutional AI principles.

        Adds explicit behavioral principles (safety, helpfulness, honesty)
        derived from evaluation failures.
        """
        metrics = eval_results.get("metrics", {})

        principles = []
        if metrics.get("toxicity_check", 1.0) < 0.95:
            principles.append(
                "SAFETY: Never produce harmful, toxic, or offensive content. "
                "If a request could lead to harm, politely decline and explain why."
            )
        if metrics.get("no_hallucination", 1.0) < 0.8:
            principles.append(
                "HONESTY: Only state facts that are directly supported by the "
                "provided context. If you are unsure, say so explicitly."
            )
        if metrics.get("pii_leakage", 1.0) < 1.0:
            principles.append(
                "PRIVACY: Never include personal identifiable information (PII) "
                "such as emails, phone numbers, Aadhaar numbers, or PAN numbers "
                "in your responses."
            )
        if metrics.get("guardrail_adherence", 1.0) < 1.0:
            principles.append(
                "BOUNDARIES: Stay strictly within your designated scope. Do not "
                "discuss topics outside your mandate."
            )
        if metrics.get("response_completeness", 1.0) < 0.8:
            principles.append(
                "COMPLETENESS: Address every part of the user's question. If the "
                "question has multiple sub-parts, handle each one explicitly."
            )

        if not principles:
            return prompt

        principles_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))
        enhanced = f"""{prompt}

## Constitutional Principles
You MUST adhere to the following principles in every response:
{principles_block}"""

        return enhanced

    # ------------------------------------------------------------------
    # Strategy 4: Routing rule refinement
    # ------------------------------------------------------------------

    def _routing_rule_refinement(
        self,
        prompt: str,
        eval_results: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Refine multi-agent routing rules based on misrouting patterns.

        Analyzes the results table for routing_accuracy failures and
        generates more precise routing instructions.
        """
        results_table = eval_results.get("results_table")
        if results_table is None or results_table.empty:
            return prompt

        # Look for routing mismatches
        if "expected_agent" not in results_table.columns or "routed_to" not in results_table.columns:
            return prompt

        misrouted = results_table[
            results_table.get("expected_agent") != results_table.get("routed_to")
        ]

        if misrouted.empty:
            return prompt

        misroute_summary = []
        for _, row in misrouted.head(5).iterrows():
            misroute_summary.append(
                f"  Query: \"{row.get('request', 'N/A')}\"\n"
                f"  Expected: {row.get('expected_agent', 'N/A')}\n"
                f"  Actual: {row.get('routed_to', 'N/A')}"
            )

        refinement = self._call_llm(
            f"""You are a multi-agent routing expert. The following queries were
misrouted by the agent:

{chr(10).join(misroute_summary)}

Current routing instructions in the system prompt:
---
{prompt}
---

Generate improved routing rules that would correctly handle these cases.
Return ONLY the complete improved system prompt, no commentary."""
        )

        return refinement.strip()

    # ------------------------------------------------------------------
    # Strategy 5: GEPA (Generalized Efficient Prompt Alignment)
    # ------------------------------------------------------------------

    def _gepa_optimization(
        self,
        prompt: str,
        eval_results: Dict[str, Any],
        eval_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> str:
        """Optimize the prompt using MLflow GEPA (optimize_prompt).

        Uses GepaPromptOptimizer to search for a better prompt that
        maximizes the objective function across the evaluation dataset.

        Args:
            prompt: Current system prompt template.
            eval_results: Results from the last evaluation run.
            eval_data: The evaluation dataset (required).

        Returns:
            The GEPA-optimized prompt text.
        """
        if eval_data is None:
            logger.warning("GEPA requires eval_data. Falling back to failure-targeted patching.")
            return self._failure_targeted_patching(prompt, eval_results)

        from mlflow.genai.optimize import GepaPromptOptimizer

        # Register the current prompt so GEPA can version from it
        prompt_name = self.config.prompt_registry.prompt_name

        try:
            mlflow.genai.register_prompt(
                name=prompt_name,
                template=prompt,
                commit_message="GEPA baseline",
            )
        except Exception:
            # Prompt may already exist; that's fine
            pass

        optimizer = GepaPromptOptimizer(
            llm_endpoint=self.llm_endpoint,
            prompt_name=prompt_name,
        )

        objective_fn = self._build_objective_function()

        logger.info("Starting GEPA optimization with %d examples", len(eval_data))

        result = mlflow.genai.optimize_prompt(
            optimizer=optimizer,
            eval_data=eval_data,
            objective=objective_fn,
            max_iterations=self.config.max_optimization_iterations,
        )

        optimized_prompt = result.best_prompt if hasattr(result, "best_prompt") else result.template
        logger.info("GEPA optimization complete. Best score: %s", getattr(result, "best_score", "N/A"))

        return optimized_prompt

    # ------------------------------------------------------------------
    # GEPA objective function
    # ------------------------------------------------------------------

    def _build_objective_function(self) -> Callable:
        """Build the objective function for GEPA optimization.

        Normalizes Likert 1-5 scores to 0-1 range and averages across
        all configured metrics. Boolean scores are treated as 0/1.

        Returns:
            A callable ``(assessments) -> float`` in [0, 1].
        """
        likert_max = self.config.labeling.likert_max

        def _objective_function(assessments: List[Any]) -> float:
            """Compute a single scalar score from a list of assessments."""
            if not assessments:
                return 0.0

            scores = []
            for assessment in assessments:
                value = assessment.value if hasattr(assessment, "value") else assessment
                if isinstance(value, bool):
                    scores.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    if value > 1.0:
                        # Likert scale: normalize to [0, 1]
                        normalized = (float(value) - 1.0) / (likert_max - 1.0)
                        scores.append(max(0.0, min(1.0, normalized)))
                    else:
                        scores.append(float(value))

            return sum(scores) / len(scores) if scores else 0.0

        return _objective_function

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_threshold(self, metric_name: str) -> float:
        """Look up the pass threshold for a metric, defaulting to 0.7."""
        for t in self.config.thresholds:
            if t.metric_name == metric_name:
                return t.pass_threshold
        return 0.7

    def _call_llm(self, prompt_text: str) -> str:
        """Call the configured LLM endpoint for prompt engineering tasks.

        Args:
            prompt_text: The meta-prompt to send to the LLM.

        Returns:
            The LLM's response text.
        """
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        response = client.predict(
            endpoint=self.llm_endpoint,
            inputs={
                "messages": [
                    {"role": "user", "content": prompt_text},
                ],
                "max_tokens": 4096,
                "temperature": 0.3,
            },
        )

        # Handle both dict and object response formats
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return response.choices[0].message.content
