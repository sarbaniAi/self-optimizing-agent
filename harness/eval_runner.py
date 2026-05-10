"""
Enhanced evaluation runner for the self-optimizing agent harness.

Extends the original EvalRunner with judge alignment, full self-optimization
loops, and run comparison capabilities.  Works with MLflow 3.1+ APIs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd

from harness.eval_config import EvalHarnessConfig, OptimizationStrategy
from harness.scorer_registry import SCORER_REGISTRY, build_scorer_list
from harness.optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates evaluation, judge alignment, and prompt optimization.

    Args:
        config: Fully-populated EvalHarnessConfig.
        agent_fn: Callable that takes a request dict and returns a response dict.
            If ``None``, the runner assumes traces are pre-recorded.
    """

    def __init__(
        self,
        config: EvalHarnessConfig,
        agent_fn: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.agent_fn = agent_fn
        self.optimizer = PromptOptimizer(config)
        self._last_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        eval_data: pd.DataFrame,
        scorers: Optional[List[Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single evaluation pass.

        Args:
            eval_data: DataFrame with at least ``request`` and ``expected``
                columns (plus any scorer-specific columns).
            scorers: List of scorer callables.  Defaults to all enabled
                scorers from config.
            experiment_name: MLflow experiment name override.

        Returns:
            Dict with ``run_id``, ``metrics`` (name -> float), and
            ``passed`` (bool indicating all thresholds met).
        """
        if scorers is None:
            scorer_dicts = [asdict(s) for s in self.config.scorers]
            scorers = build_scorer_list(scorer_dicts)

        exp_name = experiment_name or self.config.experiment_name
        mlflow.set_experiment(exp_name)

        with mlflow.start_run() as run:
            results = mlflow.genai.evaluate(
                data=eval_data,
                predict_fn=self.agent_fn,
                scorers=scorers,
            )

            metrics = self._compute_metrics(results)
            passed = self._check_thresholds(metrics)

            mlflow.log_metrics(metrics)
            mlflow.log_param("passed_all_thresholds", passed)

            self._last_run_id = run.info.run_id
            logger.info(
                "Eval run %s complete. passed=%s metrics=%s",
                run.info.run_id, passed, metrics,
            )

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "passed": passed,
            "results_table": results.tables.get("eval_results", None),
        }

    # ------------------------------------------------------------------
    # Evaluation with optimization (original)
    # ------------------------------------------------------------------

    def run_with_optimization(
        self,
        eval_data: pd.DataFrame,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Run eval, and if thresholds fail, optimize the prompt and re-eval.

        Args:
            eval_data: Evaluation dataset.
            system_prompt: Current system prompt text.

        Returns:
            Dict with keys ``initial_run``, ``optimized_prompt``,
            ``final_run``, and ``iterations``.
        """
        initial = self.run(eval_data)
        if initial["passed"]:
            return {
                "initial_run": initial,
                "optimized_prompt": system_prompt,
                "final_run": initial,
                "iterations": 0,
            }

        current_prompt = system_prompt
        last_run = initial
        for i in range(1, self.config.max_optimization_iterations + 1):
            logger.info("Optimization iteration %d / %d", i, self.config.max_optimization_iterations)
            current_prompt = self.optimizer.optimize(
                prompt=current_prompt,
                eval_results=last_run,
                strategy=self.config.optimization_strategy,
            )
            last_run = self.run(eval_data)
            if last_run["passed"]:
                break

        return {
            "initial_run": initial,
            "optimized_prompt": current_prompt,
            "final_run": last_run,
            "iterations": i,
        }

    # ------------------------------------------------------------------
    # NEW: Evaluation with judge alignment
    # ------------------------------------------------------------------

    def run_with_judge_alignment(
        self,
        eval_data: pd.DataFrame,
        labeled_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run eval, and if thresholds fail, align the judge and re-evaluate.

        This uses human-labeled data to calibrate the LLM judge via SIMBA,
        MemAlign, or Likert-SIMBA before re-running evaluation with the
        aligned scorer.

        Args:
            eval_data: Evaluation dataset.
            labeled_data: Human-labeled traces with ground-truth assessments.

        Returns:
            Dict with ``initial_run``, ``aligned_run``, ``alignment_report``.
        """
        initial = self.run(eval_data)
        if initial["passed"]:
            return {
                "initial_run": initial,
                "aligned_run": initial,
                "alignment_report": {"status": "skipped", "reason": "all thresholds passed"},
            }

        alignment_cfg = self.config.judge_alignment
        logger.info(
            "Thresholds not met. Running judge alignment with optimizer=%s",
            alignment_cfg.optimizer,
        )

        aligned_scorer = self._align_judge(labeled_data, alignment_cfg)
        aligned_run = self.run(eval_data, scorers=[aligned_scorer])

        return {
            "initial_run": initial,
            "aligned_run": aligned_run,
            "alignment_report": {
                "status": "completed",
                "optimizer": alignment_cfg.optimizer,
                "labeled_traces_used": len(labeled_data),
            },
        }

    # ------------------------------------------------------------------
    # NEW: Full self-optimization loop
    # ------------------------------------------------------------------

    def run_full_loop(
        self,
        eval_data: pd.DataFrame,
        labeled_data: pd.DataFrame,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Execute the complete self-optimization loop.

        Steps:
            1. Initial evaluation
            2. Judge alignment (if thresholds not met)
            3. Prompt optimization (if still not met)
            4. Final re-evaluation

        Args:
            eval_data: Evaluation dataset.
            labeled_data: Human-labeled traces for judge alignment.
            system_prompt: Current system prompt text.

        Returns:
            Dict capturing every stage of the loop.
        """
        logger.info("=== Self-Optimization Loop: START ===")
        loop_result: Dict[str, Any] = {}

        # Step 1: initial eval
        initial = self.run(eval_data)
        loop_result["step1_initial_eval"] = initial
        if initial["passed"]:
            loop_result["final_status"] = "passed_initial"
            logger.info("=== Self-Optimization Loop: PASSED at step 1 ===")
            return loop_result

        # Step 2: judge alignment
        alignment_cfg = self.config.judge_alignment
        aligned_scorer = self._align_judge(labeled_data, alignment_cfg)
        aligned_run = self.run(eval_data, scorers=[aligned_scorer])
        loop_result["step2_judge_alignment"] = {
            "run": aligned_run,
            "optimizer": alignment_cfg.optimizer,
        }
        if aligned_run["passed"]:
            loop_result["final_status"] = "passed_after_judge_alignment"
            logger.info("=== Self-Optimization Loop: PASSED at step 2 ===")
            return loop_result

        # Step 3: prompt optimization
        current_prompt = system_prompt
        best_run = aligned_run
        for i in range(1, self.config.max_optimization_iterations + 1):
            logger.info("Prompt optimization iteration %d", i)
            current_prompt = self.optimizer.optimize(
                prompt=current_prompt,
                eval_results=best_run,
                strategy=self.config.optimization_strategy,
            )
            best_run = self.run(eval_data, scorers=[aligned_scorer])
            if best_run["passed"]:
                break

        loop_result["step3_prompt_optimization"] = {
            "run": best_run,
            "optimized_prompt": current_prompt,
            "iterations": i,
        }

        # Step 4: final re-eval with all scorers
        final = self.run(eval_data)
        loop_result["step4_final_eval"] = final
        loop_result["final_status"] = "passed" if final["passed"] else "failed"
        logger.info("=== Self-Optimization Loop: %s ===", loop_result["final_status"].upper())

        return loop_result

    # ------------------------------------------------------------------
    # NEW: Compare two eval runs
    # ------------------------------------------------------------------

    def compare_runs(
        self,
        run_a: Dict[str, Any],
        run_b: Dict[str, Any],
        labels: Tuple[str, str] = ("run_a", "run_b"),
    ) -> pd.DataFrame:
        """Side-by-side comparison of two evaluation runs.

        Args:
            run_a: Result dict from a previous ``run()`` call.
            run_b: Result dict from another ``run()`` call.
            labels: Display labels for the two runs.

        Returns:
            DataFrame with columns: metric, <label_a>, <label_b>, delta.
        """
        metrics_a = run_a.get("metrics", {})
        metrics_b = run_b.get("metrics", {})
        all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

        rows = []
        for key in all_keys:
            val_a = metrics_a.get(key, float("nan"))
            val_b = metrics_b.get(key, float("nan"))
            rows.append({
                "metric": key,
                labels[0]: val_a,
                labels[1]: val_b,
                "delta": val_b - val_a if not (val_a != val_a or val_b != val_b) else float("nan"),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, results: Any) -> Dict[str, float]:
        """Compute aggregate metrics from an MLflow evaluation result.

        Handles both boolean assessments (converted to pass rate) and
        numeric/Likert scores (averaged and optionally normalized).
        """
        metrics: Dict[str, float] = {}

        # Use pre-computed metrics if available
        if hasattr(results, "metrics") and results.metrics:
            for k, v in results.metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            if metrics:
                return metrics

        # Fall back to computing from the results table
        eval_table = results.tables.get("eval_results", None) if hasattr(results, "tables") else None
        if eval_table is None:
            return metrics

        for col in eval_table.columns:
            if col.startswith("assessment/") or col.endswith("_score"):
                series = eval_table[col].dropna()
                if series.empty:
                    continue

                # Boolean columns -> pass rate
                if series.dtype == bool or set(series.unique()).issubset({True, False, 0, 1}):
                    metrics[col] = float(series.astype(float).mean())
                # Numeric columns -> mean (normalize Likert 1-5 to 0-1 if max > 1)
                elif pd.api.types.is_numeric_dtype(series):
                    mean_val = float(series.mean())
                    max_val = float(series.max())
                    if max_val > 1.0:
                        # Likert-style: normalize to 0-1
                        likert_max = self.config.labeling.likert_max
                        metrics[col] = (mean_val - 1.0) / (likert_max - 1.0)
                    else:
                        metrics[col] = mean_val

        return metrics

    def _check_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check whether all configured thresholds are met."""
        for threshold in self.config.thresholds:
            value = metrics.get(threshold.metric_name)
            if value is None:
                logger.warning("Metric '%s' not found in results.", threshold.metric_name)
                continue
            if value < threshold.pass_threshold:
                logger.info(
                    "Threshold FAILED: %s = %.4f < %.4f",
                    threshold.metric_name, value, threshold.pass_threshold,
                )
                return False
        return True

    def _align_judge(self, labeled_data: pd.DataFrame, cfg: Any) -> Any:
        """Align an LLM judge using the configured optimizer.

        Returns a scorer callable that incorporates human-aligned rubrics.
        """
        from mlflow.genai.judges import create_llm_judge

        optimizer_type = cfg.optimizer.lower()

        if optimizer_type == "simba":
            from mlflow.genai.optimize import SimbaJudgeOptimizer
            opt = SimbaJudgeOptimizer(
                reflection_lm=cfg.reflection_lm,
                embedding_model=cfg.embedding_model,
                max_demos=cfg.max_demos,
            )
        elif optimizer_type == "memalign":
            from mlflow.genai.optimize import MemAlignJudgeOptimizer
            opt = MemAlignJudgeOptimizer(
                reflection_lm=cfg.reflection_lm,
                max_demos=cfg.max_demos,
            )
        elif optimizer_type == "likert_simba":
            from mlflow.genai.optimize import LikertSimbaJudgeOptimizer
            opt = LikertSimbaJudgeOptimizer(
                reflection_lm=cfg.reflection_lm,
                embedding_model=cfg.embedding_model,
                max_demos=cfg.max_demos,
                likert_max=self.config.labeling.likert_max,
            )
        else:
            raise ValueError(f"Unknown judge optimizer: {optimizer_type}")

        logger.info("Aligning judge with %s on %d labeled traces", optimizer_type, len(labeled_data))
        aligned_judge = opt.optimize(
            labeled_data=labeled_data,
            batch_size=cfg.batch_size,
        )

        return aligned_judge
