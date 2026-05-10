"""
Monitoring Loop
===============
Continuous self-optimization monitoring loop with drift detection.

Uses MLflow 3.1+ evaluation APIs and Delta tables for state tracking.

Workspace : https://adb-7405619910560146.6.azuredatabricks.net
Catalog   : classic_stable_4a2ohn_azure
LLM       : databricks-claude-sonnet-4-6
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalHarnessConfig:
    """Configuration for the monitoring loop."""

    catalog: str = "classic_stable_4a2ohn_azure"
    schema: str = "self_optimizing_agent"
    monitoring_table: str = "monitoring_history"
    drift_threshold: float = 0.05
    scorers: List[str] = field(default_factory=lambda: ["relevance", "safety"])
    model_name: str = "databricks-claude-sonnet-4-6"
    auto_optimize: bool = False  # Whether to trigger re-optimization automatically.
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def monitoring_table_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.monitoring_table}"


# ---------------------------------------------------------------------------
# Monitoring Loop
# ---------------------------------------------------------------------------

class MonitoringLoop:
    """
    Continuous self-optimization monitoring loop.

    Runs evaluation cycles, detects quality drift against baselines,
    and optionally triggers re-optimization when scores degrade.

    Usage
    -----
    >>> cfg = EvalHarnessConfig(drift_threshold=0.05)
    >>> monitor = MonitoringLoop(cfg)
    >>> monitor.set_baseline({"relevance": 0.92, "safety": 0.98})
    >>> result = monitor.run_monitoring_cycle(eval_data, predict_fn, experiment_id="123")
    """

    def __init__(self, config: EvalHarnessConfig):
        self.config = config
        self.baseline_scores: Dict[str, float] = {}

    # -- Baseline management --------------------------------------------------

    def set_baseline(self, scores: Dict[str, float]) -> None:
        """
        Set baseline scores for drift detection.

        Parameters
        ----------
        scores : Dict mapping scorer name to baseline score (0-1).
        """
        self.baseline_scores = dict(scores)
        logger.info("Baseline scores set: %s", self.baseline_scores)

    # -- Core monitoring cycle ------------------------------------------------

    def run_monitoring_cycle(
        self,
        eval_data,
        predict_fn: Callable,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """
        Run one monitoring cycle: evaluate, check drift, optionally optimise.

        Parameters
        ----------
        eval_data      : Evaluation dataset (list of dicts or pandas DataFrame).
        predict_fn     : The agent's predict function ``(input) -> output``.
        experiment_id  : MLflow experiment ID.

        Returns
        -------
        Dict with ``current_scores``, ``drift_report``, ``action_taken``,
        ``timestamp``, and ``run_id``.
        """
        cycle_start = datetime.now(timezone.utc)
        logger.info(
            "Starting monitoring cycle at %s (experiment=%s).",
            cycle_start.isoformat(),
            experiment_id,
        )

        # 1. Run evaluation
        current_scores, run_id = self._run_evaluation(
            eval_data, predict_fn, experiment_id
        )

        # 2. Detect drift
        drift_report = self.detect_drift(
            current_scores,
            self.baseline_scores,
            threshold=self.config.drift_threshold,
        )

        # 3. Decide action
        action_taken = "none"
        if drift_report["drift_detected"]:
            logger.warning(
                "Quality drift detected in %d scorer(s): %s",
                len(drift_report["drifted_scorers"]),
                drift_report["drifted_scorers"],
            )
            if self.config.auto_optimize:
                action_taken = "re-optimization triggered"
                logger.info("Auto-optimize is ON. Triggering re-optimization.")
                # The actual re-optimization is delegated to the caller
                # (e.g., PromptRegistryManager + JudgeAligner).
            else:
                action_taken = "drift flagged (manual review)"
                logger.info(
                    "Auto-optimize is OFF. Flagging for manual review."
                )

        # 4. Log to monitoring table
        record = {
            "timestamp": cycle_start.isoformat(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "current_scores": current_scores,
            "baseline_scores": dict(self.baseline_scores),
            "drift_report": drift_report,
            "action_taken": action_taken,
        }
        self._log_to_monitoring_table(record)

        logger.info(
            "Monitoring cycle complete. Action: %s. Scores: %s",
            action_taken,
            current_scores,
        )

        return {
            "current_scores": current_scores,
            "drift_report": drift_report,
            "action_taken": action_taken,
            "timestamp": cycle_start.isoformat(),
            "run_id": run_id,
        }

    # -- Drift detection ------------------------------------------------------

    def detect_drift(
        self,
        current_scores: Dict[str, float],
        baseline_scores: Dict[str, float],
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect quality drift by comparing current vs baseline scores.

        A scorer is flagged as drifted if its score drops by more than
        ``threshold`` (absolute) below the baseline.

        Parameters
        ----------
        current_scores  : Current evaluation scores by scorer name.
        baseline_scores : Baseline scores by scorer name.
        threshold       : Absolute score drop threshold (default 0.05).

        Returns
        -------
        Dict with ``drift_detected`` (bool), ``drifted_scorers`` (list),
        and per-scorer ``details``.
        """
        details: Dict[str, Dict[str, Any]] = {}
        drifted: List[str] = []

        for scorer, current in current_scores.items():
            baseline = baseline_scores.get(scorer)
            if baseline is None:
                details[scorer] = {
                    "current": current,
                    "baseline": None,
                    "delta": None,
                    "drifted": False,
                    "note": "No baseline available.",
                }
                continue

            delta = current - baseline
            is_drifted = delta < -threshold

            details[scorer] = {
                "current": round(current, 4),
                "baseline": round(baseline, 4),
                "delta": round(delta, 4),
                "threshold": threshold,
                "drifted": is_drifted,
            }

            if is_drifted:
                drifted.append(scorer)
                logger.warning(
                    "Drift detected for '%s': current=%.4f baseline=%.4f delta=%.4f",
                    scorer,
                    current,
                    baseline,
                    delta,
                )

        return {
            "drift_detected": len(drifted) > 0,
            "drifted_scorers": drifted,
            "threshold": threshold,
            "details": details,
        }

    # -- Monitoring history ---------------------------------------------------

    def get_monitoring_history(
        self, experiment_id: str, lookback_hours: int = 168
    ) -> List[Dict[str, Any]]:
        """
        Get monitoring run history from the Delta table.

        Parameters
        ----------
        experiment_id  : Filter to this experiment.
        lookback_hours : How many hours back to look (default 168 = 7 days).

        Returns
        -------
        List of monitoring records (most recent first).
        """
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
        except ImportError:
            spark = None

        if spark is None:
            logger.warning(
                "SparkSession not available. "
                "Cannot read monitoring history from Delta table."
            )
            return []

        table_fqn = self.config.monitoring_table_fqn

        try:
            from datetime import timedelta

            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            ).isoformat()

            df = spark.sql(
                f"""
                SELECT *
                FROM {table_fqn}
                WHERE experiment_id = '{experiment_id}'
                  AND timestamp >= '{cutoff}'
                ORDER BY timestamp DESC
                """
            )
            rows = [row.asDict() for row in df.collect()]
            logger.info(
                "Retrieved %d monitoring records (last %d hours).",
                len(rows),
                lookback_hours,
            )
            return rows
        except Exception as exc:
            logger.warning(
                "Failed to read monitoring history from '%s': %s",
                table_fqn,
                exc,
            )
            return []

    # -- Internal helpers -----------------------------------------------------

    def _run_evaluation(
        self,
        eval_data,
        predict_fn: Callable,
        experiment_id: str,
    ) -> tuple:
        """
        Run MLflow evaluation and return ``(scores_dict, run_id)``.
        """
        try:
            import mlflow
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow and mlflow.genai are required. Install mlflow>=3.1."
            ) from exc

        # Build scorers list
        scorers = []
        for scorer_name in self.config.scorers:
            try:
                from mlflow.genai.scorers import get_scorer

                scorers.append(get_scorer(scorer_name))
            except Exception:
                logger.warning(
                    "Could not load scorer '%s'; skipping.", scorer_name
                )

        if not scorers:
            logger.error("No valid scorers loaded. Returning empty scores.")
            return {}, None

        logger.info(
            "Running evaluation with %d scorers: %s",
            len(scorers),
            [s.name if hasattr(s, "name") else str(s) for s in scorers],
        )

        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                eval_result = mlflow.genai.evaluate(
                    predict_fn=predict_fn,
                    data=eval_data,
                    scorers=scorers,
                )

                # Extract aggregate scores from the result
                scores: Dict[str, float] = {}
                metrics = getattr(eval_result, "metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        try:
                            scores[key] = float(value)
                        except (TypeError, ValueError):
                            pass

                # Fallback: try aggregated_results
                if not scores:
                    agg = getattr(eval_result, "aggregated_results", {})
                    if isinstance(agg, dict):
                        for key, value in agg.items():
                            try:
                                scores[key] = float(value)
                            except (TypeError, ValueError):
                                pass

                run_id = run.info.run_id
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            raise

        logger.info("Evaluation complete (run=%s): %s", run_id, scores)
        return scores, run_id

    def _log_to_monitoring_table(self, record: Dict[str, Any]) -> None:
        """Persist a monitoring record to the Delta table."""
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
        except ImportError:
            spark = None

        if spark is None:
            logger.info(
                "SparkSession not available. Logging monitoring record to MLflow only."
            )
            self._log_to_mlflow(record)
            return

        table_fqn = self.config.monitoring_table_fqn

        try:
            # Ensure schema exists.
            spark.sql(
                f"CREATE SCHEMA IF NOT EXISTS "
                f"{self.config.catalog}.{self.config.schema}"
            )

            # Serialise nested dicts to JSON strings for Delta storage.
            flat_record = {
                "timestamp": record["timestamp"],
                "experiment_id": record["experiment_id"],
                "run_id": record.get("run_id"),
                "current_scores_json": json.dumps(record.get("current_scores", {})),
                "baseline_scores_json": json.dumps(record.get("baseline_scores", {})),
                "drift_detected": record.get("drift_report", {}).get(
                    "drift_detected", False
                ),
                "drifted_scorers_json": json.dumps(
                    record.get("drift_report", {}).get("drifted_scorers", [])
                ),
                "action_taken": record.get("action_taken", "none"),
            }

            row_df = spark.createDataFrame([flat_record])
            row_df.write.format("delta").mode("append").saveAsTable(table_fqn)
            logger.info("Monitoring record persisted to %s.", table_fqn)
        except Exception as exc:
            logger.warning(
                "Failed to write to Delta table '%s': %s. "
                "Falling back to MLflow logging.",
                table_fqn,
                exc,
            )
            self._log_to_mlflow(record)

    @staticmethod
    def _log_to_mlflow(record: Dict[str, Any]) -> None:
        """Fallback: log monitoring record as MLflow metrics/params."""
        try:
            import mlflow

            with mlflow.start_run(
                experiment_id=record.get("experiment_id"),
                run_name=f"monitoring_{record['timestamp']}",
                nested=True,
            ):
                mlflow.log_param("action_taken", record.get("action_taken", "none"))
                mlflow.log_param(
                    "drift_detected",
                    record.get("drift_report", {}).get("drift_detected", False),
                )
                for scorer, score in record.get("current_scores", {}).items():
                    mlflow.log_metric(f"current_{scorer}", score)
                for scorer, score in record.get("baseline_scores", {}).items():
                    mlflow.log_metric(f"baseline_{scorer}", score)
        except Exception as exc:
            logger.error("Failed to log monitoring record to MLflow: %s", exc)
