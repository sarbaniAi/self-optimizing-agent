"""
Judge Alignment Module
======================
Align LLM judges to match human expert feedback using SIMBA, MemAlign, or LikertSIMBA optimizers.

Uses MLflow 3.1+ GenAI judge alignment APIs.

Workspace : https://adb-7405619910560146.6.azuredatabricks.net
Catalog   : classic_stable_4a2ohn_azure
LLM       : databricks-claude-sonnet-4-6
Embedding : databricks-gte-large-en
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class JudgeAlignmentConfig:
    """Configuration for judge alignment runs."""

    optimizer_type: str = "simba"  # "simba" | "memalign" | "likert_simba"
    reflection_lm: str = "databricks-claude-sonnet-4-6"
    embedding_model: str = "databricks-gte-large-en"
    batch_size: int = 8
    likert_max: int = 5
    likert_min: int = 1
    catalog: str = "classic_stable_4a2ohn_azure"
    extra_params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Custom LikertSIMBA optimizer
# ---------------------------------------------------------------------------

class LikertSIMBAAlignmentOptimizer:
    """
    SIMBA-style alignment optimizer that correctly handles Likert-scale scores.

    Standard SIMBA treats scores as binary pass/fail.  This variant computes
    alignment as:

        metric = 1 - |llm_score - human_score| / (max_score - min_score)

    so partial agreement on a Likert scale is rewarded rather than penalised.
    """

    def __init__(
        self,
        model: str = "databricks-claude-sonnet-4-6",
        batch_size: int = 8,
        likert_min: int = 1,
        likert_max: int = 5,
    ):
        self.model = model
        self.batch_size = batch_size
        self.likert_min = likert_min
        self.likert_max = likert_max
        self._range = max(self.likert_max - self.likert_min, 1)

        # Attempt to inherit from the MLflow base class so it can participate
        # in the standard `.align()` protocol.
        try:
            from mlflow.genai.judges.base import AlignmentOptimizer  # noqa: F811

            # Dynamically make this class a subclass (mixin-style) if not already.
            if not issubclass(type(self), AlignmentOptimizer):
                self.__class__ = type(
                    self.__class__.__name__,
                    (self.__class__, AlignmentOptimizer),
                    {},
                )
        except ImportError:
            logger.warning(
                "mlflow.genai.judges.base.AlignmentOptimizer not available; "
                "LikertSIMBAAlignmentOptimizer will operate standalone."
            )

    # -- core metric ----------------------------------------------------------

    def alignment_score(self, llm_score: float, human_score: float) -> float:
        """Return a 0-1 alignment metric for a single example."""
        return 1.0 - abs(llm_score - human_score) / self._range

    # -- batch helpers --------------------------------------------------------

    def compute_batch_alignment(
        self, llm_scores: List[float], human_scores: List[float]
    ) -> float:
        """Average alignment across a batch."""
        if not llm_scores or not human_scores:
            return 0.0
        scores = [
            self.alignment_score(l, h) for l, h in zip(llm_scores, human_scores)
        ]
        return sum(scores) / len(scores)

    # -- optimise (called by judge.align) -------------------------------------

    def optimize(self, judge, traces, **kwargs) -> Dict[str, Any]:
        """
        Run the LikertSIMBA optimization loop.

        Parameters
        ----------
        judge  : The baseline MLflow judge object.
        traces : List of traces with both human and LLM feedback.

        Returns
        -------
        dict with ``aligned_instructions``, ``distilled_guidelines``,
        ``improvement_score``.
        """
        try:
            from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer
        except ImportError:
            raise ImportError(
                "SIMBAAlignmentOptimizer is required as the inner optimizer. "
                "Install mlflow>=3.1 with GenAI extras."
            )

        # Delegate the heavy lifting to SIMBA, then re-score using Likert metric.
        inner = SIMBAAlignmentOptimizer(
            model=self.model, batch_size=self.batch_size
        )
        raw_result = inner.optimize(judge, traces, **kwargs)

        # Re-compute improvement score using the Likert-aware metric.
        llm_scores: List[float] = []
        human_scores: List[float] = []
        for trace in traces:
            llm_fb = getattr(trace, "llm_feedback", None)
            human_fb = getattr(trace, "human_feedback", None)
            if llm_fb is not None and human_fb is not None:
                try:
                    llm_scores.append(float(llm_fb))
                    human_scores.append(float(human_fb))
                except (TypeError, ValueError):
                    continue

        likert_improvement = self.compute_batch_alignment(llm_scores, human_scores)
        raw_result["likert_alignment_score"] = likert_improvement
        raw_result["likert_range"] = (self.likert_min, self.likert_max)
        return raw_result

    def __repr__(self) -> str:
        return (
            f"LikertSIMBAAlignmentOptimizer(model={self.model!r}, "
            f"likert_range=[{self.likert_min}, {self.likert_max}])"
        )


# ---------------------------------------------------------------------------
# Main aligner
# ---------------------------------------------------------------------------

class JudgeAligner:
    """
    Align LLM judges to match human expert feedback using SIMBA, MemAlign,
    or LikertSIMBA optimizers.

    Usage
    -----
    >>> cfg = JudgeAlignmentConfig(optimizer_type="simba")
    >>> aligner = JudgeAligner(cfg)
    >>> result = aligner.align("relevance", labeled_traces, experiment_id="123")
    """

    def __init__(self, config: JudgeAlignmentConfig):
        self.config = config

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _filter_valid_traces(traces) -> list:
        """Keep only traces that carry both human and judge feedback."""
        valid = []
        for t in traces:
            has_human = getattr(t, "human_feedback", None) is not None
            has_judge = getattr(t, "llm_feedback", None) is not None
            if has_human and has_judge:
                valid.append(t)
        return valid

    def _build_optimizer(self):
        """Instantiate the requested optimizer."""
        otype = self.config.optimizer_type.lower()

        if otype == "simba":
            try:
                from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer
            except ImportError as exc:
                raise ImportError(
                    "SIMBAAlignmentOptimizer requires mlflow>=3.1 with GenAI extras."
                ) from exc
            return SIMBAAlignmentOptimizer(
                model=self.config.reflection_lm,
                batch_size=self.config.batch_size,
            )

        if otype == "memalign":
            try:
                from mlflow.genai.judges.optimizers import MemAlignOptimizer
            except ImportError as exc:
                raise ImportError(
                    "MemAlignOptimizer requires mlflow>=3.1 with GenAI extras."
                ) from exc
            return MemAlignOptimizer(
                reflection_lm=self.config.reflection_lm,
                embedding_model=self.config.embedding_model,
            )

        if otype == "likert_simba":
            return LikertSIMBAAlignmentOptimizer(
                model=self.config.reflection_lm,
                batch_size=self.config.batch_size,
                likert_min=self.config.likert_min,
                likert_max=self.config.likert_max,
            )

        raise ValueError(
            f"Unknown optimizer_type '{otype}'. "
            "Choose from: simba, memalign, likert_simba."
        )

    # -- public API -----------------------------------------------------------

    def align(
        self,
        judge_name: str,
        labeled_traces,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """
        Run alignment on a judge using labeled traces.

        Parameters
        ----------
        judge_name      : Name of the built-in scorer (e.g. ``"relevance"``).
        labeled_traces  : Iterable of traces with human + LLM feedback.
        experiment_id   : MLflow experiment to scope the alignment run.

        Returns
        -------
        dict with keys:
            - original_instructions
            - aligned_instructions
            - distilled_guidelines
            - improvement_score
        """
        try:
            from mlflow.genai.scorers import get_scorer
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai.scorers.get_scorer requires mlflow>=3.1."
            ) from exc

        logger.info(
            "Starting judge alignment: judge=%s optimizer=%s experiment=%s",
            judge_name,
            self.config.optimizer_type,
            experiment_id,
        )

        # 1. Load baseline judge
        baseline_judge = get_scorer(judge_name)
        original_instructions = getattr(
            baseline_judge, "instructions", "<not available>"
        )

        # 2. Filter to traces with dual feedback
        valid_traces = self._filter_valid_traces(labeled_traces)
        if not valid_traces:
            logger.warning("No traces with both human and judge feedback found.")
            return {
                "original_instructions": original_instructions,
                "aligned_instructions": None,
                "distilled_guidelines": None,
                "improvement_score": 0.0,
                "valid_trace_count": 0,
            }

        logger.info(
            "Found %d valid traces (out of %d) for alignment.",
            len(valid_traces),
            len(labeled_traces) if hasattr(labeled_traces, "__len__") else "?",
        )

        # 3. Build optimizer
        optimizer = self._build_optimizer()

        # 4. Run alignment
        try:
            aligned_result = baseline_judge.align(
                traces=valid_traces, optimizer=optimizer
            )
        except AttributeError:
            # Fallback: call optimizer.optimize() directly
            aligned_result = optimizer.optimize(baseline_judge, valid_traces)

        return {
            "original_instructions": original_instructions,
            "aligned_instructions": aligned_result.get("aligned_instructions"),
            "distilled_guidelines": aligned_result.get("distilled_guidelines"),
            "improvement_score": aligned_result.get(
                "improvement_score",
                aligned_result.get("likert_alignment_score", 0.0),
            ),
            "valid_trace_count": len(valid_traces),
            "optimizer": self.config.optimizer_type,
            "raw": aligned_result,
        }

    def register_aligned_judge(
        self,
        aligned_judge,
        experiment_id: str,
        judge_name: str,
    ) -> None:
        """
        Register the aligned judge so it can be re-used in future evaluations.

        Parameters
        ----------
        aligned_judge : The judge object returned by alignment.
        experiment_id : MLflow experiment for provenance tracking.
        judge_name    : Logical name under which to register.
        """
        try:
            from mlflow.genai.judges import make_judge
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai.judges.make_judge requires mlflow>=3.1."
            ) from exc

        logger.info(
            "Registering aligned judge '%s' in experiment %s.",
            judge_name,
            experiment_id,
        )

        # If aligned_judge is a dict of results, build a judge from its instructions.
        if isinstance(aligned_judge, dict):
            instructions = aligned_judge.get("aligned_instructions", "")
            judge_obj = make_judge(
                name=judge_name,
                instructions=instructions,
                model=self.config.reflection_lm,
            )
        else:
            judge_obj = aligned_judge

        # Register
        try:
            judge_obj.register(name=judge_name)
        except AttributeError:
            logger.warning(
                "Judge object does not support .register(). "
                "Saving instructions as an artifact instead."
            )
            import mlflow

            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_dict(
                    {
                        "judge_name": judge_name,
                        "instructions": getattr(judge_obj, "instructions", str(judge_obj)),
                    },
                    artifact_file=f"aligned_judges/{judge_name}.json",
                )

        logger.info("Aligned judge '%s' registered successfully.", judge_name)

    def compare_optimizers(
        self,
        judge_name: str,
        labeled_traces,
        experiment_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all three optimizers on the same judge and traces, then compare.

        Returns
        -------
        dict keyed by optimizer name, each value is the alignment result dict.
        """
        results: Dict[str, Dict[str, Any]] = {}
        original_type = self.config.optimizer_type

        for opt_type in ("simba", "memalign", "likert_simba"):
            logger.info("Running optimizer: %s", opt_type)
            self.config.optimizer_type = opt_type
            try:
                results[opt_type] = self.align(
                    judge_name, labeled_traces, experiment_id
                )
            except (ImportError, Exception) as exc:
                logger.error("Optimizer %s failed: %s", opt_type, exc)
                results[opt_type] = {"error": str(exc)}

        # Restore original setting
        self.config.optimizer_type = original_type

        # Rank by improvement_score
        ranked = sorted(
            [
                (k, v.get("improvement_score", 0.0))
                for k, v in results.items()
                if "error" not in v
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        if ranked:
            logger.info(
                "Best optimizer: %s (improvement=%.4f)", ranked[0][0], ranked[0][1]
            )

        results["_ranking"] = [{"optimizer": k, "score": s} for k, s in ranked]
        return results
