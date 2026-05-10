"""Self-Optimizing Agent Evaluation Harness."""
from .eval_config import (
    EvalHarnessConfig, load_config, ScorerConfig, ThresholdConfig,
    JudgeAlignmentConfig, PromptRegistryConfig, MonitoringConfig, LabelingConfig,
)

# Import modules that may have optional MLflow dependencies
try:
    from .scorer_registry import build_scorer_list
except ImportError:
    build_scorer_list = None

try:
    from .eval_runner import EvalRunner
except ImportError:
    EvalRunner = None

try:
    from .judge_alignment import JudgeAligner
except ImportError:
    JudgeAligner = None

try:
    from .prompt_registry import PromptRegistryManager
except ImportError:
    PromptRegistryManager = None

try:
    from .trace_observer import TraceObserver
except ImportError:
    TraceObserver = None

try:
    from .labeling import LabelingManager
except ImportError:
    LabelingManager = None

try:
    from .monitoring import MonitoringLoop
except ImportError:
    MonitoringLoop = None

__all__ = [
    "EvalHarnessConfig", "load_config", "EvalRunner", "build_scorer_list",
    "JudgeAligner", "PromptRegistryManager", "TraceObserver",
    "LabelingManager", "MonitoringLoop",
]
