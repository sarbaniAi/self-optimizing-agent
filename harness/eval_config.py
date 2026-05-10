"""
Configuration loader for the Self-Optimizing Agent Harness.
Reads YAML config and provides typed access to all settings.
"""
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class ScorerConfig:
    name: str
    type: str = "builtin"  # builtin, llm_judge, code, guidelines
    enabled: bool = True
    layer: str = ""
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    overall_pass_rate: float = 0.90
    security_pass_rate: float = 0.99
    orchestrator_pass_rate: float = 0.95
    regression_tolerance: float = 0.02


@dataclass
class MonitoringScorerConfig:
    name: str
    sample_rate: float = 1.0


@dataclass
class OptimizationStrategy:
    name: str
    enabled: bool = True
    scope: str = "agent"
    description: str = ""


@dataclass
class JudgeAlignmentConfig:
    optimizer: str = "memalign"  # simba | memalign | likert_simba
    reflection_lm: str = "databricks-claude-sonnet-4-6"
    embedding_model: str = "databricks-gte-large-en"
    batch_size: int = 8
    max_demos: int = 10
    min_labeled_traces: int = 5
    judge_name: str = "domain_guidelines"


@dataclass
class PromptRegistryConfig:
    prompt_name: str = "agent_system_prompt"
    initial_version: int = 1
    production_alias: str = "production"


@dataclass
class MonitoringConfig:
    schedule: str = "daily"
    drift_threshold: float = 0.05
    auto_retrigger: bool = True
    lookback_window_hours: int = 168


@dataclass
class LabelingConfig:
    schema_name: str = "agent_quality"
    schema_type: str = "likert"  # likert | binary
    likert_max: int = 5
    session_name_prefix: str = "labeling_session"


@dataclass
class EvalHarnessConfig:
    """Complete configuration for a self-optimizing evaluation harness run."""
    # Metadata
    name: str = ""
    version: str = "1.0"
    description: str = ""
    agent_type: str = "single_agent"

    # Environment
    catalog: str = ""
    schema: str = ""
    llm_endpoint: str = ""
    strong_llm_endpoint: str = ""
    embedding_endpoint: str = ""

    # Scorers
    scorers: List[ScorerConfig] = field(default_factory=list)

    # Thresholds
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Monitoring scorers
    monitoring_scorers: List[MonitoringScorerConfig] = field(default_factory=list)

    # Optimization
    optimization_strategies: List[OptimizationStrategy] = field(default_factory=list)
    auto_optimize_trigger: str = ""
    max_iterations: int = 3
    require_human_approval: bool = True

    # NEW: Judge alignment
    judge_alignment: JudgeAlignmentConfig = field(default_factory=JudgeAlignmentConfig)

    # NEW: Prompt registry
    prompt_registry: PromptRegistryConfig = field(default_factory=PromptRegistryConfig)

    # NEW: Enhanced monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # NEW: Labeling
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    # Agent config (raw)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    feedback_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"


def load_config(yaml_path: str) -> EvalHarnessConfig:
    """Load and parse a YAML configuration file."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    meta = raw.get("metadata", {})
    env = raw.get("environment", {})
    eval_cfg = raw.get("evaluation", {})
    mon_section = raw.get("monitoring", {})
    opt = raw.get("optimization", {})

    # Parse scorers
    scorers = [
        ScorerConfig(
            name=s["name"], type=s.get("type", "builtin"), enabled=s.get("enabled", True),
            layer=s.get("layer", ""), description=s.get("description", ""),
            config=s.get("config", {})
        )
        for s in eval_cfg.get("scorers", [])
    ]

    # Parse thresholds
    thresh_raw = eval_cfg.get("thresholds", {})
    thresholds = ThresholdConfig(**{
        k: v for k, v in thresh_raw.items()
        if k in ThresholdConfig.__dataclass_fields__
    })

    # Parse monitoring scorers
    monitoring_scorers = [
        MonitoringScorerConfig(name=m["name"], sample_rate=m.get("sample_rate", 1.0))
        for m in mon_section.get("scorers", [])
    ]

    # Parse optimization strategies
    strategies = [
        OptimizationStrategy(
            name=s["name"], enabled=s.get("enabled", True),
            scope=s.get("scope", "agent"), description=s.get("description", "")
        )
        for s in opt.get("strategies", [])
    ]

    auto_opt = opt.get("auto_optimize", {})

    # Parse NEW sections
    ja_raw = raw.get("judge_alignment", {})
    judge_alignment = JudgeAlignmentConfig(**{
        k: v for k, v in ja_raw.items()
        if k in JudgeAlignmentConfig.__dataclass_fields__
    }) if ja_raw else JudgeAlignmentConfig()

    pr_raw = raw.get("prompt_registry", {})
    prompt_registry = PromptRegistryConfig(**{
        k: v for k, v in pr_raw.items()
        if k in PromptRegistryConfig.__dataclass_fields__
    }) if pr_raw else PromptRegistryConfig()

    mon_config_raw = {k: v for k, v in mon_section.items() if k != "scorers"}
    monitoring_cfg = MonitoringConfig(**{
        k: v for k, v in mon_config_raw.items()
        if k in MonitoringConfig.__dataclass_fields__
    }) if mon_config_raw else MonitoringConfig()

    lab_raw = raw.get("labeling", {})
    labeling = LabelingConfig(**{
        k: v for k, v in lab_raw.items()
        if k in LabelingConfig.__dataclass_fields__
    }) if lab_raw else LabelingConfig()

    return EvalHarnessConfig(
        name=meta.get("name", ""),
        version=meta.get("version", "1.0"),
        description=meta.get("description", ""),
        agent_type=meta.get("agent_type", "single_agent"),
        catalog=env.get("catalog", ""),
        schema=env.get("schema", ""),
        llm_endpoint=env.get("llm_endpoint", ""),
        strong_llm_endpoint=env.get("strong_llm_endpoint", ""),
        embedding_endpoint=env.get("embedding_endpoint", ""),
        scorers=scorers,
        thresholds=thresholds,
        monitoring_scorers=monitoring_scorers,
        optimization_strategies=strategies,
        auto_optimize_trigger=auto_opt.get("trigger", ""),
        max_iterations=auto_opt.get("max_iterations", 3),
        require_human_approval=auto_opt.get("require_human_approval", True),
        judge_alignment=judge_alignment,
        prompt_registry=prompt_registry,
        monitoring=monitoring_cfg,
        labeling=labeling,
        agent_config=raw.get("agent", {}),
        data_config=raw.get("data", {}),
        dataset_config=eval_cfg.get("dataset", {}),
        feedback_config=raw.get("feedback", {}),
    )
