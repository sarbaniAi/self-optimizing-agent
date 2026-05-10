# Self-Optimizing Agent

A production-grade, end-to-end pipeline for evaluating, aligning, and optimizing AI agents on Databricks — with real MLflow experiments, not simulations.

## What It Does

This framework implements a **closed-loop self-optimization cycle**:

```
Agent → Traces → Evaluate → Labels → Align Judges → Optimize Prompts → Deploy
                    ↑                                                      |
                    └──── Continuous monitoring detects drift ──────────────┘
```

**Every step runs on real Databricks infrastructure:**
- Agent calls go through Model Serving endpoints
- Traces are captured in MLflow experiments
- Evaluation uses `mlflow.genai.evaluate()` with real scorers
- Judge alignment uses SIMBA/MemAlign optimizers
- Prompt optimization uses GEPA with Prompt Registry
- Results are stored in Unity Catalog Delta tables

## Architecture

```
self-optimizing-agent/
├── app/                    # Multi-page Databricks App (FastAPI)
│   ├── app.py              # 7-page sidebar: Overview → Monitor
│   └── pages/              # Backend logic per page
├── harness/                # Core evaluation framework
│   ├── eval_runner.py      # Evaluation orchestrator
│   ├── scorer_registry.py  # 30+ scorers (code + LLM)
│   ├── judge_alignment.py  # SIMBA / MemAlign / LikertSIMBA
│   ├── optimizer.py        # GEPA + 4 strategies
│   ├── prompt_registry.py  # MLflow Prompt Registry wrapper
│   ├── trace_observer.py   # Trace analysis
│   ├── labeling.py         # Ground truth labeling sessions
│   └── monitoring.py       # Continuous drift detection
├── agents/                 # Sample agents
├── config/                 # YAML configs
├── notebooks/              # Databricks notebooks (5)
├── databricks.yml          # DAB bundle config
└── pyproject.toml          # pip installable
```

## Quick Start

```bash
# Clone
git clone https://github.com/sarbaniAi/self-optimizing-agent.git
cd self-optimizing-agent

# Install
pip install -e .

# Set environment
export DATABRICKS_HOST="https://your-workspace.azuredatabricks.net"
export DATABRICKS_TOKEN="your-token"

# Run the app locally
uvicorn app.app:app --port 8765

# Or deploy to Databricks
databricks bundle deploy
```

## Notebooks

Run these in order on Databricks:

| Notebook | What It Does |
|----------|-------------|
| `00_setup.py` | Create UC schema, seed data, set up MLflow experiment |
| `01_quickstart.py` | End-to-end: agent → eval → trace tagging |
| `02_judge_alignment.py` | SIMBA vs MemAlign deep-dive |
| `03_prompt_optimization.py` | GEPA + failure-targeted patching |
| `04_monitoring_setup.py` | Drift detection + auto-optimization |

## Workspace

- **Host**: `https://adb-7405619910560146.6.azuredatabricks.net`
- **Catalog**: `classic_stable_4a2ohn_azure`
- **Schema**: `self_optimizing_agent`
- **MLflow Experiment**: `/Users/sarbani.maiti@databricks.com/self-optimizing-agent`
- **LLM**: `databricks-claude-sonnet-4-6`
- **Embedding**: `databricks-gte-large-en`

## What's New vs agent-eval-harness v1

| Feature | v1 (agent-eval-harness) | v2 (self-optimizing-agent) |
|---------|------------------------|---------------------------|
| Judge Alignment | None | SIMBA, MemAlign, LikertSIMBA |
| Prompt Optimization | 4 LLM strategies | + GEPA with Prompt Registry |
| Ground Truth | None | MLflow Review App + labeling |
| Trace Analysis | None | DC analysis with trace explorer |
| Monitoring | Config-ready | Live drift detection + auto-retrigger |
| MLflow Integration | Answer-sheet only | Full: tracing + evaluate + registry |
| UI | Single-page KPI | 7-page interactive tutorial |
| Installable | No | `pip install -e .` |
