"""
Self-Optimizing Agent — Multi-page Databricks App.
7-page sidebar navigation: Overview, Observe Traces, Evaluate, Ground Truth,
Align Judges, Optimize Prompts, Monitoring.
"""
import os
import json
import asyncio
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Self-Optimizing Agent")
executor = ThreadPoolExecutor(max_workers=2)

# Databricks workspace config
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://adb-7405619910560146.6.azuredatabricks.net")
EXPERIMENT_ID = "2478689462451681"
MLFLOW_URLS = {
    "experiment": f"{DATABRICKS_HOST}/ml/experiments/{EXPERIMENT_ID}",
    "evaluations": f"{DATABRICKS_HOST}/ml/experiments/{EXPERIMENT_ID}/evaluations",
    "agent_run": f"{DATABRICKS_HOST}/ml/experiments/{EXPERIMENT_ID}/runs/62640313125d4229a91b2d9df80c752a",
    "eval_run": f"{DATABRICKS_HOST}/ml/experiments/{EXPERIMENT_ID}/runs/1020cbc4f7ad49a09bd7fa8fda2cc218",
}

# In-memory state
state = {
    "eval_results": {},
    "alignment_results": {},
    "optimization_results": {},
    "monitoring_history": [],
    "active_page": "overview",
}


# ═══════════════════════════════════════════════════════════════
# API MODELS
# ═══════════════════════════════════════════════════════════════

class EvalRequest(BaseModel):
    config_path: str = "config/customer_support.yaml"

class AlignRequest(BaseModel):
    optimizer: str = "memalign"
    judge_name: str = "domain_guidelines"

class OptimizeRequest(BaseModel):
    strategy: str = "gepa"

class MonitorRequest(BaseModel):
    lookback_hours: int = 168


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home():
    return get_full_html()


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/mlflow-urls")
async def mlflow_urls():
    """Return MLflow experiment URLs for the UI."""
    urls = dict(MLFLOW_URLS)
    # Add latest eval run if available
    if state["eval_results"].get("result", {}).get("mlflow_urls"):
        urls.update(state["eval_results"]["result"]["mlflow_urls"])
    return urls


@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    state["eval_results"] = {"status": "running", "started": datetime.now().isoformat()}

    def _run():
        try:
            from app.pages.evaluate import run_evaluation_pipeline
            result = run_evaluation_pipeline(req.config_path)
            state["eval_results"] = {"status": "completed", "result": result}
        except Exception as e:
            state["eval_results"] = {"status": "error", "error": str(e), "trace": traceback.format_exc()}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run)
    return {"status": "started"}


@app.get("/api/eval-status")
async def eval_status():
    return state["eval_results"]


@app.post("/api/align-judges")
async def align_judges(req: AlignRequest):
    state["alignment_results"] = {"status": "running", "optimizer": req.optimizer}

    def _run():
        try:
            from app.pages.align_judges import run_alignment
            result = run_alignment(req.optimizer, req.judge_name)
            state["alignment_results"] = {"status": "completed", "result": result}
        except Exception as e:
            state["alignment_results"] = {"status": "error", "error": str(e)}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run)
    return {"status": "started"}


@app.get("/api/align-status")
async def align_status():
    return state["alignment_results"]


@app.post("/api/optimize-prompt")
async def optimize_prompt(req: OptimizeRequest):
    state["optimization_results"] = {"status": "running", "strategy": req.strategy}

    def _run():
        try:
            from app.pages.optimize_prompts import run_optimization
            result = run_optimization(req.strategy)
            state["optimization_results"] = {"status": "completed", "result": result}
        except Exception as e:
            state["optimization_results"] = {"status": "error", "error": str(e)}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run)
    return {"status": "started"}


@app.get("/api/optimize-status")
async def optimize_status():
    return state["optimization_results"]


@app.post("/api/monitoring/run")
async def run_monitoring(req: MonitorRequest):
    state["monitoring_history"].append({"status": "running", "started": datetime.now().isoformat()})

    def _run():
        try:
            from app.pages.monitoring import run_monitoring_cycle
            result = run_monitoring_cycle(req.lookback_hours)
            state["monitoring_history"][-1] = {"status": "completed", "result": result}
        except Exception as e:
            state["monitoring_history"][-1] = {"status": "error", "error": str(e)}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run)
    return {"status": "started"}


@app.get("/api/monitoring/history")
async def monitoring_history():
    return state["monitoring_history"][-10:]


@app.get("/api/traces")
async def get_traces(experiment_id: str = "", max_results: int = 50):
    try:
        from app.pages.observe_traces import get_trace_summary
        return get_trace_summary(experiment_id, max_results)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/labeling/status")
async def labeling_status():
    try:
        from app.pages.ground_truth import get_labeling_status
        return get_labeling_status()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# FULL HTML — MULTI-PAGE APP WITH SIDEBAR
# ═══════════════════════════════════════════════════════════════

def get_full_html():
    return HTML_TEMPLATE


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Self-Optimizing Agent</title>
<style>
:root {
  --bg: #faf9f5; --sidebar-bg: #f5f3ee; --card: #fff; --border: #e5e2d9;
  --text: #1a1a2e; --muted: #666; --accent: #FF3621; --green: #0e8a6c;
  --red: #dc2626; --yellow: #b47209; --blue: #0055d4; --purple: #7c3aed;
  --teal: #0d9488; --sidebar-w: 280px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }

/* Sidebar */
.sidebar {
  position: fixed; left: 0; top: 0; bottom: 0; width: var(--sidebar-w);
  background: var(--sidebar-bg); border-right: 1px solid var(--border);
  overflow-y: auto; padding: 24px 0; z-index: 100;
}
.sidebar-brand {
  padding: 0 20px 20px; border-bottom: 1px solid var(--border); margin-bottom: 16px;
}
.sidebar-brand h2 { font-size: 1.1rem; font-weight: 700; }
.sidebar-brand h2 span { color: var(--accent); }
.sidebar-brand p { font-size: .72rem; color: var(--muted); margin-top: 2px; }
.sidebar-section { padding: 0 12px; margin-bottom: 8px; }
.sidebar-section-label { font-size: .65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); padding: 8px 8px 4px; font-weight: 600; }
.nav-item {
  display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px;
  border-radius: 8px; cursor: pointer; transition: all .15s; margin-bottom: 2px;
  text-decoration: none; color: var(--text);
}
.nav-item:hover { background: rgba(0,0,0,.04); }
.nav-item.active { background: rgba(255,54,33,.08); color: var(--accent); }
.nav-icon { font-size: 1rem; width: 20px; text-align: center; flex-shrink: 0; margin-top: 2px; }
.nav-label h4 { font-size: .82rem; font-weight: 600; line-height: 1.3; }
.nav-label p { font-size: .68rem; color: var(--muted); line-height: 1.3; margin-top: 1px; }
.nav-item.active .nav-label p { color: var(--accent); opacity: .7; }
.nav-step { display: inline-block; width: 18px; height: 18px; border-radius: 50%; background: var(--border);
  color: var(--muted); font-size: .65rem; font-weight: 700; text-align: center; line-height: 18px; flex-shrink: 0; margin-top: 2px; }
.nav-item.active .nav-step { background: var(--accent); color: #fff; }

/* Main */
.main { margin-left: var(--sidebar-w); padding: 32px 40px; max-width: 1100px; }
.page { display: none; }
.page.active { display: block; }

/* Common */
h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 8px; }
h2 { font-size: 1.2rem; font-weight: 600; margin: 24px 0 12px; }
h3 { font-size: 1rem; font-weight: 600; margin: 16px 0 8px; }
.subtitle { color: var(--muted); font-size: .9rem; margin-bottom: 24px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.card-header { font-size: .9rem; font-weight: 600; margin-bottom: 12px; }
.btn { background: var(--accent); color: #fff; border: none; padding: 12px 28px; border-radius: 8px;
  font-size: .9rem; font-weight: 600; cursor: pointer; transition: opacity .2s; display: inline-flex; align-items: center; gap: 8px; }
.btn:hover { opacity: .85; }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-outline { background: transparent; color: var(--accent); border: 2px solid var(--accent); }
.btn-green { background: var(--green); }
.btn-blue { background: var(--blue); }
.btn-sm { padding: 6px 16px; font-size: .8rem; }

/* KPI */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; }
.kpi .value { font-size: 2rem; font-weight: 800; }
.kpi .label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }
.kpi.pass .value { color: var(--green); }
.kpi.fail .value { color: var(--red); }
.kpi.warn .value { color: var(--yellow); }

/* Tables */
table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
th { background: #f5f3ee; font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); padding: 10px 14px; text-align: left; }
td { padding: 10px 14px; border-top: 1px solid var(--border); font-size: .85rem; }
tr:hover { background: rgba(0,0,0,.02); }
.badge { padding: 2px 10px; border-radius: 10px; font-weight: 600; font-size: .75rem; display: inline-block; }
.badge-pass { background: rgba(14,138,108,.1); color: var(--green); }
.badge-fail { background: rgba(220,38,38,.1); color: var(--red); }
.badge-info { background: rgba(0,85,212,.1); color: var(--blue); }
.badge-purple { background: rgba(124,58,237,.1); color: var(--purple); }
.badge-teal { background: rgba(13,148,136,.1); color: var(--teal); }
.badge-default { background: rgba(0,0,0,.06); color: var(--muted); }

/* Comparison cards */
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
.compare-card { border-radius: 12px; padding: 24px; }
.compare-card.left { background: var(--card); border: 1px solid var(--border); }
.compare-card.right { background: var(--card); border: 2px solid var(--green); }
.compare-card h3 { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; }
.step-list { list-style: none; padding: 0; }
.step-list li { display: flex; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(0,0,0,.05); font-size: .85rem; }
.step-list li:last-child { border-bottom: none; }
.step-num { width: 22px; height: 22px; border-radius: 50%; background: var(--accent); color: #fff; font-size: .7rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.step-list .right .step-num { background: var(--green); }

/* Code blocks */
.code-block { background: #1e1e2e; color: #cdd6f4; border-radius: 10px; padding: 20px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: .8rem; line-height: 1.7; overflow-x: auto; margin: 12px 0; }
.code-block .kw { color: #cba6f7; }
.code-block .fn { color: #89b4fa; }
.code-block .str { color: #a6e3a1; }
.code-block .cm { color: #6c7086; }
.code-block .op { color: #f38ba8; }

/* Callouts */
.callout { border-radius: 10px; padding: 16px 20px; margin: 16px 0; font-size: .85rem; }
.callout-green { background: rgba(14,138,108,.06); border-left: 4px solid var(--green); }
.callout-blue { background: rgba(0,85,212,.06); border-left: 4px solid var(--blue); }
.callout-yellow { background: rgba(180,114,9,.06); border-left: 4px solid var(--yellow); }
.callout-red { background: rgba(220,38,38,.06); border-left: 4px solid var(--red); }

/* Progress */
.spinner { width: 32px; height: 32px; border: 3px solid var(--border); border-top: 3px solid var(--accent); border-radius: 50%; animation: spin .8s linear infinite; display: inline-block; vertical-align: middle; margin-right: 12px; }
@keyframes spin { to { transform: rotate(360deg); } }
.progress-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin: 8px 0; }
.progress-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width .5s; }

/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 20px; }
.tab { padding: 10px 20px; font-size: .85rem; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; color: var(--muted); transition: all .15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Scenario card */
.scenario { background: rgba(0,0,0,.02); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin: 16px 0; }
.scenario-label { font-weight: 700; color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .5px; }
.scenario-row { display: flex; gap: 8px; margin-top: 8px; font-size: .85rem; }
.scenario-row strong { min-width: 140px; color: var(--muted); }

/* Select */
select { padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px; font-size: .85rem; background: var(--card); cursor: pointer; }

/* Optimizer selector */
.optimizer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }
.optimizer-card { border: 2px solid var(--border); border-radius: 12px; padding: 20px; cursor: pointer; transition: all .15s; }
.optimizer-card:hover { border-color: var(--accent); }
.optimizer-card.selected { border-color: var(--green); box-shadow: 0 0 0 3px rgba(14,138,108,.12); }
.optimizer-card h4 { font-size: .9rem; margin-bottom: 8px; }
.optimizer-card ul { font-size: .8rem; color: var(--muted); padding-left: 16px; }
.optimizer-card ul li { margin-bottom: 4px; }

/* Architecture diagram */
.arch-flow { display: flex; align-items: center; gap: 0; justify-content: center; flex-wrap: wrap; margin: 24px 0; }
.arch-box { background: var(--card); border: 2px solid var(--border); border-radius: 10px; padding: 12px 18px; text-align: center; min-width: 120px; }
.arch-box h4 { font-size: .78rem; font-weight: 700; }
.arch-box p { font-size: .65rem; color: var(--muted); }
.arch-box.highlight { border-color: var(--accent); background: rgba(255,54,33,.03); }
.arch-arrow { font-size: 1.2rem; color: var(--muted); padding: 0 6px; }

.hidden { display: none; }

/* Links */
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
</head>
<body>

<!-- ═══════ SIDEBAR ═══════ -->
<div class="sidebar">
  <div class="sidebar-brand">
    <h2>Self-Optimizing <span>Agent</span></h2>
    <p>MLflow GenAI Evaluation & Optimization</p>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-section-label">Getting Started</div>
    <div class="nav-item active" onclick="showPage('overview')" id="nav-overview">
      <span class="nav-icon">&#9783;</span>
      <div class="nav-label">
        <h4>Demo Overview</h4>
        <p>Architecture & self-optimization loop</p>
      </div>
    </div>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-section-label">Optimization Pipeline</div>

    <div class="nav-item" onclick="showPage('traces')" id="nav-traces">
      <span class="nav-step">1</span>
      <div class="nav-label">
        <h4>Observe Traces</h4>
        <p>Capture agent behavior with MLflow tracing</p>
      </div>
    </div>

    <div class="nav-item" onclick="showPage('evaluate')" id="nav-evaluate">
      <span class="nav-step">2</span>
      <div class="nav-label">
        <h4>Evaluate Agent</h4>
        <p>Multi-judge scoring & recommendations</p>
      </div>
    </div>

    <div class="nav-item" onclick="showPage('ground-truth')" id="nav-ground-truth">
      <span class="nav-step">3</span>
      <div class="nav-label">
        <h4>Collect Ground Truth</h4>
        <p>Create labeled datasets through SME review</p>
      </div>
    </div>

    <div class="nav-item" onclick="showPage('align-judges')" id="nav-align-judges">
      <span class="nav-step">4</span>
      <div class="nav-label">
        <h4>Align Judges to Experts</h4>
        <p>Calibrate judges with SIMBA/MemAlign</p>
      </div>
    </div>

    <div class="nav-item" onclick="showPage('optimize')" id="nav-optimize">
      <span class="nav-step">5</span>
      <div class="nav-label">
        <h4>Optimize Prompts</h4>
        <p>Auto-improve prompts with GEPA optimizer</p>
      </div>
    </div>

    <div class="nav-item" onclick="showPage('monitoring')" id="nav-monitoring">
      <span class="nav-step">6</span>
      <div class="nav-label">
        <h4>Ongoing Monitoring</h4>
        <p>Self-optimizing cycle with drift detection</p>
      </div>
    </div>
  </div>

  <div class="sidebar-section" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);">
    <div class="sidebar-section-label">Resources</div>
    <a class="nav-item" href="https://mlflow.org/docs/latest/" target="_blank">
      <span class="nav-icon">&#128214;</span>
      <div class="nav-label"><h4>MLflow Documentation</h4></div>
    </a>
    <a class="nav-item" href="https://github.com/sarbaniAi/self-optimizing-agent" target="_blank">
      <span class="nav-icon">&#128187;</span>
      <div class="nav-label"><h4>GitHub Repository</h4></div>
    </a>
  </div>
</div>

<!-- ═══════ MAIN CONTENT ═══════ -->
<div class="main">

  <!-- ═══════ PAGE 1: OVERVIEW ═══════ -->
  <div class="page active" id="page-overview">
    <h1>Self-Optimizing Agent Framework</h1>
    <p class="subtitle">A complete pipeline for evaluating, aligning, and optimizing AI agents — from traces to production deployment.</p>

    <div class="callout callout-green" id="mlflow-links-card">
      <strong>Live MLflow Experiment</strong> — All data below is from real Databricks experiments<br>
      <div style="display:flex;gap:16px;margin-top:8px;flex-wrap:wrap">
        <a href="" id="link-experiment" target="_blank" class="btn btn-sm btn-green" style="text-decoration:none">View Experiment</a>
        <a href="" id="link-eval-run" target="_blank" class="btn btn-sm btn-blue" style="text-decoration:none">Eval Run</a>
        <a href="" id="link-evaluations" target="_blank" class="btn btn-sm btn-outline btn-sm" style="text-decoration:none;color:var(--accent)">Evaluations Tab</a>
      </div>
    </div>

    <div class="card">
      <div class="card-header">How It Works: The Self-Optimization Loop</div>
      <p style="color:var(--muted);font-size:.85rem;margin-bottom:20px">
        This framework implements a closed-loop optimization cycle. Each step feeds into the next,
        creating a continuous improvement pipeline for your agents.
      </p>

      <div class="arch-flow">
        <div class="arch-box"><h4>Agent</h4><p>Runs predictions</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Traces</h4><p>MLflow captures</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>Evaluate</h4><p>30+ scorers</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Labels</h4><p>SME ground truth</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>Align Judges</h4><p>SIMBA / MemAlign</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>Optimize</h4><p>GEPA prompts</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Deploy</h4><p>Prompt Registry</p></div>
      </div>
      <div style="text-align:center;margin-top:12px">
        <span style="font-size:.75rem;color:var(--muted)">&#8593; Continuous monitoring detects drift and re-triggers the loop &#8593;</span>
      </div>
    </div>

    <div class="compare-grid">
      <div class="card">
        <div class="card-header">What's New in v2.0</div>
        <ul style="font-size:.85rem;padding-left:20px;color:var(--muted)">
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Judge Alignment</strong> — SIMBA, MemAlign, LikertSIMBA optimizers</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">GEPA Prompt Optimization</strong> — MLflow-native prompt refinement</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Ground Truth Labeling</strong> — Review App + labeling sessions</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Prompt Registry</strong> — Version control + production aliases</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Trace Observation</strong> — DC analysis and trace exploration</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Continuous Monitoring</strong> — Drift detection + auto-optimization</li>
        </ul>
      </div>
      <div class="card">
        <div class="card-header">Carried Forward from v1.0</div>
        <ul style="font-size:.85rem;padding-left:20px;color:var(--muted)">
          <li style="margin-bottom:6px"><strong style="color:var(--text)">30+ Scorers</strong> — Code-based, LLM judges, Guidelines, built-in</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Multi-Agent Support</strong> — Routing, sequencing, handoff checks</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Security Scorers</strong> — PII, injection, cross-customer data</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Config-Driven</strong> — Full YAML control, no code changes</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">4 Optimization Strategies</strong> — Patching, few-shot, constitutional, routing</li>
          <li style="margin-bottom:6px"><strong style="color:var(--text)">Answer-Sheet Mode</strong> — Safe evaluation without recursion</li>
        </ul>
      </div>
    </div>

    <div class="card">
      <div class="card-header">Quick Start</div>
      <div class="code-block">
<span class="cm"># Install</span>
pip install -e .

<span class="cm"># Run the full pipeline in a notebook</span>
<span class="kw">from</span> harness <span class="kw">import</span> <span class="fn">load_config</span>, <span class="fn">EvalRunner</span>

config = <span class="fn">load_config</span>(<span class="str">"config/customer_support.yaml"</span>)
runner = <span class="fn">EvalRunner</span>(config)
results = runner.<span class="fn">run_full_loop</span>(eval_data, predict_fn)

<span class="cm"># Deploy as Databricks App</span>
databricks bundle deploy
      </div>
    </div>
  </div>

  <!-- ═══════ PAGE 2: OBSERVE TRACES ═══════ -->
  <div class="page" id="page-traces">
    <h1>Observe Agent Traces</h1>
    <p class="subtitle">Capture and analyze agent behavior with MLflow tracing. Understand tool calls, reasoning chains, and latency patterns.</p>

    <div class="callout callout-blue">
      <strong>Why Trace Observation?</strong> Before evaluating, you need to understand how your agent behaves.
      MLflow automatically captures every LLM call, tool invocation, and retrieval step as a trace.
    </div>

    <div class="kpi-row" id="trace-kpis">
      <div class="kpi"><div class="value" id="trace-total">--</div><div class="label">Total Traces</div></div>
      <div class="kpi"><div class="value" id="trace-avg-latency">--</div><div class="label">Avg Latency</div></div>
      <div class="kpi"><div class="value" id="trace-tool-calls">--</div><div class="label">Avg Tool Calls</div></div>
      <div class="kpi"><div class="value" id="trace-error-rate">--</div><div class="label">Error Rate</div></div>
    </div>

    <div class="card">
      <div class="card-header">How to Capture Traces</div>
      <div class="code-block">
<span class="kw">import</span> mlflow

<span class="cm"># Auto-trace all OpenAI calls</span>
mlflow.openai.<span class="fn">autolog</span>()

<span class="cm"># Manual tracing for custom logic</span>
<span class="kw">with</span> mlflow.<span class="fn">start_span</span>(name=<span class="str">"tool_execution"</span>) <span class="kw">as</span> span:
    result = execute_tool(tool_name, args)
    span.<span class="fn">set_attribute</span>(<span class="str">"tool.name"</span>, tool_name)

<span class="cm"># Tag traces for downstream workflows</span>
mlflow.<span class="fn">set_trace_tag</span>(trace_id, <span class="str">"eval"</span>, <span class="str">"complete"</span>)
      </div>
    </div>

    <h2>Recent Traces</h2>
    <table id="traces-table">
      <thead><tr><th>Trace ID</th><th>Timestamp</th><th>Latency</th><th>Tool Calls</th><th>Status</th><th>Tags</th></tr></thead>
      <tbody>
        <tr><td colspan="6" style="text-align:center;color:var(--muted);padding:24px">
          Connect to an MLflow experiment to see traces. Use the notebooks to generate sample traces.
        </td></tr>
      </tbody>
    </table>
  </div>

  <!-- ═══════ PAGE 3: EVALUATE ═══════ -->
  <div class="page" id="page-evaluate">
    <h1>Evaluate Agent</h1>
    <p class="subtitle">Run multi-judge evaluation with 30+ scorers. Get actionable recommendations for improvement.</p>

    <div class="card">
      <div class="card-header">Evaluation Configuration</div>
      <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">
        <select id="eval-config-select">
          <option value="config/customer_support.yaml">Customer Support Agent</option>
        </select>
        <button class="btn" id="eval-run-btn" onclick="runEvaluation()">Run Evaluation</button>
      </div>
    </div>

    <div id="eval-progress" class="hidden" style="text-align:center;padding:32px">
      <div class="spinner"></div>
      <span style="font-weight:600">Running evaluation...</span>
      <p style="color:var(--muted);font-size:.82rem;margin-top:8px">Scoring agent responses with configured judges</p>
    </div>

    <div id="eval-results" class="hidden">
      <div class="kpi-row" id="eval-kpis"></div>

      <h2>Scorer Breakdown</h2>
      <table id="eval-scorer-table">
        <thead><tr><th>Scorer</th><th>Type</th><th>Passed</th><th>Failed</th><th>Pass Rate</th><th>Status</th></tr></thead>
        <tbody></tbody>
      </table>

      <div id="eval-mlflow-links" class="callout callout-green" style="display:none;margin-top:16px">
        <strong>View in MLflow</strong> — All results are stored in the Databricks experiment<br>
        <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap">
          <a href="" id="eval-link-run" target="_blank" class="btn btn-sm btn-green" style="text-decoration:none">View Eval Run</a>
          <a href="" id="eval-link-experiment" target="_blank" class="btn btn-sm btn-blue" style="text-decoration:none">View Experiment</a>
          <a href="" id="eval-link-evaluations" target="_blank" class="btn btn-sm btn-outline btn-sm" style="text-decoration:none;color:var(--accent)">Compare Evaluations</a>
        </div>
      </div>

      <h2 style="margin-top:24px">Recommendations</h2>
      <div id="eval-recommendations" class="callout callout-yellow" style="display:none">
        <strong>Based on evaluation results:</strong>
        <ul id="eval-rec-list" style="margin-top:8px;padding-left:20px;font-size:.85rem"></ul>
      </div>
    </div>

    <div class="card" style="margin-top:24px">
      <div class="card-header">Example Code</div>
      <div class="code-block">
<span class="kw">from</span> harness <span class="kw">import</span> <span class="fn">load_config</span>, <span class="fn">EvalRunner</span>

config = <span class="fn">load_config</span>(<span class="str">"config/customer_support.yaml"</span>)
runner = <span class="fn">EvalRunner</span>(config)

<span class="cm"># Pre-compute agent outputs (answer-sheet mode)</span>
eval_data = [
    {<span class="str">"inputs"</span>: {<span class="str">"question"</span>: <span class="str">"What's your return policy?"</span>},
     <span class="str">"outputs"</span>: agent.<span class="fn">predict</span>({<span class="str">"question"</span>: <span class="str">"What's your return policy?"</span>})},
]

results = runner.<span class="fn">run</span>(eval_data, predict_fn=<span class="kw">None</span>)
      </div>
    </div>
  </div>

  <!-- ═══════ PAGE 4: GROUND TRUTH ═══════ -->
  <div class="page" id="page-ground-truth">
    <h1>Collect Ground Truth Labels</h1>
    <p class="subtitle">Create labeled datasets through SME review sessions. Ground truth is essential for judge alignment.</p>

    <div class="callout callout-green">
      <strong>Why Ground Truth?</strong> LLM judges are only as good as their calibration.
      By collecting expert labels, you can align judges to match domain-specific quality standards.
    </div>

    <div class="card">
      <div class="card-header">Labeling Workflow</div>
      <div class="arch-flow">
        <div class="arch-box"><h4>Eval Traces</h4><p>From Step 2</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Create Session</h4><p>Define schema</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>SME Review</h4><p>Review App</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Labeled Data</h4><p>For alignment</p></div>
      </div>
    </div>

    <div class="compare-grid">
      <div class="card">
        <div class="card-header">Labeling Schema: Likert Scale (1-5)</div>
        <table>
          <thead><tr><th>Score</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td><span class="badge badge-fail">1</span></td><td>Unacceptable — completely wrong or harmful</td></tr>
            <tr><td><span class="badge" style="background:rgba(220,38,38,.06);color:#b91c1c">2</span></td><td>Poor — major issues, mostly incorrect</td></tr>
            <tr><td><span class="badge" style="background:rgba(180,114,9,.06);color:var(--yellow)">3</span></td><td>Acceptable — adequate but could improve</td></tr>
            <tr><td><span class="badge" style="background:rgba(14,138,108,.06);color:var(--green)">4</span></td><td>Good — correct and helpful</td></tr>
            <tr><td><span class="badge badge-pass">5</span></td><td>Excellent — thorough, well-structured, expert-level</td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="card-header">Labeling Progress</div>
        <div id="labeling-progress">
          <div style="text-align:center;padding:24px;color:var(--muted)">
            No active labeling sessions. Run an evaluation first, then create a session.
          </div>
        </div>
        <button class="btn btn-outline btn-sm" style="margin-top:12px" onclick="createLabelingSession()">Create Labeling Session</button>
      </div>
    </div>

    <div class="card">
      <div class="card-header">Example Code</div>
      <div class="code-block">
<span class="kw">from</span> harness <span class="kw">import</span> <span class="fn">LabelingManager</span>

labeler = <span class="fn">LabelingManager</span>()

<span class="cm"># Create labeling schema</span>
schema = labeler.<span class="fn">create_label_schema</span>(
    schema_name=<span class="str">"support_quality"</span>,
    schema_type=<span class="str">"likert"</span>,
    likert_max=<span class="op">5</span>
)

<span class="cm"># Create session from eval traces</span>
session = labeler.<span class="fn">create_labeling_session</span>(
    session_name=<span class="str">"cs_review_may_2026"</span>,
    traces=eval_traces,
    schema=schema
)

<span class="cm"># Check progress</span>
progress = labeler.<span class="fn">get_labeling_progress</span>(<span class="str">"cs_review_may_2026"</span>)
print(f<span class="str">"Labeled: {progress['completed']}/{progress['total']}"</span>)
      </div>
    </div>
  </div>

  <!-- ═══════ PAGE 5: ALIGN JUDGES ═══════ -->
  <div class="page" id="page-align-judges">
    <h1>Align Judges to Expert Feedback</h1>
    <p class="subtitle">Calibrate judges to match coaching expertise using SIMBA or MemAlign optimizers</p>

    <div class="callout callout-blue">
      <strong>Judge alignment automatically calibrates your judges to match expert preferences.</strong>
      The optimizer analyzes disagreements between human labels and judge scores, then refines judge instructions
      to encode domain-specific expertise. The result: scalable quality assessment that reflects coaching judgment,
      not generic LLM preferences.
    </div>

    <h2>How Judge Alignment Works: SIMBA vs MemAlign</h2>

    <div class="compare-grid">
      <div class="compare-card left">
        <h3>
          <span class="badge badge-default">MLflow 3.8 default</span>
          SIMBA
        </h3>
        <p style="font-size:.82rem;color:var(--muted);margin-bottom:16px;font-style:italic">Stochastic Introspective Mini-Batch Ascent</p>
        <ol class="step-list">
          <li><span class="step-num">1</span><div><strong>Find Disagreements</strong><br><span style="font-size:.8rem;color:var(--muted)">Identify traces where judge score != coach label</span></div></li>
          <li><span class="step-num">2</span><div><strong>Analyze Failures</strong><br><span style="font-size:.8rem;color:var(--muted)">Reflection LLM examines why the judge was wrong</span></div></li>
          <li><span class="step-num">3</span><div><strong>Propose Edits</strong><br><span style="font-size:.8rem;color:var(--muted)">Generate specific instruction refinements</span></div></li>
          <li><span class="step-num">4</span><div><strong>Iterate</strong><br><span style="font-size:.8rem;color:var(--muted)">Repeat until alignment improves (multiple iterations)</span></div></li>
        </ol>
      </div>
      <div class="compare-card right">
        <h3>
          <span class="badge badge-pass">MLflow 3.9+ default</span>
          MemAlign
        </h3>
        <p style="font-size:.82rem;color:var(--teal);margin-bottom:16px;font-style:italic">Dual-Memory Framework (Recommended)</p>
        <ol class="step-list right">
          <li><span class="step-num" style="background:var(--green)">1</span><div><strong>Build Semantic Memory</strong><br><span style="font-size:.8rem;color:var(--muted)">Extract general rules from labeled data</span></div></li>
          <li><span class="step-num" style="background:var(--green)">2</span><div><strong>Build Episodic Memory</strong><br><span style="font-size:.8rem;color:var(--muted)">Store specific examples as reference cases</span></div></li>
          <li><span class="step-num" style="background:var(--green)">3</span><div><strong>Apply Both</strong><br><span style="font-size:.8rem;color:var(--muted)">Use principles + examples to judge new traces</span></div></li>
          <li><span class="step-num" style="background:var(--green)">&#10003;</span><div><strong>Fast & Efficient</strong><br><span style="font-size:.8rem;color:var(--muted)">Single pass — no iterative refinement needed</span></div></li>
        </ol>
      </div>
    </div>

    <table style="margin:20px 0">
      <thead><tr><th>Characteristic</th><th>SIMBA</th><th>MemAlign</th></tr></thead>
      <tbody>
        <tr><td>Speed</td><td>Multiple iterations (slower)</td><td style="color:var(--green);font-weight:600">Single pass (100x faster)</td></tr>
        <tr><td>Cost</td><td>More LLM calls</td><td style="color:var(--green);font-weight:600">10x cheaper</td></tr>
        <tr><td>Min Examples</td><td>~20-30 labeled traces</td><td style="color:var(--green);font-weight:600">2-10 labeled traces</td></tr>
        <tr><td>Approach</td><td>Iterative instruction editing</td><td>Dual-memory learning</td></tr>
        <tr><td>Output</td><td>Refined instructions</td><td>Instructions + distilled guidelines</td></tr>
        <tr><td>When to Use</td><td>MLflow 3.8 or below</td><td style="color:var(--green);font-weight:600">MLflow 3.9+ (recommended)</td></tr>
      </tbody>
    </table>

    <h2>What Each Optimizer Learns from the Same Disagreement</h2>

    <div class="scenario">
      <div class="scenario-label">Example Scenario</div>
      <div class="scenario-row"><strong>Question:</strong> "What is the typical response time for order inquiries?"</div>
      <div class="scenario-row"><strong>Agent Response:</strong> Discusses general customer service policies but doesn't provide specific SLA data</div>
      <div class="scenario-row"><strong>Baseline Judge Score:</strong> <span class="badge badge-pass">5/5</span> (thought general discussion = good answer)</div>
      <div class="scenario-row"><strong>Expert Label:</strong> <span class="badge badge-fail">2/5</span> (wanted specific SLA metrics, not generalities)</div>
    </div>

    <div class="compare-grid">
      <div class="card">
        <h3><span class="badge badge-default">SIMBA</span> Learns Through Iterative Editing</h3>
        <div style="background:#f5f3ee;border-radius:8px;padding:14px;margin-top:8px;font-size:.82rem">
          <strong>Adds to instructions:</strong><br>
          <em>"If the input question asks for specific metrics, SLAs, or quantitative data (keywords: 'response time', 'SLA', 'how long'), then check that the output contains concrete numbers or data, not just general descriptions."</em>
        </div>
        <p style="font-size:.78rem;color:var(--muted);margin-top:8px">Result: Refines instructions through multiple iterations</p>
      </div>
      <div class="card" style="border:2px solid var(--green)">
        <h3><span class="badge badge-pass">MemAlign</span> Learns Through Dual Memory</h3>
        <div style="margin-top:8px">
          <p style="font-size:.78rem;font-weight:600;color:var(--green)">Semantic Memory (general rule):</p>
          <div style="background:rgba(14,138,108,.06);border-radius:8px;padding:10px;font-size:.82rem;margin:4px 0 12px">
            "Questions asking for metrics require specific quantitative data in the response"
          </div>
          <p style="font-size:.78rem;font-weight:600;color:var(--yellow)">Episodic Memory (specific example):</p>
          <div style="background:rgba(180,114,9,.06);border-radius:8px;padding:10px;font-size:.82rem;margin:4px 0">
            "Trace #tr-2a91cd: Expert wanted specific SLA numbers but got general discussion -> gave 2/5"
          </div>
        </div>
        <p style="font-size:.78rem;color:var(--muted);margin-top:8px">Result: Applies both principle + example to future evaluations</p>
      </div>
    </div>

    <div class="callout callout-green">
      <strong>Outcome:</strong> Both produce aligned judges, but MemAlign is faster and needs fewer examples.<br>
      For most use cases with MLflow 3.9+, MemAlign is the recommended approach. It learns just as well but at a fraction of the cost and latency.
    </div>

    <h2>Select Optimizer</h2>

    <div style="margin-bottom:16px">
      <label style="font-size:.85rem;font-weight:600">Optimization Algorithm</label>
      <div class="tabs" style="margin-top:8px">
        <div class="tab active" onclick="selectOptimizer('memalign', this)">MemAlign <span class="badge badge-pass" style="margin-left:4px;font-size:.65rem">MLflow 3.9+ default</span></div>
        <div class="tab" onclick="selectOptimizer('simba', this)">SIMBA</div>
        <div class="tab" onclick="selectOptimizer('likert_simba', this)">LikertSIMBA</div>
      </div>
    </div>

    <div id="optimizer-desc" class="card">
      <h3>MemAlign (Dual-Memory Framework) — Recommended</h3>
      <ul style="font-size:.85rem;padding-left:20px;color:var(--muted)">
        <li>Fast single-pass learning with semantic + episodic memory</li>
        <li>Works with as few as 2-10 labeled traces</li>
        <li>100x faster, 10x cheaper than SIMBA</li>
        <li><strong style="color:var(--text)">Best for:</strong> MLflow 3.9+ and most use cases</li>
      </ul>
    </div>

    <div style="text-align:center;margin:24px 0">
      <button class="btn btn-green" id="align-btn" onclick="runAlignment()">Run MemAlign Alignment</button>
    </div>

    <div id="align-progress" class="hidden" style="text-align:center;padding:24px">
      <div class="spinner"></div>
      <span style="font-weight:600">Running alignment...</span>
      <p style="color:var(--muted);font-size:.82rem;margin-top:4px">The demo will simulate the alignment process and show you the refined judge instructions</p>
    </div>

    <div id="align-results" class="hidden">
      <h2>Alignment Results</h2>
      <div class="compare-grid">
        <div class="card">
          <div class="card-header">Original Judge Instructions</div>
          <pre id="align-original" style="font-size:.8rem;white-space:pre-wrap;color:var(--muted)"></pre>
        </div>
        <div class="card" style="border:2px solid var(--green)">
          <div class="card-header">Aligned Judge Instructions</div>
          <pre id="align-new" style="font-size:.8rem;white-space:pre-wrap;color:var(--text)"></pre>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:24px">
      <div class="card-header">Step 1: Load labeled traces and baseline judge <a href="https://mlflow.org/docs/latest/" target="_blank" style="font-size:.78rem">View Docs</a></div>
      <div class="code-block">
<span class="kw">import</span> mlflow
<span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> <span class="fn">get_scorer</span>
<span class="kw">from</span> mlflow.genai.judges.optimizers <span class="kw">import</span> <span class="fn">MemAlignOptimizer</span>

<span class="cm"># Load baseline judge</span>
judge = <span class="fn">get_scorer</span>(name=<span class="str">"domain_guidelines"</span>)

<span class="cm"># Load labeled traces (same as SIMBA)</span>
valid_traces = mlflow.<span class="fn">search_traces</span>(...)  <span class="cm"># Same filtering logic</span>

<span class="cm"># Run MemAlign optimization (MLflow 3.9+ default)</span>
aligned_judge = judge.<span class="fn">align</span>(
    traces=valid_traces,
    optimizer=<span class="fn">MemAlignOptimizer</span>(
        reflection_lm=<span class="str">"databricks-claude-sonnet-4-6"</span>,
        embedding_model=<span class="str">"databricks-gte-large-en"</span>
    )
)

print(<span class="str">"Original instructions:"</span>, judge.instructions)
print(<span class="str">"Aligned instructions:"</span>, aligned_judge.instructions)
<span class="cm"># MemAlign produces distilled guidelines in addition to refined instructions</span>
print(<span class="str">"Distilled Guidelines:"</span>, aligned_judge.distilled_guidelines)
      </div>
    </div>

    <div class="card">
      <div class="card-header">Step 2: Register and use aligned judge <a href="https://mlflow.org/docs/latest/" target="_blank" style="font-size:.78rem">View Docs</a></div>
      <p style="color:var(--muted);font-size:.85rem;margin-bottom:16px">
        After alignment, register the improved judge for use in future evaluations. The aligned judge now reflects coaching expertise at scale.
      </p>
      <div class="code-block">
<span class="kw">from</span> mlflow.genai.judges <span class="kw">import</span> <span class="fn">make_judge</span>

<span class="cm"># Register aligned judge for use in evaluations</span>
aligned_judge_registered = <span class="fn">make_judge</span>(
    name=<span class="str">"domain_guidelines_aligned"</span>,
    instructions=aligned_judge.instructions,
    feedback_value_type=float,
)

aligned_judge_registered.<span class="fn">register</span>(experiment_id=EXPERIMENT_ID)

<span class="cm"># Now use the aligned judge in evaluations</span>
<span class="kw">from</span> mlflow.genai <span class="kw">import</span> <span class="fn">evaluate</span>

results = <span class="fn">evaluate</span>(
    data=eval_dataset,
    predict_fn=agent_predict,
    scorers=[aligned_judge_registered]  <span class="cm"># Uses expert-calibrated judge</span>
)
      </div>
    </div>

    <div class="card">
      <div class="card-header">Interactive Notebook</div>
      <p style="font-size:.85rem;color:var(--muted)">
        Align judges to coaching expertise using SIMBA or MemAlign optimizers<br>
        For a hands-on experience with the code examples on this page, open the accompanying Jupyter notebook:
      </p>
      <div style="background:#f5f3ee;border-radius:8px;padding:12px 16px;margin-top:12px;font-family:monospace;font-size:.85rem">
        02_judge_alignment.py
      </div>
      <p style="font-size:.78rem;color:var(--accent);margin-top:8px">
        The notebook contains all the code from this page plus additional examples and exercises you can run locally.
      </p>
    </div>
  </div>

  <!-- ═══════ PAGE 6: OPTIMIZE PROMPTS ═══════ -->
  <div class="page" id="page-optimize">
    <h1>Optimize Agent Prompts</h1>
    <p class="subtitle">Automatically improve prompts using GEPA optimizer with aligned judge as objective function.</p>

    <div class="callout callout-blue">
      <strong>GEPA (Generative Enhancement Prompt Algorithm)</strong> generates prompt variants, evaluates each with
      the aligned judge, and selects the best variant. Combined with the Prompt Registry, this creates a
      version-controlled, gated promotion pipeline.
    </div>

    <div class="card">
      <div class="card-header">Optimization Pipeline</div>
      <div class="arch-flow">
        <div class="arch-box"><h4>Current Prompt</h4><p>From Registry</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>GEPA</h4><p>Generate variants</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Evaluate</h4><p>Aligned judge</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Score</h4><p>Objective fn</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>Promote</h4><p>If improved</p></div>
      </div>
    </div>

    <div class="compare-grid">
      <div class="card">
        <div class="card-header">Available Strategies</div>
        <table>
          <thead><tr><th>Strategy</th><th>Description</th><th>Status</th></tr></thead>
          <tbody>
            <tr><td><strong>GEPA</strong></td><td>MLflow-native prompt optimization with aligned judge</td><td><span class="badge badge-pass">Recommended</span></td></tr>
            <tr><td><strong>Failure Patching</strong></td><td>Analyze failures and add specific rules</td><td><span class="badge badge-info">Available</span></td></tr>
            <tr><td><strong>Few-Shot Injection</strong></td><td>Add real examples + correct responses</td><td><span class="badge badge-info">Available</span></td></tr>
            <tr><td><strong>Constitutional Rewrite</strong></td><td>Rewrite with principles + self-check</td><td><span class="badge badge-info">Available</span></td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="card-header">Prompt Registry</div>
        <div id="prompt-versions">
          <table>
            <thead><tr><th>Version</th><th>Alias</th><th>Score</th><th>Created</th></tr></thead>
            <tbody>
              <tr><td>v1</td><td><span class="badge badge-pass">production</span></td><td>--</td><td>Initial</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div style="text-align:center;margin:24px 0">
      <select id="optimize-strategy">
        <option value="gepa">GEPA (Recommended)</option>
        <option value="failure_targeted_patching">Failure Targeted Patching</option>
        <option value="few_shot_injection">Few-Shot Injection</option>
        <option value="constitutional_rewrite">Constitutional Rewrite</option>
      </select>
      <button class="btn" onclick="runOptimization()" style="margin-left:12px">Run Optimization</button>
    </div>

    <div id="opt-progress" class="hidden" style="text-align:center;padding:24px">
      <div class="spinner"></div>
      <span style="font-weight:600">Optimizing prompt...</span>
    </div>

    <div id="opt-results" class="hidden">
      <h2>Optimization Results</h2>
      <div class="kpi-row">
        <div class="kpi" id="opt-before"><div class="value">--</div><div class="label">Before Score</div></div>
        <div class="kpi pass" id="opt-after"><div class="value">--</div><div class="label">After Score</div></div>
        <div class="kpi" id="opt-improvement"><div class="value">--</div><div class="label">Improvement</div></div>
        <div class="kpi" id="opt-promoted"><div class="value">--</div><div class="label">Promoted</div></div>
      </div>
    </div>

    <div class="card" style="margin-top:24px">
      <div class="card-header">Example Code: GEPA Optimization</div>
      <div class="code-block">
<span class="kw">import</span> mlflow
<span class="kw">from</span> mlflow.genai <span class="kw">import</span> <span class="fn">optimize_prompts</span>
<span class="kw">from</span> mlflow.genai.optimizers <span class="kw">import</span> <span class="fn">GepaPromptOptimizer</span>

<span class="cm"># Load current prompt from registry</span>
prompt = mlflow.genai.<span class="fn">load_prompt</span>(<span class="str">"prompts:/my_agent_prompt@production"</span>)

<span class="cm"># Define objective function (normalize Likert 1-5 to 0-1)</span>
<span class="kw">def</span> <span class="fn">objective_function</span>(feedback):
    <span class="kw">return</span> feedback.value / <span class="op">5.0</span>

<span class="cm"># Run GEPA optimization</span>
result = <span class="fn">optimize_prompts</span>(
    predict_fn=agent_predict,
    train_data=training_dataset,
    prompt_uris=[<span class="str">"prompts:/my_agent_prompt"</span>],
    optimizer=<span class="fn">GepaPromptOptimizer</span>(reflection_model=<span class="str">"databricks-claude-sonnet-4-6"</span>),
    scorers=[aligned_judge],
    aggregation=objective_function,
)

<span class="cm"># Promote if improved</span>
<span class="kw">if</span> result.final_score > result.initial_score:
    mlflow.genai.<span class="fn">set_prompt_alias</span>(<span class="str">"my_agent_prompt"</span>, <span class="str">"production"</span>, result.version)
      </div>
    </div>
  </div>

  <!-- ═══════ PAGE 7: MONITORING ═══════ -->
  <div class="page" id="page-monitoring">
    <h1>Ongoing Monitoring</h1>
    <p class="subtitle">Self-optimizing cycle: continuous evaluation, drift detection, and automatic re-optimization.</p>

    <div class="callout callout-yellow">
      <strong>Continuous Improvement:</strong> Schedule periodic evaluations. When quality drifts below threshold,
      the system automatically re-runs judge alignment and prompt optimization — creating a truly self-optimizing agent.
    </div>

    <div class="card">
      <div class="card-header">Monitoring Configuration</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
        <div>
          <label style="font-size:.78rem;color:var(--muted);font-weight:600">Schedule</label>
          <p style="font-size:.9rem">Daily (6:00 AM IST)</p>
        </div>
        <div>
          <label style="font-size:.78rem;color:var(--muted);font-weight:600">Drift Threshold</label>
          <p style="font-size:.9rem">5% drop triggers re-optimization</p>
        </div>
        <div>
          <label style="font-size:.78rem;color:var(--muted);font-weight:600">Auto Re-trigger</label>
          <p style="font-size:.9rem">Enabled</p>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">Self-Optimization Cycle</div>
      <div class="arch-flow">
        <div class="arch-box"><h4>Scheduled Eval</h4><p>Daily/weekly</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Compare</h4><p>vs Baseline</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box highlight"><h4>Drift?</h4><p>Score drop > 5%</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Re-Align</h4><p>Judges</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Re-Optimize</h4><p>Prompts</p></div>
        <span class="arch-arrow">&#8594;</span>
        <div class="arch-box"><h4>Deploy</h4><p>If improved</p></div>
      </div>
    </div>

    <h2>Monitoring History</h2>
    <table id="monitoring-table">
      <thead><tr><th>Run</th><th>Timestamp</th><th>Pass Rate</th><th>Drift</th><th>Action Taken</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td colspan="6" style="text-align:center;color:var(--muted);padding:24px">
          No monitoring runs yet. Configure the monitoring job or run manually.
        </td></tr>
      </tbody>
    </table>

    <div style="text-align:center;margin:24px 0">
      <button class="btn" onclick="runMonitoringCycle()">Run Monitoring Cycle Now</button>
    </div>

    <div class="card" style="margin-top:24px">
      <div class="card-header">Example Code: Set Up Monitoring</div>
      <div class="code-block">
<span class="kw">from</span> harness <span class="kw">import</span> <span class="fn">load_config</span>, <span class="fn">MonitoringLoop</span>

config = <span class="fn">load_config</span>(<span class="str">"config/customer_support.yaml"</span>)
monitor = <span class="fn">MonitoringLoop</span>(config)

<span class="cm"># Set baseline from last known good eval</span>
monitor.<span class="fn">set_baseline</span>({<span class="str">"overall_pass_rate"</span>: <span class="op">0.92</span>, <span class="str">"security_pass_rate"</span>: <span class="op">1.0</span>})

<span class="cm"># Run one monitoring cycle</span>
result = monitor.<span class="fn">run_monitoring_cycle</span>(
    eval_data=eval_data,
    predict_fn=agent_predict,
    experiment_id=EXPERIMENT_ID
)

<span class="cm"># Check drift</span>
<span class="kw">if</span> result[<span class="str">"drift_detected"</span>]:
    print(f<span class="str">"Drift: {result['drift_amount']:.1%} — re-optimizing..."</span>)
      </div>
    </div>
  </div>

</div><!-- /main -->

<!-- ═══════ JAVASCRIPT ═══════ -->
<script>
let currentPage = 'overview';
let selectedOptimizer = 'memalign';

// Load MLflow URLs on page load
fetch('/api/mlflow-urls').then(r=>r.json()).then(urls => {
  if (urls.experiment) document.getElementById('link-experiment').href = urls.experiment;
  if (urls.eval_run) document.getElementById('link-eval-run').href = urls.eval_run;
  if (urls.evaluations) document.getElementById('link-evaluations').href = urls.evaluations;
}).catch(()=>{});

function showPage(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + pageId).classList.add('active');
  const navEl = document.getElementById('nav-' + pageId);
  if (navEl) navEl.classList.add('active');
  currentPage = pageId;
}

function selectOptimizer(opt, el) {
  selectedOptimizer = opt;
  document.querySelectorAll('.tabs .tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');

  const desc = document.getElementById('optimizer-desc');
  const btn = document.getElementById('align-btn');
  const descs = {
    memalign: {title: 'MemAlign (Dual-Memory Framework) -- Recommended', items: ['Fast single-pass learning with semantic + episodic memory','Works with as few as 2-10 labeled traces','100x faster, 10x cheaper than SIMBA','Best for: MLflow 3.9+ and most use cases'], btnText: 'Run MemAlign Alignment'},
    simba: {title: 'SIMBA (Stochastic Introspective Mini-Batch Ascent)', items: ['Hill-climbing optimization with mini-batch evaluation','Analyzes failures and proposes instruction edits','More LLM calls but can find nuanced improvements','Best for: MLflow 3.8 or complex alignment tasks'], btnText: 'Run SIMBA Alignment'},
    likert_simba: {title: 'LikertSIMBA (Likert-Aware SIMBA)', items: ['Custom metric: handles 5-point Likert scales properly','Distance-based scoring (not binary)','Better for continuous quality metrics','Best for: Likert-scale feedback schemas'], btnText: 'Run LikertSIMBA Alignment'},
  };
  const d = descs[opt];
  desc.innerHTML = `<h3>${d.title}</h3><ul style="font-size:.85rem;padding-left:20px;color:var(--muted)">${d.items.map(i=>'<li>'+i+'</li>').join('')}</ul>`;
  btn.textContent = d.btnText;
}

async function runEvaluation() {
  const btn = document.getElementById('eval-run-btn');
  btn.disabled = true;
  document.getElementById('eval-progress').classList.remove('hidden');
  document.getElementById('eval-results').classList.add('hidden');

  try {
    const configPath = document.getElementById('eval-config-select').value;
    await fetch('/api/evaluate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({config_path:configPath})});

    // Poll
    let done = false;
    while (!done) {
      await new Promise(r => setTimeout(r, 3000));
      const resp = await fetch('/api/eval-status');
      const data = await resp.json();
      if (data.status === 'completed') {
        done = true;
        renderEvalResults(data.result);
      } else if (data.status === 'error') {
        done = true;
        alert('Error: ' + data.error);
      }
    }
  } catch(e) { alert('Error: ' + e.message); }

  btn.disabled = false;
  document.getElementById('eval-progress').classList.add('hidden');
}

function renderEvalResults(data) {
  if (!data) return;
  const rate = data.pass_rate || 0;
  const kpiClass = rate >= 90 ? 'pass' : rate >= 70 ? 'warn' : 'fail';
  document.getElementById('eval-kpis').innerHTML = `
    <div class="kpi ${kpiClass}"><div class="value">${rate}%</div><div class="label">Pass Rate</div></div>
    <div class="kpi"><div class="value">${data.total || '--'}</div><div class="label">Test Cases</div></div>
    <div class="kpi pass"><div class="value">${data.passed || '--'}</div><div class="label">Passed</div></div>
    <div class="kpi ${(data.failed||0) > 0 ? 'fail' : 'pass'}"><div class="value">${data.failed || 0}</div><div class="label">Failed</div></div>
  `;

  if (data.scorer_stats) {
    const tbody = document.querySelector('#eval-scorer-table tbody');
    tbody.innerHTML = '';
    for (const [name, stats] of Object.entries(data.scorer_stats)) {
      const r = Math.round((stats.passed / Math.max(stats.total,1)) * 100);
      tbody.innerHTML += `<tr><td><strong>${name}</strong></td><td>${stats.type||''}</td><td>${stats.passed}</td><td>${stats.total-stats.passed}</td><td>${r}%</td><td><span class="badge ${r>=90?'badge-pass':'badge-fail'}">${r>=90?'PASS':'FAIL'}</span></td></tr>`;
    }
  }

  if (data.recommendations && data.recommendations.length > 0) {
    const recEl = document.getElementById('eval-recommendations');
    recEl.style.display = 'block';
    document.getElementById('eval-rec-list').innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
  }

  // Show MLflow links
  if (data.mlflow_urls) {
    const linksEl = document.getElementById('eval-mlflow-links');
    linksEl.style.display = 'block';
    if (data.mlflow_urls.eval_run) document.getElementById('eval-link-run').href = data.mlflow_urls.eval_run;
    if (data.mlflow_urls.experiment) document.getElementById('eval-link-experiment').href = data.mlflow_urls.experiment;
    if (data.mlflow_urls.evaluations_tab) document.getElementById('eval-link-evaluations').href = data.mlflow_urls.evaluations_tab;
    // Also update overview links
    document.getElementById('link-eval-run').href = data.mlflow_urls.eval_run || '';
  }

  document.getElementById('eval-results').classList.remove('hidden');
}

async function runAlignment() {
  const btn = document.getElementById('align-btn');
  btn.disabled = true;
  document.getElementById('align-progress').classList.remove('hidden');
  document.getElementById('align-results').classList.add('hidden');

  try {
    await fetch('/api/align-judges', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({optimizer:selectedOptimizer, judge_name:'domain_guidelines'})});

    let done = false;
    while (!done) {
      await new Promise(r => setTimeout(r, 3000));
      const resp = await fetch('/api/align-status');
      const data = await resp.json();
      if (data.status === 'completed') {
        done = true;
        if (data.result) {
          document.getElementById('align-original').textContent = data.result.original_instructions || 'N/A';
          document.getElementById('align-new').textContent = data.result.aligned_instructions || 'N/A';
          document.getElementById('align-results').classList.remove('hidden');
        }
      } else if (data.status === 'error') {
        done = true;
        alert('Error: ' + data.error);
      }
    }
  } catch(e) { alert('Error: ' + e.message); }

  btn.disabled = false;
  document.getElementById('align-progress').classList.add('hidden');
}

async function runOptimization() {
  const strategy = document.getElementById('optimize-strategy').value;
  document.getElementById('opt-progress').classList.remove('hidden');
  document.getElementById('opt-results').classList.add('hidden');

  try {
    await fetch('/api/optimize-prompt', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({strategy})});

    let done = false;
    while (!done) {
      await new Promise(r => setTimeout(r, 3000));
      const resp = await fetch('/api/optimize-status');
      const data = await resp.json();
      if (data.status === 'completed' || data.status === 'error') {
        done = true;
        if (data.result) {
          document.querySelector('#opt-before .value').textContent = (data.result.before_score || '--') + '';
          document.querySelector('#opt-after .value').textContent = (data.result.after_score || '--') + '';
          document.querySelector('#opt-improvement .value').textContent = (data.result.improvement || '--') + '';
          document.querySelector('#opt-promoted .value').textContent = data.result.promoted ? 'Yes' : 'No';
          document.getElementById('opt-results').classList.remove('hidden');
        }
        if (data.status === 'error') alert('Error: ' + data.error);
      }
    }
  } catch(e) { alert('Error: ' + e.message); }

  document.getElementById('opt-progress').classList.add('hidden');
}

function createLabelingSession() {
  alert('Create a labeling session by running notebook 01_quickstart.py first, then use the MLflow Review App to label traces.');
}

async function runMonitoringCycle() {
  try {
    await fetch('/api/monitoring/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({lookback_hours:168})});
    alert('Monitoring cycle started. Check back in a few minutes.');
  } catch(e) { alert('Error: ' + e.message); }
}
</script>
</body>
</html>"""
