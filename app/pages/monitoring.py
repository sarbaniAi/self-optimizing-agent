"""Backend for the Monitoring page."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_monitoring_cycle(lookback_hours: int = 168) -> dict:
    """Run a monitoring cycle and return results."""
    try:
        return _demo_monitoring_results()
    except Exception as e:
        return {"error": str(e)}


def _demo_monitoring_results() -> dict:
    """Simulated monitoring results."""
    return {
        "pass_rate": 0.88,
        "baseline_pass_rate": 0.92,
        "drift_detected": True,
        "drift_amount": -0.04,
        "action_taken": "Re-optimization triggered",
        "timestamp": "2026-05-10T06:00:00",
    }
