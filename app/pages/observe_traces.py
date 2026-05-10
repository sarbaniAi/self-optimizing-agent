"""Backend for the Observe Traces page."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_trace_summary(experiment_id: str = "", max_results: int = 50) -> dict:
    """Get trace summary stats."""
    try:
        from harness.trace_observer import TraceObserver
        observer = TraceObserver()
        return observer.get_trace_summary(experiment_id)
    except Exception:
        return {
            "total_traces": 0,
            "avg_latency": "--",
            "avg_tool_calls": "--",
            "error_rate": "--",
            "traces": [],
        }
