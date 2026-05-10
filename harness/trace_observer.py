"""
Trace Observer
==============
Observe and analyse agent traces from MLflow for DC (Data Collection) analysis.

Uses MLflow 3.1+ tracing APIs.

Workspace : https://adb-7405619910560146.6.azuredatabricks.net
Catalog   : classic_stable_4a2ohn_azure
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceObserver:
    """
    Observe and analyse agent traces from MLflow for data-collection analysis.

    Usage
    -----
    >>> observer = TraceObserver()
    >>> traces = observer.search_traces("12345", max_results=50)
    >>> summary = observer.get_trace_summary("12345")
    """

    # -- Search ---------------------------------------------------------------

    def search_traces(
        self,
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 100,
    ) -> list:
        """
        Search traces with optional filters.

        Parameters
        ----------
        experiment_id : MLflow experiment ID.
        filters       : Optional dict with keys:
                        - ``start_time`` / ``end_time``: ISO-8601 strings or datetime.
                        - ``tags``: dict of tag key/value pairs.
                        - ``status``: trace status (e.g. ``"OK"``, ``"ERROR"``).
                        - ``filter_string``: raw MLflow filter expression.
        max_results   : Maximum traces to return.

        Returns
        -------
        List of MLflow Trace objects.
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow is required. Install mlflow>=3.1.") from exc

        filter_parts: List[str] = []
        if filters:
            if "filter_string" in filters:
                filter_parts.append(filters["filter_string"])
            if "status" in filters:
                filter_parts.append(f"status = '{filters['status']}'")
            if "tags" in filters:
                for k, v in filters["tags"].items():
                    filter_parts.append(f"tag.`{k}` = '{v}'")
            if "start_time" in filters:
                ts = self._to_epoch_ms(filters["start_time"])
                filter_parts.append(f"timestamp >= {ts}")
            if "end_time" in filters:
                ts = self._to_epoch_ms(filters["end_time"])
                filter_parts.append(f"timestamp <= {ts}")

        filter_string = " AND ".join(filter_parts) if filter_parts else None

        logger.info(
            "Searching traces: experiment=%s filter=%s max=%d",
            experiment_id,
            filter_string,
            max_results,
        )

        try:
            traces = mlflow.search_traces(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results,
            )
        except Exception as exc:
            logger.error("Failed to search traces: %s", exc)
            raise

        logger.info("Found %d traces.", len(traces) if traces is not None else 0)
        return traces if traces is not None else []

    # -- Tag ------------------------------------------------------------------

    def tag_traces(
        self, trace_ids: List[str], tag_key: str, tag_value: str
    ) -> Dict[str, bool]:
        """
        Tag traces for downstream workflows (e.g., ``eval: complete``).

        Parameters
        ----------
        trace_ids : List of trace IDs.
        tag_key   : Tag key.
        tag_value : Tag value.

        Returns
        -------
        Dict mapping trace_id to success boolean.
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow is required.") from exc

        results: Dict[str, bool] = {}
        for tid in trace_ids:
            try:
                mlflow.set_trace_tag(tid, tag_key, tag_value)
                results[tid] = True
                logger.debug("Tagged trace %s: %s=%s", tid, tag_key, tag_value)
            except Exception as exc:
                logger.warning("Failed to tag trace %s: %s", tid, exc)
                results[tid] = False

        tagged_count = sum(1 for v in results.values() if v)
        logger.info(
            "Tagged %d/%d traces with %s=%s.",
            tagged_count,
            len(trace_ids),
            tag_key,
            tag_value,
        )
        return results

    # -- Single-trace analysis ------------------------------------------------

    def analyze_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Deep analysis of a single trace: tool calls, latency, token usage.

        Parameters
        ----------
        trace_id : The MLflow trace ID.

        Returns
        -------
        Dict with keys: trace_id, status, total_latency_ms, spans,
        tool_calls, token_usage, error (if any).
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow is required.") from exc

        logger.info("Analysing trace %s", trace_id)

        try:
            trace = mlflow.get_trace(trace_id)
        except Exception as exc:
            logger.error("Failed to get trace %s: %s", trace_id, exc)
            return {"trace_id": trace_id, "error": str(exc)}

        if trace is None:
            return {"trace_id": trace_id, "error": "Trace not found."}

        # Extract spans
        spans = []
        tool_calls: List[Dict[str, Any]] = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        trace_data = getattr(trace, "data", trace)
        raw_spans = getattr(trace_data, "spans", [])

        for span in raw_spans:
            span_info: Dict[str, Any] = {
                "name": getattr(span, "name", "unknown"),
                "span_type": getattr(span, "span_type", None),
                "status": str(getattr(span, "status", "")),
                "start_time": getattr(span, "start_time_ns", None),
                "end_time": getattr(span, "end_time_ns", None),
            }

            # Compute span latency
            if span_info["start_time"] and span_info["end_time"]:
                span_info["latency_ms"] = (
                    span_info["end_time"] - span_info["start_time"]
                ) / 1e6
            else:
                span_info["latency_ms"] = None

            spans.append(span_info)

            # Identify tool calls
            stype = str(getattr(span, "span_type", "")).lower()
            if stype in ("tool", "function", "retriever"):
                tool_calls.append(
                    {
                        "name": span_info["name"],
                        "type": stype,
                        "latency_ms": span_info["latency_ms"],
                    }
                )

            # Aggregate token usage from attributes
            attrs = getattr(span, "attributes", {}) or {}
            if isinstance(attrs, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = attrs.get(key) or attrs.get(f"llm.token_count.{key}", 0)
                    try:
                        total_tokens[key] += int(val)
                    except (TypeError, ValueError):
                        pass

        # Overall latency from trace-level timestamps
        trace_info = getattr(trace, "info", trace)
        start_ts = getattr(trace_info, "timestamp_ms", None) or getattr(
            trace_info, "start_time_ns", None
        )
        end_ts = getattr(trace_info, "end_time_ms", None) or getattr(
            trace_info, "end_time_ns", None
        )
        total_latency_ms: Optional[float] = None
        if start_ts is not None and end_ts is not None:
            # Heuristic: if values are in ns, convert; otherwise treat as ms.
            if start_ts > 1e15:  # nanoseconds
                total_latency_ms = (end_ts - start_ts) / 1e6
            else:
                total_latency_ms = end_ts - start_ts

        return {
            "trace_id": trace_id,
            "status": str(getattr(trace_info, "status", "UNKNOWN")),
            "total_latency_ms": total_latency_ms,
            "span_count": len(spans),
            "tool_calls": tool_calls,
            "token_usage": total_tokens,
            "spans": spans,
        }

    # -- Summary --------------------------------------------------------------

    def get_trace_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Summary statistics across traces in an experiment.

        Returns
        -------
        Dict with keys: total_traces, avg_latency_ms, tool_call_distribution,
        error_rate, status_distribution.
        """
        traces = self.search_traces(experiment_id, max_results=500)
        if not traces:
            return {
                "total_traces": 0,
                "avg_latency_ms": None,
                "tool_call_distribution": {},
                "error_rate": 0.0,
                "status_distribution": {},
            }

        latencies: List[float] = []
        tool_counter: Dict[str, int] = {}
        status_counter: Dict[str, int] = {}
        error_count = 0

        for trace in traces:
            # Status
            trace_info = getattr(trace, "info", trace)
            status = str(getattr(trace_info, "status", "UNKNOWN"))
            status_counter[status] = status_counter.get(status, 0) + 1
            if status.upper() in ("ERROR", "INTERNAL_ERROR"):
                error_count += 1

            # Latency
            start_ts = getattr(trace_info, "timestamp_ms", None)
            end_ts = getattr(trace_info, "end_time_ms", None)
            if start_ts is not None and end_ts is not None:
                try:
                    latencies.append(float(end_ts) - float(start_ts))
                except (TypeError, ValueError):
                    pass

            # Tool calls from spans
            trace_data = getattr(trace, "data", trace)
            for span in getattr(trace_data, "spans", []):
                stype = str(getattr(span, "span_type", "")).lower()
                if stype in ("tool", "function", "retriever"):
                    name = getattr(span, "name", "unknown")
                    tool_counter[name] = tool_counter.get(name, 0) + 1

        total = len(traces)
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        summary = {
            "total_traces": total,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
            "tool_call_distribution": dict(
                sorted(tool_counter.items(), key=lambda x: x[1], reverse=True)
            ),
            "error_rate": round(error_count / total, 4) if total else 0.0,
            "status_distribution": status_counter,
        }

        logger.info(
            "Trace summary for experiment %s: %d traces, avg latency=%.1f ms, "
            "error rate=%.2f%%",
            experiment_id,
            total,
            avg_latency or 0,
            summary["error_rate"] * 100,
        )
        return summary

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _to_epoch_ms(value) -> int:
        """Convert a datetime or ISO string to epoch milliseconds."""
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        raise ValueError(f"Cannot convert {type(value)} to epoch ms.")
