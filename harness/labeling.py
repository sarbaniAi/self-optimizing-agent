"""
Labeling Manager
================
Manage ground truth labeling sessions via MLflow Review App.

Uses MLflow 3.1+ GenAI labeling APIs.

Workspace : https://adb-7405619910560146.6.azuredatabricks.net
Catalog   : classic_stable_4a2ohn_azure
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LabelingManager:
    """
    Manage ground truth labeling sessions via MLflow Review App.

    Usage
    -----
    >>> mgr = LabelingManager()
    >>> schema = mgr.create_label_schema("quality", schema_type="likert", likert_max=5)
    >>> session = mgr.create_labeling_session("round1", traces, schema)
    >>> progress = mgr.get_labeling_progress("round1")
    >>> labeled = mgr.export_labeled_traces("round1")
    """

    # -- Schema ---------------------------------------------------------------

    def create_label_schema(
        self,
        schema_name: str,
        schema_type: str = "likert",
        likert_max: int = 5,
        likert_min: int = 1,
        description: str = "",
    ) -> Any:
        """
        Create a labeling schema (Likert 1-5 or binary pass/fail).

        Parameters
        ----------
        schema_name : Unique name for this schema.
        schema_type : ``"likert"`` or ``"binary"``.
        likert_max  : Maximum value on the Likert scale (default 5).
        likert_min  : Minimum value on the Likert scale (default 1).
        description : Human-readable description.

        Returns
        -------
        The created LabelSchema object.
        """
        try:
            from mlflow.genai import label_schemas
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai.label_schemas is required. Install mlflow>=3.1."
            ) from exc

        logger.info(
            "Creating label schema '%s' (type=%s).", schema_name, schema_type
        )

        schema_type_lower = schema_type.lower()

        try:
            if schema_type_lower == "likert":
                schema = label_schemas.create_label_schema(
                    name=schema_name,
                    type="feedback",
                    title=schema_name,
                    description=description or f"Likert scale {likert_min}-{likert_max}",
                    input=label_schemas.InputCategorical(
                        options=[
                            label_schemas.InputCategoricalOption(
                                label=str(i), value=str(i)
                            )
                            for i in range(likert_min, likert_max + 1)
                        ]
                    ),
                    overwrite=True,
                )
            elif schema_type_lower == "binary":
                schema = label_schemas.create_label_schema(
                    name=schema_name,
                    type="feedback",
                    title=schema_name,
                    description=description or "Binary pass/fail",
                    input=label_schemas.InputCategorical(
                        options=[
                            label_schemas.InputCategoricalOption(
                                label="pass", value="pass"
                            ),
                            label_schemas.InputCategoricalOption(
                                label="fail", value="fail"
                            ),
                        ]
                    ),
                    overwrite=True,
                )
            else:
                raise ValueError(
                    f"Unknown schema_type '{schema_type}'. Choose 'likert' or 'binary'."
                )
        except AttributeError:
            # Fallback for slightly different API shapes in MLflow versions.
            logger.warning(
                "label_schemas API shape differs from expected. "
                "Attempting simplified creation."
            )
            schema = label_schemas.create_label_schema(
                name=schema_name,
                type=schema_type_lower,
            )

        logger.info("Label schema '%s' created successfully.", schema_name)
        return schema

    # -- Session --------------------------------------------------------------

    def create_labeling_session(
        self,
        session_name: str,
        traces: list,
        schema: Any,
        model_name: Optional[str] = None,
        assignees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a labeling session from evaluation traces.

        Parameters
        ----------
        session_name : Human-readable session name.
        traces       : List of MLflow trace objects (or trace IDs).
        schema       : The LabelSchema to use for labeling.
        model_name   : Optional model name for provenance.
        assignees    : Optional list of user emails to assign.

        Returns
        -------
        Dict with ``session_name``, ``trace_count``, ``review_app_url``,
        and ``session_id``.
        """
        try:
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai is required. Install mlflow>=3.1."
            ) from exc

        logger.info(
            "Creating labeling session '%s' with %d traces.",
            session_name,
            len(traces),
        )

        # Resolve trace IDs if full trace objects are passed.
        trace_ids: List[str] = []
        for t in traces:
            if isinstance(t, str):
                trace_ids.append(t)
            else:
                tid = getattr(getattr(t, "info", t), "request_id", None) or getattr(
                    t, "trace_id", None
                )
                if tid:
                    trace_ids.append(str(tid))

        if not trace_ids:
            logger.warning("No valid trace IDs found. Session will be empty.")

        try:
            session = mlflow.genai.create_labeling_session(
                name=session_name,
                trace_ids=trace_ids,
                label_schemas=[schema],
                model_name=model_name,
                assignees=assignees,
            )
        except TypeError:
            # Some MLflow versions accept slightly different kwargs.
            try:
                session = mlflow.genai.create_labeling_session(
                    name=session_name,
                    trace_ids=trace_ids,
                    schema=schema,
                )
            except Exception as exc:
                logger.error("Failed to create labeling session: %s", exc)
                raise
        except Exception as exc:
            logger.error("Failed to create labeling session: %s", exc)
            raise

        # Extract session metadata.
        session_id = getattr(session, "session_id", getattr(session, "id", None))
        review_url = getattr(session, "review_app_url", None)

        # Build review app URL from workspace if not returned directly.
        if review_url is None and session_id:
            review_url = (
                "https://adb-7405619910560146.6.azuredatabricks.net"
                f"#mlflow/review/{session_id}"
            )

        result = {
            "session_name": session_name,
            "session_id": str(session_id) if session_id else None,
            "trace_count": len(trace_ids),
            "review_app_url": review_url,
        }

        logger.info(
            "Labeling session created: id=%s, review_app_url=%s",
            result["session_id"],
            result["review_app_url"],
        )
        return result

    # -- Progress -------------------------------------------------------------

    def get_labeling_progress(self, session_name: str) -> Dict[str, Any]:
        """
        Check how many traces have been labeled in a session.

        Returns
        -------
        Dict with ``session_name``, ``total_traces``, ``labeled_count``,
        ``pending_count``, ``progress_pct``.
        """
        try:
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai is required. Install mlflow>=3.1."
            ) from exc

        logger.info("Checking labeling progress for session '%s'.", session_name)

        try:
            session = mlflow.genai.get_labeling_session(name=session_name)
        except AttributeError:
            logger.warning(
                "mlflow.genai.get_labeling_session not available. "
                "Cannot check progress."
            )
            return {
                "session_name": session_name,
                "total_traces": None,
                "labeled_count": None,
                "pending_count": None,
                "progress_pct": None,
                "error": "get_labeling_session API not available.",
            }
        except Exception as exc:
            logger.error("Failed to get session '%s': %s", session_name, exc)
            raise

        total = getattr(session, "total_traces", 0)
        labeled = getattr(session, "labeled_count", 0)

        # Fallback: count from trace list if attributes are not present.
        if total == 0:
            trace_list = getattr(session, "traces", [])
            total = len(trace_list)
            labeled = sum(
                1
                for t in trace_list
                if getattr(t, "labeled", False)
                or getattr(t, "feedback", None) is not None
            )

        pending = total - labeled
        progress = round(labeled / total * 100, 1) if total > 0 else 0.0

        result = {
            "session_name": session_name,
            "total_traces": total,
            "labeled_count": labeled,
            "pending_count": pending,
            "progress_pct": progress,
        }

        logger.info(
            "Session '%s': %d/%d labeled (%.1f%%).",
            session_name,
            labeled,
            total,
            progress,
        )
        return result

    # -- Export ---------------------------------------------------------------

    def export_labeled_traces(self, session_name: str) -> List[Any]:
        """
        Export labeled traces from a session for downstream use
        (e.g., judge alignment).

        Returns
        -------
        List of trace objects that have human feedback attached.
        """
        try:
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai is required. Install mlflow>=3.1."
            ) from exc

        logger.info("Exporting labeled traces from session '%s'.", session_name)

        try:
            session = mlflow.genai.get_labeling_session(name=session_name)
        except AttributeError:
            logger.warning(
                "mlflow.genai.get_labeling_session not available. "
                "Attempting fallback via search."
            )
            return self._export_fallback(session_name)
        except Exception as exc:
            logger.error("Failed to get session '%s': %s", session_name, exc)
            raise

        # Filter to only labeled traces.
        all_traces = getattr(session, "traces", [])
        labeled = [
            t
            for t in all_traces
            if getattr(t, "labeled", False)
            or getattr(t, "feedback", None) is not None
            or getattr(t, "human_feedback", None) is not None
        ]

        logger.info(
            "Exported %d labeled traces from session '%s'.",
            len(labeled),
            session_name,
        )
        return labeled

    def _export_fallback(self, session_name: str) -> List[Any]:
        """Fallback export via search_traces with a session tag."""
        try:
            import mlflow

            traces = mlflow.search_traces(
                filter_string=f"tag.`labeling_session` = '{session_name}'",
                max_results=500,
            )
            return [t for t in (traces or []) if t is not None]
        except Exception as exc:
            logger.error("Fallback export failed: %s", exc)
            return []
