"""Backend for the Ground Truth Labeling page."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_labeling_status() -> dict:
    """Get current labeling session status."""
    try:
        from harness.labeling import LabelingManager
        labeler = LabelingManager()
        return labeler.get_labeling_progress("latest")
    except Exception:
        return {
            "active_sessions": 0,
            "total_traces": 0,
            "labeled": 0,
            "pending": 0,
            "review_app_url": None,
        }
