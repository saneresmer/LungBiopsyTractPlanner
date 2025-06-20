"""Refactored entry point using modular logic classes."""

from logic.segmentation_logic import SegmentationLogic
from logic.tract_analysis_logic import TractAnalysisLogic


class LungBiopsyTractPlannerRefactored:
    """Simple container for the refactored logic classes."""

    def __init__(self, widget=None, log_callback=None):
        self.segmentation_logic = SegmentationLogic(widget=widget, logCallback=log_callback)
        self.tract_analysis_logic = TractAnalysisLogic(widget=widget, logCallback=log_callback)


__all__ = [
    "SegmentationLogic",
    "TractAnalysisLogic",
    "LungBiopsyTractPlannerRefactored",
]

