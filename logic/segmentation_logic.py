"""Segmentation related logic extracted from the original module."""

from LungBiopsyTractPlanner import LungBiopsyTractPlannerLogic
from utils.helpers import (
    get_segment_mask_as_array,
    segment_has_content,
    ras_to_ijk,
    get_smoothed_closed_surface_representation,
    iter_valid_lobes,
    read_image_data,
)
from config.segmentator_task_settings import task_segment_map, task_name_map, roi_subset


class SegmentationLogic(LungBiopsyTractPlannerLogic):
    """Wrapper exposing segmentation methods with snake_case names."""

    def crop_volume_superoinferior(self, *args, **kwargs):
        return super().crop_volume_superoinferior(*args, **kwargs)

    def rename_segments_from_header(self, *args, **kwargs):
        return super().renameSegmentsFromHeader(*args, **kwargs)

    def run_task_sequentially(self, *args, **kwargs):
        return super().runTaskSequentially(*args, **kwargs)

    def run_total_segmentator_sequentially(self, *args, **kwargs):
        return super().runTotalSegmentatorSequentially(*args, **kwargs)

    def create_emphysema_segment(self, *args, **kwargs):
        return super().createEmphysemaSegment(*args, **kwargs)

    def add_target_region_from_largest_nodule(self, *args, **kwargs):
        return super().addTargetRegionFromLargestNodule(*args, **kwargs)

    def subtract_nodules_from_lung_vessels_clean(self, *args, **kwargs):
        return super().subtractNodulesFromLungVesselsClean(*args, **kwargs)

    def initialize_lung_mask(self, *args, **kwargs):
        return super().initializeLungMask(*args, **kwargs)

    def lung_parenchyma_distance(self, *args, **kwargs):
        return super().lungParenchymaDistance(*args, **kwargs)

    def ensure_lung_kd(self):
        return super()._ensureLungKD()

