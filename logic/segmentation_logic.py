"""Segmentation related logic extracted from the original module."""
import numpy as np
import vtk
import slicer
from scipy.spatial import cKDTree
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
    """Implements segmentation routines using snake_case method names."""

    def crop_volume_superoinferior(self, *args, **kwargs):
        return super().crop_volume_superoinferior(*args, **kwargs)

    def rename_segments_from_header(self, *args, **kwargs):
        return super().renameSegmentsFromHeader(*args, **kwargs)

    def run_task_sequentially(self, *args, **kwargs):
        return super().runTaskSequentially(*args, **kwargs)

    def run_total_segmentator_sequentially(self, *args, **kwargs):
        return super().runTotalSegmentatorSequentially(*args, **kwargs)

    # ------------------------------------------------------------------
    # Segmentation utilities
    # ------------------------------------------------------------------
    def create_emphysema_segment(self, inputVolumeNode, combinedSegmentationNode):
        """Create an emphysema/bulla segment from lobar masks."""
        if not inputVolumeNode or not inputVolumeNode.GetImageData():
            slicer.util.errorDisplay("A valid input volume was not selected.")
            return None

        inputArray = slicer.util.arrayFromVolume(inputVolumeNode)
        segmentation = combinedSegmentationNode.GetSegmentation()

        lobe_segment_ids = []
        for i in range(segmentation.GetNumberOfSegments()):
            segName = segmentation.GetNthSegment(i).GetName()
            if "lobe" in segName.lower():
                segID = segmentation.GetSegmentIdBySegmentName(segName)
                lobe_segment_ids.append(segID)

        if not lobe_segment_ids:
            slicer.util.errorDisplay("Lobe segments not found.")
            return None

        combined_lung_mask = None
        for segID in lobe_segment_ids:
            tmpLabelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TmpLobeMask")
            idArray = vtk.vtkStringArray()
            idArray.InsertNextValue(segID)

            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(
                combinedSegmentationNode, idArray, tmpLabelmapNode, inputVolumeNode
            )

            lobMaskArray = slicer.util.arrayFromVolume(tmpLabelmapNode)
            slicer.mrmlScene.RemoveNode(tmpLabelmapNode)

            if combined_lung_mask is None:
                combined_lung_mask = lobMaskArray.copy()
            else:
                combined_lung_mask = np.maximum(combined_lung_mask, lobMaskArray)

        self.lungMaskNumpy = combined_lung_mask.copy()
        self._lungKD = None

        lung_hu_array = inputArray * combined_lung_mask
        emphysema_mask = np.zeros_like(lung_hu_array, dtype=np.uint8)
        emphysema_mask[(combined_lung_mask > 0) & (lung_hu_array >= -1000) & (lung_hu_array <= -930)] = 1

        if np.count_nonzero(emphysema_mask) == 0:
            slicer.util.errorDisplay("No voxels within the emphysema range. Segment was not created.")
            return None

        emphysemaLMNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "EmphysemaLM")
        slicer.util.updateVolumeFromArray(emphysemaLMNode, emphysema_mask)
        emphysemaLMNode.SetOrigin(inputVolumeNode.GetOrigin())
        emphysemaLMNode.SetSpacing(inputVolumeNode.GetSpacing())
        m = vtk.vtkMatrix4x4()
        inputVolumeNode.GetIJKToRASMatrix(m)
        emphysemaLMNode.SetIJKToRASMatrix(m)

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "EmphysemaSegmentation")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(emphysemaLMNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(emphysemaLMNode)

        segment = segmentationNode.GetSegmentation().GetNthSegment(0)
        if segment:
            segment.SetName("emphysema_bulla")
            segment.SetColor(0.0, 0.0, 139 / 255.0)

        return segmentationNode


    def add_target_region_from_largest_nodule(self, *args, **kwargs):
        return super().addTargetRegionFromLargestNodule(*args, **kwargs)

    def subtract_nodules_from_lung_vessels_clean(self, *args, **kwargs):
        return super().subtractNodulesFromLungVesselsClean(*args, **kwargs)

    def initialize_lung_mask(self, segmentationNode, volumeNode):
        self.segmentationNode = segmentationNode
        self.inputVolumeNode = volumeNode
        segmentationNode.GetSegmentation().SetConversionParameter("ReferenceImageGeometry", volumeNode.GetID())
        segmentationNode.CreateBinaryLabelmapRepresentation()

        segmentation = segmentationNode.GetSegmentation()
        lobe_masks = [
            mask for (_, mask) in iter_valid_lobes(segmentationNode, volumeNode, return_mask=True)
            if mask is not None and np.any(mask)
        ]
        if lobe_masks:
            lungMask = np.logical_or.reduce(lobe_masks).astype(np.uint8)
        else:
            self.logCallback("[WARN] No valid lung lobes were found!")
            lungMask = np.zeros(slicer.util.arrayFromVolume(volumeNode).shape, dtype=np.uint8)

        self.lungMaskNumpy = lungMask
        self.ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(self.ijkToRAS)
        self._lungKD = None

    def lung_parenchyma_distance(self, p_start, p_end):
        if self._lungKD is None:
            vox = np.argwhere(self.lungMaskNumpy > 0)
            ijkHomogeneous = np.column_stack((vox[:, 2], vox[:, 1], vox[:, 0], np.ones(len(vox))))
            ijkToRAS_np = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    ijkToRAS_np[i, j] = self.ijkToRAS.GetElement(i, j)
            rasCoords = (ijkToRAS_np @ ijkHomogeneous.T).T[:, :3]
            self._lungKD = cKDTree(rasCoords)

        p0 = np.array(p_start)
        p1 = np.array(p_end)
        L = np.linalg.norm(p1 - p0)
        if L == 0:
            return 0.0
        step_mm = 2.0
        steps = max(int(L / step_mm), 1)
        vec = (p1 - p0) / steps
        mids = p0 + vec * (np.arange(steps)[:, None] + 0.5)
        inside = self._lungKD.query_ball_point(mids, r=1.0)
        inside_mm = sum(bool(lst) for lst in inside) * np.linalg.norm(vec)
        return inside_mm

    def ensure_lung_kd(self):
        if not hasattr(self, "lungMaskNumpy"):
            self.initialize_lung_mask(self.segmentationNode, self.inputVolumeNode)
        if self._lungKD is not None:
            return
        vox = np.argwhere(self.lungMaskNumpy > 0)
        if vox.size == 0:
            self._lungKD = cKDTree(np.empty((0, 3)))
            return
        ijkHom = np.column_stack((vox[:, 2], vox[:, 1], vox[:, 0], np.ones(len(vox))))
        M = np.array([[self.ijkToRAS.GetElement(i, j) for j in range(4)] for i in range(4)])
        ras = (M @ ijkHom.T).T[:, :3]
        self._lungKD = cKDTree(ras)

