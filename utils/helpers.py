"""Utility helper functions for the refactored LungBiopsyTractPlanner."""

import numpy as np
import vtk
import slicer


def read_image_data(nifti_path: str) -> "vtkImageData":
    """Read a NIfTI image and return a vtkImageData copy."""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_path)
    reader.Update()
    image = vtk.vtkImageData()
    image.DeepCopy(reader.GetOutput())
    return image


LUNG_SEGMENTS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


def iter_valid_lobes(seg_node, reference_volume_node, *, return_mask=False, as_ids=False):
    """Yield valid lung lobe names or masks from a segmentation node."""
    seg = seg_node.GetSegmentation()
    bin_rep = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    seg.SetConversionParameter("ReferenceImageGeometry", reference_volume_node.GetID())
    if not seg.ContainsRepresentation(bin_rep):
        seg_node.CreateBinaryLabelmapRepresentation()

    for name in LUNG_SEGMENTS:
        sid = seg.GetSegmentIdBySegmentName(name)
        if not sid:
            continue
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, sid, reference_volume_node)
        if mask is None or not np.any(mask):
            continue
        if return_mask:
            yield name, mask
        elif as_ids:
            yield name, sid
        else:
            yield name


def get_segment_mask_as_array(seg_node, seg_id, ref_vol):
    """Return the binary mask of a segment as a NumPy array."""
    seg_logic = slicer.modules.segmentations.logic()
    tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    seg_logic.ExportSegmentsToLabelmapNode(seg_node, [seg_id], tmp, ref_vol)
    arr = slicer.util.arrayFromVolume(tmp).astype(np.uint8, copy=False)
    slicer.mrmlScene.RemoveNode(tmp)
    return arr


def segment_has_content(seg_node, segment_name, ref_volume_node):
    """Return True if the given segment exists and contains voxels."""
    seg = seg_node.GetSegmentation()
    seg_id = seg.GetSegmentIdBySegmentName(segment_name)
    if not seg_id:
        return False
    arr = get_segment_mask_as_array(seg_node, seg_id, ref_volume_node)
    return np.any(arr)


def ras_to_ijk(volume_node, ras_point):
    """Convert a RAS point to IJK coordinates for the given volume."""
    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    ras_vec = list(ras_point) + [1.0]
    ijk_vec = [0.0, 0.0, 0.0, 0.0]
    m.MultiplyPoint(ras_vec, ijk_vec)
    return [int(round(c)) for c in ijk_vec[:3]]


def get_smoothed_closed_surface_representation(segmentation_node, segment_name, *,
                                               iterations=50, relaxation=0.3,
                                               boundary_smoothing_on=True):
    """Return a smoothed closed surface representation of the given segment."""
    segmentation = segmentation_node.GetSegmentation()
    segment_id = segmentation.GetSegmentIdBySegmentName(segment_name)
    if not segment_id:
        print(f"[ERROR] Segment '{segment_name}' not found.")
        return None

    rep_name = "ClosedSurface"
    if not segmentation.ContainsRepresentation(rep_name):
        segmentation.CreateRepresentation(rep_name)

    current_poly = vtk.vtkPolyData()
    slicer.modules.segmentations.logic().GetSegmentClosedSurfaceRepresentation(
        segmentation_node, segment_id, current_poly
    )
    if current_poly.GetNumberOfPoints() == 0:
        print(f"[ERROR] Segment '{segment_name}' closed surface could not be retrieved.")
        return None

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputData(current_poly)
    smooth.SetNumberOfIterations(iterations)
    smooth.SetRelaxationFactor(relaxation)
    smooth.FeatureEdgeSmoothingOn()
    if boundary_smoothing_on:
        smooth.BoundarySmoothingOn()
    else:
        smooth.BoundarySmoothingOff()
    smooth.Update()

    return smooth.GetOutput()

