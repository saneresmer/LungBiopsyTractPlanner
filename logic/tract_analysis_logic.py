"""Logic related to biopsy tract analysis."""

import math
import numpy as np
import vtk
import slicer

from LungBiopsyTractPlanner import (
    LungBiopsyTractPlannerLogic,
    _getClosed,
    _getVesselImplicit,
)
from utils.helpers import iter_valid_lobes


class TractAnalysisLogic(LungBiopsyTractPlannerLogic):
    """Expose tract analysis methods with snake_case naming."""

    # ------------------------------------------------------------------
    # Wrappers around existing parent class methods
    # ------------------------------------------------------------------
    def projects_on_scapulae_posterior(
        self,
        segmentation_node,
        volume_node,
        *,
        target_name: str = "TargetRegion",
        scap_right_name: str = "scapula_right",
        scap_left_name: str = "scapula_left",
        restrict_to_scap_z: bool = True,
    ) -> bool:
        """Return True if the posterior projection of *target_name* intersects either scapula."""

        seg = segmentation_node.GetSegmentation()
        bin_rep = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
        seg.SetConversionParameter("ReferenceImageGeometry", volume_node.GetID())
        if not seg.ContainsRepresentation(bin_rep):
            segmentation_node.CreateBinaryLabelmapRepresentation()

        tid = seg.GetSegmentIdBySegmentName(target_name)
        if not tid:
            raise ValueError(f"Target segment '{target_name}' not found.")

        sid_r = seg.GetSegmentIdBySegmentName(scap_right_name)
        sid_l = seg.GetSegmentIdBySegmentName(scap_left_name)

        scap_masks = []
        if sid_r:
            mask_r = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, sid_r, volume_node
            ).astype(bool)
            if np.any(mask_r):
                scap_masks.append(mask_r)

        if sid_l:
            mask_l = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, sid_l, volume_node
            ).astype(bool)
            if np.any(mask_l):
                scap_masks.append(mask_l)

        if not scap_masks:
            return False

        scap_mask = np.logical_or.reduce(scap_masks)

        tar_mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, tid, volume_node
        ).astype(bool)

        if restrict_to_scap_z:
            z_idx = np.where(scap_mask)[0]
            zmin, zmax = int(z_idx.min()), int(z_idx.max())
            tar_mask = tar_mask[zmin : zmax + 1]
            scap_mask = scap_mask[zmin : zmax + 1]

        tar_proj = np.any(tar_mask, axis=1)
        scap_proj = np.any(scap_mask, axis=1)

        return bool(np.any(tar_proj & scap_proj))

    def analyze_and_visualize_tracts(
        self, segmentation_node, combined_segmentation_node, risk_segments
    ):
        """Wrapper around the original ``analyzeAndVisualizeTracts`` method."""
        return super().analyzeAndVisualizeTracts(
            segmentation_node, combined_segmentation_node, risk_segments
        )

    # ------------------------------------------------------------------
    # Utility helpers used by tract analysis feature computations
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_intersect(obb, p0, p1):
        """Return number of intersections for a line, handling errors."""
        try:
            if obb is None:
                return 0
            if np.allclose(p0, p1):
                return 0
            pts, ids = vtk.vtkPoints(), vtk.vtkIdList()
            return obb.IntersectWithLine(list(map(float, p0)), list(map(float, p1)), pts, ids)
        except Exception as exc:  # noqa: BLE001
            print("[EXC] IntersectWithLine:", exc)
            return 0

    # ------------------------------------------------------------------
    # Feature calculation functions
    # ------------------------------------------------------------------
    def feature_large_vessel_dist_mm(self, start_point, end_point, segmentation_node, volume_node):
        """Score distance from large vessels sampled every 10 mm along the tract."""
        imp = _getVesselImplicit(segmentation_node)
        if imp is None:
            return 0.0
        p0, p1 = np.array(start_point), np.array(end_point)
        vec = p1 - p0
        length = np.linalg.norm(vec)
        if length == 0:
            return 0.0
        step_mm = 10.0
        steps = max(int(length / step_mm), 1)
        mids = [p0 + vec * ((i + 0.5) / steps) for i in range(steps)]
        dists = np.abs([imp.EvaluateFunction(pt) for pt in mids])
        score = 0
        for d in dists:
            if d < 5:
                score += 5
            elif d < 10:
                score += 4
            elif d < 15:
                score += 3
            elif d < 20:
                score += 2
            elif d < 25:
                score += 1
        return score

    def feature_bulla_crossed(self, start_point, end_point, segmentation_node, volume_node):
        """Return 1 if the tract intersects the emphysema/bulla segment."""
        seg = segmentation_node.GetSegmentation()
        bulla_id = seg.GetSegmentIdBySegmentName("emphysema_bulla")
        if not bulla_id:
            return 0
        poly = _getClosed(segmentation_node, bulla_id)
        if poly is None:
            return 0
        obb = vtk.vtkOBBTree()
        obb.SetDataSet(poly)
        obb.BuildLocator()
        return int(self._safe_intersect(obb, start_point, end_point) > 0)

    def feature_fissure_crossed(self, start_point, end_point, segmentation_node, volume_node):
        """Return 1 if the tract crosses more than one lung lobe."""
        if np.allclose(start_point, end_point):
            return 0
        crossed = 0
        for _, sid in iter_valid_lobes(segmentation_node, volume_node, as_ids=True):
            poly = _getClosed(segmentation_node, sid)
            if poly is None or poly.GetNumberOfPoints() == 0:
                continue
            obb = vtk.vtkOBBTree()
            obb.SetDataSet(poly)
            obb.BuildLocator()
            if self._safe_intersect(obb, start_point, end_point):
                crossed += 1
                if crossed > 1:
                    return 1
        return 0

    def feature_pleural_fluid_instillation(self, start_point, end_point, segmentation_node, volume_node):
        """Return 1 if the tract intersects only pleural effusion without lung lobes."""
        if np.allclose(start_point, end_point):
            return 0
        seg = segmentation_node.GetSegmentation()
        for _, sid in iter_valid_lobes(segmentation_node, volume_node, as_ids=True):
            poly_lob = _getClosed(segmentation_node, sid)
            if poly_lob is None:
                continue
            obb_lob = vtk.vtkOBBTree()
            obb_lob.SetDataSet(poly_lob)
            obb_lob.BuildLocator()
            if self._safe_intersect(obb_lob, start_point, end_point) > 0:
                return 0
        pe_id = seg.GetSegmentIdBySegmentName("pleural_effusion")
        if not pe_id:
            return 0
        poly_pe = _getClosed(segmentation_node, pe_id)
        if poly_pe is None or poly_pe.GetNumberOfPoints() == 0:
            return 0
        obb_pe = vtk.vtkOBBTree()
        obb_pe.SetDataSet(poly_pe)
        obb_pe.BuildLocator()
        return int(self._safe_intersect(obb_pe, start_point, end_point) > 0)

    def feature_technic_diff(self, start_point, end_point, segmentation_node, volume_node):
        """Return the number of axial slices traversed by the tract."""
        ras_to_ijk_matrix = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk_matrix)

        def ras_to_ijk(ras_point):
            ras_vec = list(ras_point) + [1.0]
            ijk_vec = [0.0, 0.0, 0.0, 0.0]
            ras_to_ijk_matrix.MultiplyPoint(ras_vec, ijk_vec)
            return [int(round(c)) for c in ijk_vec[:3]]

        ijk_start = ras_to_ijk(start_point)
        ijk_end = ras_to_ijk(end_point)
        return abs(ijk_end[2] - ijk_start[2]) + 1

    def feature_anterior_entry(self, start_point, end_point, segmentation_node, volume_node):
        """Return 1 if the entry point is closest to the anterior body surface."""
        seg = segmentation_node.GetSegmentation()
        sid = seg.GetSegmentIdBySegmentName("body_trunc")
        if not sid:
            return 0
        if not seg.ContainsRepresentation("Closed surface"):
            segmentation_node.CreateClosedSurfaceRepresentation()
        poly = vtk.vtkPolyData()
        segmentation_node.GetClosedSurfaceRepresentation(sid, poly)
        xmin, xmax, ymin, ymax, *_ = poly.GetBounds()
        p0 = np.array(start_point, dtype=float)
        dist_ant = ymax - p0[1]
        dist_post = p0[1] - ymin
        dist_lat = min(p0[0] - xmin, xmax - p0[0])
        return int(dist_ant < dist_lat and dist_ant < dist_post)

    def feature_lateral_entry(self, start_point, end_point, segmentation_node, volume_node):
        """Return 1 if the entry point is closest to the lateral body surface."""
        seg = segmentation_node.GetSegmentation()
        sid = seg.GetSegmentIdBySegmentName("body_trunc")
        if not sid:
            return 0
        if not seg.ContainsRepresentation("Closed surface"):
            segmentation_node.CreateClosedSurfaceRepresentation()
        poly = vtk.vtkPolyData()
        segmentation_node.GetClosedSurfaceRepresentation(sid, poly)
        xmin, xmax, ymin, ymax, *_ = poly.GetBounds()
        p0 = np.array(start_point, dtype=float)
        dist_ant = ymax - p0[1]
        dist_post = p0[1] - ymin
        dist_lat = min(p0[0] - xmin, xmax - p0[0])
        return int(dist_lat < dist_ant and dist_lat < dist_post)

    # ------------------------------------------------------------------
    # Ranking helper
    # ------------------------------------------------------------------
    @staticmethod
    def assign_ranks(tracts, key, rank_key):
        """Assign ranks based on the given key and store under ``rank_key``."""
        for rank, tract in enumerate(sorted(tracts, key=lambda x: x[key]), start=1):
            tract[rank_key] = rank

    # ------------------------------------------------------------------
    # Risk and depth helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def compute_risk_tract(event, model, tract_features, base_eta, risk_table):
        """Return risk probability using logistic regression coefficients."""
        table = risk_table[event][model]
        eta = base_eta

        for name, value in tract_features.items():
            factor = table["factors"].get(name)
            if factor:
                eta += factor["beta"] * value
        return 1.0 / (1.0 + math.exp(-eta))

    @staticmethod
    def depth_or_continuous(mm):
        """Continuous odds ratio function for pneumothorax."""
        anchors = [(0, 0.0), (20, math.log(2.16)), (30, math.log(2.38)), (50, math.log(8.47))]
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= mm <= x1:
                t = (mm - x0) / (x1 - x0)
                return math.exp(y0 + t * (y1 - y0))

        slope = (anchors[-1][1] - anchors[-2][1]) / (anchors[-1][0] - anchors[-2][0])
        log_or = anchors[-1][1] + slope * (mm - anchors[-1][0])
        return math.exp(log_or)

    @staticmethod
    def depth_or_continuous_hmr(mm):
        """Continuous odds ratio for hemorrhage models."""
        anchors = [(0, 0.0), (30, math.log(4.558)), (50, math.log(25.641))]
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= mm <= x1:
                t = (mm - x0) / (x1 - x0)
                return math.exp(y0 + t * (y1 - y0))

        slope = (anchors[-1][1] - anchors[-2][1]) / (anchors[-1][0] - anchors[-2][0])
        log_or = anchors[-1][1] + slope * (mm - anchors[-1][0])
        return math.exp(log_or)



