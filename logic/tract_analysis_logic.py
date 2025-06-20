"""Tract analysis logic wrapper."""

from LungBiopsyTractPlanner import LungBiopsyTractPlannerLogic


class TractAnalysisLogic(LungBiopsyTractPlannerLogic):
    """Expose tract analysis methods with snake_case naming."""

    def projects_on_scapulae_posterior(self, *args, **kwargs):
        return super().projects_on_scapulae_posterior(*args, **kwargs)

    def analyze_and_visualize_tracts(self, *args, **kwargs):
        return super().analyzeAndVisualizeTracts(*args, **kwargs)

