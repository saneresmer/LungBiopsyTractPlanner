import math

# Allowed segments for thoracic biopsy tract planning
ALLOWED_SEGMENTS_THORAX = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_lower_lobe_right",
    "lung_middle_lobe_right", "pleural_effusion", 
    "subcutaneous_fat", "torso_fat", "skeletal_muscle",
    "autochthon_left", "autochthon_right", "body_trunc",
    "emphysema_bulla", "TargetRegion", "lung", "lung_nodules"
]

# Tract generation constants for thoracic biopsy

# How many surface points to skip when generating tract start points
SUBSAMPLE_EVERY_NTH = 20  # (integer)

# Minimum Euclidean distance (in mm) between candidate start points on skin
MIN_DIST_BETWEEN_POINTS = 10.0  # mm

# Maximum total tract length (from skin to lesion), excluding safety margin
NEEDLE_LEN_MAX = 190  # mm

# Minimum portion of tract required to be within the lung parenchyma
MIN_INSIDE_MM = 10  # mm


# Lung Biopsy Risk Table
# Academic sources and references included inline for traceability and citation.
# Only statistically significant factors (p < 0.05) from referenced studies are included.

RISK_TABLE = {

    # ─────────────────────────────────────────────────────────────
    # Hemorrhage risk models based on multivariate logistic regression analysis.
    # Source: Zhu et al., Quant Imaging Med Surg 2020
    # DOI: https://doi.org/10.21037/qims-19-1024
    # Only statistically significant predictors (p < 0.05) are included.
    # ─────────────────────────────────────────────────────────────
    "hemorrhage": {
        "any_grade": {
            "base": -0.795,
            "factors": {
                # Target factor — TargetRegion intersects left_lower_lobe
                "left_lower_lobe": {
                    "OR": 1.948, "CI": (1.209, 3.138), "p": 0.006,
                    "type": "target", "table": "Table 4",
                    "comment": "TargetRegion intersects left_lower_lobe",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — TargetRegion intersects right_lower_lobe
                "right_lower_lobe": {
                    "OR": 1.754, "CI": (1.125, 2.734), "p": 0.013,
                    "type": "target", "table": "Table 4",
                    "comment": "TargetRegion intersects right_lower_lobe",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — TargetRegion intersects right_hilar
                "right_hilar": {
                    "OR": 5.368, "CI": (1.518, 18.986), "p": 0.009,
                    "type": "target", "table": "Table 4",
                    "comment": "TargetRegion intersects right_hilar region",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — TargetRegion size ≤ 3 cm
                "size_le3cm": {
                    "OR": 1.628, "CI": (1.186, 2.236), "p": 0.003,
                    "type": "target", "table": "Table 4",
                    "comment": "TargetRegion maximum size is ≤ 3 cm",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Tract-dependent factor — Continuous depth from pleura
                "depth_cont_hmr": {
                    "beta": 1.0, 
                    "type": "tract", "table": "Table 4",
                    "comment": "OR interpolated from depth using Zhu 2020 anchors: 0–30–50 mm",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — Lung metastases present in TargetRegion
                "lung_metastases": {
                    "OR": 6.695, "CI": (2.618, 17.122), "p": "<0.001",
                    "type": "target", "table": "Table 4",
                    "comment": "Lesion histopathology consistent with lung metastases",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                }
            }
        },
        "high_grade": {
            "base": -2.590,
            "factors": {
                # Target factor — mPAD/AAD > 1 ratio
                "mpad_aad_gt1": {
                    "OR": 1.871, "CI": (1.063, 3.294), "p": 0.03,
                    "type": "target", "table": "Table 4",
                    "comment": "Main pulmonary artery diameter / ascending aorta diameter > 1",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — TargetRegion size ≤ 3 cm
                "size_le3cm": {
                    "OR": 1.769, "CI": (1.081, 2.897), "p": 0.023,
                    "type": "target", "table": "Table 4",
                    "comment": "TargetRegion maximum size is ≤ 3 cm",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Tract-dependent factor — Continuous depth from pleura
                "depth_cont_hmr": {
                    "beta": 1.0, 
                    "type": "tract", "table": "Table 4",
                    "comment": "OR interpolated from depth using Zhu 2020 anchors: 0–30–50 mm",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — Emphysematous lung in TargetRegion
                "emphysema": {
                    "OR": 2.810, "CI": (1.709, 4.621), "p": "<0.001",
                    "type": "target", "table": "Table 4",
                    "comment": "TargetRegion overlaps emphysematous parenchyma",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                },
                # Target factor — Lung metastases present in TargetRegion
                "lung_metastases": {
                    "OR": 6.687, "CI": (2.629, 17.011), "p": "<0.001",
                    "type": "target", "table": "Table 4",
                    "comment": "Lesion histopathology consistent with lung metastases",
                    "source": "Zhu et al. 2020", "doi": "10.21037/qims-19-1024"
                }
            }
        }
    },

    "pneumothorax": {

        # ─────────────────────────────────────────────────────────────
        # Pneumothorax Risk Models
        # Core model base values and most risk factors are derived from:
        # Huo et al., Br J Radiol 2020 – https://doi.org/10.1259/bjr.20190866
        # (See Table 1 for base rates, Table 4 & 5 for ORs)

        # Additional factors are integrated from independent publications:
        # - Pleural fluid instillation: Brönnimann et al., Eur J Radiol 2024 (DOI: 10.1016/j.ejrad.2024.111529)
        # - Pleural tail crossing: Deng et al., BMC Pulm Med 2024 (DOI: 10.1186/s12890-024-03307-z)
        # - Ipsilateral effusion: Anil et al., J Am Coll Radiol 2022 (DOI: 10.1016/j.jacr.2022.04.010)
        # ─────────────────────────────────────────────────────────────
        
        "general": {
            "base": -0.887,
            "factors": {

                # Tract factor — Entry point anterior to chest
                "anterior_entry": {
                    "OR": 1.83, "CI": (1.51, 2.21), "p": "<0.001",
                    "type": "tract", "table": "Table 4",
                    "comment": "Needle enters from anterior chest wall (Ref. = Posterior)",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Entry point lateral to chest
                "lateral_entry": {
                    "OR": 1.89, "CI": (0.43, 8.33), "p": "",
                    "type": "tract", "table": "Table 4",
                    "comment": "Computed from multiple table comparisons: Anterior–Posterior, Anterior–Lateral, Posterior–Lateral",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Tract intersects emphysematous area
                "bulla_crossed": {
                    "OR": 6.13, "CI": (3.73, 10.06), "p": "<0.001",
                    "type": "tract", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Tract intersects fissure (cross-lobar)
                "fissure_crossed": {
                    "OR": 3.75, "CI": (2.57, 5.46), "p": "<0.001",
                    "type": "tract", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Emphysema in region of interest
                "emphysema": {
                    "OR": 6.44, "CI": (4.27, 9.72), "p": "<0.01",
                    "type": "target", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Lesion size 2–3 cm
                "size_2_3cm": {
                    "OR": 0.5, "CI": (0.4, 0.64), "p": "<0.001",
                    "type": "target", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Lesion size 3–4 cm
                "size_3_4cm": {
                    "OR": 0.58, "CI": (0.5, 0.67), "p": "<0.001",
                    "type": "target", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Lesion size 4–5 cm
                "size_4_5cm": {
                    "OR": 0.48, "CI": (0.30, 0.75), "p": "<0.01",
                    "type": "target", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Lesion size >5 cm
                "size_5cm": {
                    "OR": 0.34, "CI": (0.18, 0.65), "p": "<0.01",
                    "type": "target", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Continuous OR by interpolated depth (0–50 mm)
                "depth_cont": {
                    "beta": 1.0,
                    "type": "tract", "table": "Table 4",
                    "comment": "OR interpolated from anchors: 0–20–30–50 mm",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Lesion in pleural contact
                "pleural_contact": {
                    "OR": 0.57, "CI": (0.39, 0.85), "p": "<0.01",
                    "type": "target", "table": "Table 4",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Tract passes through pleural fluid
                "pleural_fluid_instillation": {
                    "OR": 0.071, "p": 0.003,
                    "type": "tract", "table": "Table 2",
                    "comment": "Tract passes through pleural fluid",
                    "source": "Broennimann et al. 2024", "doi": "10.1016/j.ejrad.2024.111529"
                },

                # Target factor — Ipsilateral pleural effusion
                "ipsilateral_effusion": {
                    "OR": 0.65, "p": 0.05,
                    "type": "target", "table": "Supplementary Table 4",
                    "comment": "Pleural fluid present in same hemithorax as TargetRegion",
                    "source": "Anil et al. 2022", "doi": "10.1016/j.jacr.2022.04.010"
                },

                # Tract factor — Tract intersects pleural tail sign (PTS)
                "crossing_pts_risk": {
                    "OR": 2.566, "CI": (1.71, 3.851), "p": "<0.001",
                    "type": "tract", "table": "Table 3",
                    "comment": "Tract intersects pleural tail sign (PTS)",
                    "source": "Deng et al. 2024", "doi": "10.1186/s12890-024-03307-z"
                }
            }
        },

        "drain_required": {
            "base": -2.300,
            "factors": {

                # Tract factor — Anterior chest entry
                "anterior_entry": {
                    "OR": 1.94, "CI": (1.62, 2.32), "p": "<0.001",
                    "type": "tract", "table": "Table 5",
                    "comment": "Needle enters from anterior chest wall (Ref. = Posterior)",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Lateral chest entry
                "lateral_entry": {
                    "OR": 1.19, "CI": (0.9, 1.56), "p": "<0.05",
                    "type": "tract", "table": "Table 5",
                    "comment": "Computed from multiple table comparisons: Anterior–Posterior, Anterior–Lateral, Posterior–Lateral",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Bulla crossed
                "bulla_crossed": {
                    "OR": 11.04, "CI": (5.32, 22.90), "p": "<0.05",
                    "type": "tract", "table": "Table 5",
                    "comment": "Tract passes through emphysematous lung tissue",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Fissure crossed
                "fissure_crossed": {
                    "OR": 3.54, "CI": (2.32, 5.40), "p": "<0.05",
                    "type": "tract", "table": "Table 5",
                    "comment": "Tract crosses interlobar fissure (multi-lobar access)",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Emphysema in target area
                "emphysema": {
                    "OR": 6.44, "CI": (4.27, 9.72), "p": "<0.01",
                    "type": "target", "table": "Table 5",
                    "comment": "Emphysema is present in the same lung lobe as the TargetRegion",
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Target factor — Pleural contact present
                "pleural_contact": {
                    "OR": 0.53, "CI": (0.32, 0.87), "p": "<0.01",
                    "type": "target", "table": "Table 5",
                    "comment": "TargetRegion directly contacts pleural surface"
                    "source": "Huo et al. 2020", "doi": "10.1259/bjr.20190866"
                },

                # Tract factor — Tract crosses pleural effusion
                "pleural_fluid_instillation": {
                    "OR": 0.071, "p": 0.003,
                    "type": "tract", "table": "Table 2",
                    "comment": "Tract passes through pleural fluid",
                    "source": "Broennimann et al. 2024", "doi": "10.1016/j.ejrad.2024.111529"
                },

                # Target factor — Ipsilateral pleural effusion
                "ipsilateral_effusion": {
                    "OR": 0.48, "p": 0.05,
                    "type": "target", "table": "Supplementary Table 4",
                    "comment": "Pleural fluid present in same hemithorax as TargetRegion",
                    "source": "Anil et al. 2022", "doi": "10.1016/j.jacr.2022.04.010"
                }
            }
        }
    }
}

# Tract-dependent features for pneumothorax risk model
# These features are evaluated along the planned biopsy tract.
PX_TRACT_FACTORS = {
    "anterior_entry",                  # Anterior chest wall entry 
    "lateral_entry",                   # Lateral chest wall entry 
    "fissure_crossed",                 # Tract intersects interlobar fissure
    "bulla_crossed",                   # Tract intersects emphysematous area
    "depth_cont",                      # Continuous pleura-to-target depth (interpolated OR)
    "pleural_fluid_instillation",      # Tract crosses pleural fluid
    "crossing_pts_risk"                # Tract intersects pleural tail sign (PTS)
}

# Tract-dependent features for hemorrhage risk model
# These factors depend on tract geometry and depth.
HMR_TRACT_FACTORS = {
    "depth_cont_hmr"                  # Continuous depth to target, specific to hemorrhage OR function
}

# Lesion-level (target-region dependent) features
# These factors are evaluated from the anatomical location or properties of the lesion itself.
# They are considered independent of tract geometry.
LESION_FEATURES = {
    "left_lower_lobe": left_lower_lobe,               # Lesion located in left lower lobe
    "right_lower_lobe": right_lower_lobe,             # Lesion located in right lower lobe
    "right_hilar": right_hilar,                       # Lesion located in right hilar region
    "size_le3cm": size_le3cm,                         # Lesion size ≤ 3 cm (hemorrhage model)
    "size_2_3cm": size_2_3cm,                         # Lesion size 2–3 cm (pneumothorax model)
    "size_3_4cm": size_3_4cm,                         # Lesion size 3–4 cm (pneumothorax model)
    "size_4_5cm": size_4_5cm,                         # Lesion size 4–5 cm (pneumothorax model)
    "size_5cm": size_5cm,                             # Lesion size >5 cm (pneumothorax model)
    "pleural_contact": pleural_contact,               # Lesion is in direct contact with pleura
    "mpad_aad_gt1": mpad_aad_gt1,                     # mPAD/AAD ratio > 1 (vascular risk indicator)
    "emphysema": emphysema,                           # Emphysema overlaps lesion region
    "ipsilateral_effusion": ipsilateral_effusion      # Pleural effusion on same side as target
}

# Automatically add 'beta = log(OR)' to all risk factors with defined OR
for event in RISK_TABLE.values():
    for model in event.values():
        for factor in model["factors"].values():
            if "OR" in factor:
                factor["beta"] = math.log(factor["OR"])
                
                
def build_lesion_feature_vector(
    left_lower_lobe, right_lower_lobe, right_hilar,
    size_le3cm, size_2_3cm, size_3_4cm, size_4_5cm, size_5cm,
    pleural_contact, pleural_patch_too_small,
    mpad_aad_gt1, emphysema, ipsilateral_effusion
    ) -> dict:
    """
    Converts raw lesion metadata into a structured feature vector for risk models.

    All inputs must be 0/1 (or boolean); casting will be applied as needed.

    Returns:
        dict: {feature_name: binary or scalar value}
    """
    return {
        "left_lower_lobe": left_lower_lobe,
        "right_lower_lobe": right_lower_lobe,
        "right_hilar": right_hilar,
        "size_le3cm": size_le3cm,
        "size_2_3cm": size_2_3cm,
        "size_3_4cm": size_3_4cm,
        "size_4_5cm": size_4_5cm,
        "size_5cm": size_5cm,
        "pleural_contact": int(pleural_contact),
        "pleural_patch_too_small": int(pleural_patch_too_small),
        "mpad_aad_gt1": int(mpad_aad_gt1),
        "emphysema": emphysema,
        "ipsilateral_effusion": ipsilateral_effusion,
    }
