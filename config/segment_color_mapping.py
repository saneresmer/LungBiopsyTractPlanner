# config/segment_color_mapping.py

# Bone (ribs and vertebrae) segment color: white
rib_left = [f"rib_left_{i}" for i in range(1, 13)]
rib_right = [f"rib_right_{i}" for i in range(1, 13)]
vertebrae_thoracic = [f"vertebrae_T{i}" for i in range(1, 13)]
vertebrae_cervical = ["vertebrae_C7"]

segment_color_mapping = {
    # Arterial and heart structures
    "aorta": (220, 20, 60),
    "atrial_appendage_left": (205, 92, 92),
    "heart": (255, 0, 0),
    "common_carotid_artery_left": (220, 20, 60),
    "common_carotid_artery_right": (220, 20, 60),
    "brachiocephalic_trunk": (220, 20, 60),
    "subclavian_artery_left": (220, 20, 60),
    "subclavian_artery_right": (220, 20, 60),
    "pulmonary_artery": (128, 0, 128),
    "heart_myocardium": (255, 85, 130),
    "heart_atrium_left": (128, 0, 0),
    "heart_ventricle_left": (139, 0, 0),
    "heart_atrium_right": (128, 0, 32),
    "heart_ventricle_right": (101, 0, 11),

    # Venous structures
    "brachiocephalic_vein_left": (65, 105, 225),
    "brachiocephalic_vein_right": (65, 105, 225),
    "inferior_vena_cava": (30, 144, 255),
    "superior_vena_cava": (30, 144, 255),
    "pulmonary_vein": (100, 149, 237),

    # Airway
    "trachea": (0, 255, 0),
    "lung_trachea_bronchia": (34, 139, 34),

    # Lung lobes
    "lung_lower_lobe_left": (176, 196, 222),
    "lung_lower_lobe_right": (176, 196, 222),
    "lung_middle_lobe_right": (172, 192, 218),
    "lung_upper_lobe_left": (174, 194, 220),
    "lung_upper_lobe_right": (174, 194, 220),
    "lung": (178, 198, 224),
    "lung_nodules": (255, 140, 0),  # Orange tone for nodules

    # Soft tissue
    "body_trunc": (255, 228, 196),
    "body_extremities": (255, 228, 196),
    "subcutaneous_fat": (255, 230, 198),
    "torso_fat": (254, 227, 195),
    "skeletal_muscle": (253, 226, 194),
    "autochthon_left": (253, 226, 194),
    "autochthon_right": (253, 226, 194),
    "intermuscular_fat": (255, 229, 197),

    # Spleen
    "spleen": (255, 0, 0),

    # Esophagus and stomach
    "esophagus": (255, 165, 0),  # Orange
    "stomach": (255, 140, 0),    # Orange

    # Shoulder, scapula, clavicle, humerus
    "scapula_left": (255, 255, 255),
    "scapula_right": (255, 255, 255),
    "clavicula_left": (255, 255, 255),
    "clavicula_right": (255, 255, 255),
    "humerus_left": (255, 255, 255),
    "humerus_right": (255, 255, 255),

    # Other
    "lung_vessels": (128, 0, 128),
    "pericardial_effusion": (255, 153, 153),
    "pleural_effusion": (255, 182, 193),
    "thyroid_gland": (255, 182, 193),
    "sternum": (255, 255, 255),
    "costal_cartilages": (255, 255, 255),
}

# Dynamically add rib and vertebrae colors
for rib in rib_left + rib_right:
    segment_color_mapping[rib] = (255, 255, 255)
for vertebra in vertebrae_thoracic + vertebrae_cervical:
    segment_color_mapping[vertebra] = (255, 255, 255)
