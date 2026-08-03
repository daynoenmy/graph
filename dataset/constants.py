import os


BMAD_BASE_PATH = r"E:\datasets\bmad\BMAD"


# Retina is the existing Retina_RESC benchmark name used by this repository.
BMAD_DATASETS = (
    "Brain",
    "Liver",
    "Retina",
    "Chest",
    "Retina_OCT2017",
    "Histopathology",
)

DATA_PATH = {
    "Brain": os.path.join(BMAD_BASE_PATH, "Brain"),
    "Liver": os.path.join(BMAD_BASE_PATH, "Liver"),
    "Retina": os.path.join(BMAD_BASE_PATH, "Retina_RESC"),
    "Chest": os.path.join(BMAD_BASE_PATH, "Chest"),
    "Retina_OCT2017": os.path.join(BMAD_BASE_PATH, "Retina_OCT2017"),
    "Histopathology": os.path.join(BMAD_BASE_PATH, "Histopathology"),
}

CLASS_NAMES = {
    "Brain": ["Brain"],
    "Liver": ["Liver"],
    "Retina": ["Retina"],
    "Chest": ["Chest"],
    "Retina_OCT2017": ["Retina_OCT2017"],
    "Histopathology": ["Histopathology"],
}
DOMAINS = {
    "Brain": "Medical",
    "Liver": "Medical",
    "Retina": "Medical",
    "Chest": "Medical",
    "Retina_OCT2017": "Medical",
    "Histopathology": "Medical",
}
REAL_NAMES = {
    "Brain": {"Brain": "scan"},
    "Liver": {"Liver": "scan"},
    "Retina": {"Retina": "retinal OCT scan"},
    "Chest": {"Chest": "chest X-ray"},
    "Retina_OCT2017": {"Retina_OCT2017": "retinal OCT scan"},
    "Histopathology": {"Histopathology": "histopathology image"},
}
PROMPTS = {
    "prompt_normal": ["{}", "a normal {}", "a healthy {}"],
    "prompt_abnormal": [
        "an abnormal {}",
        "a pathological {}",
        "a {} with lesion",
        "a {} with abnormality",
        "a {} with disease",
    ],
    "prompt_templates": [
        "{}.",
        "a medical image of {}.",
    ],
}
