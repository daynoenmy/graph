import os


BASE_PATH = os.environ.get("GRAPH_DATA_ROOT", "/home/sunhan")
BMAD_DATA_PATH_OVERRIDE = os.environ.get("BMAD_DATA_PATH")
BMAD_BASE_PATH = BMAD_DATA_PATH_OVERRIDE or f"{BASE_PATH}/data/BMAD"


def _bmad_or_legacy(subdirectory, legacy_path):
    if BMAD_DATA_PATH_OVERRIDE:
        return f"{BMAD_BASE_PATH}/{subdirectory}"
    return legacy_path


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
    "Brain": os.environ.get(
        "BRAIN_DATA_PATH",
        _bmad_or_legacy("Brain", f"{BASE_PATH}/dataset/Brain_AD"),
    ),
    "Liver": os.environ.get(
        "LIVER_DATA_PATH",
        _bmad_or_legacy("Liver", f"{BASE_PATH}/dataset/Liver_AD"),
    ),
    "Retina": os.environ.get(
        "RETINA_RESC_DATA_PATH",
        _bmad_or_legacy("Retina_RESC", f"{BASE_PATH}/data/MedAD/Retina_RESC_AD"),
    ),
    "Chest": os.environ.get(
        "CHEST_DATA_PATH",
        f"{BMAD_BASE_PATH}/Chest",
    ),
    "Retina_OCT2017": os.environ.get(
        "OCT2017_DATA_PATH",
        f"{BMAD_BASE_PATH}/Retina_OCT2017",
    ),
    "Histopathology": os.environ.get(
        "HISTOPATHOLOGY_DATA_PATH",
        f"{BMAD_BASE_PATH}/Histopathology",
    ),
    "Colon_clinicDB": f"{BASE_PATH}/data/Colon/CVC-ClinicDB",
    "Colon_colonDB": f"{BASE_PATH}/data/Colon/CVC-ColonDB",
    "Colon_cvc300": f"{BASE_PATH}/data/Colon/CVC-300",
    "Colon_Kvasir": f"{BASE_PATH}/data/Colon/Kvasir",
    "BTAD": f"{BASE_PATH}/data/BTech_Dataset_transformed",
    "MPDD": f"{BASE_PATH}/data/MPDD",
    "MVTec": f"{BASE_PATH}/data/mvtec_ad",
    "VisA": f"{BASE_PATH}/data/VisA_20220922",
    "DDTI": os.environ.get("DDTI_DATA_PATH", f"{BASE_PATH}/data/DDTI"),
}

CLASS_NAMES = {
    "Brain": ["Brain"],
    "Liver": ["Liver"],
    "Retina": ["Retina"],
    "Chest": ["Chest"],
    "Retina_OCT2017": ["Retina_OCT2017"],
    "Histopathology": ["Histopathology"],
    "Colon_clinicDB": ["Colon_clinicDB"],
    "Colon_colonDB": ["Colon_colonDB"],
    "Colon_Kvasir": ["Kvasir"],
    "Colon_cvc300": ["CVC-300"],
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "toothbrush",
        "wood",
        "zipper",
    ],
    "VisA": [
        "candle",
        "pcb3",
        "capsules",
        "pipe_fryum",
        "pcb4",
        "macaroni2",
        "pcb2",
        "chewinggum",
        "macaroni1",
        "cashew",
        "fryum",
        "pcb1",
    ],
    "MPDD": [
        "connector",
        "tubes",
        "metal_plate",
        "bracket_white",
        "bracket_brown",
        "bracket_black",
    ],
    "BTAD": ["01", "02", "03"],
    "DDTI": ["DDTI"],
}
DOMAINS = {
    "VisA": "Industrial",
    "BTAD": "Industrial",
    "MPDD": "Industrial",
    "MVTec": "Industrial",
    "Brain": "Medical",
    "Liver": "Medical",
    "Retina": "Medical",
    "Chest": "Medical",
    "Retina_OCT2017": "Medical",
    "Histopathology": "Medical",
    "Colon_clinicDB": "Medical",
    "Colon_colonDB": "Medical",
    "Colon_Kvasir": "Medical",
    "Colon_cvc300": "Medical",
    "DDTI": "Medical",
}
REAL_NAMES = {
    "Brain": {"Brain": "scan"},
    "Liver": {"Liver": "scan"},
    "Retina": {"Retina": "retinal OCT scan"},
    "Chest": {"Chest": "chest X-ray"},
    "Retina_OCT2017": {"Retina_OCT2017": "retinal OCT scan"},
    "Histopathology": {"Histopathology": "histopathology image"},
    "DDTI": {"DDTI": "thyroid ultrasound scan"},
    "MVTec": {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
    },
    "VisA": {
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    },
    "Colon_clinicDB": {
        "Colon_clinicDB": "colon endoscopy image",
    },
    "Colon_colonDB": {
        "Colon_colonDB": "colon endoscopy image",
    },
    "Colon_cvc300": {"CVC-300": "colon endoscopy image"},
    "Colon_Kvasir": {"Kvasir": "colon endoscopy image"},
    "MPDD": {
        "connector": "metal clamps with black adjustment knobs",
        "tubes": "scattered metal objects",
        "metal_plate": "blue rectangular metal plate with a notch on one side",
        "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
        "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
        "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
    },
    "BTAD": {
        "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
        "02": "vertical fabric lines in warm, dusty pink and beige tones",
        "03": "oval concentric circular rings in gradient shades of blue and white",
    },
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
