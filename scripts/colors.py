import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ---
# --- CENTRAL CONTROL PANEL FOR COLORS, CATEGORIES, & STYLES ---
# ---

UNIVERSAL_FONTSIZE = 50
MIN_FONT_SIZE = 8 
RADAR_REFERENCE_ANGLE = 20 

HSL_PARAMS = {
    "start_angle_degrees": 30.0,
    "lightness_start": 0.6,
    "lightness_end": 0.85,
    "sat_start": 0.7,
    "sat_end": 0.4,
}

# --- ADD THIS ---
HSL_PARAMS_ENTITY = {
    "start_angle_degrees": 180.0,  # <-- Different start angle
    "lightness_start": 0.6,
    "lightness_end": 0.85,
    "sat_start": 0.7,
    "sat_end": 0.4,
}

# --- Sunburst Color Sampling ---
INNER_RING_GRADIENT_POS = 0.1
OUTER_RING_GRADIENT_RANGE = [0.7, 1.0]

def create_hsl_colormaps(category_names, 
                         start_angle_degrees=0,
                         lightness_start=0.5,
                         lightness_end=0.5,
                         sat_start=0.2,
                         sat_end=0.6):
    """
    Generates a dict of distinct colormaps based on HSL properties.
    """
    num_categories = len(category_names)
    if num_categories == 0:
        return {}
        
    cmap_dict = {}
    start_hue = start_angle_degrees / 360.0 # Convert degrees (0-360) to HLS scale (0-1)

    for i, category_name in enumerate(category_names):
        hue = (start_hue + (i / float(num_categories))) % 1.0        
        start_rgb = colorsys.hls_to_rgb(hue, lightness_start, sat_start)
        end_rgb = colorsys.hls_to_rgb(hue, lightness_end, sat_end)
        cmap_name = f'hls_grad_{i}'
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [start_rgb, end_rgb])
        cmap_dict[category_name] = cmap
        
    return cmap_dict

def sample_colormaps(base_cmaps, position):
    """
    Samples each colormap in the dict at a specific position (0-1)
    to create a new dict of single {category_name: (r,g,b,a)} colors.
    """
    return {
        category_name: cmap(position) 
        for category_name, cmap in base_cmaps.items()
    }



# ---
# --- NEW 3-LEVEL HIERARCHY (USER REFACTOR) ---
# ---

def generate_discrete_hsl_palette(names, hues_degrees, saturation, lightness):
    """
    Generates a dictionary of discrete {name: (r, g, b)} colors from HSL values.
    'hues_degrees' can be a list of hues (0-360) or a single start angle
    to be evenly distributed.
    """
    palette = {}
    num_names = len(names)
    
    if isinstance(hues_degrees, (int, float)):
        # Evenly distribute hues starting from the given angle
        start_hue_deg = hues_degrees
        # Use endpoint=False for linspace to avoid duplicating 0 and 360
        hues_list = np.linspace(start_hue_deg, start_hue_deg + 360, num=num_names, endpoint=False) % 360.0
    elif len(hues_degrees) == num_names:
        # Use the provided list of hues
        hues_list = hues_degrees
    else:
        raise ValueError("hues_degrees must be a single number or a list of the same length as names")

    for name, hue_deg in zip(names, hues_list):
        hue_01 = hue_deg / 360.0  # Convert to 0-1 scale for colorsys
        # Note: colorsys.hls_to_rgb expects H, L, S
        rgb_01 = colorsys.hls_to_rgb(hue_01, lightness, saturation)
        palette[name] = rgb_01 # These are (r, g, b) tuples from 0-1
        
    return palette

# --- 1. Category Definitions ---
# These lists define the categories and, crucially, their *order*
NEW_L1_NAMES = [
    "spatial", 
    "temporal"
]
NEW_L2_NAMES = [
    "human-human",
    "human-object",
    "human-animal",
    "object-animal",
    "object-object",
    "animal-animal",
    "self interaction",
    "no interaction"
]
NEW_L3_NAMES = [
    "Affective",
    "Antagonistic",
    "Body Motion",
    "Communicative",
    "Competitive",
    "Cooperative",
    "Relational Movement",
    "Observation",
    "Passive",
    "Physical",
    "Provisioning",
    "Proximity",
    "Social",
    "Supportive"
]

# --- 2. Color Generation Controls (HLS Parameters) ---

# Toned down all levels. Inner rings are now softer.
NEW_HIERARCHY_HLS_PARAMS = {
    1: {'saturation': 0.70, 'lightness': 0.60}, # L1
    2: {'saturation': 0.55, 'lightness': 0.70}, # L2
    3: {'saturation': 0.40, 'lightness': 0.80}, # L3
}

# Independent rotation for each ring's color wheel
NEW_HIERARCHY_HUE_STARTS = {
    1: 0.0,   # Not used, L1 is hardcoded
    2: 0.0,   # L2 wheel starts at 0 degrees (red)
    3: 40.0,  # L3 wheel starts at 40 degrees (orange/yellow)
}

def generate_discrete_hsl_palette(names, start_hue_degrees, saturation, lightness):
    """
    Generates a dictionary of {name: (r, g, b)} colors,
    sampling evenly around the color wheel.
    """
    palette = {}
    num_names = len(names)
    if num_names == 0:
        return {}
        
    # Use endpoint=False to avoid duplicating 0 and 360
    hues_list = np.linspace(start_hue_degrees, start_hue_degrees + 360, num=num_names, endpoint=False) % 360.0

    for name, hue_deg in zip(names, hues_list):
        hue_01 = hue_deg / 360.0
        rgb_01 = colorsys.hls_to_rgb(hue_01, lightness, saturation)
        palette[name] = rgb_01
        
    return palette

# --- 3. Generate Independent Color Maps ---

# --- Level 1 Colors (Hardcoded) ---
_l1_params = NEW_HIERARCHY_HLS_PARAMS[1]
NEW_L1_COLOR_MAP = {
    # --- MODIFIED: Changed Hues ---
    "spatial": colorsys.hls_to_rgb(30/360.0, _l1_params['lightness'], _l1_params['saturation']),  # Was 240 (Blue), now 30 (Orange)
    "temporal": colorsys.hls_to_rgb(190/360.0, _l1_params['lightness'], _l1_params['saturation']) # Was 120 (Green), now 190 (Cyan)
}

# --- Level 2 Colors (Independent Wheel) ---
NEW_L2_COLOR_MAP = generate_discrete_hsl_palette(
    names=NEW_L2_NAMES,
    start_hue_degrees=NEW_HIERARCHY_HUE_STARTS[2],
    **NEW_HIERARCHY_HLS_PARAMS[2]
)

# --- Level 3 Colors (Independent Wheel) ---
NEW_L3_COLOR_MAP = generate_discrete_hsl_palette(
    names=NEW_L3_NAMES,
    start_hue_degrees=NEW_HIERARCHY_HUE_STARTS[3],
    **NEW_HIERARCHY_HLS_PARAMS[3]
)

# --- Fallback Color ---
FALLBACK_COLOR = (0.5, 0.5, 0.5) # Neutral Grey
NEW_L1_COLOR_MAP["__NEW_CATEGORY__"] = FALLBACK_COLOR
NEW_L2_COLOR_MAP["__NEW_CATEGORY__"] = FALLBACK_COLOR
NEW_L3_COLOR_MAP["__NEW_CATEGORY__"] = FALLBACK_COLOR