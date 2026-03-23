# # %%
# import pandas as pd 

# fp = "table1_level1and2.csv"

# df = pd.read_csv(fp)
# df.drop(['pred_tube', 'gt_tube', 'caption', ], axis=1, inplace=True)

# %%

# import pandas as pd
# import numpy as np

# # Assuming your DataFrame is loaded into a variable named 'df'
# # Example: df = pd.read_csv('your_data.csv') or however you loaded it

# # 1. Filter out the 'freeform' task
# df_filtered = df

# # 2. Define columns that make an entry unique (excluding 'model')
# # This ensures we count each benchmark annotation once.
# unique_entry_cols = [
#     'dataset', 'task', 'caption', 'video_path', 
#     'st_level0_cls', 'entity_level0_cls', '0', '1', '2', '3'
# ]

# # 3. Drop duplicates to get a single row per unique benchmark entry
# df_benchmark = df_filtered.drop_duplicates(subset=unique_entry_cols)

# # 4. Define the specific class columns you want to count
# class_cols = ['st_level0_cls', 'entity_level0_cls', '0', '1', '2', '3']

# # 5. Get the DataFrame with just these class columns
# df_classes = df_benchmark[class_cols]

# # 6. Count occurrences of each unique value across all selected columns
# # .apply(pd.Series.value_counts) counts values per column (ignoring NaNs)
# # .sum(axis=1) sums these counts across the columns for a total
# total_class_counts = df_classes.apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)

# # Convert to integer (counts should be whole numbers)
# total_class_counts = total_class_counts.astype(int)

# print("--- Total Class Counts (Benchmark-wide, 'freeform' excluded) ---")
# print(total_class_counts)


# # --- Optional: Filter for only the categories you listed ---

# user_categories = [
#     'Spatial', 'Temporal',
#     'Animal-Animal', 'Animal-Object', 'Human-Animal', 'Human-Human', 
#     'Human-Object', 'Human-Self', 'Object-Object', 'No Interaction',
#     'Antagonistic', 'Affective', 'Body Motion', 'Communicative', 'Competitive',
#     'Cooperative', 'Observation', 'Passive', 'Physical', 'Provisioning', 
#     'Proximity', 'Social', 'Supportive', 'Relational Movement'
# ]

# # Use .reindex() to show only your categories, filling with 0 if they don't exist
# user_category_counts = total_class_counts.reindex(user_categories).fillna(0).astype(int)

# print("\n--- Counts for Specified Categories ---")
# print(user_category_counts)
# %%




import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker # <-- ADD THIS LINE
# %%

# ---
# --- USER-PROVIDED COLOR GENERATION CODE ---
# ---
# This code is pasted directly from your prompt.

FONTSIZE = 55

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
    
    (Note: This definition from the prompt overwrites the previous one.)
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

# ---
# --- END OF USER-PROVIDED CODE ---
# ---

# %%
def plot_class_distribution():
    """
    Creates and saves a bar chart for the class distribution,
    sectioned and colored according to the taxonomy.
    """
    
    # --- 1. Define Data and Mappings ---
    
    # Section 1: Spatio-temporal Interaction
    data_s_t = {
        "Spatial": 2845,
        "Temporal": 4571,
    }
    # Map data labels to the color map keys
    map_s_t = {
        "Spatial": "spatial",
        "Temporal": "temporal",
    }
    
    # Section 2: Entity
    data_entity = {
        "Human-Human": 2166,
        "Human-Object": 1541,
        "No Interaction": 1701,
        "Human-Animal": 395,
        "Animal-Animal": 810,
        "Animal-Object": 365,
        "Object-Object": 277,
        "Human-Self": 160,
    }
    # Map data labels to the color map keys
    map_entity = {
        "Human-Human": "human-human",
        "Human-Object": "human-object",
        "No Interaction": "no interaction",
        "Human-Animal": "human-animal",
        "Animal-Animal": "animal-animal",
        "Animal-Object": "object-animal",
        "Object-Object": "object-object",
        "Human-Self": "self interaction",
    }

    # Section 3: Interaction Type
    data_interaction = {
        "Relational Movement": 3738,
        "Observation": 2040,
        "Physical Interaction": 1902,
        "Body Motion": 1879,
        "Proximity": 1158,
        "Communicative": 487,
        "Passive": 347,
        "Supportive": 256,
        "Affective": 245,
        "Antagonistic": 236,
        "Provisioning": 160,
        "Social": 140,
        "Cooperative": 123,
        "Competitive": 28,
    }
    # Map data labels to the color map keys
    map_interaction = {
        "Relational Movement": "Relational Movement",
        "Observation": "Observation",
        "Physical Interaction": "Physical", # Key mapping
        "Body Motion": "Body Motion",
        "Proximity": "Proximity",
        "Communicative": "Communicative",
        "Passive": "Passive",
        "Supportive": "Supportive",
        "Affective": "Affective",
        "Antagonistic": "Antagonistic",
        "Provisioning": "Provisioning",
        "Social": "Social",
        "Cooperative": "Cooperative",
        "Competitive": "Competitive",
    }

    # --- 2. Prepare Data for Plotting ---
    
    # Sort each section by count (descending)
    sorted_s_t = sorted(data_s_t.items(), key=lambda item: item[1], reverse=True)
    sorted_entity = sorted(data_entity.items(), key=lambda item: item[1], reverse=True)
    sorted_interaction = sorted(data_interaction.items(), key=lambda item: item[1], reverse=True)

    
    # Combine all data into single lists
    all_labels = []
    all_counts = []
    all_colors = []
    
    # Add Section 1 data
    for label, count in sorted_s_t:
        all_labels.append(label)
        all_counts.append(count)
        color_key = map_s_t.get(label)
        all_colors.append(NEW_L1_COLOR_MAP.get(color_key, FALLBACK_COLOR))
        
    # Add Section 2 data
    for label, count in sorted_entity:
        all_labels.append(label)
        all_counts.append(count)
        color_key = map_entity.get(label)
        all_colors.append(NEW_L2_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    # Add Section 3 data
    for label, count in sorted_interaction:
        all_labels.append(label)
        all_counts.append(count)
        color_key = map_interaction.get(label)
        all_colors.append(NEW_L3_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    all_labels.reverse()
    all_counts.reverse()
    all_colors.reverse()

    # --- 3. Create the Plot ---
    
    plt.figure(figsize=(36, 12)) # Wide figure to fit labels

    ax = plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    indices = np.arange(len(all_labels))
    plt.bar(indices, all_counts, color=all_colors, edgecolor='black', linewidth=0.0, width=0.7)
    
    # Add count text on top of each bar
    for i, count in enumerate(all_counts):
        plt.text(i, count + 20, str(count), ha='center', va='bottom', fontsize=22)

    # --- 4. Add Section Lines and Labels ---
    
    # Calculate boundary positions
    len_s_t = len(sorted_s_t)
    len_entity = len(sorted_entity)
    len_interaction = len(sorted_interaction)
    
    line1_pos = len_s_t - 0.5
    line2_pos = len_s_t + len_entity - 0.5
    
    # Add dashed lines
    plt.axvline(line1_pos, color='black', linestyle='--', linewidth=2, alpha=0.5)
    plt.axvline(line2_pos, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add section labels
    max_y = max(all_counts)
    label_y_pos = max_y * 1.15 # Position labels above bars
    
    plt.text(-0.4, label_y_pos, 
             "spatio-temporal\ninteraction", ha='center', fontsize=FONTSIZE, weight='bold')
             
    plt.text(line1_pos + (len_entity / 2), label_y_pos, 
             "entity", ha='center', fontsize=FONTSIZE, weight='bold')
             
    plt.text(line2_pos + (len_interaction / 2), label_y_pos, 
             "interaction type", ha='center', fontsize=FONTSIZE, weight='bold')

    # --- 5. Final Formatting ---
    
    # plt.title('Class Distribution by Taxonomy', fontsize=20, weight='bold')
    plt.ylabel('Count', fontsize=FONTSIZE)
    plt.xticks(indices, all_labels, rotation=90, ha='center', fontsize=FONTSIZE)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.xlim(-0.5, len(all_labels) - 0.5)
    plt.ylim(0, max_y * 1.25) # Give space for text labels
    plt.yticks(fontsize=FONTSIZE)
    
    plt.tight_layout()
    
    # Save and show the figure
    output_filename = "class_distribution_taxonomy.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved as '{output_filename}'")

# ---
# --- Run the script ---
# ---

# plot_class_distribution()
# %%
def plot_class_distribution():
    """
    Creates and saves a bar chart for the class distribution,
    sectioned and colored according to the taxonomy.
    """
    
    # --- 1. Define Data and Mappings ---
    
    # Section 1: Spatio-temporal Interaction
    data_s_t = {
        "Spatial": 2845,
        "Temporal": 4571,
    }
    map_s_t = {
        "Spatial": "spatial",
        "Temporal": "temporal",
    }
    
    # Section 2: Entity
    data_entity = {
        "Human-Human": 2167,
        "Human-Object": 1541,
        "No Interaction": 1701,
        "Human-Animal": 395,
        "Animal-Animal": 810,
        "Animal-Object": 365,
        "Object-Object": 277,
        "Human-Self": 160,
    }
    map_entity = {
        "Human-Human": "human-human",
        "Human-Object": "human-object",
        "No Interaction": "no interaction",
        "Human-Animal": "human-animal",
        "Animal-Animal": "animal-animal",
        "Animal-Object": "object-animal",
        "Object-Object": "object-object",
        "Human-Self": "self interaction",
    }

    # Section 3: Interaction Type
    data_interaction = {
        "Relational Movement": 3738,
        "Observation": 2040,
        "Physical Interaction": 1902,
        "Body Motion": 1879,
        "Proximity": 1158,
        "Communicative": 487,
        "Passive": 347,
        "Supportive": 256,
        "Affective": 245,
        "Antagonistic": 236,
        "Provisioning": 160,
        "Social": 140,
        "Cooperative": 123,
        "Competitive": 28,
    }
    map_interaction = {
        "Relational Movement": "Relational Movement",
        "Observation": "Observation",
        "Physical Interaction": "Physical",
        "Body Motion": "Body Motion",
        "Proximity": "Proximity",
        "Communicative": "Communicative",
        "Passive": "Passive",
        "Supportive": "Supportive",
        "Affective": "Affective",
        "Antagonistic": "Antagonistic",
        "Provisioning": "Provisioning",
        "Social": "Social",
        "Cooperative": "Cooperative",
        "Competitive": "Competitive",
    }

    # --- 2. Prepare Data for Plotting ---
    
    # Sort each section by count (descending)
    sorted_s_t = sorted(data_s_t.items(), key=lambda item: item[1], reverse=True)
    sorted_entity = sorted(data_entity.items(), key=lambda item: item[1], reverse=True)
    sorted_interaction = sorted(data_interaction.items(), key=lambda item: item[1], reverse=True)
    
    # Combine all data into single lists
    all_labels = []
    all_counts = []
    all_colors = []

    
    # Add Section 1 data
    for label, count in sorted_s_t:
        all_labels.append(label)
        all_counts.append(count)
        color_key = map_s_t.get(label)
        all_colors.append(NEW_L1_COLOR_MAP.get(color_key, FALLBACK_COLOR))
        
    # Add Section 2 data
    for label, count in sorted_entity:
        all_labels.append(label)
        all_counts.append(count)
        color_key = map_entity.get(label)
        all_colors.append(NEW_L2_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    # Add Section 3 data
    for label, count in sorted_interaction:
        all_labels.append(label)
        all_counts.append(count)
        color_key = map_interaction.get(label)
        all_colors.append(NEW_L3_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    
    all_labels.reverse()
    all_counts.reverse()
    all_colors.reverse()

    # --- 3. Create the Plot ---
    
    plt.figure(figsize=(12, 42)) 
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
    
    # Remove top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    indices = np.arange(len(all_labels))
    # CHANGE 1: Use plt.barh(). Height controls bar thickness.
    plt.barh(indices, all_counts, color=all_colors, edgecolor='none', height=0.7) 
    
    # Add count text on the bars
    X_OFFSET = 100 # Offset for text to be outside the bar
    for i, count in enumerate(all_counts):
        # CHANGE 2: Swap X (count) and Y (index) coordinates, align text horizontally
        plt.text(count + X_OFFSET, i, str(count), ha='left', va='center', fontsize=FONTSIZE-10)

    # --- 4. Add Section Lines and Labels ---
    
    # --- 4. Add Section Lines and Labels ---

    # Calculate boundary lengths
    len_s_t = len(sorted_s_t) # 2
    len_entity = len(sorted_entity) # 8
    len_interaction = len(sorted_interaction) # 14

    # CHANGE: Recalculate positions based on the new reversed order
    # Line 1 (between Interaction Type and Entity)
    line1_pos = len_interaction - 0.5 

    # Line 2 (between Entity and Spatio-temporal Interaction)
    line2_pos = len_interaction + len_entity - 0.5

    # Add horizontal dashed lines
    # CHANGE 3: Use plt.axhline()
    plt.axhline(line1_pos, color='black', linestyle='--', linewidth=2.5)
    plt.axhline(line2_pos, color='black', linestyle='--', linewidth=2.5)
    
    # Calculate max X for label positioning
    max_x = max(all_counts)
    label_x_pos = max_x * 1.15 # Position labels to the right of the graph

    # Section labels (Vertical graph runs from bottom-up)
    # CHANGE 4: Swap coordinates, align vertically with rotation=90
    
    # Spatio-temporal (x=fixed high value, y=center of section 1)
    ax.text(1.05, 0.94, 
             "spatio-temporal", 
             ha='center', va='center', 
             fontsize=FONTSIZE, weight='bold', rotation=0,
             transform=ax.transAxes) # KEY CHANGE: Use ax.transAxes for normalized coordinates
             
    # 2. Entity (e.g., Center of the plot)
    ax.text(0.65, 0.75, 
             "entity", 
             ha='center', va='center', 
             fontsize=FONTSIZE, weight='bold', rotation=0,
             transform=ax.transAxes) # KEY CHANGE
             
    # 3. Interaction Type (e.g., Bottom right corner of the plot)
    ax.text(0.65, 0.25, 
             "interaction Type", 
             ha='center', va='center', 
             fontsize=FONTSIZE, weight='bold', rotation=0,
             transform=ax.transAxes) # KEY CHANGE

    # --- 5. Final Formatting ---
    
    # CHANGE 5: Swap X and Y labels
    plt.xlabel('count', fontsize=FONTSIZE)
    plt.ylabel(None) # Remove Y-axis label as the category names are the labels
    plt.xticks(fontsize=FONTSIZE - 20)
    
    # CHANGE 6: Use plt.yticks() for category labels (vertical axis)
    plt.yticks(indices, [label.lower() for label in all_labels], fontsize=FONTSIZE)
    
    # CHANGE 7: Grid lines should be vertical (axis='x')
    plt.grid(axis='x', linestyle=':', alpha=0.7)

    # CHANGE 8: Adjust limits
    plt.xlim(0, max_x * 1.35) # Max X for space for text labels
    plt.ylim(-0.5, len(all_labels) - 0.5) # Y limits
    
    # plt.tight_layout()
    
    # Save and show the figure
    output_filename = "class_distribution_vertical_taxonomy.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved as '{output_filename}'")
# %%

plot_class_distribution()

# %%

def plot_class_distribution_percent():
    """
    Creates and saves a bar chart for the class distribution,
    sectioned and colored according to the taxonomy, showing percentages.
    """
    
    # --- 1. Define Data and Mappings ---
    
    # Section 1: Spatio-temporal Interaction
    data_s_t = {
        "Spatial": 2845,
        "Temporal": 4571,
    }
    map_s_t = {
        "Spatial": "spatial",
        "Temporal": "temporal",
    }
    
    # Section 2: Entity
    data_entity = {
        "Human-Human": 2167,
        "Human-Object": 1541,
        "No Interaction": 1701,
        "Human-Animal": 395,
        "Animal-Animal": 810,
        "Animal-Object": 365,
        "Object-Object": 277,
        "Human-Self": 160,
    }
    map_entity = {
        "Human-Human": "human-human",
        "Human-Object": "human-object",
        "No Interaction": "no interaction",
        "Human-Animal": "human-animal",
        "Animal-Animal": "animal-animal",
        "Animal-Object": "object-animal",
        "Object-Object": "object-object",
        "Human-Self": "self interaction",
    }

    # Section 3: Interaction Type
    data_interaction = {
        "Relational Movement": 3738,
        "Observation": 2040,
        "Physical Interaction": 1902,
        "Body Motion": 1879,
        "Proximity": 1158,
        "Communicative": 487,
        "Passive": 347,
        "Supportive": 256,
        "Affective": 245,
        "Antagonistic": 236,
        "Provisioning": 160,
        "Social": 140,
        "Cooperative": 123,
        "Competitive": 28,
    }
    map_interaction = {
        "Relational Movement": "Relational Movement",
        "Observation": "Observation",
        "Physical Interaction": "Physical",
        "Body Motion": "Body Motion",
        "Proximity": "Proximity",
        "Communicative": "Communicative",
        "Passive": "Passive",
        "Supportive": "Supportive",
        "Affective": "Affective",
        "Antagonistic": "Antagonistic",
        "Provisioning": "Provisioning",
        "Social": "Social",
        "Cooperative": "Cooperative",
        "Competitive": "Competitive",
    }

    # --- 2. Prepare Data for Plotting ---
    
    # NEW: Calculate grand total
    grand_total = (
        sum(data_s_t.values()) + 
        sum(data_entity.values()) + 
        sum(data_interaction.values())
    )
    
    # Sort each section by count (descending)
    sorted_s_t = sorted(data_s_t.items(), key=lambda item: item[1], reverse=True)
    sorted_entity = sorted(data_entity.items(), key=lambda item: item[1], reverse=True)
    sorted_interaction = sorted(data_interaction.items(), key=lambda item: item[1], reverse=True)
    
    # Combine all data into single lists
    all_labels = []
    all_percentages = [] # RENAMED from all_counts
    all_colors = []

    
    # Add Section 1 data
    for label, count in sorted_s_t:
        all_labels.append(label)
        all_percentages.append((count / grand_total) * 100) # CHANGED
        color_key = map_s_t.get(label)
        all_colors.append(NEW_L1_COLOR_MAP.get(color_key, FALLBACK_COLOR))
        
    # Add Section 2 data
    for label, count in sorted_entity:
        all_labels.append(label)
        all_percentages.append((count / grand_total) * 100) # CHANGED
        color_key = map_entity.get(label)
        all_colors.append(NEW_L2_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    # Add Section 3 data
    for label, count in sorted_interaction:
        all_labels.append(label)
        all_percentages.append((count / grand_total) * 100) # CHANGED
        color_key = map_interaction.get(label)
        all_colors.append(NEW_L3_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    
    all_labels.reverse()
    all_percentages.reverse() # RENAMED
    all_colors.reverse()

    # --- 3. Create the Plot ---
    
    plt.figure(figsize=(12, 42)) 
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    indices = np.arange(len(all_labels))
    
    # CHANGED: Plot all_percentages
    plt.barh(indices, all_percentages, color=all_colors, edgecolor='none', height=0.7) 
    
    # Add percentage text on the bars
    X_OFFSET = 0.1 # CHANGED: Offset is now relative to percentage
    for i, percent in enumerate(all_percentages):
        # CHANGED: Format as percentage string
        text_label = f"{percent:.1f}%"
        plt.text(percent + X_OFFSET, i, text_label, ha='left', va='center', fontsize=FONTSIZE-10)

    # --- 4. Add Section Lines and Labels ---

    len_s_t = len(sorted_s_t)
    len_entity = len(sorted_entity)
    len_interaction = len(sorted_interaction)

    line1_pos = len_interaction - 0.5 
    line2_pos = len_interaction + len_entity - 0.5

    plt.axhline(line1_pos, color='black', linestyle='--', linewidth=2.5)
    plt.axhline(line2_pos, color='black', linestyle='--', linewidth=2.5)
    
    # Calculate max X for label positioning
    max_percent = max(all_percentages) # RENAMED
    
    # Section labels (Vertical graph runs from bottom-up)
    ax.text(1.1, 0.94, 
             "spatio-temporal", 
             ha='center', va='center', 
             fontsize=FONTSIZE, weight='bold', rotation=0,
             transform=ax.transAxes)
             
    ax.text(0.65, 0.75, 
             "entity", 
             ha='center', va='center', 
             fontsize=FONTSIZE, weight='bold', rotation=0,
             transform=ax.transAxes)
             
    ax.text(0.80, 0.25, 
             "interaction Type", 
             ha='center', va='center', 
             fontsize=FONTSIZE, weight='bold', rotation=0,
             transform=ax.transAxes)

    # --- 5. Final Formatting ---
    
    # CHANGED: X-axis label
    plt.xlabel('percentage (%)', fontsize=FONTSIZE) 
    plt.ylabel(None)
    plt.xticks(fontsize=FONTSIZE - 20)
    
    plt.yticks(indices, [label.lower() for label in all_labels], fontsize=FONTSIZE)
    
    plt.grid(axis='x', linestyle=':', alpha=0.7)

    # CHANGED: Adjust limits based on max_percent
    plt.xlim(0, max_percent * 1.35)
    plt.ylim(-0.5, len(all_labels) - 0.5)
    
    # Save and show the figure
    output_filename = "class_distribution_vertical_taxonomy_percent.png" # CHANGED
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved as '{output_filename}'")

# %%
FONTSIZE=60
plot_class_distribution_percent()

# %%

def plot_class_distribution_percent_per_section():
    """
    Creates and saves a bar chart for the class distribution,
    sectioned and colored, showing percentages *within each section*.
    """
    
    # --- 1. Define Data and Mappings ---
    
    # Section 1: Spatio-temporal Interaction
    data_s_t = {
        "Spatial": 2845,
        "Temporal": 4571,
    }
    map_s_t = {
        "Spatial": "spatial",
        "Temporal": "temporal",
    }
    
    # Section 2: Entity
    data_entity = {
        "Human-Human": 2167,
        "Human-Object": 1541,
        "No Interaction": 1701,
        "Human-Animal": 395,
        "Animal-Animal": 810,
        "Animal-Object": 365,
        "Object-Object": 277,
        "Human-Self": 160,
    }
    map_entity = {
        "Human-Human": "human-human",
        "Human-Object": "human-object",
        "No Interaction": "no interaction",
        "Human-Animal": "human-animal",
        "Animal-Animal": "animal-animal",
        "Animal-Object": "object-animal",
        "Object-Object": "object-object",
        "Human-Self": "self interaction",
    }

    # Section 3: Interaction Type
    data_interaction = {
        "Relational Movement": 3738,
        "Observation": 2040,
        "Physical Interaction": 1902,
        "Body Motion": 1879,
        "Proximity": 1158,
        "Communicative": 487,
        "Passive": 347,
        "Supportive": 256,
        "Affective": 245,
        "Antagonistic": 236,
        "Provisioning": 160,
        "Social": 140,
        "Cooperative": 123,
        "Competitive": 28,
    }
    map_interaction = {
        "Relational Movement": "Relational Movement",
        "Observation": "Observation",
        "Physical Interaction": "Physical",
        "Body Motion": "Body Motion",
        "Proximity": "Proximity",
        "Communicative": "Communicative",
        "Passive": "Passive",
        "Supportive": "Supportive",
        "Affective": "Affective",
        "Antagonistic": "Antagonistic",
        "Provisioning": "Provisioning",
        "Social": "Social",
        "Cooperative": "Cooperative",
        "Competitive": "Competitive",
    }

    # --- 2. Prepare Data for Plotting ---
    
    # CHANGED: Calculate totals for each section individually
    total_s_t = sum(data_s_t.values())
    total_entity = sum(data_entity.values())
    total_interaction = sum(data_interaction.values())
    
    # Sort each section by count (descending)
    sorted_s_t = sorted(data_s_t.items(), key=lambda item: item[1], reverse=True)
    sorted_entity = sorted(data_entity.items(), key=lambda item: item[1], reverse=True)
    sorted_interaction = sorted(data_interaction.items(), key=lambda item: item[1], reverse=True)
    
    # Combine all data into single lists
    all_labels = []
    all_percentages = [] 
    all_colors = []

    
    # Add Section 1 data
    for label, count in sorted_s_t:
        all_labels.append(label)
        # CHANGED: Divide by section total
        all_percentages.append((count / total_s_t) * 100) 
        color_key = map_s_t.get(label)
        all_colors.append(NEW_L1_COLOR_MAP.get(color_key, FALLBACK_COLOR))
        
    # Add Section 2 data
    for label, count in sorted_entity:
        all_labels.append(label)
        # CHANGED: Divide by section total
        all_percentages.append((count / total_entity) * 100) 
        color_key = map_entity.get(label)
        all_colors.append(NEW_L2_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    # Add Section 3 data
    for label, count in sorted_interaction:
        all_labels.append(label)
        # CHANGED: Divide by section total
        all_percentages.append((count / total_interaction) * 100)
        color_key = map_interaction.get(label)
        all_colors.append(NEW_L3_COLOR_MAP.get(color_key, FALLBACK_COLOR))

    
    all_labels.reverse()
    all_percentages.reverse()
    all_colors.reverse()

    # --- 3. Create the Plot ---
    
    plt.figure(figsize=(13, 42)) 
    ax = plt.gca()
    # Set max 5 ticks on x-axis, as you had in your code
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    indices = np.arange(len(all_labels))
    
    plt.barh(indices, all_percentages, color=all_colors, edgecolor='none', height=0.7) 
    
    # Add percentage text on the bars
    X_OFFSET = 0.5 # Increased offset slightly for larger fonts
    for i, percent in enumerate(all_percentages):
        text_label = f"{percent:.1f}"
        plt.text(percent + X_OFFSET, i, text_label, ha='left', va='center', fontsize=FONTSIZE-15)

    # --- 4. Add Section Lines and Labels ---

    len_s_t = len(sorted_s_t)
    len_entity = len(sorted_entity)
    len_interaction = len(sorted_interaction)

    line1_pos = len_interaction - 0.5 
    line2_pos = len_interaction + len_entity - 0.5

    plt.axhline(line1_pos, color='black', linestyle='--', linewidth=2.5)
    plt.axhline(line2_pos, color='black', linestyle='--', linewidth=2.5)
    
    max_percent = max(all_percentages)
    
    # Using the coordinates from your provided script
    ax.text(1.2, 0.94, 
             "spatio-temporal", 
             ha='center', va='center', 
             fontsize=FONTSIZE-5, weight='bold', rotation=0,
             transform=ax.transAxes)
             
    ax.text(0.9, 0.75, 
             "entity", 
             ha='center', va='center', 
             fontsize=FONTSIZE-5, weight='bold', rotation=0,
             transform=ax.transAxes)
             
    ax.text(0.9, 0.25, 
             "interaction type", 
             ha='center', va='center', 
             fontsize=FONTSIZE-5, weight='bold', rotation=0,
             transform=ax.transAxes)

    # --- 5. Final Formatting ---
    
    plt.xlabel('percentage (%)', fontsize=FONTSIZE) 
    plt.ylabel(None)
    plt.xticks(fontsize=FONTSIZE - 10)
    
    plt.yticks(indices, [label.lower() for label in all_labels], fontsize=FONTSIZE)
    
    plt.grid(axis='x', linestyle=':', alpha=0.7)

    # Set x-limit to go to 100% (or slightly more) since it's percentage
    # Or base it on the max_percent, which might be < 100
    # Let's set it to 100, which is a standard ceiling for percentages.
    # You can change this to `max_percent * 1.35` if you prefer.
    plt.xlim(0, 105) # Set fixed 100% axis
    plt.ylim(-0.5, len(all_labels) - 0.5)
    
    # Save and show the figure
    output_filename = "class_distribution_final.png" # CHANGED
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved as '{output_filename}'")

# %%
# Set the global FONTSIZE variable as you did
FONTSIZE = 90
plot_class_distribution_percent_per_section()

# %%