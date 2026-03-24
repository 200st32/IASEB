# %%
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.patches import Wedge
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as path_effects

# =====================================================================
# CONTROL PANEL - Adjust all visual parameters here
# =====================================================================

CONTROL_PANEL = {
    # Figure settings
    'use_cvpr_proportions': True,  # If True, uses CVPR figure size
    'fig_size_pt': 237.13594,      # CVPR column width in points
    
    # Ring dimensions
    'donut_hole_radius': 0.3,      # Size of center hole (0 = no hole)
    'l1_ring_width': 0.5,          # Width of innermost ring
    'l2_ring_width': 0.8,          # Width of second ring
    'l3_ring_width': 0.8,          # Width of third ring
    'l4_ring_width': 0.8,          # Width of outermost ring
    
    # Spacing between rings
    'l1_l2_gap': 0.05,             # Gap between L1 and L2
    'l2_l3_gap': 0.05,             # Gap between L2 and L3
    'l3_l4_gap': 0.05,             # Gap between L3 and L4
    
    # Rotation
    'l1_startangle': 90,           # Starting angle for L1 (degrees)
    'l2_rotation_offset': 0,       # Additional rotation for L2
    'l3_rotation_offset': 0,       # Additional rotation for L3
    'l4_rotation_offset': 0,       # Additional rotation for L4
    
    # Visual styling
    'edge_color': 'white',         # Color of lines between wedges
    'edge_linewidth': 1.5,         # Width of lines between wedges
    
    # Text settings
    'font_size': 8,                # Base font size for labels
    'text_color': 'black',         # Color of text labels
    'min_angle_for_label': 3,      # Minimum wedge angle to show label (degrees)
    'enable_text_wrapping': True,  # Enable curved text (experimental)
}

# =====================================================================
# TAXONOMY DEFINITION
# =====================================================================

# Level 1 (Innermost ring)
L1_NAMES = [
    "spatial", 
    "temporal"
]

# Level 2 (Second ring)
L2_NAMES = [
    "human-human",
    "human-object",
    "human-animal",
    "object-animal",
    "object-object",
    "animal-animal",
    "self interaction",
    "no interaction"
]

# Level 3 (Third ring)
L3_NAMES = [
    "Emotional and Social",
    "Physical and Action-Oriented",
    "Observational and Passive"
]

# Level 4 (Outermost ring)
L4_NAMES = [
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

# =====================================================================
# COLOR GENERATION
# =====================================================================

def generate_discrete_hsl_palette(names, start_hue_degrees, saturation, lightness):
    """Generate evenly distributed colors around the color wheel."""
    palette = {}
    num_names = len(names)
    if num_names == 0:
        return {}
        
    hues_list = np.linspace(start_hue_degrees, start_hue_degrees + 360, 
                           num=num_names, endpoint=False) % 360.0

    for name, hue_deg in zip(names, hues_list):
        hue_01 = hue_deg / 360.0
        rgb_01 = colorsys.hls_to_rgb(hue_01, lightness, saturation)
        palette[name] = rgb_01
        
    return palette

# Color parameters for each level
HLS_PARAMS = {
    1: {'saturation': 0.70, 'lightness': 0.60},
    2: {'saturation': 0.55, 'lightness': 0.70},
    3: {'saturation': 0.45, 'lightness': 0.75},
    4: {'saturation': 0.40, 'lightness': 0.80},
}

HUE_STARTS = {
    1: 0.0,    # Not used (L1 is hardcoded)
    2: 0.0,    # L2 starts at red
    3: 120.0,  # L3 starts at green
    4: 40.0,   # L4 starts at orange/yellow
}

# Generate color maps
_l1_params = HLS_PARAMS[1]
L1_COLOR_MAP = {
    "spatial": colorsys.hls_to_rgb(30/360.0, _l1_params['lightness'], _l1_params['saturation']),
    "temporal": colorsys.hls_to_rgb(190/360.0, _l1_params['lightness'], _l1_params['saturation'])
}

L2_COLOR_MAP = generate_discrete_hsl_palette(
    names=L2_NAMES,
    start_hue_degrees=HUE_STARTS[2],
    **HLS_PARAMS[2]
)

L3_COLOR_MAP = generate_discrete_hsl_palette(
    names=L3_NAMES,
    start_hue_degrees=HUE_STARTS[3],
    **HLS_PARAMS[3]
)

L4_COLOR_MAP = generate_discrete_hsl_palette(
    names=L4_NAMES,
    start_hue_degrees=HUE_STARTS[4],
    **HLS_PARAMS[4]
)

# =====================================================================
# SUNBURST CHART CREATION
# =====================================================================

def create_sunburst_chart():
    """Create a 3-level sunburst chart with the defined taxonomy."""
    
    cp = CONTROL_PANEL  # Shorthand
    
    # Calculate figure size
    if cp['use_cvpr_proportions']:
        fig_height_in = cp['fig_size_pt'] / 72.27
        fig_size = (fig_height_in, fig_height_in)
    else:
        fig_size = (8, 8)
    
    # Calculate radii for each ring
    l1_inner = cp['donut_hole_radius']
    l1_outer = l1_inner + cp['l1_ring_width']
    
    l2_inner = l1_outer + cp['l1_l2_gap']
    l2_outer = l2_inner + cp['l2_ring_width']
    
    l3_inner = l2_outer + cp['l2_l3_gap']
    l3_outer = l3_inner + cp['l3_ring_width']
    
    l4_inner = l3_outer + cp['l3_l4_gap']
    l4_outer = l4_inner + cp['l4_ring_width']
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(aspect="equal"))
    ax.set_xlim(-l4_outer*1.1, l4_outer*1.1)
    ax.set_ylim(-l4_outer*1.1, l4_outer*1.1)
    ax.axis('off')
    
    # Prepare data (equal counts for each category)
    l1_data = {name: 1 for name in L1_NAMES}
    l2_data = {name: 1 for name in L2_NAMES}
    l3_data = {name: 1 for name in L3_NAMES}
    l4_data = {name: 1 for name in L4_NAMES}
    
    l1_total = sum(l1_data.values())
    l2_total = sum(l2_data.values())
    l3_total = sum(l3_data.values())
    l4_total = sum(l4_data.values())
    
    # Calculate angles for each wedge
    def calc_wedges(data, total, startangle):
        wedges = []
        current_angle = startangle
        for name, value in data.items():
            angle_extent = 360 * (value / total)
            wedges.append({
                'name': name,
                'start': current_angle,
                'extent': angle_extent,
                'end': current_angle + angle_extent
            })
            current_angle += angle_extent
        return wedges
    
    l1_wedges = calc_wedges(l1_data, l1_total, cp['l1_startangle'])
    l2_wedges = calc_wedges(l2_data, l2_total, cp['l1_startangle'] + cp['l2_rotation_offset'])
    l3_wedges = calc_wedges(l3_data, l3_total, cp['l1_startangle'] + cp['l3_rotation_offset'])
    l4_wedges = calc_wedges(l4_data, l4_total, cp['l1_startangle'] + cp['l4_rotation_offset'])
    
    # Draw wedges
    def draw_ring(wedges, inner_r, outer_r, color_map):
        for w in wedges:
            color = color_map.get(w['name'], (0.5, 0.5, 0.5))
            wedge = Wedge(
                center=(0, 0),
                r=outer_r,
                theta1=w['start'],
                theta2=w['end'],
                width=outer_r - inner_r,
                facecolor=color,
                edgecolor=cp['edge_color'],
                linewidth=cp['edge_linewidth']
            )
            ax.add_patch(wedge)
            w['wedge'] = wedge
    
    draw_ring(l1_wedges, l1_inner, l1_outer, L1_COLOR_MAP)
    draw_ring(l2_wedges, l2_inner, l2_outer, L2_COLOR_MAP)
    draw_ring(l3_wedges, l3_inner, l3_outer, L3_COLOR_MAP)
    draw_ring(l4_wedges, l4_inner, l4_outer, L4_COLOR_MAP)
    
    # Add labels
    def add_labels(wedges, radius):
        for w in wedges:
            if w['extent'] < cp['min_angle_for_label']:
                continue
                
            angle_mid = (w['start'] + w['end']) / 2
            angle_rad = np.deg2rad(angle_mid)
            
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            
            # Rotation for text
            rotation = angle_mid
            if 90 < angle_mid < 270:
                rotation = angle_mid + 180
            
            # Truncate long labels
            label = w['name'][:15]
            
            text = ax.text(
                x, y, label,
                rotation=rotation,
                ha='center',
                va='center',
                fontsize=cp['font_size'],
                color=cp['text_color'],
                weight='normal'
            )
            text.set_clip_on(True)
    
    # Calculate label radii (middle of each ring)
    l1_label_r = (l1_inner + l1_outer) / 2
    l2_label_r = (l2_inner + l2_outer) / 2
    l3_label_r = (l3_inner + l3_outer) / 2
    l4_label_r = (l4_inner + l4_outer) / 2
    
    add_labels(l1_wedges, l1_label_r)
    add_labels(l2_wedges, l2_label_r)
    add_labels(l3_wedges, l3_label_r)
    add_labels(l4_wedges, l4_label_r)
    
    plt.tight_layout()
    return fig, ax

# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    print("Creating sunburst chart...")
    print(f"  L1 categories: {len(L1_NAMES)}")
    print(f"  L2 categories: {len(L2_NAMES)}")
    print(f"  L3 categories: {len(L3_NAMES)}")
    print(f"  L4 categories: {len(L4_NAMES)}")
    
    fig, ax = create_sunburst_chart()
    
    # Save the figure
    output_filename = 'taxonomy.png'
    # plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_filename, dpi=300, bbox_inches=None)
    # print(f"\nChart saved as: {output_filename}")
    
    plt.show()
    print("Done!")
# %%
