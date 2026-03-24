# %%
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.patches import Wedge, FancyArrowPatch
import matplotlib.patheffects as path_effects

# =====================================================================
# CONTROL PANEL - Adjust all visual parameters here
# =====================================================================

CONTROL_PANEL = {
    # Figure settings
    'use_cvpr_proportions': True,
    'fig_size_pt': 237.13594,      # CVPR column width in points

    # Ring dimensions
    'donut_hole_radius': 0.25,
    'l1_ring_width': 0.38,
    'l2_ring_width': 0.55,
    'l3_ring_width': 0.55,
    'l4_ring_width': 0.55,

    # Spacing between rings
    'l1_l2_gap': 0.04,
    'l2_l3_gap': 0.04,
    'l3_l4_gap': 0.04,

    # Rotation
    'l1_startangle': 90,
    'l2_rotation_offset': 0,
    'l3_rotation_offset': 0,
    'l4_rotation_offset': 0,

    # Visual styling
    'edge_color': 'white',
    'edge_linewidth': 1.2,

    # Text settings per level
    'l1_font_size': 6.5,
    'l2_font_size': 5.5,
    'l3_font_size': 5.0,
    'l4_font_size': 4.5,
    'text_color': 'black',
    'min_angle_for_label': 3,
}

# =====================================================================
# TAXONOMY DEFINITION
# =====================================================================

L1_NAMES = ["spatial", "temporal"]

L2_NAMES = [
    "human-human", "human-object", "human-animal", "object-animal",
    "object-object", "animal-animal", "self interaction", "no interaction"
]

L3_NAMES = [
    "Emot. & Social",
    "Phys. & Action",
    "Observ. & Passive"
]

# Full names for legend, abbreviated for wedge labels
L4_NAMES = [
    "Affective", "Antagonistic", "Body Motion", "Communicative",
    "Competitive", "Cooperative", "Rel. Movement", "Observation",
    "Passive", "Physical", "Provisioning", "Proximity",
    "Social", "Supportive"
]

# Abbreviations for labels that are too long for their wedge
L2_ABBREV = {
    "self interaction": "self\ninteract.",
    "no interaction": "no\ninteract.",
    "human-human": "H-H",
    "human-object": "H-O",
    "human-animal": "H-A",
    "object-animal": "O-A",
    "object-object": "O-O",
    "animal-animal": "A-A",
}

L4_ABBREV = {
    "Antagonistic": "Antag.",
    "Communicative": "Comm.",
    "Competitive": "Compet.",
    "Cooperative": "Coop.",
    "Rel. Movement": "Rel. Mov.",
    "Provisioning": "Provis.",
    "Supportive": "Support.",
    "Body Motion": "Body M.",
    "Observation": "Observ.",
}

# =====================================================================
# COLOR GENERATION
# =====================================================================

def generate_discrete_hsl_palette(names, start_hue_degrees, saturation, lightness):
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

HLS_PARAMS = {
    1: {'saturation': 0.70, 'lightness': 0.60},
    2: {'saturation': 0.55, 'lightness': 0.70},
    3: {'saturation': 0.45, 'lightness': 0.75},
    4: {'saturation': 0.40, 'lightness': 0.80},
}

HUE_STARTS = {1: 0.0, 2: 0.0, 3: 120.0, 4: 40.0}

_l1_params = HLS_PARAMS[1]
L1_COLOR_MAP = {
    "spatial": colorsys.hls_to_rgb(30/360.0, _l1_params['lightness'], _l1_params['saturation']),
    "temporal": colorsys.hls_to_rgb(190/360.0, _l1_params['lightness'], _l1_params['saturation'])
}

L2_COLOR_MAP = generate_discrete_hsl_palette(L2_NAMES, HUE_STARTS[2], **HLS_PARAMS[2])
L3_COLOR_MAP = generate_discrete_hsl_palette(L3_NAMES, HUE_STARTS[3], **HLS_PARAMS[3])
L4_COLOR_MAP = generate_discrete_hsl_palette(L4_NAMES, HUE_STARTS[4], **HLS_PARAMS[4])

# =====================================================================
# SUNBURST CHART CREATION
# =====================================================================

def create_sunburst_chart():
    cp = CONTROL_PANEL

    if cp['use_cvpr_proportions']:
        fig_height_in = cp['fig_size_pt'] / 72.27
        fig_size = (fig_height_in, fig_height_in)
    else:
        fig_size = (8, 8)

    # Calculate radii
    l1_inner = cp['donut_hole_radius']
    l1_outer = l1_inner + cp['l1_ring_width']

    l2_inner = l1_outer + cp['l1_l2_gap']
    l2_outer = l2_inner + cp['l2_ring_width']

    l3_inner = l2_outer + cp['l2_l3_gap']
    l3_outer = l3_inner + cp['l3_ring_width']

    l4_inner = l3_outer + cp['l3_l4_gap']
    l4_outer = l4_inner + cp['l4_ring_width']

    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(aspect="equal"))
    # Extra margin for external labels
    margin = l4_outer * 1.35
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.axis('off')

    # Equal-weight data
    l1_data = {name: 1 for name in L1_NAMES}
    l2_data = {name: 1 for name in L2_NAMES}
    l3_data = {name: 1 for name in L3_NAMES}
    l4_data = {name: 1 for name in L4_NAMES}

    def calc_wedges(data, startangle):
        total = sum(data.values())
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

    l1_wedges = calc_wedges(l1_data, cp['l1_startangle'])
    l2_wedges = calc_wedges(l2_data, cp['l1_startangle'] + cp['l2_rotation_offset'])
    l3_wedges = calc_wedges(l3_data, cp['l1_startangle'] + cp['l3_rotation_offset'])
    l4_wedges = calc_wedges(l4_data, cp['l1_startangle'] + cp['l4_rotation_offset'])

    def draw_ring(wedges, inner_r, outer_r, color_map):
        for w in wedges:
            color = color_map.get(w['name'], (0.5, 0.5, 0.5))
            wedge = Wedge(
                center=(0, 0), r=outer_r,
                theta1=w['start'], theta2=w['end'],
                width=outer_r - inner_r,
                facecolor=color,
                edgecolor=cp['edge_color'],
                linewidth=cp['edge_linewidth']
            )
            ax.add_patch(wedge)

    draw_ring(l1_wedges, l1_inner, l1_outer, L1_COLOR_MAP)
    draw_ring(l2_wedges, l2_inner, l2_outer, L2_COLOR_MAP)
    draw_ring(l3_wedges, l3_inner, l3_outer, L3_COLOR_MAP)
    draw_ring(l4_wedges, l4_inner, l4_outer, L4_COLOR_MAP)

    # --- Internal labels (L1, L2, L3) ---
    def add_internal_labels(wedges, radius, font_size, abbrev_map=None):
        for w in wedges:
            if w['extent'] < cp['min_angle_for_label']:
                continue
            angle_mid = (w['start'] + w['end']) / 2
            angle_rad = np.deg2rad(angle_mid)

            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)

            rotation = angle_mid
            if 90 < angle_mid % 360 < 270:
                rotation = angle_mid + 180

            label = w['name']
            if abbrev_map and label in abbrev_map:
                label = abbrev_map[label]

            ax.text(
                x, y, label,
                rotation=rotation,
                ha='center', va='center',
                fontsize=font_size,
                color=cp['text_color'],
                weight='normal',
                clip_on=True
            )

    # --- External labels with leader lines (L4) ---
    def add_external_labels(wedges, inner_r, outer_r, font_size, abbrev_map=None):
        leader_start_r = outer_r + 0.03
        leader_end_r = outer_r + 0.25

        for w in wedges:
            if w['extent'] < cp['min_angle_for_label']:
                continue
            angle_mid = (w['start'] + w['end']) / 2
            angle_rad = np.deg2rad(angle_mid)

            # Leader line start (just outside ring)
            x0 = leader_start_r * np.cos(angle_rad)
            y0 = leader_start_r * np.sin(angle_rad)

            # Leader line end
            x1 = leader_end_r * np.cos(angle_rad)
            y1 = leader_end_r * np.sin(angle_rad)

            # Draw leader line
            ax.plot([x0, x1], [y0, y1], color='gray', linewidth=0.5, alpha=0.6)

            # Text alignment based on which side of the chart
            ha = 'left' if np.cos(angle_rad) >= 0 else 'right'

            label = w['name']
            if abbrev_map and label in abbrev_map:
                label = abbrev_map[label]

            ax.text(
                x1, y1, label,
                ha=ha, va='center',
                fontsize=font_size,
                color=cp['text_color'],
                weight='normal'
            )

    l1_label_r = (l1_inner + l1_outer) / 2
    l2_label_r = (l2_inner + l2_outer) / 2
    l3_label_r = (l3_inner + l3_outer) / 2

    add_internal_labels(l1_wedges, l1_label_r, cp['l1_font_size'])
    add_internal_labels(l2_wedges, l2_label_r, cp['l2_font_size'], L2_ABBREV)
    add_internal_labels(l3_wedges, l3_label_r, cp['l3_font_size'])
    add_external_labels(l4_wedges, l4_inner, l4_outer, cp['l4_font_size'], L4_ABBREV)

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

    output_filename = 'taxonomy.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Chart saved as: {output_filename}")

    # Also save PNG for quick preview
    plt.savefig('taxonomy.png', dpi=300, bbox_inches='tight')
    print("Preview saved as: taxonomy.png")

    plt.show()
    print("Done!")
# %%
