# %%
# ============================================
# CELL 1 — CONSTANTS, DATA LOADING, FUNCTIONS
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Explicitly disable LaTeX and set font
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

# ----------------------------
# GLOBAL CONSTANTS
# ----------------------------

# FIG2_WIDTH = 150
# FIG2_WIDTH_IN = FIG2_WIDTH /  72.27

# TEXT_WIDTH = 496.85625
# FIG_WIDTH_PT = TEXT_WIDTH - FIG2_WIDTH
# FIG_WIDTH_IN = FIG_WIDTH_PT / 72.27      # convert pt → inches (~3.28 in)


FIG_HEIGHT_PT = 237.13594
FIG_HEIGHT_IN = FIG_HEIGHT_PT / 72.27

# Radar chart positioning (in figure coordinates 0-1)
RADAR_CENTER_X = 0.487    # Center x position (0.5 = middle)
RADAR_CENTER_Y = 0.6    # Center y position (0.5 = middle)
RADAR_RADIUS = 0.35    # Radius of the radar chart (0.35 = 35% of figure)

FONTSIZE_CAT   = 8.5    # category label size
FONTSIZE_TICKS = 8     # numeric tick label size
LINEWIDTH      = 1.5  # line width

# ----------------------------
# YOUR ACRONYMS
# ----------------------------
entity_acronyms = {
    'Animal-Animal': 'AA',
    'Animal-Object': 'AO',
    'Human-Animal': 'HA',
    'Human-Human': 'HH',
    'Human-Object': 'HO',
    'Human-Self': 'HS',
    'No Interaction': 'NI',
    'Object-Object': 'OO'
}

other_acronyms = {
    'Antagonistic': 'ANT',
    'Affective': 'AFF',
    'Body Motion': 'BM',
    'Communicative': 'COM',
    'Competitive': 'CMP',
    'Cooperative': 'COP',
    'Observation': 'OBS',
    'Passive': 'PAS',
    'Physical Interaction': 'PHY',
    'Provisioning': 'PRV',
    'Proximity': 'PRX',
    'Social': 'SOC',
    'Supportive': 'SUP',
    'Relational Movement': 'RM'
}

# ----------------------------
# LOAD YOUR DATA
# ----------------------------
df = pd.read_csv("table1_finalized.csv", header=1, skiprows=[2])
df = df.rename(columns={'category': 'model'})
df = df.set_index("model")
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(how='all')

# ENTITY DATA
entity_cols = [c for c in entity_acronyms.keys() if c in df.columns]
df_entity = df[entity_cols].dropna(how='all')

# OTHER INTERACTIONS
other_cols = [c for c in other_acronyms.keys() if c in df.columns]
df_other = df[other_cols].dropna(how='all')

models = df.index.tolist()

# ----------------------------
# RADAR PLOTTING FUNCTION
# ----------------------------

def plot_radar(ax, df_subset, label_map, title):
    labels_full = df_subset.columns.tolist()
    labels_short = [c.lower().replace('-', '\n').replace(' ', '\n') for c in labels_full]

    N = len(labels_short)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    # Better color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(df_subset.index)))
    
    # Plot each model
    for idx, model in enumerate(df_subset.index):
        vals = df_subset.loc[model].fillna(0).values
        vals = np.concatenate([vals, [vals[0]]])

        ax.plot(angles, vals, linewidth=LINEWIDTH, label=model, color=colors[idx])
        ax.fill(angles, vals, alpha=0.15, color=colors[idx])

    # Orientation - start from top and go clockwise
    ax.set_theta_offset((np.pi / 2) + 30)
    # ax.set_theta_offset(45)
    ax.set_theta_direction(-1)

    # Category labels with better positioning
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_short, fontsize=FONTSIZE_CAT)
    ax.tick_params(axis='x', pad=2.5)  # Push labels out more
    
    # Clean up radial axis - fewer ticks, lighter styling
    ax.set_ylim(0, None)
    # Reduce number of radial ticks for cleaner look
    y_max = ax.get_ylim()[1]
    n_ticks = 4
    y_ticks = np.linspace(0, y_max, n_ticks + 1)[1:]  # Skip 0
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y)}' for y in y_ticks], 
                       fontsize=FONTSIZE_TICKS, color='gray', alpha=0.7)
    
    # Lighter grid
    ax.grid(True, linewidth=0.5, alpha=0.3, color='gray', linestyle='-')
    ax.spines['polar'].set_visible(False)
    

    legend = ax.legend(loc='center', bbox_to_anchor=(0.5, -0.25),
                      fontsize=FONTSIZE_TICKS, frameon=True, fancybox=True,
                      shadow=False, framealpha=0.9, ncol=3, labelspacing=0.25, columnspacing=0.25)
    legend.set_clip_on(True)


    # # Get all handles and labels
    # handles, labels = ax.get_legend_handles_labels()
    
    # # Split into two halves
    # mid_point = len(handles) // 2
    
    # # First legend (top half)
    # legend1 = ax.legend(handles[:mid_point], labels[:mid_point],
    #                    loc='upper left', bbox_to_anchor=(1.15, 1.0),
    #                    fontsize=FONTSIZE_TICKS, frameon=True, fancybox=True,
    #                    shadow=False, framealpha=0.9)
    # legend1.set_clip_on(True)
    
    # # Second legend (bottom half)
    # legend2 = ax.legend(handles[mid_point:], labels[mid_point:],
    #                    loc='upper left', bbox_to_anchor=(1.15, 0.4),
    #                    fontsize=FONTSIZE_TICKS, frameon=True, fancybox=True,
    #                    shadow=False, framealpha=0.9)
    # legend2.set_clip_on(True)
    
    # # Add first legend back (matplotlib removes it when creating the second)
    # ax.add_artist(legend1)

# %%

# ============================================
# CELL 2 — GENERATE & SAVE SEPARATE FIGURES
# ============================================

# ------- FIGURE 1: ENTITY INTERACTIONS -------
fig1 = plt.figure(figsize=(FIG_HEIGHT_IN, FIG_HEIGHT_IN+0.3))
# Position radar chart based on center and radius
left1 = RADAR_CENTER_X - RADAR_RADIUS
bottom1 = RADAR_CENTER_Y - RADAR_RADIUS
width1 = RADAR_RADIUS * 2
height1 = RADAR_RADIUS * 2
ax1 = fig1.add_axes([left1, bottom1, width1, height1], polar=True)

plot_radar(
    ax1,
    df_entity,
    entity_acronyms,
    title="Entity Interactions"
)

import os
save_dir = 'radar_final'
os.makedirs(save_dir, exist_ok=True)
savepath = os.path.join(save_dir, 'entity.png')

fig1.savefig(savepath, dpi=300, bbox_inches=None)
plt.close(fig1)
plt.show()


# %%
# ------- FIGURE 2: OTHER INTERACTIONS -------
# fig2 = plt.figure(figsize=(FIG_HEIGHT_IN, FIG_HEIGHT_IN+0.2))
# # Position radar chart based on center and radius
# left2 = RADAR_CENTER_X - RADAR_RADIUS
# bottom2 = RADAR_CENTER_Y - RADAR_RADIUS
# width2 = RADAR_RADIUS * 2
# height2 = RADAR_RADIUS * 2
# ax2 = fig2.add_axes([left2, bottom2, width2, height2], polar=True)

# plot_radar(
#     ax2,
#     df_other,
#     other_acronyms,
#     title="Other Interactions"
# )

# save_dir = 'radar_final'
# os.makedirs(save_dir, exist_ok=True)
# savepath = os.path.join(save_dir, 'fine.png')
# fig2.savefig(savepath, dpi=300, bbox_inches=None)
# plt.close(fig2)
# plt.show()

# %%