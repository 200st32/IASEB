"""
Figure 3: VISTA Benchmark Statistics — generates 3 separate images.

Layout in the paper (LaTeX subfigure):
  Left column (48% col width):  (a) class_distribution.png — tall bar chart
  Right column (48% col width): (b) dataset_distribution.png — pie chart
                                (c) caption_size_distribution.png — word count histogram

Usage:
    python scripts/visualization/figure3_stats.py
    python scripts/visualization/figure3_stats.py --csv _archive/adaw_eval_results_coarse_only.csv
    python scripts/visualization/figure3_stats.py --outdir images/figures
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import colorsys
import argparse
import pandas as pd
import os

# =====================================================================
# SIZING — CVPR single column is ~3.4 inches
# Left subfigure = 48% of column width, right = 48%
# =====================================================================
COL_WIDTH_IN = 3.4
SUBFIG_WIDTH = COL_WIDTH_IN * 0.48  # ~1.63 inches

LABEL_FONTSIZE = 8
TICK_FONTSIZE = 7
SECTION_FONTSIZE = 7
BAR_LABEL_FONTSIZE = 5.5

# =====================================================================
# COLOR GENERATION (matches colors.py)
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

_l1_params = {'saturation': 0.70, 'lightness': 0.60}
L1_COLOR_MAP = {
    "spatial": colorsys.hls_to_rgb(30/360.0, _l1_params['lightness'], _l1_params['saturation']),
    "temporal": colorsys.hls_to_rgb(190/360.0, _l1_params['lightness'], _l1_params['saturation'])
}
L2_COLOR_MAP = generate_discrete_hsl_palette(
    ["human-human", "human-object", "human-animal", "object-animal",
     "object-object", "animal-animal", "self interaction", "no interaction"],
    0.0, saturation=0.55, lightness=0.70
)
L3_COLOR_MAP = generate_discrete_hsl_palette(
    ["Affective", "Antagonistic", "Body Motion", "Communicative", "Competitive",
     "Cooperative", "Relational Movement", "Observation", "Passive", "Physical",
     "Provisioning", "Proximity", "Social", "Supportive"],
    40.0, saturation=0.40, lightness=0.80
)
FALLBACK_COLOR = (0.5, 0.5, 0.5)

DATASET_COLORS = {
    'HC-STVG-1': '#66c2a5',
    'HC-STVG-2': '#fc8d62',
    'VidSTG': '#8da0cb',
    'VidVRD': '#e78ac3',
    'MeViS': '#a6d854',
    'R-YT-VOS': '#ffd92f',
}

# =====================================================================
# (a) HORIZONTAL BAR CHART — class distribution (percent per section)
# =====================================================================

def generate_bar_chart(outdir):
    data_s_t = {"Spatial": 2845, "Temporal": 4571}
    map_s_t = {"Spatial": "spatial", "Temporal": "temporal"}

    data_entity = {
        "Human-Human": 2167, "Human-Object": 1541, "No Interaction": 1701,
        "Human-Animal": 395, "Animal-Animal": 810, "Animal-Object": 365,
        "Object-Object": 277, "Human-Self": 160,
    }
    map_entity = {
        "Human-Human": "human-human", "Human-Object": "human-object",
        "No Interaction": "no interaction", "Human-Animal": "human-animal",
        "Animal-Animal": "animal-animal", "Animal-Object": "object-animal",
        "Object-Object": "object-object", "Human-Self": "self interaction",
    }

    data_interaction = {
        "Rel. Movement": 3738, "Observation": 2040, "Physical": 1902,
        "Body Motion": 1879, "Proximity": 1158, "Communicative": 487,
        "Passive": 347, "Supportive": 256, "Affective": 245,
        "Antagonistic": 236, "Provisioning": 160, "Social": 140,
        "Cooperative": 123, "Competitive": 28,
    }
    map_interaction = {
        "Rel. Movement": "Relational Movement", "Observation": "Observation",
        "Physical": "Physical", "Body Motion": "Body Motion",
        "Proximity": "Proximity", "Communicative": "Communicative",
        "Passive": "Passive", "Supportive": "Supportive",
        "Affective": "Affective", "Antagonistic": "Antagonistic",
        "Provisioning": "Provisioning", "Social": "Social",
        "Cooperative": "Cooperative", "Competitive": "Competitive",
    }

    total_s_t = sum(data_s_t.values())
    total_entity = sum(data_entity.values())
    total_interaction = sum(data_interaction.values())

    sorted_s_t = sorted(data_s_t.items(), key=lambda x: x[1], reverse=True)
    sorted_entity = sorted(data_entity.items(), key=lambda x: x[1], reverse=True)
    sorted_interaction = sorted(data_interaction.items(), key=lambda x: x[1], reverse=True)

    all_labels, all_pcts, all_colors = [], [], []

    for label, count in sorted_s_t:
        all_labels.append(label)
        all_pcts.append(count / total_s_t * 100)
        all_colors.append(L1_COLOR_MAP.get(map_s_t[label], FALLBACK_COLOR))
    for label, count in sorted_entity:
        all_labels.append(label)
        all_pcts.append(count / total_entity * 100)
        all_colors.append(L2_COLOR_MAP.get(map_entity[label], FALLBACK_COLOR))
    for label, count in sorted_interaction:
        all_labels.append(label)
        all_pcts.append(count / total_interaction * 100)
        all_colors.append(L3_COLOR_MAP.get(map_interaction[label], FALLBACK_COLOR))

    all_labels.reverse()
    all_pcts.reverse()
    all_colors.reverse()

    # Tall figure for left column
    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH * 1.8, SUBFIG_WIDTH * 3.5))

    indices = np.arange(len(all_labels))
    ax.barh(indices, all_pcts, color=all_colors, edgecolor='none', height=0.7)

    for i, pct in enumerate(all_pcts):
        ax.text(pct + 0.8, i, f"{pct:.0f}%", ha='left', va='center', fontsize=BAR_LABEL_FONTSIZE)

    len_int = len(sorted_interaction)
    len_ent = len(sorted_entity)
    ax.axhline(len_int - 0.5, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(len_int + len_ent - 0.5, color='black', linestyle='--', linewidth=0.8)

    # Section labels positioned inside the plot area
    ax.text(0.97, 0.96, "spatio-temporal",
            ha='right', va='top', fontsize=SECTION_FONTSIZE, weight='bold',
            transform=ax.transAxes)
    ax.text(0.97, 0.73, "entity",
            ha='right', va='center', fontsize=SECTION_FONTSIZE, weight='bold',
            transform=ax.transAxes)
    ax.text(0.97, 0.28, "interaction type",
            ha='right', va='center', fontsize=SECTION_FONTSIZE, weight='bold',
            transform=ax.transAxes)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('percentage (%)', fontsize=LABEL_FONTSIZE)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.5, len(all_labels) - 0.5)
    ax.set_yticks(indices)
    ax.set_yticklabels([l.lower() for l in all_labels], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    path = os.path.join(outdir, 'class_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# =====================================================================
# (b) PIE CHART — dataset distribution (legend, no inline labels)
# =====================================================================

def generate_pie_chart(outdir):
    labels = ['HC-STVG-1', 'HC-STVG-2', 'VidVRD', 'VidSTG', 'MeViS', 'R-YT-VOS']
    sizes = [2300, 4000, 1776, 2288, 616, 834]
    colors = [DATASET_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH * 1.6, SUBFIG_WIDTH * 1.6))

    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
    )

    total = sum(sizes)
    legend_labels = [f"{l}  ({s/total*100:.1f}%)" for l, s in zip(labels, sizes)]
    ax.legend(
        wedges, legend_labels,
        loc='center left', bbox_to_anchor=(1.0, 0.5),
        fontsize=TICK_FONTSIZE, frameon=False, labelspacing=0.6,
        handlelength=1.0, handleheight=1.0
    )
    ax.set_aspect('equal')

    path = os.path.join(outdir, 'dataset_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# =====================================================================
# (c) CAPTION WORD COUNT HISTOGRAM
# =====================================================================

def generate_word_count_histogram(outdir, csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, usecols=['caption', 'dataset', 'model'])
        unique = df.drop_duplicates(subset=['caption', 'dataset'])
        word_counts = unique['caption'].str.split().str.len().values
    else:
        # Fallback: load pre-extracted word counts
        import json
        wc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'caption_word_counts.json')
        if os.path.exists(wc_path):
            word_counts = np.array(json.load(open(wc_path)))
        else:
            np.random.seed(42)
            word_counts = np.random.normal(11.9, 6.4, 7233).clip(1, 51).astype(int)
            print("  Warning: no data source found, using approximate distribution")

    fig, ax = plt.subplots(figsize=(SUBFIG_WIDTH * 1.6, SUBFIG_WIDTH * 1.2))

    ax.hist(word_counts, bins=range(1, 55), color='#8da2cc', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('caption word count', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('count', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 52)

    path = os.path.join(outdir, 'caption_size_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 3 images (3 separate files)")
    parser.add_argument('--csv', type=str, default='_archive/adaw_eval_results_coarse_only.csv',
                        help='Path to coarse CSV for word count histogram')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for images')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("Generating Figure 3 images...")
    generate_bar_chart(args.outdir)
    generate_pie_chart(args.outdir)
    generate_word_count_histogram(args.outdir, csv_path=args.csv)
    print("Done!")
