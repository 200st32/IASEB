# %%
import pandas as pd
import json
import matplotlib.pyplot as plt
from constants import ST_HIERARCHY, ENTITY_HIERARCHY

# --- 1. LOAD DATA BASED ON MODE ---
# (Unchanged, correctly sets up coarse_categories_map and coarse_color_map_for_plot)
CONTROL_PANEL = {
    # --- File Paths ---
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_st_class_v1.json",
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_v1.json",
}

results_data = "/home/aparcedo/IASEB/interaction_analysis/alejandro_dalton_anirudh_wen_table1.csv"

# %%
# --- 1. LOAD AND VALIDATE CLASSIFICATION DATA ---

st_hierarchy = ST_HIERARCHY
entity_hierarchy = ENTITY_HIERARCHY

def is_valid_path(hierarchy, path_str):
    """
    Checks if a category path string (e.g., "1.2.1") is valid
    by traversing the given hierarchy.
    """
    if not path_str:
        return False
        
    # Handle paths that might have notes, e.g., "1.2.1 Note..."
    path_str_clean = path_str.split(" ")[0] 
    
    try:
        current_level_dict = hierarchy
        levels = path_str_clean.split('.')
        for level_key_str in levels:
            key = int(level_key_str)
            node = current_level_dict[key]
            current_level_dict = node['children']
        return True
    except (KeyError, ValueError, TypeError, AttributeError):
        return False

# Load ST Classification Map (Raw)
st_classification_data = json.load(open(CONTROL_PANEL["st_data_path"], 'r'))

# need a dict of caption : category
raw_st_cls_data_map = {}
for entry in st_classification_data:
    caption = entry["caption"]
    key = caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
    raw_st_cls_data_map[key] = entry.get("category", entry.get("st_class_raw"))
    
# Load Entity Classification Map (Raw)
entity_classification_data = json.load(open(CONTROL_PANEL["entity_data_path"], 'r'))
raw_entity_cls_data_map = {}
for entry in entity_classification_data:
    caption = entry["caption"]
    key = caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
    raw_entity_cls_data_map[key] = entry.get("category", entry.get("entity_class_raw"))

print(f"Original ST classifications: {len(raw_st_cls_data_map)}")
print(f"Original Entity classifications: {len(raw_entity_cls_data_map)}")

# --- Filter ST Map for valid paths ---
st_cls_data_map = {}
invalid_st_count = 0
for caption, category_str in raw_st_cls_data_map.items():
    if is_valid_path(st_hierarchy, category_str):
        st_cls_data_map[caption] = category_str
    else:
        print(f"Invalid ST Path: {category_str} for caption: {caption}")
        invalid_st_count += 1

# --- Filter Entity Map for valid paths ---
entity_cls_data_map = {}
invalid_entity_count = 0
for caption, category_str in raw_entity_cls_data_map.items():
    if is_valid_path(entity_hierarchy, category_str):
        entity_cls_data_map[caption] = category_str
    else:
        # print(f"Invalid Entity Path: {category_str} for caption: {caption}")
        invalid_entity_count += 1

print("--- VALIDATION COMPLETE ---")
print(f"Invalid ST classifications removed: {invalid_st_count}")
print(f"Invalid Entity classifications removed: {invalid_entity_count}")
print(f"Clean ST classifications remaining: {len(st_cls_data_map)}")
print(f"Clean Entity classifications remaining: {len(entity_cls_data_map)}")
print("---------------------------")

# %%

df = pd.read_csv(results_data)
df['st_fine_category'] = df['caption'].map(st_cls_data_map)
df['entity_fine_category'] = df['caption'].map(entity_cls_data_map)

# --- 4. Define helper functions for hierarchy parsing ---
# This is much cleaner than putting the logic in the main script.

def get_st_levels(st_category_str):
    """
    Parses a ST category string and returns a dict of new levels,
    including both the 'short_name' (for labels) and the
    'key' path (for sorting).
    """
    levels = {}
    if not isinstance(st_category_str, str): 
        return levels
    
    st_levels_list = st_category_str.split(" ")[0].split('.')
    try:
        current_level_dict = ST_HIERARCHY
        current_path_parts = [] # To build the key path string
        
        for level_idx, category_num_str in enumerate(st_levels_list):
            key = int(category_num_str)
            node = current_level_dict[key]
            
            # Build the key path (e.g., "1", then "1.2", then "1.2.1")
            current_path_parts.append(category_num_str)
            current_key_str = ".".join(current_path_parts)
            
            # Store the short name (label)
            levels[f'st_level{level_idx}_cls'] = node['short_name']
            
            # Store the key path (for sorting)
            levels[f'st_level{level_idx}_key'] = current_key_str
            
            current_level_dict = node['children']
    except (KeyError, ValueError, TypeError):
        # Handle cases where the path might be bad
        pass 
    return levels

def get_entity_levels(entity_category_str):
    """
    Parses an Entity category string and returns a dict of new levels,
    including both the 'short_name' (for labels) and the
    'key' path (for sorting).
    """
    levels = {}
    if not isinstance(entity_category_str, str):
        return levels
        
    entity_levels_list = entity_category_str.split(" ")[0].split('.')
    try:
        current_level_dict = ENTITY_HIERARCHY
        current_path_parts = [] # To build the key path string
        
        for level_idx, category_num_str in enumerate(entity_levels_list):
            key = int(category_num_str)
            node = current_level_dict[key]
            
            # Build the key path (e.g., "1", then "1.1")
            current_path_parts.append(category_num_str)
            current_key_str = ".".join(current_path_parts)

            # Store the short name (label)
            levels[f'entity_level{level_idx}_cls'] = node['short_name']
            
            # Store the key path (for sorting)
            levels[f'entity_level{level_idx}_key'] = current_key_str
            
            current_level_dict = node['children']
    except (KeyError, ValueError, TypeError):
        pass
    return levels
# %%
# --- 5. Apply the functions and join the results ---

# .apply(get_st_levels) creates a Series of dictionaries
# .apply(pd.Series) expands that Series of dicts into a new DataFrame
st_level_data = df['st_fine_category'].apply(get_st_levels).apply(pd.Series)
entity_level_data = df['entity_fine_category'].apply(get_entity_levels).apply(pd.Series)

# --- 6. Concatenate the new columns back to the original DataFrame ---
# axis=1 joins horizontally (by column)
df = pd.concat([df, st_level_data, entity_level_data], axis=1)

# (Optional) Replicate your 'continue' logic by dropping rows that had no match
# df = df.dropna(subset=['st_fine_category', 'entity_fine_category'])

print("DataFrame processing complete.")
print(f"Total rows: {len(df)}")
df.head()
df.to_csv('alejandro_dalton_anirudh_wen_table1_categorized.csv', index=False)
# %%
import pandas as pd
df = pd.read_csv("/home/aparcedo/IASEB/interaction_analysis/alejandro_dalton_anirudh_wen_results_levels1_2_categorization.csv")

df = df[df['task'] != 'referral']

# --- Move all 'temporal' categories to 'spatio-temporal' ---
l1_categories_to_move = ['Sequential', 'Composite', 'Durational', 'Actor State', 'Object State']
df.loc[df['st_level1_cls'].isin(l1_categories_to_move), 'st_level0_cls'] = 'Spatio-Temporal'

df['st_level0_cls'] = df['st_level0_cls'].replace('Spatio-Temporal', 'Temporal')


# %%
df.drop(['pred_tube', 'gt_tube', 'st_fine_category', 'entity_fine_category', 'st_level1_cls', 'st_level1_key', 'st_level2_cls', 'st_level2_key', 'entity_level1_cls', 'entity_level1_key', 'st_level0_key', 'entity_level0_key'], axis=1, inplace=True)
# %%
df.to_csv("/home/aparcedo/IASEB/interaction_analysis/alejandro_dalton_anirudh_wen_results_levels1_2_categorization_clean.csv", index=False)

# %%
# %%
# %%
import json
import pandas as pd

fp = "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_uniq_captions_v17.json"
taxonomy_v2_data = json.load(open(fp, 'r'))

category_map = {entry["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '') : entry["category"] for entry in taxonomy_v2_data}

results_fp = "/home/aparcedo/IASEB/interaction_analysis/alejandro_dalton_anirudh_wen_results_levels1_2_categorization_clean.csv"

def get_categories(caption):
    caption = caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
    # Use .get(key, None) to return None if the key is not found
    # This prevents the script from crashing with a KeyError
    return category_map.get(caption, None)

df = pd.read_csv(results_fp)

# 1. Apply the safe function. This Series will have lists and None values.
new_categories_list = df['caption'].apply(get_categories)

# 2. Count the failures (None values)
# .isna().sum() counts all the None/NaN values
fail_count = new_categories_list.isna().sum()
success_count = new_categories_list.notna().sum()

print(f"--- Processing Report ---")
print(f"Total rows: {len(df)}")
print(f"Successfully matched: {success_count}")
print(f"Failed to find match: {fail_count} (will be set to NaN)")
print("-------------------------")

# 3. Now, apply pd.Series. This will not crash.
# Rows that were None will become rows of NaN
category_df = new_categories_list.apply(pd.Series)

df = pd.concat([df, category_df], axis=1)

# print(df.head())
df.to_csv("table1.csv", index=False)
# %%
# All columns to keep as identifiers
df = pd.read_csv('table1_level1and2.csv')
# (Assuming the category columns 'st_level0_cls', etc., are still in your df)
id_columns = [
    'model', 
    'dataset', 
    'task', 
    'caption', 
    'video_path', 
    'mvIoU',
    'pred_tube',
    'gt_tube'
]

# List of all columns that contain category labels
category_columns = [
    'st_level0_cls', 
    'entity_level0_cls', 
    '0', 
    '1', 
    '2', 
    '3'
]

# Filter out any id_columns that might not exist in the DataFrame
# (e.g., if pred_tube/gt_tube were excluded)
id_columns = [col for col in id_columns if col in df.columns]

# Melt the DataFrame to stack all categories into a single column
df_melted = df.melt(
    id_vars=id_columns, 
    value_vars=category_columns, 
    value_name='category'
)

# Drop rows where the category was NaN
df_cleaned = df_melted.dropna(subset=['category'])

# Group by model, dataset, AND the new category column
df_agg = df_cleaned.groupby(['model', 'category'])['mvIoU'].agg(['mean'])

# Unstack the 'category' level to pivot it into columns
df_final_result = df_agg.unstack(level='category')

# Display the final result
print(df_final_result)
# %%



# %%
cols_to_average = [('mean', 'Spatial'), ('mean', 'Movement')]
df_final_result[('mean', 'Relational Movement')] = df_final_result[[('mean', 'Spatial'), ('mean', 'Movement')]].mean(axis=1)
df_final_result = df_final_result.drop(columns=cols_to_average)
# %%
df_final_result = df_final_result.rename(columns={"Spatial (Static)": "Spatial"})
# %%
df = df_final_result.applymap(lambda x: round(x * 100, 2))
# %%
df.to_csv('table1_finalized.csv')

# %%
