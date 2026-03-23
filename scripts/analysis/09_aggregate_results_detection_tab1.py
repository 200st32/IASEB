# %%
import pandas as pd

# Show all rows (instead of truncating in the middle)
pd.set_option('display.max_rows', None)

# Show all columns (instead of truncating in the middle)
pd.set_option('display.max_columns', None)

# Show the full content of each column (no '...')
pd.set_option('display.max_colwidth', None)

# Expand the width of the display to fit more columns
pd.set_option('display.width', 1000) # or None for auto-detect

# %%
# Aggregate all the detection and results for IASEB 
import sys
sys.path.insert(0, '/home/aparcedo/IASEB')

# %%
import json
from collections import Counter
import os
from tqdm import tqdm 
from pathlib import Path 
from IASEB.datasets import DATASET_PATHS

CONTROL_PANEL = {
    "dalton_results_dir": "/home/aparcedo/IASEB/_archive/all_final_results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/_archive/all_final_results/final_aka_on_paper_alejandro/detection",
    "wen_results_dir": "/home/aparcedo/IASEB/data/results/stvg_output_bbox_wen/QwenVL_interpolated",
    "anirudh_results_dir": "/home/aparcedo/IASEB/_archive/all_final_results/STVG_results_anirudh"
}

# %%
# WE'RE GONNA USE THIS TO GET THE GT_TUBE
metadata_fp = '/home/aparcedo/IASEB/data/stvg_metadata_with_gt_tubes.json'

metadata = json.load(open(metadata_fp, 'r'))

counts = Counter([entry_md["caption"] for entry_md in metadata])
single_ocurrence_captions = {item: count for item, count in counts.items() 
                 if count ==1}

metadata_map = {
                entry_metadata["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', ''): entry_metadata 
                for entry_metadata in metadata 
                if entry_metadata["caption"] in single_ocurrence_captions}
# %%
# VIDSTG QUALITY FILTERING
# some samples are just subject instead of subject+attribute

stg_ff_fp = '/home/aparcedo/IASEB/vidstg_filtered_freeform_subset.json'
stg_r_fp = '/home/aparcedo/IASEB/vidstg_filtered_referral_subset.json'
vrd_ff_fp = '/home/aparcedo/IASEB/vidvrd_filtered_freeform_subset.json'
vrd_r_fp = '/home/aparcedo/IASEB/vidvrd_filtered_referral_subset.json'

stg_ff_data = json.load(open(stg_ff_fp, 'r'))
stg_r_data = json.load(open(stg_r_fp, 'r'))
vrd_ff_data = json.load(open(vrd_ff_fp, 'r'))
vrd_r_data = json.load(open(vrd_r_fp, 'r'))

stg_ff_data_map = set([entry["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '') for entry in stg_ff_data]) # 
stg_r_data_map = set([entry["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '') for entry in stg_r_data]) # 
vrd_ff_data_map = set([entry["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '') for entry in vrd_ff_data]) # 
vrd_r_data_map = set([entry["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '') for entry in vrd_r_data]) # 
# %%
# PART 1: DETECTION
# Detection datasets: HC-STVG1&2, VidVRD, VidSTG, MeViS (converted), and Ref-Youtube-VOS (converted)
# Detection models: CogVLM, Ferret, Shikra, Qwen-VL, LLaVA-G, InternVL2.5, Minigpt, Sphinx2
processed_records = []  

anirudh_filepaths = list(Path(CONTROL_PANEL["anirudh_results_dir"]).rglob('*.json'))
alejandro_filepaths = [Path(CONTROL_PANEL['alejandro_results_dir']) / f for f in os.listdir(CONTROL_PANEL['alejandro_results_dir']) if f.endswith('.json')]
dalton_filepaths = [Path(CONTROL_PANEL['dalton_results_dir']) / f for f in os.listdir(CONTROL_PANEL['dalton_results_dir']) if f.endswith('.json')]
wen_filepaths = [Path(CONTROL_PANEL['wen_results_dir']) / f for f in os.listdir(CONTROL_PANEL['wen_results_dir']) if f.endswith('.json')]

filepaths = anirudh_filepaths + alejandro_filepaths + dalton_filepaths + wen_filepaths
# %%
cnt = 0
for fp in tqdm(filepaths, desc="Processing all results files"):
    limit = None
    
    fp_str = str(fp)
    data = json.load(open(fp, 'r'))
        
    if 'anirudh' in fp_str:
        relative_parts = fp.relative_to(CONTROL_PANEL["anirudh_results_dir"]).parts
        dataset = relative_parts[0]
        task = relative_parts[1]
        model = fp.stem
        loop_data = data.get("results", [])
            
    elif 'alejandro' in fp_str:
        model = fp.stem.split("_")[1]
        dataset = fp.stem.split("_")[2]
        task = fp.stem.split("_")[3]
        loop_data = data.get("results", [])
        
    elif 'dalton' in fp_str:

        model = fp.stem.split("_")[0]
        dataset = fp.stem.split("_")[1]
        task = fp.stem.split("_")[2]
        loop_data = data
    elif 'wen' in fp_str:
        model = fp.stem.split("_")[0]
        dataset = fp.stem.split("_")[1]
        task = fp.stem.split("_")[2]
        loop_data = data.get("results", [])
        # import code; code.interact(local=locals())

    if dataset == 'ytvos': dataset = 'rvos'

    if dataset == 'vidvrd': limit = 888 # these are the maximum number of unique keys we have for these two 
    if dataset == 'vidstg': limit = 1144 # we'll use only these numbers for final results
        
    # --- 2. Process all samples for this file ---
    index = 0
    for sample_dict in loop_data:
        if limit and index >= limit:
            break
        
        caption_raw = None
        mvIoU = None
        predictions = None # interpolated for all except rvos
        gt_tube = None

        if 'model' in ['llava', 'gdino']:
            predictions = sample_dict.get("interpolated_boxes") # dalton
        elif model in ['sphinxv2', 'internvl2.5', 'minigptv2']:
            predictions = sample_dict.get("predicted_boxes") # anirudh
        elif model in ['shikra', 'ferret', 'cogvlm', 'qwenvl']:
            predictions = sample_dict.get("interpolated_predictions") # wen & alejandro
        
        if 'entry' in sample_dict and 'metrics' in sample_dict: # alejandro rvos mevis
            mvIoU = sample_dict["metrics"]["mv_iou"]
            sample_dict = sample_dict['entry']
        elif 'entry' in sample_dict and 'mvIoU' in sample_dict: # anirudh 
            mvIoU = sample_dict["mvIoU"]
            sample_dict = sample_dict["entry"]
        else:
            mvIoU = sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))

        caption_raw = sample_dict["caption"]
    
        if 'video_path' in sample_dict:
            if os.path.dirname(sample_dict["video_path"]): # vidvrd vidstg
                video_path = sample_dict["video_path"]
            else:
                video_path = os.path.join(DATASET_PATHS[dataset]['video'], sample_dict["video_path"])
        else:
            video_path = sample_dict["video_id"] # rvos and mevis are not actual video paths so we'll just save the directory id

        caption = caption_raw.strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        if caption in metadata_map:
            gt_tube = metadata_map[caption]['gt_tube']

        if dataset == 'vidstg' and task == 'freeform' and caption not in stg_ff_data_map: continue
        if dataset == 'vidstg' and task == 'referral' and caption not in stg_r_data_map: continue
        if dataset == 'vidvrd' and task == 'freeform' and caption not in vrd_ff_data_map: continue
        if dataset == 'vidvrd' and task == 'referral' and caption not in vrd_r_data_map: continue
                
        # --- 5. Build the record ---
        record = {
            "model": model,
            "dataset": dataset,
            "task": task,
            "caption": caption,
            "video_path": video_path,
            "mvIoU": mvIoU,
            "pred_tube": predictions,
            "gt_tube": gt_tube
        }
        processed_records.append(record)
        index += 1
# %%
import pandas as pd
df = pd.DataFrame(processed_records)
print(f'Length of data frame: {len(df)}')
df.head()
# %%
df.to_csv('alejandro_dalton_anirudh_wen_table1.csv', index=False)
# %%

# Task wise performance
# # Group by both 'model' AND 'task', then calculate the mean of 'mvIoU'
task_averages = df.groupby(by=['model'])['mvIoU'].mean()


print(task_averages)
# # %%

# # Combined performance (R&F) referral and freeform
# # Calculate the combined (overall) average mvIoU per model
# combined_avg = df.groupby('model')['mvIoU'].mean()

# %%
mean_scores_df = df[
    (df['dataset'] == 'vidstg') & (df['task'] == 'referral')
].groupby(by='model')['mvIoU'].mean().reset_index()

# This gives you a nice DataFrame:
#      model  mvIoU
# 0  Model_A   0.85
# 1  Model_B   0.72
# 2  Model_C   0.91
print(mean_scores_df)
# %%

# This groups by all three columns and counts the rows in each group
sample_counts = df.groupby(['model', 'dataset', 'task']).size()

print(sample_counts)
# %%
