# CLAUDE.md

## Project Overview

IASEB (Interaction-Aware Spatio-temporal Entity Benchmark) evaluates vision-language models on spatio-temporal video grounding (STVG). Models localize objects in video frames given natural language descriptions across 6 datasets, multiple models, and 2 task types (referral/freeform).

## Running Evaluation

```bash
python -m IASEB.run_eval \
    --dataset <dataset> \
    --model <model> \
    --task_type <referral|freeform> \
    --config config.yaml \
    --output_path results/output.json \
    --checkpoint_path results/output.jsonl \
    --device cuda \
    --entry_index <start> \
    --max_iters <count>
```

- Datasets: `hcstvg1`, `hcstvg2`, `vidstg`, `vidvrd`, `mevis`, `rvos`
- Models: `cogvlm`, `shikra`, `ferret`, `qwen3vl`, `mimovl`
- MeViS and RVOS are **freeform-only** — no referral captions exist
- `--checkpoint_path` enables resume on restart (reads .jsonl, skips processed indices)
- `--config` points to `config.yaml` with cluster-specific dataset paths

## Model Paths

Set via environment variables or defaults to HuggingFace IDs:
```bash
IASEB_QWEN3VL_PATH=/home/aparcedo/Qwen3-VL-8B-Instruct   # or Qwen/Qwen3-VL-8B-Instruct
IASEB_MIMOVL_PATH=/home/aparcedo/MiMo-VL-7B-RL            # or XiaomiMiMo/MiMo-VL-7B-RL
IASEB_SHIKRA_PATH=...
IASEB_FERRET_PATH=...
```

Shikra and Ferret require their repos cloned (`shikra/`, `ml_ferret/`). Imports are lazy — only loaded when that model is selected.

## SLURM Array Job Pattern

All experiments use the same pattern. SLURM scripts are in `slurm/`:
```bash
sbatch slurm/mimovl_hcstvg1_a100.slurm   # freeform
sbatch slurm/mimovl_referral_hcstvg1.slurm # referral
```

Array jobs split datasets: `TOTAL_SAMPLES / NUM_JOBS = CHUNK_SIZE`, each array gets `--entry_index` and `--max_iters`. Always include `--checkpoint_path` for resume.

**GPU selection**: Use `-C gmem80` (A100 80GB) not `gmem48` (A6000). MiMo-VL is ~3.7x faster on A100 (~18.6s/sample vs ~67.5s on A6000). Qwen3-VL is ~2x faster.

## Cluster Commands

```bash
sbatch slurm/<job>.slurm        # Submit
squeue --me                      # Check status
scancel <jobid>                  # Cancel
```

## Data Pipeline

### Table 1 (main results table)

**Source of truth**: `_archive/adaw_eval_results_coarse_only.csv` (101,093 rows, 9 models)

Column definitions:
- **R, F, R&F**: mean mvIoU by task type (referral, freeform, combined)
- **S**: `Spatial (Static)` category only
- **T**: `Temporal (State)` + `Spatio-Temporal` combined (everything non-Spatial)
- **Entity columns** (AA, AO, HA, HH, HO, HS, NI, OO): from `entity_level0_cls`

The coarse CSV has 3 ST categories: `Spatial (Static)`, `Temporal (State)`, `Spatio-Temporal`. The paper collapses the latter two into T.

### Aggregation pipeline

1. Raw collaborator JSONs in `_archive/all_final_results/` (Anirudh, Alejandro, Dalton, Wen)
2. `scripts/analysis/09_aggregate_results_detection_tab1.py` → base CSV
3. `scripts/analysis/10_categorize_tab1.py` → coarse categorized CSV
4. New model results added via custom scripts → `results/combined_results_with_qwen3vl.csv`

### Quality filtering

VidSTG and VidVRD require filtering to a quality-controlled subset:
- Filter files: `vidstg_filtered_freeform_subset.json`, `vidstg_filtered_referral_subset.json`, `vidvrd_filtered_freeform_subset.json`, `vidvrd_filtered_referral_subset.json`
- VidVRD: 888 valid IDs
- VidSTG: 879 unique `(video_path, target_id, caption)` tuples

## Results Naming Convention

```
results/{model}_{dataset}_{jobid}_{arrayid}.json          # freeform
results/{model}_referral_{dataset}_{jobid}_{arrayid}.json  # referral
results/{model}_{dataset}_{jobid}_{arrayid}.jsonl          # checkpoint (incomplete)
```

`.json` = final output (written when array completes). `.jsonl` = incremental checkpoint.

## Key File Locations

| Purpose | Path |
|---------|------|
| Core package | `IASEB/` (models.py, datasets.py, run_eval.py, utils.py) |
| Dataset config | `config.yaml` (cluster-specific, not tracked) |
| Config template | `config.example.yaml` |
| SLURM scripts | `slurm/` |
| Table 1 source CSV | `_archive/adaw_eval_results_coarse_only.csv` |
| Base 9-model CSV | `_archive/adaw_eval_results_raw_updated.csv` |
| Combined 10-model CSV | `results/combined_results_with_qwen3vl.csv` |
| Collaborator raw results | `_archive/all_final_results/` |
| Quality filter files | `vid*_filtered_*.json` (project root) |
| Analysis scripts | `scripts/analysis/` |
| Taxonomy definitions | `scripts/constants.py` |

## Known Table 1 Errors (fix for camera-ready)

Copy-paste errors found in the submitted paper:
- **Shikra T**: paper says 23.4, correct is **32.4** (Ferret's value was pasted)
- **Ferret S**: paper says 29.9, correct is **20.9** (Shikra's value was pasted)
- **LLaVA T**: paper says 28.1, correct is **31.9** (S value was duplicated into T)
- **MiniGPT-v2 S**: paper says 44.3, correct is **43.1** (T value was duplicated)

## Coordinate Conventions

- Dataset boxes: `[x, y, w, h]` (HC-STVG, VidSTG, VidVRD) or `[xmin, ymin, xmax, ymax]`
- Internal: `[xmin, ymin, xmax, ymax]`
- Model output: 0-1000 normalized space (all models convert to this)
- MiMo-VL: outputs PIXEL coordinates, converted to 0-1000 in wrapper

## Git

- `release` branch: clean open-source code + experiment SLURM scripts
- Push: `git push origin release`
- Old branches (alejandro, master, etc.) are stale

## Workflow

- Always use `--checkpoint_path` for long runs
- Pipe long commands to log files: `python script.py > /tmp/output.log 2>&1 &`
- Check GPU type before submitting: `gmem80` >> `gmem48` for inference speed
