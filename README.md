# IASEB: Interaction-Aware Spatio-temporal Entity Benchmark

A comprehensive evaluation framework for benchmarking vision-language models on **spatio-temporal video grounding (STVG)** tasks. IASEB evaluates models' ability to localize objects in video frames given natural language descriptions, with fine-grained analysis across interaction types and entity categories.

## Key Features

- **6 Video Grounding Datasets**: HC-STVG v1, HC-STVG v2, VidSTG, VidVRD, MeViS, Refer-YouTube-VOS
- **4 Model Wrappers**: CogVLM, Shikra, Ferret, Qwen3-VL (extensible to new models)
- **2 Task Types**: Referral (explicit object names) and Freeform (descriptive phrases)
- **Interaction-Aware Taxonomy**: Entity type hierarchy (Human-Human, Human-Object, Animal-Animal, etc.) and spatiotemporal relationship hierarchy (Spatial/Static, Temporal/Dynamic)
- **Statistical Analysis Scripts**: Bootstrap confidence intervals, pairwise Wilcoxon tests, Mann-Whitney U tests
- **Visualization Tools**: Radar charts, sunburst diagrams, performance bar charts

## Installation

```bash
git clone https://github.com/200st32/IASEB.git
cd IASEB
pip install -e .
```

### Optional Dependencies

```bash
# For statistical analysis scripts
pip install -e ".[analysis]"

# For GPT-based taxonomy classification
pip install -e ".[classification]"

# For Qwen3-VL model
pip install -e ".[qwen]"

# Everything
pip install -e ".[all]"
```

## Dataset Setup

IASEB supports 6 video grounding datasets. You must download them separately:

| Dataset | Source |
|---------|--------|
| HC-STVG v1 | [HC-STVG](https://github.com/tzhhhh123/HC-STVG) |
| HC-STVG v2 | [HC-STVG](https://github.com/tzhhhh123/HC-STVG) |
| VidSTG | [VidSTG](https://github.com/Guaranteer/VidSTG-Dataset) |
| VidVRD | [VidOR/VidVRD](https://xdshang.github.io/docs/vidor-dataset/getting-started.html) |
| MeViS | [MeViS](https://github.com/henghuiding/MeViS) |
| Refer-YouTube-VOS | [RVOS](https://youtube-vos.org/dataset/rvos/) |

After downloading, create a config file:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your local dataset paths
```

## Model Setup

### CogVLM
Downloaded automatically from HuggingFace (`zai-org/cogvlm-grounding-generalist-hf`).

### Qwen3-VL
```bash
pip install qwen-vl-utils
# Model downloaded from HuggingFace, or set IASEB_QWEN3VL_PATH to local path
```

### Shikra
Requires cloning the [Shikra repository](https://github.com/shikras/shikra) and setting:
```bash
export IASEB_SHIKRA_PATH=/path/to/shikra-7b
```

### Ferret
Requires cloning [ml-ferret](https://github.com/apple/ml-ferret) and setting:
```bash
export IASEB_FERRET_PATH=/path/to/ferret-7b-v1-3
```

## Usage

### Running Evaluation

```bash
python -m IASEB.run_eval \
    --dataset hcstvg1 \
    --model qwen3vl \
    --task_type freeform \
    --config config.yaml \
    --output_path results/qwen3vl_hcstvg1.json \
    --device cuda
```

**Arguments:**
- `--dataset`: `hcstvg1`, `hcstvg2`, `vidstg`, `vidvrd`, `mevis`, `rvos`
- `--model`: `cogvlm`, `shikra`, `ferret`, `qwen3vl`
- `--task_type`: `referral` or `freeform`
- `--config`: Path to dataset config YAML file
- `--frame_step`: Frame sampling interval (default: 5)
- `--checkpoint_path`: Path for incremental checkpointing (.jsonl), enables resume on restart

### SLURM Cluster

Example SLURM scripts are provided in `slurm/`:
- `example_single_gpu.slurm` — Single model/dataset evaluation
- `example_array_job.slurm` — Array job splitting a dataset across GPUs

### Adding a New Model

1. Create a new class in `IASEB/models.py` with a `run_inference(image, question)` method
2. The method must return `(text_output, boxes, query, response)` where `boxes` is a list of tensors with `[x1, y1, x2, y2]` coordinates in 0-1000 normalized scale
3. Add the model to the selection logic in `IASEB/run_eval.py`

## Project Structure

```
IASEB/
  IASEB/                    # Core evaluation package
    run_eval.py             # Main evaluation entry point
    models.py               # Model wrappers
    datasets.py             # Dataset loaders
    utils.py                # IoU computation, box rescaling
  scripts/
    analysis/               # Statistical analysis
    classification/         # GPT-based taxonomy annotation
    data_processing/        # Data pipeline utilities
    visualization/          # Figure generation (radar, sunburst, etc.)
    constants.py            # Taxonomy hierarchy definitions
  slurm/                    # Example SLURM job scripts
  config.example.yaml       # Dataset path template
```

## Citation

If you use IASEB in your research, please cite:

```bibtex
@article{parcedo2025iaseb,
  title={IASEB: Interaction-Aware Spatio-temporal Entity Benchmark for Video Grounding},
  author={Parcedo, Alejandro and others},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
