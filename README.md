# Knowledge-distilled compact models for edge-deployable bone scaffold design optimization

This repository contains the dual-teacher compact crystal graph network used for formation-energy, bulk-modulus, shear-modulus, elastic-tensor, and scaffold inverse-design experiments. The implementation combines fixed-topology crystal graphs, gated teacher fusion, a positive-definite elastic-tensor objective, Gibson–Ashby scaling, mixed INT8/FP16 training, sensitivity-based precision assignment, and an on-device CMA-ES search loop.

## Environment

Python 3.10, PyTorch 2.4.1, and CUDA 12.1 are the pinned training stack.

```bash
conda env create -f environment.yml
conda activate compact-scaffold
pip install -e '.[quality]'
```

The container uses the same PyTorch and CUDA pairing.

```bash
docker build -t compact-scaffold .
```

## Data

Verified canonical access points and licenses are listed in `dataset_links.txt`. The main tasks use MatBench v0.1 formation energy, bulk modulus, shear modulus, and band gap. MatBench Discovery v1 supplies the WBM training partition. Teacher targets use the 2024Q4 Materials Project and OQMD snapshots. ChEMBL and PubChem are consulted only by the post-prediction compatibility filter. No patient records or institutional clinical data are required.

Expected split sizes are 106,202/13,275/13,275 for formation energy, 8,432/1,054/1,054 for each modulus task, 85,442/10,680/10,680 for band gap, and 256,963/16,413/16,413 for WBM. Generate a SHA-256 manifest after retrieval and retain it beside the local data root.

## Training

The reported configuration uses 4 A100 80 GB GPUs, batch size 64 per process, AdamW at `3e-4`, weight decay `1e-5`, gradient clipping at `1.0`, and cosine annealing for 200 epochs. Epochs 1–40 are FP32 warmup, fake quantization begins at epoch 40, and the target INT8/FP16 map is reached by epoch 100. Seeds are 1, 2, 3, 4, and 5.

```bash
torchrun --standalone --nproc-per-node=4 -m compact_scaffold.commands.train --config settings/main.yaml
```

Teacher caching takes about 18 hours on one A100 80 GB. FP32 warmup takes about 27 hours on four A100 80 GB GPUs, joint distillation and quantization take about 96 hours, and Hessian-trace profiling takes about 3 hours on one A100. The complete run is approximately 144 wall-clock hours and 543 GPU-hours.

## Model and objectives

The student uses 128-dimensional atomic embeddings, six message-passing blocks, 256-dimensional dense paths, two directional attention heads, SiLU activation, sum pooling, and a two-layer readout. The training objective is

`L = L_KD + 0.25 (L_SPD + L_GA) + L_task`.

Distillation temperature follows a cosine trajectory from 4.0 to 1.0. The Gibson–Ashby exponents are 2.08 for HAP, 1.95 for TCP, and 1.82 for bioactive glass. The sensitivity scan assigns FP16 to attention and readout layers and fills a 30% FP16 parameter budget; remaining dense layers use symmetric per-channel INT8 fake quantization.

## Evaluation

The primary five-seed targets are formation-energy MAE `0.027 ± 0.0011 eV/atom`, bulk-modulus `R² 0.89 ± 0.004`, and band-gap MAE `0.24 ± 0.008 eV`. Deployment timing uses 100 warmup calls and 1,000 measured batch-one calls, reporting median and interquartile range. Reference latencies are `281 ± 11 ms` on Raspberry Pi 4, `138 ± 6 ms` on Jetson Nano, `26 ± 2 ms` on Jetson Orin Nano, `18 ± 3 ms` on Snapdragon 8 Gen 3-class Hexagon NPU, and `12 ± 1 ms` on Intel NUC i5.

The inverse-design search uses a population of 12, a 30-second device budget, 30–85% porosity, and three returned candidates. The reported Raspberry Pi 4 and Jetson Orin Nano end-to-end times are `27.4 ± 1.3 s` and `4.2 ± 0.3 s`.

## Verification

```bash
pytest -q
ruff check .
mypy --strict compact_scaffold
```

The test suite covers periodic graph construction, tensor shapes, elastic-tensor symmetry, physics losses, gradient-preserving fake quantization, CMA-ES convergence, and a two-update training integration path.
