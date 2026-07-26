# Fine-Tuning Improves Information Conveyance in Language Models

## Repository layout

```
src/
  generation_space/  Calculate per-prompt stats and pool across prompts
  metrics/           Tree based measure and semantic diversity
  stats/             Bootstrap and regression utilities
  visualization/     Plot helpers
scripts/
  generate_rollouts/ Sample rollouts
  process_rollouts/  Embed (and score) rollouts
  compute_gen_space/ Compute generation space metrics
  stats/             Bootstrap and regression fits
  plotting/          Plot figures
configs/             Hydra configs
data/                Prompt files (coding, completion, math, stories)
reproduce.sh         Reproduction entry point
```

## Setup

Install Python and R dependencies:

```bash
pip install -r requirements.txt
Rscript -e 'install.packages(c("glmmTMB", "DHARMa"))'
```

### `.env`

Copy `.env.example` to `.env` and fill in four absolute paths.

```
DATA_DIR=/abs/path/to/BranchingEntropy/data   # the prompt .jsonl files shipped in this repo
MODEL_DIR=/abs/path/to/models                 # directory of pre-downloaded models
OUTPUTS_DIR=/abs/path/to/scratch/outputs      # rollouts, embeddings, reward scores, per-prompt metrics
RESULTS_DIR=/abs/path/to/scratch/results      # final tables and figures
```

`OUTPUTS_DIR` could accumulate heavy data across the full sweep — point it at scratch storage, not your home directory.

### Model checkpoints

The pipeline loads weights from `${MODEL_DIR}/<name>/`. Before running, download each checkpoint into `MODEL_DIR` under exactly these directory names:

| Family       | Base                | Instruct                      |
| ------------ | ------------------- | ----------------------------- |
| Qwen2.5-7B   | `Qwen2.5-7B`        | `Qwen2.5-7B-Instruct`         |
| Qwen3-8B     | `Qwen3-8B-Base`     | `Qwen3-8B`                    |
| Llama-3.1-8B | `Llama-3.1-8B`      | `Llama-3.1-8B-Instruct`       |
| Gemma-3-12B  | `gemma-3-12b-pt`    | `gemma-3-12b-it`              |

Plus two auxiliary models used by the `process` stage:

- `modernbert-embed-large` — sentence embeddings for semantic diversity.
- (`Skywork-Reward-V2-Llama-3.1-8B` — reward-model scoring used for filtering.)
  
## Reproducing paper results

End-to-end:

```bash
./reproduce.sh
```

Run a single stage:

```bash
./reproduce.sh generate   # sample rollouts
./reproduce.sh process    # sentence-transformer embeddings
./reproduce.sh compute    # generation space metrics calculation
./reproduce.sh stats      # base-vs-instruct bootstrap + beta-interaction regression
./reproduce.sh plots      # paper figures
```

The default sweep is `all-families-matrix` (6 models × 4 datasets). To replicate a single family with smaller compute:

```bash
SWEEP=gemma-3-12b-matrix    ./reproduce.sh
SWEEP=llama-3.1-8b-matrix   ./reproduce.sh
SWEEP=qwen3-8b-matrix       ./reproduce.sh
```

## Developer commands

Fast smoke test:

```bash
python -m scripts.generate_rollouts.generate_rollouts +sweep=debug
```

Force a sweep to start from scratch instead of resuming:

```bash
python -m scripts.generate_rollouts.generate_rollouts +sweep=all-families-matrix resume=false force=true
```

Tail logs of a running sweep:

```bash
./scripts/tail-sweep.sh generate_rollouts all-families-matrix
./scripts/tail-sweep.sh embed_rollouts all-families-matrix
```
