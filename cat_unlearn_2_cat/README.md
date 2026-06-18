# Memory masking vs overwriting in procedural categorization

Code, data, analysis outputs, and manuscript files for the paper:

**Memory masking vs overwriting in procedural categorization**

## Directory Structure

- **experiment/**
  Experiment runtime code.
  - `run_exp.py` - pygame experiment runtime used to generate subject CSVs
  - `util_func_stimcat.py` - stimulus generation and grating utilities

- **analysis/**
  Analysis scripts, DBM fitting code, figure scripts, and Gadi submission files.
  - `fit_dbm.py` - fits DBMs locally or as one Gadi array chunk
  - `fit_dbm_gadi_array.pbs` - Gadi wrapper for `fit_dbm.py`
  - `merge_dbm_fit.py` - merges DBM fit chunks
  - `fit_dbm_recovery.py` - empirical DBM recovery locally or as one Gadi chunk
  - `fit_dbm_recovery_gadi_array.pbs` - Gadi wrapper for recovery
  - `merge_dbm_recovery.py` - merges empirical recovery chunks
  - `make_figure_accuracy.py` - accuracy figure
  - `make_figure_dbm_heatmap.py` - DBM transition heatmaps
  - `make_figure_dbm_recovery.py` - empirical recovery heatmaps
  - `make_figure_bayes_post.py` - Bayesian posterior comparison figure
  - `dbm_models.py` - decision-bound model likelihoods and simulation functions
  - `trial_data.py` - trial-level data loading and cleaning
  - `dbm_results.py` - best-fit DBM loading, classification, and exclusions

- **data/**
  Trial-level data files, one CSV per subject (for example `sub_1_data.csv`).

- **dbm_fits/**
  Saved DBM fitting results written by the analysis script.
  - `dbm_results.csv` - fits per subject x block x model

- **figures/**
  Analysis outputs currently present in this subtree.
  - `subjects_accuracy_include_all_models_learning_nonguessers.png`
  - `best_model_class_heatmap.png`
  - `bayesian_comparison.png`

- **write/**
  Compiled manuscript output currently included here.
  - `main.pdf`

## Setup

For software requirements, see the pinned `requirements.txt` or read below.

Analysis code uses:

- `numpy==1.26.4`
- `pandas==2.2.2`
- `matplotlib==3.8.4`
- `scipy==1.13.1`
- `seaborn==0.13.2`
- `pingouin==0.5.4`
- `Pillow==10.3.0`
- `pygame==2.5.2`

Example setup from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

Run analysis scripts from the `analysis/` directory so relative
paths resolve correctly.

### Main analysis workflow

Figure scripts can be run directly:

```bash
cd analysis
python make_figure_accuracy.py
python make_figure_dbm_heatmap.py
python make_figure_bayes_post.py
```

### Model Recovery On NCI Gadi

Run the recovery workflow from `analysis/` after `../dbm_fits/dbm_results.csv`
exists. The empirical DBM fits are now deterministic when fit with the same
base seed, and recovery uses the original trial order to define blocks. DBM
fits are run for blocks `0`, `2`, and `6`.

The checked-in Gadi scripts are configured for a cautious first real run:

- `4` array tasks total, matching PBS array range `0-3`
- `1` recovery repetition per group
- all empirical subject/block groups are processed by default

This keeps the cluster setup simple while expanding beyond the tiny pilot cap.
Add `MAX_GROUPS=<n>` at submission time only if you want to reintroduce a hard
limit for a smaller test.

Before submitting to Gadi, replace `<PROJECT_CODE>` in both PBS scripts with
your actual NCI project code.

1. Generate or refresh empirical DBM fits locally:

```bash
cd analysis
python fit_dbm.py --seed 462 --out-path ../dbm_fits/dbm_results.csv
```

2. Submit the recovery array job on Gadi:

```bash
cd analysis
qsub -v BLOCK=6,N_REPS=1,SEED=462 fit_dbm_recovery_gadi_array.pbs
```

3. Wait for all `4` array tasks to finish successfully, then merge locally or
interactively instead of submitting another PBS job:

```bash
cd analysis
python3 merge_dbm_recovery.py --block 6 --num-chunks 4
```

Outputs land in:

- `../dbm_fits/recovery_chunks/` - one CSV per chunk
- `../dbm_fits/fit_dbm_recovery_block_<BLOCK>_results.csv`
- `../dbm_fits/fit_dbm_recovery_block_<BLOCK>_family_counts.csv`
- `../dbm_fits/fit_dbm_recovery_block_<BLOCK>_family_props.csv`
- `../dbm_fits/fit_dbm_recovery_block_<BLOCK>_model_counts.csv`
- `../dbm_fits/fit_dbm_recovery_block_<BLOCK>_model_props.csv`

The local merge script fails if any chunk is missing or if chunk filenames do
not match the expected `4`-chunk array layout, so do not run the merge until
the full array has completed cleanly.

To plot quick pilot-recovery heatmaps after merging:

```bash
cd analysis
python3 make_figure_dbm_recovery.py --block 6
```

This writes:

- `../figures/dbm_recovery_pilot_block_<BLOCK>_family_props.png`
- `../figures/dbm_recovery_pilot_block_<BLOCK>_model_props.png`

To scale beyond the demo:

- edit `fit_dbm_recovery_gadi_array.pbs`
- increase the fixed `NUM_CHUNKS`
- pass the matching `--num-chunks` value to `merge_dbm_recovery.py`
- pass `MAX_GROUPS=<n>` at submission time if you want to cap the run again
- increase `N_REPS` only after confirming the small run is acceptable

### Refitting DBMs On NCI Gadi

The current DBM fit input includes blocks `0`, `2`, and `6`. The same fitting
script can be run locally or as one chunk of a Gadi array job.

For a full local/interative run:

```bash
cd analysis
python fit_dbm.py --seed 462 --out-path ../dbm_fits/dbm_results.csv
```

To benchmark or rerun the empirical DBM fits on Gadi, submit the array job from
`analysis/`:

```bash
cd analysis
qsub -v SEED=462 fit_dbm_gadi_array.pbs
```

Default behavior:

- requests `1` CPU, `4GB` RAM, and `08:00:00` walltime per array task
- runs `4` array tasks, one per chunk of subject/block groups
- uses `DE_WORKERS=1` by default inside `differential_evolution`
- writes chunk files into `../dbm_fits/dbm_results_chunks/`
- each array task calls `fit_dbm.py` with its `--chunk-index`

After all `4` array tasks finish, merge the chunk files locally or
interactively:

```bash
cd analysis
python3 merge_dbm_fit.py --num-chunks 4
```

This writes `../dbm_fits/dbm_results_gadi.csv` by default.

Useful overrides at submission time:

- `DE_WORKERS=<n>` to change optimizer parallelism
- edit `fit_dbm_gadi_array.pbs` if you want a different fixed `NUM_CHUNKS`
- pass `--out-path ../dbm_fits/dbm_results.csv` to `merge_dbm_fit.py` if
  you do want to overwrite the main fit table
- `SEED=<n>` to rerun with a different deterministic base seed

Current module layout:

- `dbm_models.py` - model fitting code
- `trial_data.py` - data loading helpers
- `dbm_results.py` - DBM result wrangling helpers

### Running the experiment

```bash
cd experiment
python run_exp.py
```

- Uses `pygame` and opens a fullscreen display.
- Subject number is set directly in the script.
- Condition assignment is currently determined in-script from `condition_list`.
- Trial data are written to `../data/sub_<subject>_data.csv`.

## Raw Data Format

Each CSV file in `data/` is a single subject's trial-level data and has the following format:

| Column Name | Column Description |
| --- | --- |
| `experiment` | Experiment identifier: `1` or `2` |
| `condition` | Condition label: `relearn` or `new_learn` |
| `subject` | Subject identifier |
| `trial` | Trial number|
| `cat` | True category label: `A` or `B` |
| `x` | Stimulus dimension 1 in abstract stimulus space|
| `y` | Stimulus dimension 2 in abstract stimulus space|
| `xt` | Spatial frequency in cycles per degree |
| `yt` | Orientation in radians |
| `resp` | Participant response: `A` or `B` |
| `rt` | Response time in ms |
| `fb` | Feedback: `Correct` or `Incorrect` |

The analysis code assumes each CSV contains **899 rows** in
chronological order:

- `Learn`: 300 trials
- `Intervention`: 300 trials
- `Test`: 299 trials
