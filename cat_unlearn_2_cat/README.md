# Memory masking vs overwriting in procedural categorization

Code, data, analysis outputs, and manuscript files for the paper:

**Memory masking vs overwriting in procedural categorization**

## Directory Structure

- **code/**
  Analysis scripts, utilities, and experiment runtime code.
  - `inspect_results.py` - thin entry-point script for running analysis functions
  - `util_func_figs.py` - figure-generation functions
  - `util_func_dbm.py` - decision-bound model (DBM) fitting and likelihood functions
  - `util_func_wrangle.py` - data loading and DBM-result wrangling helpers
  - `util_func_stimcat.py` - stimulus generation and grating utilities
  - `make_example_trials_fig.py` - standalone example-trials figure script
  - `generate_example_trial.py` - standalone raster example-trial image builder
  - `run_exp.py` - experiment runtime used to generate subject CSVs

- **data/**
  Trial-level data files, one CSV per subject (for example `sub_1_data.csv`).

- **dbm_fits/**
  Saved DBM fitting results written by the analysis script.
  - `dbm_results.csv` - fits per subject x block x model

- **figures/**
  Analysis outputs currently present in this subtree.
  - `subjects_accuracy_all.png`
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

Run analysis scripts from the `code/` directory so relative
paths resolve correctly.

### Main analysis workflow

```bash
cd code
python inspect_results.py
```

`inspect_results.py` is now a thin runner that imports top-level functions from the utility modules and exposes them as commented calls in its `__main__` block.

Any generated output is controlled by the `__main__` block near the end of `inspect_results.py`.

- `make_fig_acc_all()` - writes `subjects_accuracy_all.png`
- `make_fig_acc_talk()` - writes two talk-style PDF figures
- `fit_dbm_top()` - writes `dbm_results.csv`
- `make_fig_dbm()` - writes `best_model_class_heatmap.png`
- `make_fig_dbm_bayes()` - writes `bayesian_comparison.png`

### Model Recovery On NCI Gadi

Run the recovery workflow from `code/` after `../dbm_fits/dbm_results.csv`
exists. The empirical DBM fits are now deterministic when fit with the same
base seed, and recovery uses the original trial order to define blocks.

The checked-in Gadi scripts are configured for a conservative demo run:

- `4` array tasks total, matching PBS array range `0-3`
- `1` recovery repetition per group
- only the first `8` empirical subject/block groups are processed

This keeps the default recovery workload small for a limited allocation. Scale
up only after the demo run works cleanly.

Before submitting to Gadi, replace `<PROJECT_CODE>` in both PBS scripts with
your actual NCI project code.

1. Generate or refresh empirical DBM fits locally:

```bash
cd code
python -c "from util_func_dbm import fit_dbm_top; fit_dbm_top(seed=462)"
```

2. Submit the recovery array job on Gadi:

```bash
cd code
qsub -v BLOCK=6,N_REPS=1,SEED=462,MAX_GROUPS=8 run_dbm_recovery_gadi.pbs
```

3. Wait for all `4` array tasks to finish successfully, then submit the merge:

```bash
cd code
qsub -v BLOCK=6 merge_dbm_recovery_gadi.pbs
```

Outputs land in:

- `../dbm_fits/recovery_chunks/` - one CSV per chunk
- `../dbm_fits/dbm_recovery_empirical_block_<BLOCK>_results.csv`
- `../dbm_fits/dbm_recovery_empirical_block_<BLOCK>_family_counts.csv`
- `../dbm_fits/dbm_recovery_empirical_block_<BLOCK>_family_props.csv`
- `../dbm_fits/dbm_recovery_empirical_block_<BLOCK>_model_counts.csv`
- `../dbm_fits/dbm_recovery_empirical_block_<BLOCK>_model_props.csv`

The merge script now fails if any chunk is missing or if chunk filenames do not
match the expected `4`-chunk array layout, so do not run the merge until the
full array has completed cleanly.

To scale beyond the demo:

- edit `run_dbm_recovery_gadi.pbs` and `merge_dbm_recovery_gadi.pbs` together
- increase the fixed `NUM_CHUNKS`
- increase or remove `MAX_GROUPS`
- increase `N_REPS` only after confirming the small run is acceptable

Current module layout:

- `inspect_results.py` - manual entry point for running selected analyses
- `util_func_figs.py` - figure functions
- `util_func_dbm.py` - model fitting code
- `util_func_wrangle.py` - data wrangling helpers
- `util_func_stimcat.py` - stimulus-generation helpers

### Example-trials figure

```bash
cd code
python make_example_trials_fig.py
```

This writes `../figures/example_trials_figure.png`.

### Running the experiment

```bash
cd code
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
