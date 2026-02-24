# cat_unlearn (2-category version)

Code, data, analysis outputs, and manuscript files for the paper (in preparation):

**Memory masking vs overwriting in procedural categorization**

## Authors

Matthew J. Crossley^{1, 2, 3}
Kayla C. Rail^{1}
Jack Mair^{1}
David M. Kaplan^{1, 2, 3}

**1** School of Psychological Sciences, Macquarie
University, Sydney, Australia

**2** Performance and Expertise Research Centre, Macquarie
University, Sydney, Australia

**3** Macquarie Minds and Intelligences Initiative,
Macquarie University, Sydney, Australia

---

## Directory Structure

- **code/**
  Analysis scripts, utilities, and experiment runtime code.
  - `inspect_results_clean.py` — main analysis script
  - `util_func_dbm.py` — decision-bound model (DBM) likelihoods and fitting utilities
  - `util_func.py` — stimulus generation / visualization utilities
  - `run_exp.py` — experiment runtime (Pygame; used to run the experiment and generate data)

- **data/**
  Trial-level data files, one CSV per subject (e.g., `sub_1_data.csv`).

- **dbm_fits/**
  Saved DBM fitting results (CSV) produced by `fit_dbm()`:
  - `dbm_results.csv` — full output (fits per subject × block × model)
  - `dbm_results_short.csv` — reduced/summary version (if used)

- **figures/**
  Output figures saved by analysis scripts.

- **write/**
  LaTeX manuscript sources and compiled PDF.

- **consent/**
  Participant consent form template (no signed forms or other
  identifying information contained here).

---

## Setup

This project does not currently include a pinned environment file.
The analysis scripts import:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `seaborn`
- `pingouin`
- `pygame` (runtime script only)

Example (from repo root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy seaborn pingouin pygame
```

---

## How To Run

From the `code/` directory, run:

```bash
python inspect_results_clean.py
```

Edit the `__main__` section in `inspect_results_clean.py` to enable the analyses/figures you want.

Typical workflow:

1. Generate descriptive figures:

   * `make_fig_cat_struct()`
   * `make_fig_acc_all()`

2. Fit DBMs (writes `../dbm_fits/dbm_results.csv`):

   * `fit_dbm()`

3. Make model-selection / transition plots:

   * `make_fig_dbm()`
   * `make_fig_acc_proc()`

4. Generate Bayesian comparison figure:

   * `make_fig_dbm_state()`

Outputs are written to `../figures/` and `../dbm_fits/`.

### Running the experiment (data collection)

From the `code/` directory:

```bash
python run_exp.py
```

Notes:

- Uses `pygame` and opens a fullscreen display.
- Subject number and condition assignment are currently set in-script.
- Trial data are saved to `../data/`.

---

## Raw Data Format (CSV)

Each file in `data/` is a single subject's trial-level data. The main analysis script (`inspect_results_clean.py`) expects:

* Each subject CSV contains **899 rows** in chronological order:

  * **Learn:** 300 trials
  * **Intervention:** 300 trials
  * **Test:** 299 trials

* The script assigns a `phase` column based on row position:

  * `phase = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"] * 299`

### Expected columns

The analysis relies on these columns being present:

| Column       | Meaning / usage                                                           |
| ------------ | ------------------------------------------------------------------------- |
| `experiment` | Experiment identifier (e.g., 1 or 2). Used for grouping/plot panels.      |
| `condition`  | Condition label (e.g., `relearn`, `new_learn`). Used for grouping.        |
| `subject`    | Subject ID. Used for grouping and exclusions.                             |
| `trial`      | Trial index within file/session. Used for filtering late-learning trials. |
| `cat`        | True category label (`A`/`B`), recoded to 0/1 for modeling.               |
| `x`          | Stimulus dimension 1 (used in plots and DBM fits).                        |
| `y`          | Stimulus dimension 2 (used in plots and DBM fits).                        |
| `xt`         | Same as `x` but transformed to cycles per cm                              |
| `yt`         | Same as `y` but transformed to radians                                    |
| `resp`       | Participant response (`A`/`B` or numeric), recoded to 0/1 for modeling.   |
| `rt`         | Participant response time in ms                                           |
| `fb`         | Feedback received (e.g., `correct`/`incorrect`). Used for accuracy checks.|

The script adds:

* `phase` - Learn / Intervention / Test
* `acc` - boolean accuracy (`cat == resp` after recoding)
