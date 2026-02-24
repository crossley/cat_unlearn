# cat_unlearn (4-category version)

Code, data, and in-progress analysis/modeling utilities for the paper (in preparation):

**Memory masking vs overwriting in procedural categorization**

This is the 4-category task variant. Data, figures, and model-fit outputs
are intentionally cleared here to avoid mixing with earlier 2-category
materials.

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
  Experiment runtime and utilities for the 4-category task (plus DBM utilities).
  - `util_func_dbm.py` — decision-bound model (DBM) likelihoods and fitting utilities
  - `util_func.py` — stimulus generation / visualization utilities
  - `run_exp.py` — experiment runtime (Pygame; used to run the experiment and generate data)

- **data/**
  Trial-level data files, one CSV per subject. (This directory is currently empty.)

- **dbm_fits/**
  Saved DBM fitting results (CSV) produced by analysis scripts. (Currently empty.)

- **figures/**
  Output figures saved by analysis scripts. (Currently empty.)

- **write/**
  LaTeX manuscript sources and compiled PDF. (Currently empty.)

- **consent/**
  Participant consent form template (no signed forms or other
  identifying information contained here).

---

## Setup

This project does not currently include a pinned environment file.
The code imports:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `seaborn`
- `pingouin`
- `pygame`

Example (from repo root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy seaborn pingouin pygame
```

---

## How To Run

The 4-category runtime is currently the main executable component.

From the `code/` directory:

```bash
python run_exp.py
```

What it does:

- Creates a randomized subject/session ID and metadata JSON.
- Assigns a condition (`relearn` / `new_learn`) programmatically.
- Runs the fullscreen `pygame` experiment.
- Writes trial data and session metadata to `../data/`.

Analysis workflow for the 4-category task is still under active development.
Scripts and outputs will be added/expanded here as they are finalized.

---

## Raw Data Format (CSV)

Each file in `data/` is a single subject's trial-level data. Analysis scripts will document any
task-specific expectations as they are finalized.

* Each subject CSV contains **899 rows** in chronological order:

  * **Learn:** 300 trials
  * **Intervention:** 300 trials
  * **Test:** 299 trials

* The script assigns a `phase` column based on row position:

  * `phase = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"] * 299`

### Expected Columns

The runtime currently writes columns compatible with the 2-category project structure (with 4-category labels in `cat` / `resp` where applicable):

| Column       | Meaning / usage                                                           |
| ------------ | ------------------------------------------------------------------------- |
| `experiment` | Experiment identifier (e.g., 1 or 2). Used for grouping/plot panels.      |
| `condition`  | Condition label (e.g., `relearn`, `new_learn`). Used for grouping.        |
| `subject`    | Subject ID. Used for grouping and exclusions.                             |
| `trial`      | Trial index within file/session. Used for filtering late-learning trials. |
| `cat`        | True category label (4-category labels in this variant).                   |
| `x`          | Stimulus dimension 1 (used in plots and DBM fits).                        |
| `y`          | Stimulus dimension 2 (used in plots and DBM fits).                        |
| `xt`         | Same as `x` but transformed to cycles per cm                              |
| `yt`         | Same as `y` but transformed to radians                                    |
| `resp`       | Participant response (key-coded / category-coded response value).          |
| `rt`         | Participant response time in ms                                           |
| `fb`         | Feedback received (e.g., `correct`/`incorrect`). Used for accuracy checks.|

Downstream analysis scripts are expected to add:

* `phase` - Learn / Intervention / Test
* `acc` - boolean accuracy (`cat == resp` after recoding)
