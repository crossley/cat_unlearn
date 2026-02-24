# cat_unlearn (4-category version)

Code, data, and manuscript files for the paper (in preparation):

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

## Directory structure

- **code/**
  Analysis scripts, utilities, and experiment runtime code for the
  4-category task.
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
  Participant consent forum (no signed forums or other
  identifiying information contained here).

---

## How to run

Analysis workflow for the 4-category task is under active development.
Scripts and outputs will be added here as they are finalized.

---

## Raw data format (CSV)

Each file in `data/` is a single subject's trial-level data. Analysis scripts will document any
task-specific expectations.

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
