from util_func_dbm import fit_dbm_top
from util_func_figs import (make_fig_acc_all, make_fig_acc_talk,
                            make_fig_dbm, make_fig_dbm_bayes)


if __name__ == "__main__":

    print("Running analysis functions")

    # Choose which analysis outputs to generate.
    # Most functions write figures to ../figures/ and/or write DBM fit tables to ../dbm_fits/.

    # Learning curves for all included subjects (with exclusion rule applied):
    # make_fig_acc_all()
    # make_fig_acc_talk()

    # Fit DBMs and write ../dbm_fits/dbm_results.csv (can take time; uses differential evolution):
    # fit_dbm_top()

    # Model-class transition heatmaps (requires dbm_results.csv):
    # make_fig_dbm()

    # Bayesian-style comparison plot for selected proportions (uses hard-coded counts at present):
    make_fig_dbm_bayes()
