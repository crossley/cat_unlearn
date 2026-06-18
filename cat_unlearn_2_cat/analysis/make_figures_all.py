import subprocess
import sys


python = sys.executable

commands = [
    [
        python,
        "make_figure_accuracy.py",
    ],
    [
        python,
        "make_figure_dbm_heatmap.py",
        "--out-path",
        "../figures/best_model_heatmap_exclude_gcc_glc_procedural_unidim_rulebased.png",
        "--exclude-models",
        "nll_gcc_eq_0,nll_gcc_eq_1,nll_gcc_eq_2,nll_gcc_eq_3",
        "--procedural-models",
        "nll_glc_0,nll_glc_1",
        "--rule-based-models",
        "nll_unix_0,nll_unix_1,nll_uniy_0,nll_uniy_1",
    ],
    [
        python,
        "make_figure_dbm_heatmap.py",
        "--out-path",
        "../figures/best_model_heatmap_include_gcc_glc_procedural_unidim_gcc_rulebased.png",
        "--procedural-models",
        "nll_glc_0,nll_glc_1",
        "--rule-based-models",
        "nll_unix_0,nll_unix_1,nll_uniy_0,nll_uniy_1,nll_gcc_eq_0,nll_gcc_eq_1,nll_gcc_eq_2,nll_gcc_eq_3",
    ],
    [
        python,
        "make_figure_dbm_heatmap.py",
        "--out-path",
        "../figures/best_model_heatmap_include_gcc_all_models_raw.png",
    ],
    [
        python,
        "make_figure_bayes_post.py",
        "--out-path",
        "../figures/bayesian_comparison_exclude_gcc_glc_procedural.png",
        "--exclude-models",
        "nll_gcc_eq_0,nll_gcc_eq_1,nll_gcc_eq_2,nll_gcc_eq_3",
        "--procedural-models",
        "nll_glc_0,nll_glc_1",
    ],
    [
        python,
        "make_figure_bayes_post.py",
        "--out-path",
        "../figures/bayesian_comparison_include_gcc_glc_procedural.png",
        "--procedural-models",
        "nll_glc_0,nll_glc_1",
    ],
    [
        python,
        "make_figure_bayes_post.py",
        "--out-path",
        "../figures/bayesian_comparison_include_gcc_glc_gcc_procedural.png",
        "--procedural-models",
        "nll_glc_0,nll_glc_1,nll_gcc_eq_0,nll_gcc_eq_1,nll_gcc_eq_2,nll_gcc_eq_3",
    ],
    [
        python,
        "make_figure_dbm_recovery.py",
        "--block",
        "6",
    ],
]

for command in commands:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
