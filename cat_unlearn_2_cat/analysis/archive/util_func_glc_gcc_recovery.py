import numpy as np


GLC_SLOPE_GRID = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
GLC_DIAGONAL_GRID = np.linspace(30, 70, 5)
GLC_NOISE_GRID = np.array([2.5, 5.0, 10.0])
GLC_SIDE_GRID = [0, 1]

GCC_XC_GRID = np.linspace(30, 70, 5)
GCC_YC_GRID = np.linspace(30, 70, 5)
GCC_NOISE_GRID = np.array([2.5, 5.0, 10.0])
GCC_SIDE_GRID = [0, 1, 2, 3]


def make_cat_trials(n_per_cat):
    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    theta = np.arctan(1)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    rA = np.sqrt(np.random.uniform(0, 9, n_per_cat))
    angA = np.random.uniform(0, 2 * np.pi, n_per_cat)
    xA = rA * np.cos(angA) * std_major
    yA = rA * np.sin(angA) * std_minor
    ptsA = np.dot(rotation_matrix, np.vstack([xA, yA]))
    ptsA[0, :] += 40
    ptsA[1, :] += 60

    rB = np.sqrt(np.random.uniform(0, 9, n_per_cat))
    angB = np.random.uniform(0, 2 * np.pi, n_per_cat)
    xB = rB * np.cos(angB) * std_major
    yB = rB * np.sin(angB) * std_minor
    ptsB = np.dot(rotation_matrix, np.vstack([xB, yB]))
    ptsB[0, :] += 60
    ptsB[1, :] += 40

    x = np.concatenate([ptsA[0, :], ptsB[0, :]])
    y = np.concatenate([ptsA[1, :], ptsB[1, :]])
    cat = np.concatenate(
        [np.zeros(n_per_cat, dtype=int), np.ones(n_per_cat, dtype=int)]
    )

    p = np.random.permutation(2 * n_per_cat)
    x = x[p]
    y = y[p]
    cat = cat[p]

    return x, y, cat


def glc_slope_diag_to_params(slope, diag):
    # Treat diag as the point where the boundary crosses y=x, so the explored
    # line locations move along the diagonal category structure.
    origin_intercept = diag - (slope * diag)
    a1 = -slope / np.sqrt(1 + slope**2)
    a2 = np.sqrt(1 - a1**2)
    b = -origin_intercept * a2
    return a1, b


def make_glc_gcc_recovery_jobs(n_reps):
    jobs = []
    job_id = 0

    for side in GLC_SIDE_GRID:
        for slope in GLC_SLOPE_GRID:
            for diag in GLC_DIAGONAL_GRID:
                for noise in GLC_NOISE_GRID:
                    for rep in range(n_reps):
                        jobs.append({
                            "job_id": job_id,
                            "true_model": f"nll_glc_{side}",
                            "true_family": "GLC",
                            "true_side": side,
                            "true_slope": slope,
                            "true_diag": diag,
                            "true_xc": np.nan,
                            "true_yc": np.nan,
                            "true_noise": noise,
                            "rep": rep,
                        })
                        job_id += 1

    for side in GCC_SIDE_GRID:
        for xc in GCC_XC_GRID:
            for yc in GCC_YC_GRID:
                for noise in GCC_NOISE_GRID:
                    for rep in range(n_reps):
                        jobs.append({
                            "job_id": job_id,
                            "true_model": f"nll_gcc_eq_{side}",
                            "true_family": "GCC_eq",
                            "true_side": side,
                            "true_slope": np.nan,
                            "true_diag": np.nan,
                            "true_xc": xc,
                            "true_yc": yc,
                            "true_noise": noise,
                            "rep": rep,
                        })
                        job_id += 1

    return jobs
