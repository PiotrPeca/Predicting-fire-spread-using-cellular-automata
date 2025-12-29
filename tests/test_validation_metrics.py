import numpy as np

from fire_spread.validation.metrics import (
    burned_mask,
    calculate_fitness,
    confusion_matrix,
    toa_error_metrics,
)


def test_toa_error_metrics_perfect_match():
    real = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.nan, 3.0, np.nan],
        ]
    )
    sim = real.copy()

    m = toa_error_metrics(sim, real)
    assert m.n == 4
    assert m.bias == 0.0
    assert m.mae == 0.0
    assert m.rmse == 0.0


def test_toa_error_metrics_constant_time_shift():
    real = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.nan, 3.0, np.nan],
        ]
    )
    sim = real + 2.0

    m = toa_error_metrics(sim, real)
    assert m.n == 4
    assert np.isclose(m.bias, 2.0)
    assert np.isclose(m.mae, 2.0)
    assert np.isclose(m.rmse, 2.0)


def test_toa_error_metrics_handles_no_overlap():
    real = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    sim = np.array([[0.0, 1.0], [2.0, 3.0]])

    m = toa_error_metrics(sim, real)
    assert m.n == 0
    assert m.bias == 0.0
    assert m.mae == 0.0
    assert m.rmse == 0.0


def test_burned_mask_default():
    toa = np.array([[0.0, 5.0, np.nan]])
    assert burned_mask(toa).tolist() == [[True, True, False]]


def test_confusion_matrix_basic_counts():
    # real burns in (0,0) and (0,1); sim burns in (0,0) and (1,1)
    real = np.array([[0.0, 1.0], [np.nan, np.nan]])
    sim = np.array([[0.0, np.nan], [np.nan, 2.0]])

    # Default domain is the real burned mask => only (0,0) and (0,1) are evaluated.
    c = confusion_matrix(sim, real)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 0, 1, 0)
    assert np.isclose(c.precision, 1.0)
    assert np.isclose(c.recall, 0.5)
    assert np.isclose(c.f1, 2 / 3)
    assert np.isclose(c.iou, 0.5)


def test_confusion_matrix_full_grid_when_mask_provided():
    real = np.array([[0.0, 1.0], [np.nan, np.nan]])
    sim = np.array([[0.0, np.nan], [np.nan, 2.0]])
    mask = np.ones((2, 2), dtype=bool)

    c = confusion_matrix(sim, real, mask=mask)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert np.isclose(c.precision, 0.5)
    assert np.isclose(c.recall, 0.5)
    assert np.isclose(c.f1, 0.5)
    assert np.isclose(c.iou, 1 / 3)


def test_calculate_fitness_rmse_mode():
    real = np.array([[0.0, 1.0], [np.nan, 3.0]])
    sim = np.array([[1.0, 2.0], [np.nan, 4.0]])  # diff +1 on overlap

    score = calculate_fitness(sim, real, mode="rmse")
    assert np.isclose(score, 1.0)


def test_calculate_fitness_rmse_plus_burn_penalizes_fp_fn():
    real = np.array([[0.0, np.nan], [np.nan, np.nan]])
    sim = np.array([[0.0, 2.0], [np.nan, np.nan]])

    # If we evaluate on the full grid, the extra burned cell is an FP.
    comps = calculate_fitness(
        sim,
        real,
        mode="rmse+burn",
        mask=np.ones((2, 2), dtype=bool),
        return_components=True,
    )
    # overlap RMSE is 0 (only (0,0) compared)
    assert np.isclose(comps.rmse, 0.0)
    # one FP among 3 negatives => fp_rate = 1/3
    assert np.isclose(comps.fp_rate, 1 / 3)
    # no FN because real has only one burned cell and sim matches it
    assert np.isclose(comps.fn_rate, 0.0)
    assert np.isclose(comps.score, 1 / 3)


def test_calculate_fitness_with_mask_limits_burn_penalty_area():
    real = np.array([[0.0, np.nan], [np.nan, np.nan]])
    sim = np.array([[0.0, 2.0], [np.nan, np.nan]])
    mask = np.array([[True, False], [False, False]])

    comps = calculate_fitness(sim, real, mode="rmse+burn", mask=mask, return_components=True)
    # Within mask there is exactly 1 cell and it's TP; no negatives, so fp_rate=0
    assert np.isclose(comps.fp_rate, 0.0)
    assert np.isclose(comps.fn_rate, 0.0)
    assert np.isclose(comps.score, 0.0)
