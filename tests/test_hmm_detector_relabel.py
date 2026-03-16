import numpy as np
import pandas as pd

from regime.hmm_detector import HMMRegimeDetector


def test_relabel_by_global_rv_quantiles_yields_all_4_labels():
    detector = HMMRegimeDetector(n_states=4)
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    rv = pd.Series(np.linspace(0.01, 1.0, len(idx)), index=idx)

    labels = detector._relabel_by_global_rv_quantiles(rv)

    assert labels.min() == 0
    assert labels.max() == 3
    assert labels.nunique() == 4


def test_relabel_by_global_rv_quantiles_handles_tied_values_in_4_state_mode():
    detector = HMMRegimeDetector(n_states=4)
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    # Deliberately many repeated values to simulate quantile ties.
    vals = np.array([0.2] * 10 + [0.4] * 10 + [0.6] * 10 + [0.8] * 10)
    rv = pd.Series(vals, index=idx)

    labels = detector._relabel_by_global_rv_quantiles(rv)

    assert labels.min() == 0
    assert labels.max() == 3
    assert labels.nunique() == 4
