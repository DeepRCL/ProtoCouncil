import numpy as np
import pandas as pd

from protocouncil import council as C
from protocouncil.metrics import sensitivity_at_specificity


def table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    gg = rng.choice([0, 0, 0, 1, 2, 3, 4], n)
    t = pd.DataFrame(dict(core_id=[f'c{i}' for i in range(n)], patient_id=[f'p{i // 4}' for i in range(n)], grade_group=gg,
                          involvement=np.where(gg > 0, .5, 0.)))
    base = (gg >= 3) * 1.5 + rng.normal(size=n)
    t['p_cspca'] = 1 / (1 + np.exp(-base)); t['p_cancer'] = t.p_cspca
    for e in C.ENCODERS:
        t[f'p_cspca_{e}'] = 1 / (1 + np.exp(-(base + rng.normal(scale=.7, size=n))))
    return t


def test_thresholds_give_80_percent_specificity():
    t = table()
    th = C.calibrate([t])
    neg = t.grade_group < 3
    assert abs((t.p_cspca[neg] < th['fused']).mean() - .8) < .01


def test_gate_abstains_exactly_when_encoders_disagree():
    t = C.council(table(), C.calibrate([table()]))
    calls = np.column_stack([t[f'p_cspca_{e}'] >= C.calibrate([table()])[e] for e in C.ENCODERS])
    assert (t.abstain.values == ~(calls.all(1) | (~calls).all(1))).all()
    m = C.selective_metrics(t)
    assert abs(m['council_coverage'] - m['confidence_coverage']) < 1e-9


def test_sensitivity_at_specificity():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([.1, .2, .3, .9, .8, .95])
    assert sensitivity_at_specificity(y, s, .8) == .5    # no false positive allowed
    assert sensitivity_at_specificity(y, s, .75) == 1.   # one of four negatives may score higher


def test_retained_balanced_accuracy_undefined_without_both_classes():
    t = C.council(table(), C.calibrate([table()]))
    assert np.isnan(C.retained(t, np.zeros(len(t)))['bacc'])
    assert np.isnan(C.retained(t, (t.grade_group < 3).values)['bacc'])


def test_external_names_cannot_replace_training_splits():
    import pytest
    from protocouncil.extract import parse_external
    assert parse_external(['a/b.csv:cohort_b']) == {'cohort_b': 'a/b.csv'}
    for bad in (['x.csv:train'], ['x.csv:val'], ['x.csv:a', 'y.csv:a'], ['x.csv:../a'], ['x.csv'], ['x.csv:-cohort']):
        with pytest.raises(ValueError):
            parse_external(bad)


def test_prediction_ids_keep_leading_zeros(tmp_path):
    t = table(8)
    t['patient_id'] = ['0007'] * 4 + ['7'] * 4
    t['center'] = '01'
    t.to_csv(tmp_path / 'p.csv', index=False)
    r = C.read_predictions(tmp_path / 'p.csv')
    assert r.patient_id.nunique() == 2 and r.center.iloc[0] == '01'
