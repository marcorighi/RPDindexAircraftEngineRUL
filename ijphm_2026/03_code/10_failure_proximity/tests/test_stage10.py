#!/usr/bin/env python3
"""Tests for exact RPD prefix equivalence, temporal isolation, CSV checks."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Accept either package layout (code/stage10.py) or installed layout (stage10.py).
parent = Path(__file__).resolve().parent.parent
source = parent / 'code' if (parent / 'code' / 'stage10.py').exists() else parent
sys.path.insert(0, str(source))
from stage10 import (rpdgep_prefix_fast, rpdgep_bruteforce, ols_prefix_slope,
                     stable_alarm_cycles, construct_prefix_features,
                     read_dataset, run, SENSORS, SETTINGS)


class ExactFormulaTests(unittest.TestCase):
    def test_all_up_down_constant(self):
        for x, expected in [([.1,.3,.6,.7], 1), ([.9,.6,.3,.2], -1),
                            ([.3,.3,.3,.3], 0)]:
            out = rpdgep_prefix_fast(np.array(x))
            self.assertAlmostEqual(float(out[-1]), expected, places=13)

    def test_every_prefix_matches_exact_quadratic_formula(self):
        rng = np.random.default_rng(84)
        for _ in range(24):
            a = rng.choice(np.linspace(0,1,11), size=29, replace=True)
            fast = rpdgep_prefix_fast(a)
            for i in range(1,len(a)+1):
                self.assertAlmostEqual(float(fast[i-1]), rpdgep_bruteforce(a[:i]), places=11)

    def test_no_future_in_prefix_score(self):
        a = np.array([.1,.4,.2,.6,.3,.5,.7])
        b = np.r_[a, [9999.,-3000.,.1]]
        np.testing.assert_allclose(rpdgep_prefix_fast(a), rpdgep_prefix_fast(b)[:len(a)], rtol=1e-13, atol=1e-13)

    def test_not_unweighted_signed_variation(self):
        x = [.2,.8,.5]
        self.assertAlmostEqual(float(rpdgep_prefix_fast(np.array(x))[-1]), 5/7, places=12)

    def test_linear_slope(self):
        t = np.arange(1,8)
        v = 2*t + 5
        o = ols_prefix_slope(t,v)
        np.testing.assert_allclose(o[1:], 2, atol=1e-12)
        self.assertEqual(o[0],0)

    def test_stable_alert_two_consecutive_physical_cycles(self):
        assert stable_alarm_cycles(np.array([1,3,4,7,8]),np.array([.9,.9,.9,.1,.9]),.5,2)==[4]


def artificial_data(engines=8, cycles=52, seed=22):
    rng = np.random.default_rng(seed)
    rows = []
    for eng in range(1, engines+1):
        lifetime = cycles + eng % 7
        for t in range(1,lifetime+1):
            rul = lifetime-t
            data = {'id':eng,'cycle':t,'setting1':0.1*eng,'setting2':0.0,'setting3':0.0}
            for i, sensor in enumerate(SENSORS):
                data[sensor] = (0.02*i + 0.003*t + 0.002*eng +
                                0.009*rng.normal() + 0.04*max(0,10-rul)/10 * ((i % 3)-1))
            rows.append(data)
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_engine_grouping_and_terminal_exclusion(self):
        data = artificial_data(8, 50)
        prefixes, excluded = construct_prefix_features(data, horizon=10,min_prefix=8)
        self.assertEqual(len(excluded),0)
        self.assertTrue((prefixes['rul']>=1).all())
        self.assertTrue((prefixes['cycle']>=8).all())
        self.assertEqual(prefixes['id'].nunique(),8)
        for _,g in prefixes.groupby('id'):
            self.assertEqual(g['event_within_horizon'].sum(),10)

    def test_immutable_input_validation(self):
        data = artificial_data(6,45)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'sample.csv'
            data.to_csv(path,index=False)
            _ = read_dataset(path)
            a=data.copy()
            a.loc[0,'cycle'] = 0
            a.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                read_dataset(path)

    def test_end_to_end_small_synthetic_only(self):
        with tempfile.TemporaryDirectory() as root:
            project=Path(root)
            (project/'02_raw_data').mkdir()
            (project/'03_code').mkdir()
            (project/'04_results').mkdir()
            artificial_data(8, 51).to_csv(project/'02_raw_data'/'PM_train.csv',index=False)
            args=SimpleNamespace(project_root=str(project), dataset='FD001',
                                 output=str(project/'04_results'/'stage10_test'),
                                 horizon=10,min_prefix=8, confirmations=2,
                                 threshold=0.5,folds=4,bootstrap=100,seed=123,
                                 save_features=False)
            out=run(args)
            self.assertTrue((out/'SHA256SUMS').is_file())
            self.assertTrue((out/'model_summary.csv').is_file())
            summary=pd.read_csv(out/'model_summary.csv')
            self.assertEqual(len(summary),6)
            self.assertTrue(summary['mean_engine_average_precision'].between(0,1).all())
            self.assertFalse((out/'progressive_prefix_features.csv.gz').exists())


if __name__=='__main__': unittest.main(verbosity=2)
