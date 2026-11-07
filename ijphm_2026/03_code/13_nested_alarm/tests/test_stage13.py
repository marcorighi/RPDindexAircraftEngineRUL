#!/usr/bin/env python3
"""Stage 13 tests - mathematical semantics, isolation, and tiny synthetic run."""
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import stage13 as s


def data_for_fixture(neng=12, start=42):
    rng=np.random.default_rng(187)
    records=[]
    for e in range(1,neng+1):
        last=start+e%8
        for cycle in range(1,last+1):
            r=last-cycle
            d={'id':e,'cycle':cycle,'setting1':0.1*e,'setting2':0.0,'setting3':0.0}
            for j,sens in enumerate(s.s10.SENSORS):
                d[sens]=float(0.02*j+.004*cycle+.01*rng.normal()+.07*(r<9)*((j%3)-1))
            records.append(d)
    return pd.DataFrame(records)


class ThresholdTests(unittest.TestCase):
    def test_first_alarm_grid_matches_exact(self):
        # second adjacent hit at cycle=4, RUL=8; gap in engine 2 cannot confirm.
        df=pd.DataFrame({'id':[1]*6+[2]*5,
                         'cycle':[1,2,3,4,5,6,1,3,4,6,7],
                         'rul':[11,10,9,8,7,6,11,9,8,6,5],
                         'p':[.1,.8,.1,.9,.91,.6,.9,.9,.8,.1,.8]})
        for th in (0.5,.7,.9,.9999):
            exact=s.grouped_first_alarm(df,'p',th,horizon=8,min_lead=5)
            grid=s.first_alarm_grid(df,'p',8,5,np.array([th])).iloc[0]
            self.assertAlmostEqual(grid['premature_fraction'],exact['premature'].mean())
            self.assertAlmostEqual(grid['actionable_fraction'],exact['actionable'].mean())
            self.assertAlmostEqual(grid['late_fraction'],exact['late'].mean())
            self.assertAlmostEqual(grid['never_fraction'],exact['never'].mean())

    def test_threshold_selection_enforces_premature_cap(self):
        grid=pd.DataFrame({'threshold':[.5,.8,1.000001],
                           'premature_fraction':[.8,.2,0],
                           'actionable_fraction':[.7,.6,0],
                           'median_actionable_lead':[20,12,0]})
        r=s.choose_threshold(grid,0.2)
        self.assertEqual(r.threshold,.8)

    def test_no_alarm_fallback(self):
        grid=pd.DataFrame({'threshold':[.8,1.000001],
                           'premature_fraction':[.9,0],
                           'actionable_fraction':[.1,0],
                           'median_actionable_lead':[12,0]})
        self.assertGreater(s.choose_threshold(grid,0.2).threshold,1)

    def test_original_formula_preserved(self):
        x=np.array([.2,.8,.5],dtype=float)
        self.assertAlmostEqual(float(s.s10.rpdgep_prefix_fast(x)[-1]),5/7)

    def test_end_to_end_synthetic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)
            (project/'02_raw_data').mkdir()
            (project/'03_code'/'10_failure_proximity').mkdir(parents=True)
            (project/'04_results').mkdir()
            shutil.copy2(s.STAGE10_DIR/'stage10.py',project/'03_code'/'10_failure_proximity'/'stage10.py')
            data_for_fixture().to_csv(project/'02_raw_data'/'PM_train.csv',index=False)
            args=SimpleNamespace(project_root=str(project),dataset='FD001',
              output=str(project/'04_results'/'stage13_test'),outer_folds=3,
              inner_folds=2,horizon=8,min_lead=2,min_prefix=8,
              max_premature=.30,bootstrap=50,seed=72)
            folder=s.run(args)
            summary=pd.read_csv(folder/'model_summary.csv')
            thresholds=pd.read_csv(folder/'thresholds_selected_inner_oof.csv')
            alarms=pd.read_csv(folder/'first_alarms_outer_oof.csv')
            self.assertEqual(len(summary),2)
            self.assertEqual(len(thresholds),6)
            self.assertEqual(len(alarms),24)
            self.assertTrue((thresholds.inner_premature_fraction<=.30+1e-12).all())
            self.assertTrue((summary[['premature_first_fraction','actionable_first_fraction']].to_numpy()<=1).all())
            with self.assertRaises(ValueError):s.run(args)


if __name__=='__main__':unittest.main(verbosity=2)
