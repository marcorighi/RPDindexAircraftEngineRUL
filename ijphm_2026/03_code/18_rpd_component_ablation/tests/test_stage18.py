#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import stage18 as s

class MathematicalChecks(unittest.TestCase):
    def test_global_decomposition_matches_frozen_source(self):
        rng=np.random.default_rng(554)
        examples=[np.array([1.,3.,2.,2.,5.,0.]), np.ones(16),
                  np.array([0,1,0,1,0,1.],float), np.arange(12.),
                  np.arange(12.)[::-1]]
        examples.extend(rng.integers(-3,4,size=rng.integers(5,24)).astype(float)
                        for _ in range(60))
        for x in examples:
            g,e,t=s.component_prefix(x)
            r=s.s10.rpdgep_prefix_fast(x)
            self.assertTrue(np.allclose((g+e)/(1+g*e),r,rtol=1e-9,atol=1e-9))
            self.assertTrue(np.all((-1-1e-12<=t)&(t<=1+1e-12)))

    def test_kendall_tau_b_with_ties_and_constants(self):
        for x in ([0,1,1,4,2,2,6], [2]*9, [5,3,3,2,0,1], list(range(1,9))):
            x=np.array(x,dtype=float)
            tau=s.component_prefix(x)[2]
            for k in range(2,len(x)+1):
                true=kendalltau(np.arange(k),x[:k],variant='b').statistic
                if np.isnan(true):true=0.
                self.assertAlmostEqual(tau[k-1],true,places=12)

    def test_no_future_and_affine_invariance(self):
        x=np.array([1,4,1,3,3,5,4,10,4,5,9,7,2,0,9,1,7,6],dtype=float)
        allc=s.component_prefix(x)
        for k in range(1,len(x)+1):
            prev=s.component_prefix(x[:k])
            for a,b in zip(prev,allc):
                self.assertAlmostEqual(a[-1],b[k-1],places=12)
        g,e,t=s.component_prefix(x)
        a,b,c=s.component_prefix(2.5*x+100.)
        np.testing.assert_allclose(g,a,atol=1e-12)
        np.testing.assert_allclose(e,b,atol=1e-12)
        np.testing.assert_allclose(t,c,atol=1e-12)

    def test_theilsen_fixed_window_exact(self):
        rng=np.random.default_rng(381)
        x=rng.normal(size=47)
        ans=s.theilsen_trailing(x)
        for k in range(len(x)):
            start=max(0,k-s.WINDOW+1)
            z=x[start:k+1]
            expected=0 if len(z)<2 else theilslopes(z,np.arange(len(z))).slope
            self.assertAlmostEqual(float(ans[k]),float(expected),places=12)
        for k in range(1,len(x)+1):
            self.assertAlmostEqual(ans[k-1],s.theilsen_trailing(x[:k])[-1],places=12)

    def test_engine_bootstrap_matches_explicit_replication_with_ties(self):
        from sklearn.metrics import average_precision_score
        rng=np.random.default_rng(2026)
        rows=[]
        for eid in range(1,7):
            for k in range(1,13):
                d={'id':eid,'cycle':k,'rul':13-k,
                   'event_within_horizon':int(k>8)}
                for j,name in enumerate(s.FEATURES):
                    # Scores intentionally rounded to generate genuine ties.
                    d[name]=float(round(rng.uniform(.001,.999),2))
                rows.append(d)
        pred=pd.DataFrame(rows)
        reps=35;seed=314
        result=s.bootstrap_contrasts(pred,'FD001',reps,seed)
        groups=[g for _,g in pred.groupby('id',sort=True)]
        chooser=np.random.default_rng(seed)
        expected={ (a,b):[] for a,b,_ in s.CONTRASTS }
        for k in range(reps):
            selected=[groups[i] for i in chooser.integers(0,len(groups),len(groups))]
            samp=pd.concat(selected,ignore_index=True)
            yy=samp.event_within_horizon.to_numpy()
            ap={name:average_precision_score(yy,samp[name]) for name in s.FEATURES}
            for a,b,_ in s.CONTRASTS:
                expected[(a,b)].append(ap[b]-ap[a])
        for row in result.itertuples():
            sims=expected[(row.base,row.alternative)]
            self.assertAlmostEqual(row.bootstrap_low,float(np.quantile(sims,.025)),places=11)
            self.assertAlmostEqual(row.bootstrap_high,float(np.quantile(sims,.975)),places=11)

    def test_full_data_construct_matches_frozen_base_and_prior_fold(self):
        rng=np.random.default_rng(42)
        rows=[]
        for eid in range(1,13):
            last=44+eid%6
            for cyc in range(1,last+1):
                item={'id':eid,'cycle':cyc, 'setting1':eid/12.,
                      'setting2':0.,'setting3':0.}
                for j,sensor in enumerate(s.s10.SENSORS):
                    item[sensor]=j*0.02+cyc*0.015+float(rng.normal(0,.02))
                rows.append(item)
        raw=pd.DataFrame(rows)
        feat, ex=s.features_for_dataset(raw,horizon=8,min_prefix=8)
        frozen,_=s.s10.construct_prefix_features(raw,8,8)
        self.assertTrue(frozen[['id','cycle']].equals(feat[['id','cycle']]))
        for sensor in s.s10.SENSORS:
            np.testing.assert_allclose(frozen[f'rpd_{sensor}'],feat[f'rpd_{sensor}'])
        self.assertEqual(feat.id.nunique(),12)
        oo=s.train_oof(feat,outer_folds=3)
        summ=s.summarize(oo,'FD001')
        self.assertEqual(len(summ),len(s.FEATURES))
        self.assertEqual(len(oo),len(feat))
        self.assertTrue(((oo[list(s.FEATURES)]>0)&(oo[list(s.FEATURES)]<1)).all().all())
        self.assertTrue((oo.outer_fold>0).all())
        # Exactly the same five-fold grouping family as Stage13, for any fold count.
        from sklearn.model_selection import GroupKFold
        a=list(GroupKFold(n_splits=3).split(feat,feat.event_within_horizon,feat.id))
        b=list(GroupKFold(n_splits=3).split(feat,feat.event_within_horizon,feat.id))
        for (_,v1),(_,v2) in zip(a,b): np.testing.assert_array_equal(v1,v2)

if __name__=='__main__':unittest.main(verbosity=2)
