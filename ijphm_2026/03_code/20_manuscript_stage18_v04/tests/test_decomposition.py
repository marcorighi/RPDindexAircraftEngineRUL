#!/usr/bin/env python3
"""Independent algebraic checks; no dependence on training data or model code."""
import random
import unittest


def pairs(x, mode):
    if mode == 'global':
        return [x[j]-x[i] for i in range(len(x)) for j in range(i+1,len(x))]
    return [x[j]-x[j-1] for j in range(1,len(x))]

def direct_rpd(x, mode):
    d=pairs(x,mode)
    pos=[z for z in d if z>0]
    neg=[-z for z in d if z<0]
    P=len(pos)*sum(pos)
    N=len(neg)*sum(neg)
    return (P-N)/(P+N) if P+N else 0.0

def components(x, mode):
    d=pairs(x,mode)
    pos=[z for z in d if z>0]
    neg=[-z for z in d if z<0]
    c=len(pos)+len(neg)
    m=sum(pos)+sum(neg)
    if c==0: return 0.,0.
    return (len(pos)-len(neg))/c,(sum(pos)-sum(neg))/m

def reconstructed_rpd(x, mode):
    gamma,eta=components(x,mode)
    return (gamma+eta)/(1+gamma*eta)

class ExpostIdentity(unittest.TestCase):
    def test_examples_exact(self):
        self.assertAlmostEqual(direct_rpd([0,3,1],'global'),3/5)
        self.assertAlmostEqual(direct_rpd([0,100,99],'global'),397/399)
        self.assertEqual(components([0,3,1],'global')[0],components([0,100,99],'global')[0])
    def test_random_and_ties(self):
        rng=random.Random(20261104)
        cases=[[2,2,2],[1,2],[2,1],[1,3,2,4],[0,1,0,1],[1,1,3,3,4],[-3,-2,0,2], [1,3,0]]
        for n in range(2,31):
            for k in range(16):
                cases.append([rng.randint(-4,4) if k%2 else rng.uniform(-10,10) for _ in range(n)])
        for x in cases:
            for mode in ('local','global'):
                with self.subTest(n=len(x), mode=mode):
                    self.assertAlmostEqual(direct_rpd(x,mode),reconstructed_rpd(x,mode),places=12)
    def test_scale(self):
        for mode in ('local','global'):
            x=[0.3,0.55,0.3,0.7,0.1]
            self.assertAlmostEqual(direct_rpd(x,mode),direct_rpd([7.5*z+11 for z in x],mode))
            self.assertAlmostEqual(direct_rpd(x,mode),-direct_rpd([-3*z+11 for z in x],mode))
    def test_local_telescoping(self):
        for x in ([1,2,3],[0,3,1],[0,1,0,1],[1,1,1]):
            _,eta=components(x,'local')
            tv=sum(abs(x[j]-x[j-1]) for j in range(1,len(x)))
            self.assertAlmostEqual(eta, (x[-1]-x[0])/tv if tv else 0.)
    def test_binary_special_case(self):
        for x in ([0,1,0,1],[1,0,1,1,0,0,1],[0,0,1,0,1]):
            for mode in ('global','local'):
                gamma,eta=components(x,mode)
                self.assertAlmostEqual(gamma,eta)
                self.assertAlmostEqual(direct_rpd(x,mode),2*gamma/(1+gamma*gamma))
if __name__=='__main__': unittest.main()
