import numpy as np
import pytest
from src.geometry import (controller, metric, regret, spectral_compress, Radii,
    certificate, oracle_bound, gaussian_radii, fit_ols, lift_lti)


def test_exact_regret_against_cost():
    rng = np.random.default_rng(2)
    d, b = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))
    f, bh = rng.normal(size=d.shape), rng.normal(size=b.shape)
    k, _ = controller(b); kh, _ = controller(bh)
    u, us = -kh @ f, -k @ d
    direct = np.linalg.norm(d+b@u, 'fro')**2 + np.linalg.norm(u, 'fro')**2
    direct -= np.linalg.norm(d+b@us, 'fro')**2 + np.linalg.norm(us, 'fro')**2
    assert np.isclose(regret(d,b,f,bh), direct)


def test_global_lipschitz_and_controller_weighted_bound():
    rng = np.random.default_rng(4)
    for scale in [0.01, 1, 100]:
        for _ in range(50):
            b = rng.normal(size=(5, 3)) * scale
            bh = b + rng.normal(size=b.shape)
            e = np.linalg.norm(bh-b, 2)
            k,h = controller(b); kh,_ = controller(bh)
            assert np.linalg.norm(metric(bh)-metric(b),2) <= e+1e-9
            assert np.linalg.norm(kh-k,2) <= e+1e-9
            l = np.linalg.cholesky(h)
            assert np.linalg.norm(l.T@(kh-k),2) <= 1.5*e+1e-9


def test_spectral_tail_optimum_and_rank():
    rng=np.random.default_rng(5)
    d,b=rng.normal(size=(5,4)),rng.normal(size=(5,3))
    w=metric(b)
    for r in range(5):
        f,p=spectral_compress(d,w,r)
        assert np.linalg.matrix_rank(f, tol=1e-9)<=r
        assert np.allclose(p@p,p)
        tail=np.linalg.eigvalsh(d.T@w@d)[:4-r].sum()
        assert np.isclose(regret(d,b,f,b),tail,atol=1e-9)


def test_uniform_certificate_and_oracle_bound():
    rng=np.random.default_rng(6)
    for _ in range(50):
        d,b=rng.normal(size=(5,4)),rng.normal(size=(5,2))
        dh=d+rng.normal(size=d.shape)*0.08
        bh=b+rng.normal(size=b.shape)*0.1
        rad=Radii(np.linalg.norm(dh-d,'fro'),np.linalg.norm(bh-b,2),.05)
        for r in range(5):
            f,_=spectral_compress(dh,metric(bh)+rad.eta*np.eye(5),r)
            assert regret(d,b,f,bh)<=certificate(dh,bh,f,rad)+1e-8
            assert regret(d,b,f,bh)<=oracle_bound(d,b,dh,r,rad)+1e-8
        f=rng.normal(size=d.shape)
        assert regret(d,b,f,bh)<=certificate(dh,bh,f,rad)+1e-8


def test_no_false_domination_claim():
    d=np.diag([2.,1.]); b=np.diag([0.,1.]); w=metric(b)
    fp,_=spectral_compress(d,w,1)
    fr,_=spectral_compress(d,w+.25*np.eye(2),1)
    assert np.isclose(regret(d,b,fp,b),0)
    assert np.isclose(regret(d,b,fr,b),.5)


def test_reset_ols_and_invalid_coverage():
    rng=np.random.default_rng(9)
    theta=rng.normal(size=(3,5)); z=rng.normal(size=(5,50))
    assert np.allclose(fit_ols(z,theta@z),theta)
    with pytest.raises(ValueError): gaussian_radii(3,3,2,np.zeros((5,5)),1)
    with pytest.raises(ValueError): fit_ols(np.zeros((5,50)),theta@z)


def test_lift_matches_rollout():
    rng=np.random.default_rng(11)
    a=rng.normal(size=(3,3))*.2; b=rng.normal(size=(3,2))
    ds,bs=lift_lti(a,b,4,action_cost=2)
    x=rng.normal(size=3); u=rng.normal(size=8)
    states=[]; state=x.copy()
    for i in range(4):
        state=a@state+b@(u[2*i:2*i+2]/np.sqrt(2)); states.extend(state)
    assert np.allclose(ds@x+bs@u,states)


def test_orthogonal_output_coordinate_invariance():
    rng=np.random.default_rng(12)
    d,b=rng.normal(size=(5,4)),rng.normal(size=(5,2))
    q,_=np.linalg.qr(rng.normal(size=(5,5)))
    f,_=spectral_compress(d,metric(b)+.1*np.eye(5),2)
    f2,_=spectral_compress(q@d,metric(q@b)+.1*np.eye(5),2)
    assert np.allclose(q@f,f2)
    assert np.isclose(regret(d,b,f,b),regret(q@d,q@b,f2,q@b))


def test_gap_sensitive_projector_excess():
    rng=np.random.default_rng(13)
    l=np.diag([4.,2.,1.,.1]); gap=1.
    for _ in range(50):
        e=rng.normal(size=(4,4))*.1; e=(e+e.T)/2
        _,v=np.linalg.eigh(l+e); p=v[:,-2:]@v[:,-2:].T
        p0=np.diag([1.,1.,0.,0.])
        excess=np.trace(l@(p0-p))
        assert excess<=8*np.linalg.norm(e,2)**2/gap+1e-10
