"""Execute the explicit fixed-weight path and verify the smaller nuisance networks."""
from unittest.mock import patch
import numpy as np
import pytest
import torch
from mediencoder import estimation, models

def test_nuisance_defaults_use_two_hidden_layers_in_actual_fit():
    torch.set_num_threads(1)
    rng=np.random.default_rng(815)
    x=rng.normal(size=(32,3))
    y=x[:,0]+rng.normal(size=32)
    model,_,_=models.train_nuisance_nn(x[:24],y[:24],x[24:],y[24:],epochs=1)
    layers=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
    assert [(m.in_features,m.out_features) for m in layers]==[(3,300),(300,300),(300,1)]

@pytest.mark.parametrize("weights",[(.45,.45,.1),(.5,.5,0.)])
def test_actual_fixed_lambda_estimation_skips_candidate_selection(monkeypatch,weights):
    torch.set_num_threads(1)
    monkeypatch.setitem(estimation.SHARED_TRAIN_CFG,"epochs",1)
    monkeypatch.setattr(estimation,"SHARED_HIDDEN",(4,))
    rng=np.random.default_rng(918)
    n=80
    x,m=rng.normal(size=(n,4)),rng.normal(size=(n,3))
    a=np.arange(n)%2
    y=x[:,0]+m[:,0]+a+rng.normal(size=n)
    with patch.object(estimation,"_select_lambda_for_fold",side_effect=AssertionError("Fixed policy must not tune")):
        result=estimation.estimate_triply_IF(
            x,m,a,y,tilde_p=2,tilde_q=2,factor_method="mediencoder",
            fixed_lambda=weights,seed=14,
            nn_cfg=estimation._shared_cfg(hidden_dims=(4,),eps=1e-8),
            me_cfg={"hidden_dims_XM":(4,)})
    assert result["selected_lambda_A"]==result["selected_lambda_B"]==weights
    assert result["estimator_contract"]["lambda_policy"]=="fixed"
    assert result["estimator_contract"]["alignment_normalization"]=="row_l2"
    assert "fold_tuning" not in result
    assert np.isfinite(result["crossfit_scores"]).all()
    assert sum(len(f["estimation"]) for f in result["fold_indices"])==n

@pytest.mark.parametrize("kwargs",[
    {"fixed_lambda":(.45,.45,.1),"lambda_grid":[(.45,.45,.1)]},
    {"fixed_lambda":(.5,.5,.1)},
    {"fixed_lambda":(0.,.5,.5)},
    {"fixed_lambda":(.45,.45,float("nan"))},
    {"factor_method":"projection","fixed_lambda":(.45,.45,.1)},
])
def test_invalid_fixed_policy_is_rejected_before_any_training(kwargs):
    x=np.ones((8,2))
    args={"factor_method":"mediencoder",**kwargs}
    with patch.object(estimation,"_split_indices_4fold") as split:
        with pytest.raises(ValueError):
            estimation.estimate_triply_IF(x,x,np.arange(8)%2,np.arange(8.),tilde_p=1,tilde_q=1,**args)
        split.assert_not_called()
