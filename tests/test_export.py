import numpy as np
import pytest
from rawfeat.export import verify_discrete_case


def features(points, scores=None):
    return {**{f'points_{v}': np.array(points, dtype=np.float32) for v in ('a','b')},
            **{f'descriptors_{v}': np.eye(len(points), dtype=np.float32) for v in ('a','b')},
            **{f'match_indices_{v}': np.arange(len(points)) for v in ('a','b')}}


def test_fusion_accepts_measured_near_tie_order_and_rejects_large_inversion():
    a=features([[8,8],[20,20]]);b={k: v[::-1].copy() if k.startswith(('points','descriptors')) else v.copy() for k,v in a.items()}
    old=np.zeros((2,32,32),np.float32);new=old.copy()
    old[:,8,8]=.02+1e-8;old[:,20,20]=.02
    new[:,8,8]=.02;new[:,20,20]=.02+1e-8
    r=verify_discrete_case(a,b,old,new)
    assert r['point_order_changed'] and not r['point_set_changed'] and not r['match_coordinate_set_changed']
    old[:,8,8]=.03;new=old.copy()
    with pytest.raises(RuntimeError,match='non-tied score ranking'):verify_discrete_case(a,b,old,new)


def test_fusion_accepts_neighbor_nms_tie_but_rejects_unexplained_shift():
    a=features([[8,8]]);b=features([[8,9]])
    old=np.zeros((2,32,32),np.float32);new=old.copy()
    old[:,8,8]=.02+1e-8;old[:,9,8]=.02
    new[:,8,8]=.02;new[:,9,8]=.02+1e-8
    assert verify_discrete_case(a,b,old,new)['point_set_changed']
    old[:,8,8]=.03;new=old.copy()
    with pytest.raises(RuntimeError,match='away from NMS'):verify_discrete_case(a,b,old,new)
