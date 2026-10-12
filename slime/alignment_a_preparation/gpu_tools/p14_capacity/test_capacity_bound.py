import pytest
from capacity_bound import validate_bound


def test_cap16_conservative_bound():assert validate_bound(16,4,[4,4,4,4])==16

def test_cap8_requires_every_rank_at_most_two():
    assert validate_bound(8,4,[2,2,2,2])==8
    assert validate_bound(8,4,[0,1,2,0])==3
    with pytest.raises(ValueError):validate_bound(8,4,[3,0,0,0])


@pytest.mark.parametrize('cap,ep,rows',[(4,4,[1]*4),(8,8,[1]*8),(8,4,[2]*3),(8,4,[-1,2,2,2])])
def test_unproved_scope_rejected(cap,ep,rows):
    with pytest.raises(ValueError):validate_bound(cap,ep,rows)
