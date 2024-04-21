# coding: utf-8
# create by tongshiwei on 2020-10-13

import pytest

from longling.dm import RankGaussianNormalizer


def test_rank_gaussian_normalizer():
    rgn = RankGaussianNormalizer()

    with pytest.raises(ValueError, match='precision must be a data type, e.g.: np.float64'):
        rgn.precision = "error"

    rgn.fit([])

    rgn = RankGaussianNormalizer()

    with pytest.raises(ValueError, match="0 < p < 1. The value of p was: -0.1"):
        rgn._normal_cdf_inverse(-0.1)

    assert rgn._vd_erf_inv_single_01(0) == 0
    rgn._vd_erf_inv_single_01(-0.9)
