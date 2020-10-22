# coding: utf-8
# create by tongshiwei on 2020-10-13

import pytest

from longling.DM import RankGaussianNormalizer


def test_rank_gaussian_normalizer():
    rgn = RankGaussianNormalizer()

    with pytest.raises(ValueError):
        rgn.precision = "error"

    rgn.fit([])

    rgn = RankGaussianNormalizer()

    with pytest.raises(ValueError):
        rgn._normal_cdf_inverse(-0.1)

    assert rgn._vd_erf_inv_single_01(0) == 0
    rgn._vd_erf_inv_single_01(-0.9)
