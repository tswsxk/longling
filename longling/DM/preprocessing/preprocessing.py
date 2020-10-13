# coding: utf-8
# 2020/8/14 @ tongshiwei
from collections import Counter, OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

__all__ = ["RankGaussianNormalizer", "rank_gaussian_normalize"]


class NPRankGaussianNormalizer(BaseEstimator, TransformerMixin):  # pragma: no cover
    """
    Rank Gaussian Normalization

    time-consuming, deprecated
    """

    def __init__(self, precision=np.float32):
        # data: 1D array or list
        self.precision = precision
        self._trans_map = None

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, p):
        if not isinstance(p, type):
            raise ValueError('precision must be a data type, e.g.: np.float64')
        self._precision = p

    @staticmethod
    def _binary_search(keys, val):
        start, end = 0, len(keys) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if val < keys[mid]:
                end = mid
            else:
                start = mid
        return keys[start], keys[end]

    @staticmethod
    def _rational_approximation(t: float) -> float:
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)

    def _normal_cdf_inverse(self, p: np.ndarray):
        assert ((0.0 < p) & (p < 1.0)).all(), "The value of p should be in (0, 1)"
        ret = np.zeros_like(p)
        ret[p < 0.5] = -self._rational_approximation(np.sqrt(-2.0 * np.log(p[p < 0.5])))
        ret[p >= 0.5] = self._rational_approximation(np.sqrt(-2.0 * np.log(1 - p[p >= 0.5])))
        return ret

    def _vd_erf_inv(self, x: np.ndarray):
        ret = np.zeros_like(x)
        ret[x < 0] = - self._normal_cdf_inverse(-x[x < 0]) * 0.7
        ret[x > 0] = self._normal_cdf_inverse(x[x > 0]) * 0.7
        return ret

    @staticmethod
    def to_array(X, one_column):
        if not one_column:
            return check_array(X)
        else:
            return np.asarray(X)

    def fit(self, X, y=None, one_column=False):
        X = self.to_array(X, one_column)

        trans_map = OrderedDict()
        hist = Counter(X)
        if len(hist) == 0:
            pass
        elif len(hist) == 1:
            key = list(hist.keys())[0]
            trans_map[key] = 0.0
        elif len(hist) == 2:
            keys = sorted(list(hist.keys()))
            trans_map[keys[0]] = 0.0
            trans_map[keys[1]] = 1.0
        else:
            keys = list(sorted(hist.keys()))
            values = [hist[k] for k in keys]

            cnt = np.cumsum([0] + values[:-1])
            rank_v = self._vd_erf_inv(cnt / len(X) * 0.998 + 1e-3)
            assert ((-3.0 <= rank_v) & (rank_v <= 3.0)).all()
            mean = sum(np.asarray(values) * rank_v) / len(X)
            for i, key in enumerate(keys):
                trans_map[key] = rank_v[i] - mean

        self._trans_map = trans_map

    def transform(self, X, y=None, one_column=False):
        X = self.to_array(X, one_column)

        trans_map = self._trans_map
        keys = np.asarray(list(trans_map.keys()))
        if len(keys) == 0:
            raise Exception('No transformation map')

        ret = np.zeros_like(X)

        indices = np.digitize(X, keys)
        ret[indices == 0] = trans_map[keys[0]]
        ret[indices == len(keys)] = trans_map[keys[-1]]
        legal_indices = (0 < indices) & (indices < len(keys))
        x1 = keys[indices[legal_indices] - 1]
        x2 = keys[indices[legal_indices]]
        y1 = np.asarray([trans_map[index] for index in x1])
        y2 = np.asarray([trans_map[index] for index in x2])
        ret[legal_indices] = y1 + (X[legal_indices] - x1) * (y2 - y1) / (x2 - x1)

        data_out = np.asarray(ret, dtype=self.precision)
        return data_out

    def fit_transform(self, X, y=None, one_column=False, **fit_params):
        self.fit(X, one_column=one_column)
        return self.transform(X, one_column=one_column)


class RankGaussianNormalizer(BaseEstimator, TransformerMixin):
    """
    Rank Gaussian Normalization

    Examples
    --------
    >>> import numpy as np
    >>> array = [0, 2, 1, 3]
    >>> rgn = RankGaussianNormalizer()
    >>> np.around(rgn.fit_transform(array), 3)
    array([-1.623,  0.541,  0.07 ,  1.012], dtype=float32)
    >>> np.around(rgn.transform([0, 1, 2, 3]), 3)
    array([-1.623,  0.07 ,  0.541,  1.012], dtype=float32)
    """

    def __init__(self, precision=np.float32):
        # data: 1D array or list
        self.precision = precision
        self._trans_map = None

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, p):
        if not isinstance(p, type):
            raise ValueError('precision must be a data type, e.g.: np.float64')
        self._precision = p

    @staticmethod
    def _binary_search(keys, val):
        """

        Parameters
        ----------
        keys
        val

        Returns
        -------

        Examples
        --------
        >>> RankGaussianNormalizer._binary_search([1, 2, 3, 4, 5], 3)
        (3, 4)
        """
        start, end = 0, len(keys) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if val < keys[mid]:
                end = mid
            else:
                start = mid
        return keys[start], keys[end]

    @staticmethod
    def _rational_approximation(t: float) -> float:
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)

    def _normal_cdf_inverse(self, p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            raise ValueError('0 < p < 1. The value of p was: {}'.format(p))
        if p < 0.5:
            return -self._rational_approximation(np.sqrt(-2.0 * np.log(p)))
        return self._rational_approximation(np.sqrt(-2.0 * np.log(1 - p)))

    def _vd_erf_inv_single_01(self, x: float) -> float:
        if x == 0:
            return 0
        elif x < 0:
            return -self._normal_cdf_inverse(-x) * 0.7
        else:
            return self._normal_cdf_inverse(x) * 0.7

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None, **kwargs):
        # X = check_array(X)

        trans_map = OrderedDict()
        hist = Counter(X)
        if len(hist) == 0:
            pass
        elif len(hist) == 1:
            key = list(hist.keys())[0]
            trans_map[key] = 0.0
        elif len(hist) == 2:
            keys = sorted(list(hist.keys()))
            trans_map[keys[0]] = 0.0
            trans_map[keys[1]] = 1.0
        else:
            N = cnt = 0
            for it in hist:
                N += hist[it]
            assert (N == len(X))
            mean = 0.0
            for it in sorted(list(hist.keys())):
                rank_v = cnt / N
                rank_v = rank_v * 0.998 + 1e-3
                rank_v = self._vd_erf_inv_single_01(rank_v)
                assert -3.0 <= rank_v <= 3.0
                mean += hist[it] * rank_v
                trans_map[it] = rank_v
                cnt += hist[it]
            mean /= N
            for it in trans_map:
                trans_map[it] -= mean
        self._trans_map = trans_map

    def transform(self, X):
        # X = check_array(X)

        data_out = []
        transform_map = self._trans_map
        keys = list(transform_map.keys())
        if len(keys) == 0:
            raise Exception('No transformation map')
        for i in range(len(X)):
            val = X[i]
            if val <= keys[0]:
                trans_val = transform_map[keys[0]]
            elif val >= keys[-1]:
                trans_val = transform_map[keys[-1]]
            elif val in transform_map:
                trans_val = transform_map[val]
            else:
                lower_key, upper_key = self._binary_search(keys, val)
                x1, y1 = lower_key, transform_map[lower_key]
                x2, y2 = upper_key, transform_map[upper_key]

                trans_val = y1 + (val - x1) * (y2 - y1) / (x2 - x1)
            data_out.append(trans_val)
        data_out = np.asarray(data_out, dtype=self.precision)
        return data_out


def rank_gaussian_normalize(array, y=None):
    """

    Parameters
    ----------
    array
    y

    Returns
    -------

    Examples
    --------
    >>> array = [
    ...     -19.9378, 10.5341, -32.4515, 33.0969, 24.3530, -1.1830, -1.4106, -4.9431,
    ...     14.2153, 26.3700, -7.6760, 60.3346, 36.2992, -126.8806, 14.2488, -5.0821,
    ...     1.6958, -21.2168, -49.1075, -8.3084, -1.5748, 3.7900, -2.1561, 4.0756,
    ...     -9.0289, -13.9533, -9.8466, 79.5876, -13.3332, -111.9568, -24.2531, 120.1174
    ... ]
    >>> import numpy as np
    >>> np.around(rank_gaussian_normalize(array), 3)
    array([-0.552,  0.409, -0.852,  0.773,  0.61 ,  0.177,  0.122, -0.042,
            0.472,  0.687, -0.155,  0.987,  0.87 , -2.096,  0.538, -0.098,
            0.233, -0.637, -1.002, -0.213,  0.068,  0.29 ,  0.013,  0.348,
           -0.274, -0.474, -0.337,  1.137, -0.403, -1.227, -0.735,  1.363],
          dtype=float32)
    >>> np.around(rank_gaussian_normalize([1]), 3)
    array([0.], dtype=float32)
    >>> np.around(rank_gaussian_normalize([1, 2]), 3)
    array([0., 1.], dtype=float32)
    """
    return RankGaussianNormalizer().fit_transform(array, y)


if __name__ == '__main__':
    data = [-19.9378, 10.5341, -32.4515, 33.0969, 24.3530, -1.1830, -1.4106, -4.9431,
            14.2153, 26.3700, -7.6760, 60.3346, 36.2992, -126.8806, 14.2488, -5.0821,
            1.6958, -21.2168, -49.1075, -8.3084, -1.5748, 3.7900, -2.1561, 4.0756,
            -9.0289, -13.9533, -9.8466, 79.5876, -13.3332, -111.9568, -24.2531, 120.1174] * 100

    from longling import print_time

    rgn = NPRankGaussianNormalizer()
    with print_time("test new"):
        for _ in range(1000):
            rgn.fit_transform(data, one_column=True).tolist()
    rgn = RankGaussianNormalizer()
    with print_time("test old"):
        for _ in range(1000):
            rgn.fit_transform(data).tolist()
