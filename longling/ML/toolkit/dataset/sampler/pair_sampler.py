# coding: utf-8
# create by tongshiwei on 2020-11-21

from collections import Iterable

import pandas as pd
from longling import as_list
from numpy.random import default_rng
from tqdm import tqdm

"""
rating matrix
(query, key, value): value could be omit

contrastive triplet
(query, pos, neg)
"""

__all__ = ["Sampler", "TripletPairSampler", "UserSpecificPairSampler", "ItemSpecificSampler"]


class Sampler(object):
    def __init__(self, random_state=10, pad_value=0):
        self.random_state = default_rng(random_state)
        self.pad_value = pad_value

    def padding(self, sampled: list, n, *args, **kwargs) -> list:
        pad_value = kwargs.get("pad_value", self.pad_value)
        return sampled + [pad_value] * (n - len(sampled))


class TripletPairSampler(Sampler):
    """
    Examples
    --------
    >>> # implicit feedback
    >>> import pandas as pd
    >>> triplet_df = pd.DataFrame({
    ...     "query": [0, 1, 2],
    ...     "pos": [[1], [3, 0, 2], [1]],
    ...     "neg": [[], [], []]
    ... })
    >>> sampler = TripletPairSampler(triplet_df, "query", set_index=True)
    >>> rating_matrix = pd.DataFrame({
    ...     "query": [0, 1, 1, 1, 2],
    ...     "key": [1, 3, 0, 2, 1]
    ... })
    >>> triplet_df = TripletPairSampler.rating2triplet(rating_matrix, query_field="query", key_field="key")
    >>> triplet_df   # doctest: +NORMALIZE_WHITESPACE
                 pos neg
    query
    0            [1]  []
    1      [3, 0, 2]  []
    2            [1]  []
    >>> sampler = TripletPairSampler(triplet_df, "query")
    >>> sampler(0)
    (0, [0])
    >>> sampler(0, 3)
    (0, [0, 0, 0])
    >>> sampler(0, 3, padding=False)
    (0, [])
    >>> sampler = TripletPairSampler(triplet_df, "query", query_range=3, key_range=4)
    >>> sampler(0)
    (0, [0])
    >>> sampler(0, 3)
    (0, [0, 0, 0])
    >>> sampler(0, 3, padding=False)
    (0, [])
    >>> sampler(0, 5, padding=False, implicit=True)
    (3, [2, 3, 0])
    >>> sampler(0, 5, padding=False, implicit=True, excluded_key=[3])
    (2, [0, 2])
    >>> sampler(0, 5, padding=True, implicit=True, excluded_key=[3])
    (2, [2, 0, 0, 0, 0])
    >>> sampler(0, 5, implicit=True, pad_value=-1)
    (3, [2, 3, 0, -1, -1])
    >>> sampler(0, 5, implicit=True, fast_implicit=True, pad_value=-1)
    (3, [0, 2, 3, -1, -1])
    >>> sampler(0, 5, implicit=True, fast_implicit=True, with_n_implicit=3, pad_value=-1)
    (3, [0, 2, 3, -1, -1, -1, -1, -1])
    >>> sampler(0, 5, implicit=True, fast_implicit=True, with_n_implicit=3, pad_value=-1, padding_implicit=True)
    (3, [0, 2, 3, -1, -1, -1, -1, -1])
    >>> rating_matrix = pd.DataFrame({
    ...     "query": [0, 1, 1, 1, 2],
    ...     "key": [1, 3, 0, 2, 1],
    ...     "score": [1, 0, 1, 1, 0]
    ... })
    >>> triplet_df = TripletPairSampler.rating2triplet(
    ...     rating_matrix,
    ...     "query", "key",
    ...     value_field="score"
    ... )
    >>> triplet_df   # doctest: +NORMALIZE_WHITESPACE
                pos  neg
    query
    0           [1]   []
    1        [0, 2]  [3]
    2            []  [1]
    >>> sampler = TripletPairSampler(triplet_df, "query", query_range=3, key_range=4)
    >>> sampler([0, 1, 2], 5, implicit=True, pad_value=-1)
    [(3, [2, 3, 0, -1, -1]), (1, [1, -1, -1, -1, -1]), (3, [3, 0, 2, -1, -1])]
    >>> sampler([0, 1, 2], 5, pad_value=-1)
    [(0, [-1, -1, -1, -1, -1]), (1, [3, -1, -1, -1, -1]), (1, [1, -1, -1, -1, -1])]
    >>> sampler([0, 1, 2], 5, neg=False, pad_value=-1)
    [(1, [1, -1, -1, -1, -1]), (2, [0, 2, -1, -1, -1]), (0, [-1, -1, -1, -1, -1])]
    >>> sampler(rating_matrix["query"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["key"], pad_value=-1)
    [(0, [-1, -1]), (2, [2, 0]), (1, [3, -1]), (1, [3, -1]), (0, [-1, -1])]
    >>> sampler(rating_matrix["query"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["key"], pad_value=-1, return_column=True)
    ((0, 2, 1, 1, 0), ([-1, -1], [0, 2], [3, -1], [3, -1], [-1, -1]))
    >>> sampler(rating_matrix["query"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["key"], pad_value=-1, return_column=True, split_sample_to_column=True)
    ((0, 2, 1, 1, 0), [(-1, 0, 3, 3, -1), (-1, 2, -1, -1, -1)])
    >>> rating_matrix = pd.DataFrame({
    ...     "query": [0, 1, 1, 1, 2],
    ...     "key": [1, 3, 0, 2, 1],
    ...     "score": [0.8, 0.4, 0.7, 0.5, 0.1]
    ... })
    >>> TripletPairSampler.rating2triplet(
    ...     rating_matrix,
    ...     "query", "key",
    ...     value_field="score",
    ...     value_threshold=0.5
    ... )   # doctest: +NORMALIZE_WHITESPACE
              pos  neg
    query
    0         [1]   []
    1      [0, 2]  [3]
    2          []  [1]
    """

    def __init__(self, triplet_df: pd.DataFrame,
                 query_field, pos_field="pos", neg_field="neg", set_index=False,
                 query_range: (int, tuple, list) = None, key_range: (int, tuple, list) = None,
                 random_state=10):
        super(TripletPairSampler, self).__init__(random_state)
        self.df = triplet_df
        self.query_field = query_field
        self.pos_field = pos_field
        self.neg_field = neg_field

        def _get_range(value: (None, int, tuple, list)) -> tuple:
            if isinstance(value, int):
                return 0, value
            else:
                return value

        self.query_range = _get_range(query_range)
        self.key_range = _get_range(key_range)
        if set_index:
            self.df.set_index(self.query_field, inplace=True)

    def __call__(self, query: (int, str, list), n=1, excluded_key=None, neg=True, implicit=False, padding=True,
                 *args, verbose=True, return_column=False, split_sample_to_column=False,
                 padding_implicit=False, fast_implicit=False, with_n_implicit=None, **kwargs):
        """
        To get only explicit neg (pos) samples, set implicit as False, padding as False,
        padding_implicit as False and with_n_implicit as None.

        To get at maximum of n samples, where the insufficient ones are implicit sampled,
        set padding and padding_implicit as True, and with_n_implicit as None.

        To get explicit samples with at maximum of n implicit samples,
        set implicit as False, with_n_implicit as n.

        Parameters
        ----------
        query
        n
        excluded_key
        neg: bool
            negative sampling (set neg as True) or positive sampling (set neg as False)
        implicit: bool
            explicit sampling (set implicit as False) or implicit sampling (set implicit as True)
        padding
        args
        verbose
        return_column
        split_sample_to_column
        padding_implicit
        fast_implicit
        with_n_implicit
        kwargs

        Returns
        -------

        """
        if isinstance(query, (int, str)):
            sample = self.sample(
                query=query,
                n=n,
                excluded_key=excluded_key,
                neg=neg
            ) if not implicit else self.implicit_sample(
                query=query,
                n=n,
                excluded_key=excluded_key,
                fast_mode=fast_implicit
            )
            n_sample = len(sample)
            padding_n = n
            if with_n_implicit is not None:
                padding_n += with_n_implicit
                implicit_sample = self.implicit_sample(
                    query=query, n=with_n_implicit, excluded_key=excluded_key,
                    samples=sample,
                    fast_mode=fast_implicit)
                n_sample += len(implicit_sample)
                sample += implicit_sample

            if padding is True:
                if padding_implicit is True:
                    if n_sample < padding_n:
                        padding_sample = self.implicit_sample(query=query, n=padding_n - n_sample,
                                                              excluded_key=excluded_key, samples=sample,
                                                              fast_mode=fast_implicit)
                        n_padding_sample = len(padding_sample)
                        return n_sample + n_padding_sample, self.padding(sample + padding_sample, padding_n, **kwargs)
                else:
                    return n_sample, self.padding(sample, padding_n, **kwargs)

            return n_sample, sample
        else:
            neg = neg if isinstance(neg, Iterable) else [neg] * len(query)
            excluded_key = excluded_key if isinstance(excluded_key, Iterable) else [excluded_key] * len(query)
            n_and_sample = [
                self(
                    _query, n, _excluded_key, _neg, implicit, padding,
                    *args,
                    padding_implicit=padding_implicit,
                    fast_implicit=fast_implicit,
                    with_n_implicit=with_n_implicit,
                    **kwargs)
                for _query, _excluded_key, _neg in tqdm(
                    zip(query, excluded_key, neg), "sampling", disable=not verbose
                )
            ]
            if return_column:
                n_sample, sample = zip(*n_and_sample)
                if split_sample_to_column:
                    return n_sample, list(zip(*sample))
                return n_sample, sample
            return n_and_sample

    def sample(self, query: (int, str, list), n=1, excluded_key=None, neg=True, *args, **kwargs):
        candidates = self.df.loc[query][self.neg_field] if neg else self.df.loc[query][self.pos_field]

        if excluded_key is not None:
            candidates = list(set(candidates) - set(as_list(excluded_key)))

        sampled = self.random_state.choice(candidates, min(n, len(candidates)), replace=False).tolist()
        # rs = default_rng(10)
        # sampled = rs.choice(candidates, min(n, len(candidates)), replace=False).tolist()
        return sampled

    def implicit_sample(self, query: (int, str, list), n=1, excluded_key=None, fast_mode=False, samples=None,
                        *args, fast_max_try=100, **kwargs):
        exclude = set(self.df.loc[query][self.pos_field]) | set(self.df.loc[query][self.neg_field])
        if excluded_key is not None:
            exclude |= set(as_list(excluded_key))
        if samples is not None:
            exclude |= set(samples)
        if fast_mode is False:
            candidates = list(set(range(*self.key_range)) - exclude)
            sampled = self.random_state.choice(candidates, min(n, len(candidates)), replace=False).tolist()
            return sampled
        else:
            sampled = set()
            try_cnt = 0
            while len(sampled) < n and try_cnt < n + fast_max_try:
                _sample = self.random_state.integers(*self.key_range)
                if _sample not in exclude and _sample not in sampled:
                    sampled.add(_sample)
                try_cnt += 1
            return list(sampled)

    @staticmethod
    def rating2triplet(rating_matrix: pd.DataFrame, query_field, key_field, value_field=None,
                       value_threshold=None,
                       pos_field="pos", neg_field="neg",
                       *args, verbose=True, **kwargs):
        if isinstance(rating_matrix, pd.DataFrame):
            triplet = []
            for _, group in tqdm(rating_matrix.groupby(query_field), "rating2triplet", disable=not verbose):
                query_key = group[query_field].unique()[0]
                if value_field is None:
                    pos = group[key_field]
                    neg = pd.Series([], dtype=pos.dtype)
                elif value_threshold:
                    pos = group[group[value_field] >= value_threshold][key_field]
                    neg = group[group[value_field] < value_threshold][key_field]
                else:
                    pos = group[group[value_field] == 1][key_field]
                    neg = group[group[value_field] == 0][key_field]
                triplet.append([query_key, pos.tolist(), neg.tolist()])
        else:
            raise TypeError()

        df = pd.DataFrame(triplet, columns=[query_field, pos_field, neg_field])
        return df.set_index(query_field)


class UserSpecificPairSampler(TripletPairSampler):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> user_num = 3
    >>> item_num = 4
    >>> rating_matrix = pd.DataFrame({
    ...     "user_id": [0, 1, 1, 1, 2],
    ...     "item_id": [1, 3, 0, 2, 1]
    ... })
    >>> triplet_df = UserSpecificPairSampler.rating2triplet(rating_matrix)
    >>> triplet_df   # doctest: +NORMALIZE_WHITESPACE
                   pos neg
    user_id
    0              [1]  []
    1        [3, 0, 2]  []
    2              [1]  []
    >>> sampler = UserSpecificPairSampler(triplet_df)
    >>> sampler(1)
    (0, [0])
    >>> sampler = UserSpecificPairSampler(triplet_df, item_id_range=item_num)
    >>> sampler(0, implicit=True)
    (1, [3])
    >>> sampler(0, 5, implicit=True)
    (3, [3, 2, 0, 0, 0])
    >>> sampler(0, 5, implicit=True, pad_value=-1)
    (3, [3, 2, 0, -1, -1])
    >>> sampler([0, 1, 2], 5, implicit=True, pad_value=-1)
    [(3, [2, 3, 0, -1, -1]), (1, [1, -1, -1, -1, -1]), (3, [2, 0, 3, -1, -1])]
    >>> rating_matrix = pd.DataFrame({
    ...     "user_id": [0, 1, 1, 1, 2],
    ...     "item_id": [1, 3, 0, 2, 1],
    ...     "score": [1, 0, 1, 1, 0]
    ... })
    >>> triplet_df = UserSpecificPairSampler.rating2triplet(rating_matrix=rating_matrix, value_field="score")
    >>> triplet_df   # doctest: +NORMALIZE_WHITESPACE
                pos  neg
    user_id
    0           [1]   []
    1        [0, 2]  [3]
    2            []  [1]
    >>> sampler = UserSpecificPairSampler(triplet_df)
    >>> sampler([0, 1, 2], 5, pad_value=-1)
    [(0, [-1, -1, -1, -1, -1]), (1, [3, -1, -1, -1, -1]), (1, [1, -1, -1, -1, -1])]
    >>> sampler([0, 1, 2], 5, neg=False, pad_value=-1)
    [(1, [1, -1, -1, -1, -1]), (2, [0, 2, -1, -1, -1]), (0, [-1, -1, -1, -1, -1])]
    >>> sampler(rating_matrix["user_id"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["item_id"], pad_value=-1)
    [(0, [-1, -1]), (2, [2, 0]), (1, [3, -1]), (1, [3, -1]), (0, [-1, -1])]
    >>> sampler(rating_matrix["user_id"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["item_id"], pad_value=-1, return_column=True)
    ((0, 2, 1, 1, 0), ([-1, -1], [0, 2], [3, -1], [3, -1], [-1, -1]))
    >>> sampler(rating_matrix["user_id"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["item_id"], pad_value=-1, return_column=True, split_sample_to_column=True)
    ((0, 2, 1, 1, 0), [(-1, 2, 3, 3, -1), (-1, 0, -1, -1, -1)])
    """

    def __init__(self, triplet_df: pd.DataFrame,
                 query_field="user_id", pos_field="pos", neg_field="neg", set_index=False,
                 user_id_range=None, item_id_range=None,
                 random_state=10):
        super(UserSpecificPairSampler, self).__init__(
            triplet_df=triplet_df,
            query_field=query_field,
            pos_field=pos_field,
            neg_field=neg_field,
            set_index=set_index,
            query_range=user_id_range,
            key_range=item_id_range,
            random_state=random_state)

    @staticmethod
    def rating2triplet(rating_matrix: pd.DataFrame,
                       query_field="user_id", key_field="item_id", value_field=None,
                       value_threshold=None, pos_field="pos", neg_field="neg",
                       *args, verbose=True, **kwargs):
        return TripletPairSampler.rating2triplet(
            rating_matrix=rating_matrix,
            query_field=query_field,
            key_field=key_field,
            value_field=value_field,
            value_threshold=value_threshold,
            pos_field=pos_field,
            neg_field=neg_field,
            *args,
            verbose=verbose,
            **kwargs
        )


class ItemSpecificSampler(TripletPairSampler):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> user_num = 3
    >>> item_num = 4
    >>> rating_matrix = pd.DataFrame({
    ...     "user_id": [0, 1, 1, 1, 2],
    ...     "item_id": [1, 3, 0, 2, 1]
    ... })
    >>> triplet_df = ItemSpecificSampler.rating2triplet(rating_matrix)
    >>> triplet_df   # doctest: +NORMALIZE_WHITESPACE
                pos neg
    item_id
    0           [1]  []
    1        [0, 2]  []
    2           [1]  []
    3           [1]  []
    >>> triplet_df.index
    Int64Index([0, 1, 2, 3], dtype='int64', name='item_id')
    >>> sampler = ItemSpecificSampler(triplet_df)
    >>> sampler(1)
    (0, [0])
    >>> sampler = ItemSpecificSampler(triplet_df, user_id_range=user_num)
    >>> sampler(0, implicit=True)
    (1, [2])
    >>> sampler(0, 5, implicit=True)
    (2, [2, 0, 0, 0, 0])
    >>> sampler(0, 5, implicit=True, pad_value=-1)
    (2, [0, 2, -1, -1, -1])
    >>> sampler([0, 1, 2], 5, implicit=True, pad_value=-1)
    [(2, [0, 2, -1, -1, -1]), (1, [1, -1, -1, -1, -1]), (2, [0, 2, -1, -1, -1])]
    >>> rating_matrix = pd.DataFrame({
    ...     "user_id": [0, 1, 1, 1, 2],
    ...     "item_id": [1, 3, 0, 2, 1],
    ...     "score": [1, 0, 1, 1, 0]
    ... })
    >>> triplet_df = ItemSpecificSampler.rating2triplet(rating_matrix=rating_matrix, value_field="score")
    >>> triplet_df   # doctest: +NORMALIZE_WHITESPACE
             pos  neg
    item_id
    0        [1]   []
    1        [0]  [2]
    2        [1]   []
    3         []  [1]
    >>> sampler = UserSpecificPairSampler(triplet_df)
    >>> sampler([0, 1, 2], 5, pad_value=-1)
    [(0, [-1, -1, -1, -1, -1]), (1, [2, -1, -1, -1, -1]), (0, [-1, -1, -1, -1, -1])]
    >>> sampler([0, 1, 2], 5, neg=False, pad_value=-1)
    [(1, [1, -1, -1, -1, -1]), (1, [0, -1, -1, -1, -1]), (1, [1, -1, -1, -1, -1])]
    >>> sampler(rating_matrix["item_id"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["user_id"], pad_value=-1)
    [(1, [2, -1]), (0, [-1, -1]), (0, [-1, -1]), (0, [-1, -1]), (1, [0, -1])]
    >>> sampler(rating_matrix["item_id"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["user_id"], pad_value=-1, return_column=True)
    ((1, 0, 0, 0, 1), ([2, -1], [-1, -1], [-1, -1], [-1, -1], [0, -1]))
    >>> sampler(rating_matrix["item_id"], 2, neg=rating_matrix["score"],
    ...     excluded_key=rating_matrix["user_id"], pad_value=-1, return_column=True, split_sample_to_column=True)
    ((1, 0, 0, 0, 1), [(2, -1, -1, -1, 0), (-1, -1, -1, -1, -1)])
    """

    def __init__(self, triplet_df: pd.DataFrame,
                 query_field="item_id", pos_field="pos", neg_field="neg", set_index=False,
                 item_id_range=None, user_id_range=None,
                 random_state=10):
        super(ItemSpecificSampler, self).__init__(
            triplet_df=triplet_df,
            query_field=query_field,
            pos_field=pos_field,
            neg_field=neg_field,
            set_index=set_index,
            query_range=item_id_range,
            key_range=user_id_range,
            random_state=random_state)

    @staticmethod
    def rating2triplet(rating_matrix: pd.DataFrame,
                       query_field="item_id", key_field="user_id", value_field=None,
                       value_threshold=None, pos_field="pos", neg_field="neg",
                       *args, verbose=True, **kwargs):
        return TripletPairSampler.rating2triplet(
            rating_matrix=rating_matrix,
            query_field=query_field,
            key_field=key_field,
            value_field=value_field,
            value_threshold=value_threshold,
            pos_field=pos_field,
            neg_field=neg_field,
            *args,
            verbose=verbose,
            **kwargs
        )
