# coding: utf-8
# 2019/12/20 @

import itertools
from longling import as_list
from longling.lib.candylib import dict2pv, list2dict, get_dict_by_path


def extract_params_combinations(candidates: dict, external=None):
    """
    >>> candidates = {'b': [1, 2], 'c': [0, 3], 'd': '$b'}
    >>> list(extract_params_combinations(candidates))
    [{'b': 1, 'c': 0, 'd': 1}, {'b': 1, 'c': 3, 'd': 1}, {'b': 2, 'c': 0, 'd': 2}, {'b': 2, 'c': 3, 'd': 2}]
    >>> candidates = {'a': [1, 2], 'b': '$c'}
    >>> external = {'c': 3}
    >>> list(extract_params_combinations(candidates, external))
    [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
    """
    external = {} if external is None else external
    params_paths, params_values = dict2pv(candidates)
    params_values = [as_list(value) for value in params_values]

    for params in itertools.product(*params_values):
        _params = {}

        for p, v in zip(params_paths, params):
            list2dict(p, v, _params)

        for p, v in zip(params_paths, params):
            if isinstance(v, str) and v[0] == '$':
                map_key_path = v.lstrip('$').split(":")
                _dict_obj = get_dict_by_path(_params, p[:-1])
                for map_dict in [_params, external]:
                    try:
                        _map_dict_obj = get_dict_by_path(map_dict, map_key_path)
                        _dict_obj[p[-1]] = _map_dict_obj
                        break
                    except KeyError:
                        try:
                            _map_dict_obj = get_dict_by_path(map_dict, p[:-1] + map_key_path)
                            _dict_obj[p[-1]] = _map_dict_obj
                            break
                        except KeyError:
                            pass

                else:
                    raise KeyError(
                        "The mapped key should be in either candidates or external, but cannot find %s" % v
                    )

        yield _params
