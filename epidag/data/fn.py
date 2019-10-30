__author__ = 'TimeWz667'
__all__ = ['add_data_fn', 'find_data_sampler']


DATA_FN = dict()


def add_data_fn(fn_name, fn):
    assert callable(fn) or callable(fn.get_sampler)
    assert fn_name not in DATA_FN
    DATA_FN[fn_name] = fn


def find_data_sampler(fn_name, loc):
    fn = DATA_FN[fn_name]
    return fn.get_sampler(loc)
