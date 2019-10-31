__author__ = 'TimeWz667'
__all__ = ['add_data_func', 'find_data_sampler']


DATA_FUNC = dict()


def add_data_func(fn_name, fn):
    assert callable(fn) or callable(fn.get_sampler)
    assert fn_name not in DATA_FUNC
    DATA_FUNC[fn_name] = fn


def find_data_sampler(fn_name, loc):
    fn = DATA_FUNC[fn_name]
    return fn.get_sampler(loc)
