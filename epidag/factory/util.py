import re

__author__ = 'TimeWz667'


def parse_function(fn, env=None, loc=None):
    # todo to use ast
    fn = fn.replace(' ', '')
    mat = re.match(r'(\w+)\((\S+)\)', fn)
    if mat is None:
        raise ValueError

    f, pars = mat.group(1), mat.group(2)

    args = list()
    kw_lock = False
    pars = ','+pars
    while len(pars):
        pars = pars[1:]
        mat = re.match(r'(\w+)=', pars)
        key = None
        if mat:
            kw_lock = True
            key = mat.group(1)
            pars = re.sub(r'\w+=', '', pars, 1)
        else:
            if kw_lock:
                raise ValueError

        # parse value
        for pat in [r'{\S+}', r'\[\S+]', r'[\w\*\+\-\^\/\(\)\*\.]+']:
            mat = re.match(pat, pars)
            if mat:
                try:
                    value = eval(mat.group(0), env, loc)
                except NameError:
                    raise ValueError
                pars = re.sub(pat, '', pars, 1)
                break
        args.append((key, value) if key else value)
    return f, args


if __name__ == '__main__':
    print(parse_function('exp(0.4)'))
    print(parse_function('gamma(0.4, 1/1000)'))
    print(parse_function('gamma(0.4, v=1/1000)'))
    print(parse_function('cat(kv={"a": 4, "b": 3})'))
    print(parse_function('list(vs=[2, "hh", 4])'))
