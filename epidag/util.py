import numpy as np
from numpy.random import uniform, choice
from scipy.misc import logsumexp
import math
import ast
import astunparse
import re

__author__ = 'TimeWz667'
__all__ = ['add_math_func', 'MATH_FUNC',
           'ScriptException', 'resample',
           'parse_parents', 'parse_function', 'parse_math_express']


MATH_FUNC = {
    'hypot': np.hypot,
    'exp': np.exp,
    'log': np.log,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'ceil': np.ceil,
    'floor': np.floor,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'erf': math.erf,
    'pow': math.pow
}


def add_math_func(fn_name, fn):
    if callable(fn):
        MATH_FUNC[fn_name] = fn


class ScriptException(Exception):
    def __init__(self, err):
        self.Err = err

    def __repr__(self):
        return self.Err


def resample(wts, hs, pars=None, log=True):
    size = len(wts)
    if log:
        wts = np.array(wts)
        lse = logsumexp(wts)
        wts -= lse
        sel = choice(size, size, True, np.exp(wts))
    else:
        lse = np.sum(wts)
        wts /= lse
        lse = np.log(lse)
        sel = choice(size, size, True, wts)
    if pars:
        return [hs[i] for i in sel], [pars[i] for i in sel], lse - np.log(size)
    else:
        return [hs[i] for i in sel], lse - np.log(size)


def find_ast_parents(seq_ast):
    v, f = set(), set()

    for s in ast.walk(seq_ast):
        if isinstance(s, ast.Name):
            v.add(s.id)
        elif isinstance(s, ast.Call):
            f.add(s.func.id)
    return v - f, f


def parse_parents(seq):
    return find_ast_parents(ast.parse(seq))


class MathExpression:
    def __init__(self, eq, var, func):
        self.Expression = eq
        self.Var = var
        self.Func = func

    def __call__(self, loc=None, glo=None):
        try:
            return self.execute(loc, glo)
        except NameError:
            return self.Expression

    def execute(self, loc=None, glo=None):
        return eval(self.Expression, loc, glo)

    def is_executable(self, loc):
        return all(v in loc for v in self.Var) and all(f in MATH_FUNC for f in self.Func)

    def __str__(self):
        return self.Expression

    __repr__ = __str__


def parse_math_express(seq):
    v, f = parse_parents(seq)
    return MathExpression(seq, v, f)


def ast_to_math_express(seq_ast, seq=None):
    v, f = find_ast_parents(seq_ast)
    seq = seq if seq else astunparse.unparse(seq_ast)[:-1]
    return MathExpression(seq, v, f)


class ParsedFunction:
    def __init__(self, src, func, args):
        self.Source = src
        self.Function = func
        self.Arguments = args

    def sort_arguments(self, order):
        args = dict()
        for i, arg in enumerate(self.Arguments):
            try:
                key = arg['key']
            except KeyError:
                arg['key'] = key = order[i]
            args[key] = arg
        self.Arguments = [args[key] for key in order if key in args]

    def get_arguments(self, loc=None, glo=None):
        args = list()
        for arg in self.Arguments:
            # todo opti
            arg = dict(arg)
            try:
                arg['value'] = arg['value'].execute(loc, glo)
            except NameError:
                raise NameError("Parent nodes are not fully defined")
            args.append(arg)
        return args

    def to_blueprint(self, name, loc=None, glo=None):
        return {
            'Name': name,
            'Type': self.Function,
            'Args': self.get_arguments(loc, glo)
        }

    def to_json(self, loc=None):
        return {
            'Source': self.Source,
            'Type': self.Function,
            'Args': [arg['value'](loc) for arg in self.Arguments]
        }

    def __str__(self):
        return self.Source

    __repr__ = __str__


def parse_function(seq):
    seq = re.sub(r'\s+', '', seq)
    try:
        seq_ast = ast.parse(seq)
    except SyntaxError:
        raise SyntaxError

    f, pars = None, list()
    start = False

    # extract arguments
    for s in ast.walk(seq_ast):
        if not start and isinstance(s, ast.Name):
            f = s.id
            start = True
        elif start:
            try:
                if isinstance(s, ast.keyword):
                    pars.append({
                        'key': s.arg,
                        'value': ast_to_math_express(s.value)
                    })
                else:
                    pars.append({'value': ast_to_math_express(s)})
            except AttributeError:
                break
    return ParsedFunction(seq, f, pars)


if __name__ == '__main__':

    print('Find parents')
    print(parse_parents('x+y/2 * max(z, 5)'), '\n')

    print('Math expression')
    me = parse_math_express('x+y/2 * max(z, 5)')
    print(me)
    print(me({'x': 2, 'y': 4, 'z': 10}), '\n')

    fn = parse_function('my_func(4*a, "k", k, t=5, s=False)')
    print(fn)
    print(fn.to_json())
    print(fn.to_json({'k': 7}))
    print(fn.to_json({'k': 7, 'a': 10}))
    fn.sort_arguments(['1', '2', '3', 's', 't'])
    print(fn.to_json(), '\n')

    print(resample([0, -1.2, -1], ['a', 'b', 'c']))
