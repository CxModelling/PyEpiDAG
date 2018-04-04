import numpy as np
from numpy.random import choice
import scipy.special as sp
import math
import ast
import astunparse
import re

__author__ = 'TimeWz667'
__all__ = ['add_math_func', 'MATH_FUNC',
           'ScriptException', 'resample',
           'parse_parents', 'parse_math_expression',
           'parse_function', 'evaluate_function']


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
    'pow': math.pow,
    'logit': sp.logit,
    'expit': sp.expit
}


def add_math_func(fn_name, fn):
    if callable(fn):
        MATH_FUNC[fn_name] = fn


class ScriptException(Exception):
    def __init__(self, err):
        self.Err = err

    def __repr__(self):
        return self.Err


def resample(wts, hs, pars=None, log=True, new_size=None):
    size = len(wts)
    new_size = max(new_size, 1) if new_size else size

    if log:
        wts = np.array(wts)
        lse = sp.logsumexp(wts)
        wts -= lse
        sel = choice(size, new_size, replace=True, p=np.exp(wts))
    else:
        lse = np.sum(wts)
        wts /= lse
        lse = np.log(lse)
        sel = choice(size, new_size, replace=True, p=wts)
    if pars:
        return [hs[i] for i in sel], [pars[i] for i in sel], lse - np.log(new_size)
    else:
        return [hs[i] for i in sel], lse - np.log(new_size)


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

    def __call__(self, loc=None):
        try:
            return self.execute(loc)
        except NameError:
            return self.Expression

    def execute(self, loc=None):
        loc = dict(loc) if loc else dict()
        return eval(self.Expression, MATH_FUNC, loc)

    @property
    def Parents(self):
        return self.Var

    def is_executable(self, loc):
        return all(v in loc for v in self.Var) and all(f in MATH_FUNC for f in self.Func)

    def __str__(self):
        return self.Expression

    __repr__ = __str__


def parse_math_expression(seq):
    v, f = parse_parents(seq)
    return MathExpression(seq, v, f)


def ast_to_math_expression(seq_ast, seq=None):
    v, f = find_ast_parents(seq_ast)
    seq = seq if seq else astunparse.unparse(seq_ast)[:-1]
    return MathExpression(seq, v, f)


class ParsedFunction:
    def __init__(self, src, func, args):
        self.Source = src
        self.Function = func
        self.Arguments = args

    def get_arguments(self, loc=None):
        args = list()
        for arg in self.Arguments:
            # todo opti
            arg = dict(arg)
            try:
                arg['value'] = arg['value'].execute(loc)
            except NameError:
                raise NameError("Parent nodes are not fully defined")
            args.append(arg)
        return args

    def to_blueprint(self, name, loc=None):
        return {
            'Name': name,
            'Type': self.Function,
            'Args': self.get_arguments(loc)
        }

    @property
    def Parents(self):
        return set.union(*[arg['value'].Var for arg in self.Arguments])

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
    keylock = False
    # extract arguments
    for s in ast.walk(seq_ast):
        if not start and isinstance(s, ast.Name):
            f = s.id
            start = True
        elif start:
            try:
                if isinstance(s, ast.Load):
                    break
                elif isinstance(s, ast.keyword):
                    pars.append({
                        'key': s.arg,
                        'value': ast_to_math_expression(s.value)
                    })
                    keylock = True
                elif not keylock:
                    pars.append({'value': ast_to_math_expression(s)})

            except AttributeError:
                pass
    return ParsedFunction(seq, f, pars)


class EvaluatedFunction:
    def __init__(self, src, func, args):
        self.Source = src
        self.Function = func
        self.Arguments = args

    def to_blueprint(self, name):
        return {
            'Name': name,
            'Type': self.Function,
            'Args': self.Arguments
        }

    def to_json(self):
        return {
            'Source': self.Source,
            'Type': self.Function,
            'Args': [arg['value'] for arg in self.Arguments]
        }

    def __str__(self):
        return self.Source

    __repr__ = __str__


def evaluate_function(pf: ParsedFunction, loc=None):
    args = pf.get_arguments(loc)
    src_arg = [('{}={}'.format(arg['key'], arg['value']) if 'key' in arg else str(arg['value'])) for arg in args]
    # src = '{}({})'.format(pf.Function, ', '.join(src_arg))
    return EvaluatedFunction(pf.Source, pf.Function, args)


if __name__ == '__main__':

    print('Find parents')
    print(parse_parents('x+y/2 * max(z, 5)'), '\n')

    print('Math expression')
    me = parse_math_expression('x+y/2 * max(z, 5)')
    print(me)
    print(me({'x': 2, 'y': 4, 'z': 10}), '\n')

    fn = parse_function('my_func(4*a, "k", k, t=5, s=False)')
    print(fn)
    print(fn.to_json())
    print(fn.to_json({'k': 7}))
    print(fn.to_json({'k': 7, 'a': 10}))

    fn = evaluate_function(fn, {'k': 7, 'a': 10})
    print(fn.to_json(), '\n')

    print(resample([0, -1.2, -1], ['a', 'b', 'c']))
