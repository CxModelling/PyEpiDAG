from abc import ABCMeta, abstractmethod
from epidag.bayesnet import parse_distribution, MATH_FUNC
import ast

__author__ = 'TimeWz667'
__all__ = ['ValueLoci', 'ExoValueLoci', 'DistributionLoci', 'FunctionLoci', 'PseudoLoci',
           'parse_parents']


def parse_parents(seq):
    sen = ast.parse(seq)
    v, f = set(), set()

    for s in ast.walk(sen):
        if isinstance(s, ast.Name):
            v.add(s.id)
        elif isinstance(s, ast.Call):
            f.add(s.func.id)
    return v - f, f


class Loci(metaclass=ABCMeta):
    def __init__(self, name):
        self.Name = name

    def __call__(self, parent=None):
        return self.sample(parent)

    @abstractmethod
    def sample(self, parent=None):
        pass

    @abstractmethod
    def evaluate(self, parent=None):
        pass

    def fill(self, gene):
        gene[self.Name] = self.sample(gene)

    @property
    @abstractmethod
    def Parents(self):
        pass

    @property
    @abstractmethod
    def Definition(self):
        pass

    @abstractmethod
    def to_json(self):
        return {'Name': self.Name, 'Def': self.Definition}


class ValueLoci(Loci):
    def __init__(self, name, val):
        Loci.__init__(self, name)
        self.Value = eval(val, MATH_FUNC) if isinstance(val, str) else val

    @property
    def Parents(self):
        return list()

    @property
    def Definition(self):
        return self.Value

    def sample(self, pas=None):
        return self.Value

    def evaluate(self, pas=None):
        return 0

    def to_json(self):
        js = Loci.to_json(self)
        js['Type'] = 'Value'
        return js

    def __repr__(self):
        return '{} = {}'.format(self.Name, self.Value)

    __str__ = __repr__


class ExoValueLoci(Loci):
    def __init__(self, name):
        Loci.__init__(self, name)

    @property
    def Parents(self):
        return set()

    @property
    def Definition(self):
        return ''

    def sample(self, pas=None):
        try:
            return pas[self.Name]
        except TypeError:
            raise KeyError('Must have input value')
        except KeyError:
            raise KeyError('Exogenous variable not found')

    def evaluate(self, pas=None):
        return 0

    def to_json(self):
        return {'Name': self.Name, 'Type': 'ExoValue'}

    def __repr__(self):
        return self.Name

    __str__ = __repr__


class DistributionLoci(Loci):
    def __init__(self, name, val, pas=None):
        Loci.__init__(self, name)
        self.Func = val
        if pas:
            self.Parent = pas
        else:
            self.Parent, _ = parse_parents(val)

    @property
    def Parents(self):
        return self.Parent

    @property
    def Definition(self):
        return self.Func

    def get_distribution(self, pas):
        try:
            pas = dict(pas)
        except TypeError:
            pas = dict()
        return parse_distribution(self.Func, glo=MATH_FUNC, loc=pas)

    def sample(self, pas=None):
        return self.get_distribution(pas).sample()

    def fill(self, gene):
        gene[self.Name] = self.sample(gene.Locus)

    def evaluate(self, pas=None):
        return self.get_distribution(pas).logpdf(pas[self.Name])

    def to_json(self):
        js = Loci.to_json(self)
        js['Type'] = 'Distribution'
        js['Parents'] = list(self.Parents)
        return js

    def __repr__(self):
        return '{} ~ {}'.format(self.Name, self.Func)

    __str__ = __repr__


class FunctionLoci(Loci):
    def __init__(self, name, val, pas=None):
        Loci.__init__(self, name)
        self.Func = val
        if pas:
            self.Parent = pas
        else:
            self.Parent, _ = parse_parents(val)

    @property
    def Parents(self):
        return self.Parent

    @property
    def Definition(self):
        return self.Func

    def sample(self, pas=None):
        pas = dict(pas) if pas else dict()
        try:
            return eval(self.Func, MATH_FUNC, pas)
        except NameError:
            raise KeyError('Parent node not found')

    def evaluate(self, pas=None):
        return 0

    def to_json(self):
        js = Loci.to_json(self)
        js['Type'] = 'Function'
        js['Parents'] = list(self.Parents)
        return js

    def __repr__(self):
        return '{} = {}'.format(self.Name, self.Func)

    __str__ = __repr__


class PseudoLoci(Loci):
    def __init__(self, name, val, pas=None):
        Loci.__init__(self, name)

        if pas:
            self.Parent = pas
        else:
            self.Parent, _ = parse_parents(val)
        self.Func = 'f(' + ', '.join(self.Parents) + ')'

    @property
    def Parents(self):
        return self.Parent

    @property
    def Definition(self):
        return self.Func

    def sample(self, pas=None):
        raise AttributeError('Pseudo node can not be implemented')

    def evaluate(self, pas=None):
        raise AttributeError('Pseudo node can not be evaluated')

    def fill(self, gene):
        raise AttributeError('Pseudo node can not be implemented')

    def to_json(self):
        js = Loci.to_json(self)
        js['Type'] = 'Pseudo'
        js['Parents'] = list(self.Parents)
        return js

    def __repr__(self):
        return '{} = {}'.format(self.Name, self.Func)

    __str__ = __repr__


if __name__ == '__main__':
    d1 = '1/0.01 + exp(k) + u'
    d1 = FunctionLoci('s1', d1)
    print(d1.Parents)
    print(d1.sample({'k': 2, 'u': 5}))
    print(d1.to_json())

    d2 = 'gamma(cos(1/0.01), ss)'
    d2 = DistributionLoci('s2', d2)
    print(d2.Parents)
    print(d2.sample({'ss': 10, 'u': 5}))
    print(d2.to_json())

    d3 = '1/0.01 + exp(k) + u'
    d3 = PseudoLoci('s3', d3)
    print(d3.Parents)
    print(d3.Func)
    print(d3.to_json())
