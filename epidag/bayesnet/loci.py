from abc import ABCMeta, abstractmethod
from epidag.bayesnet import parse_distribution, MATH_FUNC
import ast

__author__ = 'TimeWz667'
__all__ = ['ValueLoci', 'ExoValueLoci', 'DistributionLoci', 'FunctionLoci',
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
    @abstractmethod
    def sample(self, parent=None):
        pass

    @abstractmethod
    def evaluate(self, parent=None):
        pass

    @abstractmethod
    def fill(self, gene):
        pass

    @property
    @abstractmethod
    def Parents(self):
        pass

    @abstractmethod
    def to_json(self):
        pass


class ValueLoci(Loci):
    def __init__(self, name, val):
        self.Name = name
        self.Value = eval(val, MATH_FUNC) if isinstance(val, str) else val

    @property
    def Parents(self):
        return list()

    def sample(self, pas=None):
        return self.Value

    def evaluate(self, pas=None):
        return 0

    def fill(self, gene):
        gene[self.Name] = self.sample(gene.Locus)

    def to_json(self):
        return {'Type': 'Value', 'Def': self.Value}

    def __repr__(self):
        return '{} = {}'.format(self.Name, self.Value)

    __str__ = __repr__


class ExoValueLoci(Loci):
    def __init__(self, name):
        self.Name = name

    @property
    def Parents(self):
        return set()

    def sample(self, pas=None):
        return pas[self.Name]

    def evaluate(self, pas=None):
        return 0

    def fill(self, gene):
        gene[self.Name] = self.sample(gene.Locus)

    def to_json(self):
        return {'Type': 'ExoValue'}

    def __repr__(self):
        return self.Name

    __str__ = __repr__


class DistributionLoci(Loci):
    def __init__(self, name, val, pas=None):
        self.Name = name
        self.Func = val
        if pas:
            self.Parent = pas
        else:
            self.Parent, _ = parse_parents(val)

    @property
    def Parents(self):
        return self.Parent

    def get_distribution(self, pas):
        dd = dict(pas)
        return parse_distribution(self.Func, glo=MATH_FUNC, loc=dict(pas))

    def sample(self, pas=None):
        return self.get_distribution(pas).sample()

    def fill(self, gene):
        gene[self.Name] = self.sample(gene.Locus)

    def evaluate(self, pas=None):
        return self.get_distribution(pas).logpdf(pas[self.Name])

    def to_json(self):
        return {'Type': 'Distribution', 'Def': self.Func, 'Parents': self.Parents}

    def __repr__(self):
        return '{} ~ {}'.format(self.Name, self.Func)

    __str__ = __repr__


class FunctionLoci(Loci):
    def __init__(self, name, val, pas=None):
        self.Name = name
        self.Func = val
        if pas:
            self.Parent = pas
        else:
            self.Parent, _ = parse_parents(val)

    @property
    def Parents(self):
        return self.Parent

    def sample(self, pas=None):
        try:
            return eval(self.Func, MATH_FUNC, pas)
        except NameError:
            return eval(self.Func, MATH_FUNC, dict(pas))

    def evaluate(self, pas=None):
        return 0

    def fill(self, gene):
        gene[self.Name] = self.sample(gene.Locus)

    def to_json(self):
        return {'Type': 'Function', 'Def': self.Func, 'Parents': self.Parents}

    def __repr__(self):
        return '{} = {}'.format(self.Name, self.Func)

    __str__ = __repr__


if __name__ == '__main__':
    d1 = '1/0.01 + exp(k) + u'
    d1 = FunctionLoci('ss', d1)
    print(d1.Parents)
    print(d1.sample({'k': 2, 'u': 5}))
    print(d1.to_json())

    d2 = 'gamma(cos(1/0.01), ss)'
    d2 = DistributionLoci('sss', d2)
    print(d2.Parents)
    print(d2.sample({'ss': 10, 'u': 5}))
    print(d2.to_json())
