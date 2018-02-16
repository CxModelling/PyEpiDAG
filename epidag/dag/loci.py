from abc import ABCMeta, abstractmethod
import re
from epidag.distribution import parse_distribution

__author__ = 'TimeWz667'


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
        self.Value = eval(val) if isinstance(val, str) else val

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


class DistributionLoci(Loci):
    def __init__(self, name, val, pas=None):
        self.Name = name
        self.Func = val
        self.Parent = pas if pas else DistributionLoci.get_parents(self.Func)

    @property
    def Parents(self):
        return self.Parent

    def get_distribution(self, pas):
        fun = self.Func
        for p in self.Parent:
            fun = re.sub(r'\b{}\b'.format(p), str(pas[p]), fun)
        return parse_distribution(fun)

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

    @staticmethod
    def get_parents(expr):
        di = re.sub(r'\s*', '', expr)
        di = di.split('(', 1)[1][:-1]
        di = di.split(',')
        pa = list()
        for d in di:
            va = None
            while not va:
                try:
                    va = eval(d)
                except NameError as e:
                    los = re.match(r"name '(\w+)'", e.args[0])
                    los = los.group(1)
                    exec('{} = 0.87'.format(los))
                    if los not in pa:
                        pa.append(los)
                except SyntaxError as e:
                    raise e
        return pa


class FunctionLoci(Loci):
    def __init__(self, name, val, pas=None):
        self.Name = name
        self.Func = val
        self.Parent = pas if pas else FunctionLoci.get_parents(self.Func)

    @property
    def Parents(self):
        return self.Parent

    def sample(self, pas=None):
        fun = self.Func
        for p in self.Parent:
            fun = re.sub(r'\b{}\b'.format(p), str(pas[p]), fun)
        return eval(fun)

    def evaluate(self, pas=None):
        return 0

    def fill(self, gene):
        gene[self.Name] = self.sample(gene.Locus)

    def to_json(self):
        return {'Type': 'Function', 'Def': self.Func, 'Parents': self.Parents}

    def __repr__(self):
        return '{} = {}'.format(self.Name, self.Func)

    __str__ = __repr__

    @staticmethod
    def get_parents(expr):
        va = None
        pa = list()
        while not va:
            try:
                va = eval(expr)
            except NameError as e:
                los = re.match(r"name '(\w+)'", e.args[0])
                los = los.group(1)
                exec('{} = 0.87'.format(los))
                pa.append(los)
            except SyntaxError as e:
                raise e
        return pa


if __name__ == '__main__':
    d1 = '1/0.01 + k / u'
    d1 = FunctionLoci('ss', d1)
    print(d1.Parents)
    print(d1.sample({'k': 50, 'u': 5}))
    print(d1.to_json())

    d2 = 'gamma(1/0.01, ss)'
    d2 = DistributionLoci('sss', d2)
    print(d2.Parents)
    print(d2.sample({'ss': 10, 'u': 5}))
    print(d2.to_json())
