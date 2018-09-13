from abc import ABCMeta, abstractmethod
import re
from epidag import MATH_FUNC, parse_math_expression, parse_function, ScriptException
from epidag.bayesnet.distribution import parse_distribution

__author__ = 'TimeWz667'
__all__ = ['ValueLoci', 'ExoValueLoci', 'DistributionLoci', 'FunctionLoci', 'PseudoLoci',
           'loci_from_json', 'parse_loci']


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
        self.Func = parse_function(val)
        self.__parents = pas if pas else self.Func.Parents

    @property
    def Parents(self):
        return self.__parents

    @property
    def Definition(self):
        return self.Func.Source

    def get_distribution(self, pas=None):
        loc = {pa: pas[pa] for pa in self.Parents}
        return parse_distribution(self.Func, loc=loc)

    def sample(self, pas=None):
        return self.get_distribution(pas).sample()

    def fill(self, gene):
        gene[self.Name] = self.sample(gene)

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
        self.Func = parse_math_expression(val)
        self.__parents = pas if pas else self.Func.Parents

    @property
    def Parents(self):
        return self.__parents

    @property
    def Definition(self):
        return self.Func.Expression

    def sample(self, pas=None):
        try:
            loc = {pa: pas[pa] for pa in self.Parents}
            return self.Func.execute(loc)
        except (NameError, KeyError) as e:
            raise KeyError('Exogenous variable {} should be defined'.format(e.args[0]))

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
        val = parse_function(val)
        self.__parents = pas if pas else val.Parents
        self.Func = 'f(' + ', '.join(self.Parents) + ')'

    @property
    def Parents(self):
        return self.__parents

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


def loci_from_json(js):
    name, tp = js['Name'], js['Type']
    if tp == 'Value':
        return ValueLoci(name, js['Def'])
    elif tp == 'ExoValue':
        return ExoValueLoci(name)
    elif tp == 'Function':
        return FunctionLoci(name, js['Def'], js['Parents'])
    elif tp == 'Distribution':
        return DistributionLoci(name, js['Def'], js['Parents'])
    elif tp == 'Pseudo':
        return PseudoLoci(name, js['Def'], js['Parents'])
    else:
        raise AttributeError('Unknown format')


def parse_loci(df):
    df = df.replace(' ', '')

    if '#' in df:
        df, des = re.match(r'\A(\S+)#(\S+)', df).groups()
    else:
        des = ''

    if '~' in df:
        name, loci = re.match(r'\A(\w+)~(\S+)', df).groups()
        loci = DistributionLoci(name, loci)
    elif re.match(r'\A\w+=\S+', df):
        name, loci = re.match(r'\A(\w+)=(\S+)', df).groups()
        if loci.startswith('f('):
            loci = PseudoLoci(name, loci)
        else:
            try:
                loci = ValueLoci(name, loci)
            except NameError:
                loci = FunctionLoci(name, loci)
    elif re.match(r'\A\w+\Z', df):
        loci = ExoValueLoci(df)
    else:
        raise ScriptException('Ill-defined variable')
    return loci, des


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
    print(loci_from_json(d3.to_json()))

    lc, description = parse_loci(df='x ~ binom(prob=0.5, size=5) # Distribution')
    print(lc, description)

    lc, description = parse_loci(df='x = exp(3*k)')
    print(lc, lc.Parents, description)

    lc, description = parse_loci(df='x = f(y, z) # Pseudo function')
    print(lc, lc.Parents, description)

    lc, description = parse_loci(df='x # Exogenous Variable')
    print(lc, description)

    lc, description = parse_loci(df='x = 5 # Single Value Variable')
    print(lc, description)
