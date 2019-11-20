from abc import ABCMeta, abstractmethod
from epidag.distribution import parse_distribution

__author__ = 'TimeWz667'
__all__ = ['LinearCombination', 'Regression', 'LinearRegression']


class Regressor(metaclass=ABCMeta):
    def __init__(self, name):
        self.Name = name

    @abstractmethod
    def effect(self, value):
        pass


class Boolean(Regressor):
    def __init__(self, js):
        Regressor.__init__(self, js['Name'])
        self.Coefficient = js['Value']

    def effect(self, value):
        return self.Coefficient if value else 0

    def to_json(self):
        return {
            'Name': self.Name,
            'Value': self.Coefficient,
            'Type': 'Boolean'
        }


class Continuous(Regressor):
    def __init__(self, js):
        Regressor.__init__(self, js['Name'])
        self.Coefficient = js['Value']

    def effect(self, value):
        return self.Coefficient*value

    def to_json(self):
        return {
            'Name': self.Name,
            'Value': self.Coefficient,
            'Type': 'Continuous'
        }


class Categorical(Regressor):
    def __init__(self, js):
        Regressor.__init__(self, js['Name'])
        self.Coefficients = js['Values']
        self.Coefficients = [float(n) for n in self.Coefficients]
        self.Labels = js['Labels']
        self.Reference = {l: i for i, l in enumerate(self.Labels)}[js['Ref']]

    def effect(self, value):
        return self.Coefficients[int(value)]

    def to_json(self):
        return {
            'Name': self.Name,
            'Values': self.Coefficients,
            'Labels': self.Labels,
            'Ref': self.Labels[self.Reference],
            'Type': 'Categorical'
        }


class LinearCombination:
    def __init__(self, js):
        self.Regressors = list()
        for ent in js:
            self.__append_regressor(ent)

    def __append_regressor(self, ent):
        tp = ent['Type']
        if tp == 'Boolean':
            self.Regressors.append(Boolean(ent))
        elif tp == 'Categorical':
            self.Regressors.append(Categorical(ent))
        else:
            self.Regressors.append(Continuous(ent))

    def predict(self, kvs):
        es = [v.effect(kvs[v.Name]) for v in self.Regressors]
        return sum(es)

    def list_regressors(self):
        return [v.Name for v in self.Regressors]

    def __str__(self):
        return '~' + '+'.join(self.list_regressors())


class Regression(metaclass=ABCMeta):
    @abstractmethod
    def get_variable_type(self):
        pass

    @abstractmethod
    def expectation(self, xs):
        pass

    @abstractmethod
    def predict(self, xs):
        pass

    @abstractmethod
    def get_sampler(self, xs):
        pass


class LinearRegression(Regression):
    def __init__(self, js):
        self.Intercept = js['Intercept']
        self.LC = LinearCombination(js['Regressors'])
        self.Error = js['SE']

    def get_variable_type(self):
        return 'Double'

    def expectation(self, xs):
        return self.LC.predict(xs) + self.Intercept

    def predict(self, xs, n=1):
        n = max(n, 1)
        return self.get_sampler(xs).sample(n)

    def get_sampler(self, xs):
        return parse_distribution('norm(mu, err)', {'mu': self.expectation(xs), 'err': self.Error})

    def __str__(self):
        return 'y{}+{}'.format(str(self.LC), self.Intercept)


if __name__ == '__main__':
    reg = [
        {'Name': 'Age', 'Type': 'Continuous', 'Value': 5},
        {'Name': 'Male', 'Type': 'Boolean', 'Value': 5},
    ]

    lc = LinearCombination(reg)

    case1 = {'Age': 5, 'Male': True}
    case2 = {'Age': 2, 'Male': False}
    print(lc)
    print(lc.predict(case1))
    print(lc.predict(case2))

    lm = LinearRegression({
        'Intercept': 0.5,
        'SE': 2.5,
        'Regressors': reg
    })

    print(lm)
    print(lm.expectation(case1))
    print(lm.predict(case1, 1000).mean())

    sampler = lm.get_sampler(case1)
    print(sampler)
    print(sampler.sample(1000).mean())
