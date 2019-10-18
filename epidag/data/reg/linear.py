from abc import ABCMeta, abstractmethod

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
