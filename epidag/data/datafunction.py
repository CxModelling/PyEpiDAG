from abc import ABCMeta, abstractmethod

__author__ = 'TimeWz667'


class AbsDataFunction(metaclass=ABCMeta):
    def __init__(self, name, cols, df):
        self.Name = name
        self.Selectors = cols
        self.RawData = df

    @property
    def Type(self):
        return str(type(self))

    @property
    def NIndices(self):
        return len(self.Selectors)

    @abstractmethod
    def getSampler(self, **kwargs):
        pass

    def to_json(self):
        js = dict()
        js['Name'] = self.Name
        js['Type'] = self.Type
        js['Selectors'] = list(self.Selectors)
        js['RawData'] = self.RawData
