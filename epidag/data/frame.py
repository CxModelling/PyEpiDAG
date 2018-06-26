from abc import ABCMeta, abstractmethod

__author__ = 'TimeWz667'
__all__ = ['AbsDataSet']


class AbsDataSet(metaclass=ABCMeta):
    def __init__(self, mat):
        self.SourceMatrix = mat

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    def sample(self, n=1, **kwargs):
        if n == 1:
            return self(**kwargs)
        else:
            return [self(**kwargs) for _ in range(n)]

    def __str__(self):
        return str(self.SourceMatrix)
