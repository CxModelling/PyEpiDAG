from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as rd
import scipy.stats as sts


class AbsMutator(metaclass=ABCMeta):
    def __init__(self, name, lo, up):
        self.Name = name
        self.Lower = lo
        self.Upper = up
        self.Scale = 1

    @abstractmethod
    def set_scale(self, vs):
        pass

    @abstractmethod
    def proposal(self, v):
        pass

    @abstractmethod
    def kernel(self, v1, v2):
        pass


class BinaryMutator(AbsMutator):
    def __init__(self, name):
        AbsMutator.__init__(self, name, 0, 1)

    def set_scale(self, vs):
        self.Scale = np.mean(vs)
        self.Scale = max(min(self.Scale, 1), 0)

    def proposal(self, v):
        return 1 if rd.random(1) < self.Scale else 0

    def kernel(self, v1, v2):
        if v2 is 1:
            return self.Scale
        else:
            return 1 - self.Scale


class DoubleMutator(AbsMutator):
    def __init__(self, name, lo, up):
        AbsMutator.__init__(self, name, lo, up)

    def set_scale(self, vs):
        x = np.array(vs)
        hi = x.std()
        lo = min(sts.iqr(x), sts.iqr(x) / 1.34)
        if not lo:
            lo = hi if hi else 1

        self.Scale = 0.9 * lo * np.power(len(x), -0.2)

    def proposal(self, v):
        v1 = rd.normal(v, self.Scale)
        return max(min(v1, 1), 0)

    def kernel(self, v1, v2):
        return sts.norm.pdf(v1, v2, self.Scale)


class IntegerMutator(AbsMutator):
    def __init__(self, name, lo, up):
        AbsMutator.__init__(self, name, lo, up)

    def set_scale(self, vs):
        x = np.array(vs)
        hi = x.std()
        lo = min(sts.iqr(x), sts.iqr(x) / 1.34)
        if not lo:
            lo = hi if hi else 1

        self.Scale = 0.9 * lo * np.power(len(x), -0.2)

    def proposal(self, v):
        v1 = round(rd.normal(v, self.Scale))
        return max(min(v1, 1), 0)

    def kernel(self, v1, v2):
        return sts.norm.pdf(v1, v2, self.Scale)


if __name__ == '__main__':
    xs = [0.1, 0.2, 0.3, 0.4]

    mut = DoubleMutator('X', 0, 1)
    mut.set_scale(xs)
    print(mut.Scale)
    print(xs)
    print([mut.proposal(x) for x in xs])

    xs = [0, 1, 1, 0, 1]

    mut = BinaryMutator('X')
    mut.set_scale(xs)
    print(mut.Scale)
    print(xs)
    print([mut.proposal(x) for x in xs])
