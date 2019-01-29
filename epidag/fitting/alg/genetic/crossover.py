from abc import ABCMeta, abstractmethod
import numpy.random as rd


class AbsCrossover(metaclass=ABCMeta):
    @abstractmethod
    def crossover(self, p1, p2, bn):
        pass


class AverageCrossover(AbsCrossover):
    def crossover(self, p1, p2, bn):
        pco = p1.clone()

        locus = dict()
        for root in bn.Roots:
            if bn.is_rv(root):
                locus[root] = (p1[root] + p2[root]) / 2
        pco.impulse(locus, bn)
        return [pco, pco.clone()]


class ShuffleCrossover(AbsCrossover):
    def crossover(self, p1, p2, bn):
        po1 = p1.clone()
        po2 = p2.clone()

        locus1, locus2 = dict(), dict()
        for root in bn.Roots:
            if bn.is_rv(root) and rd.random() < 0.5:
                locus1[root] = p2[root]
                locus2[root] = p1[root]

        if locus1:
            po1.impulse(locus1, bn)
            po2.impulse(locus2, bn)
        return [po1, po2]
