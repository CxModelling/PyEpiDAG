from abc import ABCMeta, abstractmethod
import numpy.random as rd


class AbsCrossover(metaclass=ABCMeta):
    def __init__(self, nodes):
        self.Nodes = nodes

    @abstractmethod
    def crossover(self, p1, p2, bn):
        pass


class AverageCrossover(AbsCrossover):
    def crossover(self, p1, p2, bn):
        pco = p1.clone()

        locus = dict()
        for node in self.Nodes:
            locus[node] = (p1[node] + p2[node]) / 2
        pco.impulse(locus, bn)
        return [pco, pco.clone()]


class ShuffleCrossover(AbsCrossover):
    def crossover(self, p1, p2, bn):
        po1 = p1.clone()
        po2 = p2.clone()

        locus1, locus2 = dict(), dict()
        for node in self.Nodes:
            if rd.random() < 0.5:
                locus1[node] = p2[node]
                locus2[node] = p1[node]

        if locus1:
            po1.impulse(locus1, bn)
            po2.impulse(locus2, bn)
        return [po1, po2]
