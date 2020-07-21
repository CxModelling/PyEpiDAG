from epidag.fitting.alg.fitter import EvolutionaryFitter
from epidag.fitting.alg.genetic import *
from epidag.util import resample


class GA(EvolutionaryFitter):
    DefaultParameters = {
        'n_population': 100,
        'n_generation': 20,
        'p_mutation': 0.1,
        'p_crossover': 0.1,
        'target': 'MLE'
    }

    def __init__(self, model):
        EvolutionaryFitter.__init__(self, model)
        self.Population = list()
        self.p_mutation = GA.DefaultParameters['p_mutation']
        self.p_crossover = GA.DefaultParameters['p_crossover']
        self.n_cycle = GA.DefaultParameters['n_cycle']

        self.Moveable = self.Model.get_movable_nodes()
        self.Mutators = list()
        self.Crossover = AverageCrossover([d['Name'] for d in self.Moveable])
        self.Target = GA.DefaultParameters['target']

        for d in self.Moveable:
            loci, lo, up = d['Name'], d['Lower'], d['Upper']
            if d['Type'] is 'Double':
                self.Mutators.append(DoubleMutator(loci, lo, up))
            elif d['Type'] is 'Integer':
                self.Mutators.append(IntegerMutator(loci, lo, up))
            elif d['Type'] is 'Binary':
                self.Mutators.append(BinaryMutator(loci))

        self.Series = list()
        self.Generation = 0
        self.Stay = 0
        self.MaxFitness = -float('inf')
        self.MeanFitness = -float('inf')

    def initialise(self):
        self.Population = list()
        self.Series = list()
        self.Generation = 0
        self.MaxFitness = -float('inf')
        self.MeanFitness = -float('inf')

    def fit(self, niter, **kwargs):
        self.info('Initialising')
        if 'n_cycle' in kwargs:
            self.n_cycle = int(kwargs['n_cycle'])
        if 'p_mutation' in kwargs:
            self.p_mutation = kwargs['p_mutation']
        if 'p_crossover' in kwargs:
            self.p_mutation = kwargs['p_crossover']
        if 'target' in kwargs:
            self.Target = 'MLE' if kwargs['target'] == 'MLE' else 'MAP'

        self.initialise()
        self.__genesis(niter)
        for i in range(self.n_cycle):
            self.Generation += 1
            self.__crossover()
            self.__mutation()
            self.__selection()
            self.__find_elitism()
            if self.__termination():
                break

        self.info('Finished')

    def update(self, n, **kwargs):
        self.info('Updating')
        self.Stay = 0
        for i in range(n):
            self.Generation += 1
            self.__crossover()
            self.__mutation()
            self.__selection()
            self.__find_elitism()
            if self.__termination():
                break

        self.info('Finished')

    def __genesis(self, n):
        for _ in range(n):
            p = self.Model.sample_prior()
            p.LogLikelihood = self.Model.evaluate_likelihood(p)
            self.Population.append(p)

    def __crossover(self):
        pop = self.Population
        n = len(pop)
        sel = rd.binomial(1, self.p_crossover, int(n / 2))

        for i, s in enumerate(sel):
            if s:
                p1, p2 = self.Crossover.crossover(pop[i * 2], pop[i * 2 + 1], self.Model.BN)
                self.Model.evaluate_prior(p1)
                self.Model.evaluate_prior(p2)
                p1.LogLikelihood = self.Model.evaluate_likelihood(p1)
                p2.LogLikelihood = self.Model.evaluate_likelihood(p2)
                pop[i * 2], pop[i * 2 + 1] = p1, p2

    def __mutation(self):
        for node, mut in zip(self.Moveable, self.Mutators):
            i = node['Name']
            vs = [gene[i] for gene in self.Population]
            mut.set_scale(vs)

        pop = self.Population
        n = len(pop)
        sel = rd.binomial(1, self.p_mutation, n)

        for i, s in enumerate(sel):
            if s:
                p = pop[i] = pop[i].clone()
                loc = dict()
                for mut in self.Mutators:
                    loc[mut.Name] = mut.proposal(p[mut.Name])
                p.impulse(loc, self.Model.BN)
                self.Model.evaluate_prior(p)
                p.LogLikelihood = self.Model.evaluate_likelihood(p)

    def __selection(self):
        for p in self.Population:
            if p.LogLikelihood is 0:
                p.LogLikelihood = self.Model.evaluate_likelihood(p)

        if self.Target == 'MAP':
            wts = [p.LogPosterior for p in self.Population]
        else:
            wts = [p.LogLikelihood for p in self.Population]
        pop, mean = resample(wts, self.Population)
        self.Population = [p.clone() for p in pop]
        self.MeanFitness = mean

    def __find_elitism(self):
        if self.Target == 'MAP':
            self.BestFit = max(self.Population, key=lambda x: x.LogPosterior)
        else:
            self.BestFit = max(self.Population, key=lambda x: x.LogLikelihood)

        fitness = self.BestFit.LogPosterior if self.Target == 'MAP' else self.BestFit.LogLikelihood

        if fitness == self.MaxFitness:
            self.Stay += 1

        self.MaxFitness = fitness
        self.Series.append({
            'Generation': self.Generation,
            'Max fitness': self.MaxFitness,
            'Mean fitness': self.MeanFitness
        })
        self.info('Generation: {}, Mean fitness: {:.2E}, Max fitness: {:.2E}'.format(
            self.Generation, self.MeanFitness, self.MaxFitness))

    def __termination(self):
        if self.Stay > 5:
            return True

    def summarise_fitness(self):
        print('Model:', self.Model)
        print('Target:', self.Target)
        print('Best fitting', self.BestFit)
        print('Max fitness', self.MaxFitness)
