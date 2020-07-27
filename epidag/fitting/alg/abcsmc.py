from epidag.fitting import BayesResult
from epidag.fitting.alg.fitter import Fitter
import numpy as np
import numpy.random as rd
from abc import ABCMeta, abstractmethod


__author__ = 'TimeWz667'
__all__ = ['ABCSMC']


class AbsStepper(metaclass=ABCMeta):
    def __init__(self, name, lo, up):
        self.Name = name
        self.Lower, self.Upper = lo, up

    @abstractmethod
    def propose(self, v, scale):
        pass


class BinaryStepper(AbsStepper):
    def propose(self, v, scale):
        return v


class DoubleStepper(AbsStepper):
    def propose(self, v, scale):
        return v + rd.normal() * scale


class IntegerStepper(AbsStepper):
    def propose(self, v, scale):
        return np.round(rd.normal(v, scale))


class Steppers:
    def __init__(self, model):
        self.Nodes = dict()

        temp = model.sample_prior()
        for p in model.ParameterNodes:
            pp = model.BN[p].get_distribution(temp)
            if pp.Type == 'Binary':
                step = BinaryStepper(p, 0, 1)
            elif pp.Type == 'Integer':
                step = IntegerStepper(p, pp.Lower, pp.Upper)
            else:
                step = DoubleStepper(p, pp.Lower, pp.Upper)
            self.Nodes[p] = step

    def mutate(self, model, gene, scales):
        new = gene.clone()

        mutated = dict()
        for key, node in self.Nodes.items():
            while True:
                proposed = node.propose(new[key], scales[key])
                if node.Lower < proposed < node.Upper:
                    mutated[key] = proposed
                    break
        new.impulse(mutated)
        new.LogPrior = model.evaluate_prior(new)
        return new

def wt_sd(vs, wts):
    vs = np.array(vs)
    wts = wts / wts.sum()
    mu = (wts * vs) / len(wts)

    return np.sqrt(np.sum(wts * (vs - mu) * (vs - mu)))


class ABCSMC(Fitter):
    def __init__(self, name_logger="ABCSMC", alpha=0.9, p_thres=0.7):
        Fitter.__init__(self, name_logger, alpha=alpha, p_thres=p_thres)

    def fit(self, model, **kwargs):
        n_post = kwargs['n_post']

        max_stay = kwargs['max_stay'] if 'max_stay' in kwargs else 5
        max_round = kwargs['max_round'] if 'max_round' in kwargs else 20

        trj = list()
        n_round = n_stay = 0
        n_eval = n_post
        steppers = Steppers(model)

        self.info('Initialising')
        post, d0, wts, eps0 = self.__initialise(n_post, model)

        rec = {
            'Round': n_round,
            'Eval': n_eval,
            'Eps':eps0,
            'ESS': 1 / np.sum(wts * wts),
            'ACC': 1
        }

        self.info('Round {}, ESS {:0.0f}, Eps {:0.4g}, Acceptance {:.1f}%'.format(
            rec['Round'], rec['ESS'], rec['Eps'], rec['ACC'] * 100
        ))
        trj.append(rec)

        # Iteration
        while True:
            # Update eps
            n_round += 1
            eps1 = self.find_eps(d0, eps0)
            if eps1 > eps0:
                n_stay += 1
                eps1 = eps0
            else:
                n_stay = 0

            # Step 1 Updating weight
            act_np0, a, wts = self.__update_weights(d0, wts, eps0, eps1)

            # Step 2 Resampling
            post, d1, wts = self.__resample(post, d0, wts)

            # Step 3 MH stepping
            post, d1, n_eval, acc = self.__step_mh(post, wts, d0, eps1, n_eval, steppers, model)

            rec = {
                'Round': n_round,
                'Eval': n_eval,
                'Eps': eps1,
                'ESS': 1 / np.sum(wts * wts),
                'ACC': acc
            }
            trj.append(rec)
            self.info('Round {}, ESS {:0.4g}, Eps {:0.4g}, Acceptance {:.1f}%'.format(
                rec['Round'], rec['ESS'], rec['Eps'], rec['ACC'] * 100
            ))

            if n_round >= max_round or (n_stay >= max_stay and n_round > 3):
                break
            eps0, d0 = eps1, d1

        self.info('Completed')

        res = BayesResult(nodes=post, model=model, alg=self)
        res.Benchmarks.update(rec)
        res.Benchmarks['Niter'] = n_post
        return res

    def is_updatable(self):
        return True

    def update(self, res, **kwargs):
        pass

    def find_eps(self, ds, eps):
        e0 = sum(d for d in ds if d < eps) / len(ds)
        et = self.Parameters['alpha'] * e0
        ds = list(ds)
        ds.sort(key = lambda x : -x)
        for eps1 in ds:
            e1 = sum(d for d in ds if d < eps1) / len(ds)
            if e1 <= et:
                return eps1
        else:
            return eps

    def __initialise(self, n_post, model):
        post = list()
        while len(post) < n_post:
            p = model.sample_prior()
            p.LogPrior = model.evaluate_prior(p)
            di = model.evaluate_distance(p)
            if np.isfinite(di):
                p.LogLikelihood = - di
                post.append(p)

        d0 = [-p.LogLikelihood for p in post]

        wts = np.ones(n_post) / n_post

        eps0 = float('inf')
        return post, d0, wts, eps0

    def __update_weights(self, ds, wts, eps0, eps1):
        ds = np.array(ds)
        act_np0 = ds < eps0
        act_np1 = ds < eps1
        a = act_np0 > 0

        wts[a] *= act_np1[a] / act_np0[a]
        wts[1 - act_np0] = 0
        wts /= wts.sum()
        return act_np0, a, wts

    def __resample(self, post, ds, wts):
        n_post = len(post)
        ess_thres = len(post) * self.Parameters['p_thres']

        ds = np.array(ds)
        if ess_thres * sum(wts * wts) > 1:
            assert sum(wts > 0) > 2
            alive = wts > 0
            ind = [k for k, v in enumerate(alive) if v]
            re_ind = rd.choice(ind, size=n_post, replace=True, p=wts[wts > 0])

            post = [post[i].clone() for i in re_ind]
            d1 = ds[re_ind]
            wts = np.ones(n_post) / n_post
        else:
            d1 = ds

        return post, d1, wts

    def __step_mh(self, post, wts, ds, eps1, n_eval, steppers, model):
        n_post = len(post)
        tau = {p: wt_sd([d[p] for d in post], wts) for p in model.ParameterNodes}

        post_p = list(post)

        dp = np.zeros(n_post)

        for i in range(n_post):
            di = float('inf')
            pars = post[i]
            while np.isinf(di):
                pars = steppers.mutate(model, post[i], tau)
                di = model.evaluate_distance(pars)
                n_eval += 1

            pars.LogLikelihood = - di
            dp[i] = di

        act_npp = dp < eps1

        # MH acceptance ratio
        act_np0 = np.array(ds) < eps1
        alive = act_np0 > 0

        acc = np.zeros(n_post)
        acc[alive] = act_npp[alive] / act_np0[alive]

        # Update accepted proposals
        a = rd.random(n_post) < acc

        if a.sum() > 2:
            for i, inc in enumerate(a):
                if inc:
                    post[i] = post_p[i]
                    ds[i] = dp[i]

        acc = a.sum() / alive.sum()
        return post, ds, n_eval, acc
