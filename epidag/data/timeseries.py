import numpy as np
from scipy.interpolate import interp1d
from .frame import AbsDataSet
from epidag.bayesnet.distribution import CategoricalRV

__author__ = 'TimeWz667'
__all__ = ['TimeSeries', 'TimeSeriesVector', 'TimeSeriesProbabilityTable', 'LeeCarter']


class TimeSeries(AbsDataSet):
    def __init__(self, mat, i_time, i_x, kind='nearest'):
        AbsDataSet.__init__(self, mat)
        ts = np.array(mat[i_time])
        xs = np.array(mat[i_x])
        self.IndexTime = i_time
        self.IndexX = i_x
        self.Line = interp1d(x=ts, y=xs, kind=kind, bounds_error=False, fill_value=(xs[0], xs[-1]))

    def __call__(self, t):
        return self.Line(t)

    def __repr__(self):
        return 'TimeSeries: {} ~ {}'.format(self.IndexX, self.IndexTime)


class TimeSeriesVector(AbsDataSet):
    def __init__(self, mat, i_time, i_xs, kind='nearest'):
        AbsDataSet.__init__(self, mat)
        self.IndexTime = i_time
        self.IndexXs = i_xs
        ts = np.array(mat[i_time])
        self.Lines = dict()

        for i_x in self.IndexXs:
            xs = np.array(mat[i_x])
            self.Lines[i_x] = interp1d(x=ts, y=xs, kind=kind, bounds_error=False, fill_value=(xs[0], xs[-1]))

    def __call__(self, t):
        return {x: self.Lines[x](t) for x in self.IndexXs}

    def __repr__(self):
        return 'TimeSeriesVector: {} ~ {}'.format(', '.join(self.IndexXs), self.IndexTime)


class TimeSeriesProbabilityTable(AbsDataSet):
    def __init__(self, mat, i_time, i_xs):
        AbsDataSet.__init__(self, mat)
        self.IndexTime = i_time
        self.IndexXs = i_xs
        ts = np.array(mat[i_time])
        self.Ts = interp1d(x=ts, y=ts, kind='nearest', bounds_error=False, fill_value=(ts[0], ts[-1]))
        self.PTs = dict()

        for i, ir in mat.iterrows():
            ps = {x: ir[x] for x in i_xs}
            ti = ir[i_time]
            self.PTs[ti] = CategoricalRV(ps)

    def __call__(self, t):
        ti = self.Ts(t) + 0
        return self.PTs[ti]()

    def __repr__(self):
        return 'TimeSeriesProbabilityTable: {} ~ {}'.format(', '.join(self.IndexXs), self.IndexTime)


class LeeCarter(AbsDataSet):
    def __init__(self, mat_t, mat_a, i_time='Time',
                 i_age='Age', i_al='Alpha', i_be='Beta', i_ka='Kappa', kind='nearest'):
        AbsDataSet.__init__(self, mat_t)

        self.Times = np.array(mat_t[i_time])
        self.Kappa = np.array(mat_t[i_ka])
        self.Ages = list(mat_a[i_age])
        self.Alpha = {ent[i_age]: ent[i_al] for _, ent in mat_a.iterrows()}
        self.Beta = {ent[i_age]: ent[i_be] for _, ent in mat_a.iterrows()}
        self.KappaFn = interp1d(x=self.Times, y=self.Kappa, kind=kind,
                                bounds_error=False, fill_value=(self.Kappa[0], self.Kappa[-1]))

    def __call__(self, t):
        return {a: self.get_rate(t, a) for a in self.Ages}

    def get_rate(self, t, a):
        mu = self.Alpha[a] + self.KappaFn(t) * self.Beta[a]
        return np.exp(mu)

    def __repr__(self):
        return 'Lee Carter'
