import logging
import pandas as pd

__author__ = 'TimeWz667'
__all__ = ['Monitor']


class Monitor:
    def __init__(self, name):
        self.Title = name
        self.Logger = logging.getLogger(name)
        self.Logger.setLevel(logging.INFO)
        self.Records = []
        self.Time = 0
        self.Last = dict()

    def info(self, msg, *arg, **kwargs):
        self.Logger.info(msg, *arg, **kwargs)

    def warning(self, msg, *arg, **kwargs):
        self.Logger.warning(msg, *arg, **kwargs)

    def error(self, msg, *arg, **kwargs):
        self.Logger.error(msg, *arg, **kwargs)

    def set_log_path(self, filename):
        fhl = logging.FileHandler(filename)
        self.add_handler(fhl)

    def add_handler(self, handler):
        if not handler.formatter:
            handler.setFormatter(
                logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                  '%d-%m-%Y %H:%M:%S'))
        self.Logger.addHandler(handler)

    def __getitem__(self, item):
        return self.Last[item]

    def reset(self, time: int=0):
        self.Time = time
        self.Records.clear()
        self.Last = dict()

    def step(self, time=None):
        time = time if time else self.Time + 1
        if time < self.Time:
            raise KeyError('Backward time specified')
        elif time == self.Time:
            return

        self.Last['Time'] = self.Time
        self.Records.append(self.Last)
        self.Time = time
        self.Last = dict()
        self.Logger.info('Step to {}'.format(self.Time))

    def keep(self, **kwargs):
        self.Last.update(kwargs)

    @property
    def Trajectories(self):
        dat = pd.DataFrame(self.Records)
        return dat.set_index('Time')

    def save_trajectories(self, filename):
        self.Trajectories.to_csv(filename)


if __name__ == '__main__':
    mon = Monitor('Test')
    mon.add_handler(logging.StreamHandler())

    mon.keep(Size=4)
    mon.step()
    mon.keep(Size=6)
    mon.step()
    print(mon.Trajectories)
